#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rnn_shape_sweep_mackeyglass_superposition_agop.py
=================================================

Goal
----
Fixed-parameter GRU shape sweep on multi-step NARMA-30 forecasting. Tests the
AOFE hypothesis: under fixed N, different (depth, hidden_size) shapes reach
similar fitted loss, mediated by AOFE.

Task (NARMA-30 multi-step forecasting)
--------------------------------------
A 1-D nonlinear autoregressive benchmark driven by exogenous random inputs.

    input  : T=80 consecutive steps × 2 channels  (u_t, y_t)   [B, T, 2]
    output : next H=8 target values y_{t+1:t+H}                [B, H, 1]

Why replace Lorenz-63
---------------------
The previous Lorenz setup had become too easy at N≈1M: all shapes converged to
test MSE around 1e-6, collapsing the loss dynamic range. NARMA-30 is a classic
RNN benchmark with both long memory and multiplicative nonlinear interactions,
which makes shape differences visible while still being learnable.

Output-space AGOP (fixed dimension across all shapes)
------------------------------------------------------
    J = d(y_flat)/d(x_in_flat),   y_flat ∈ R^{H×1 = 8}
    AGOP = E_data[J J^T] ∈ R^{8×8}

With B=256, proj_samples=64: 256×64=16384 rank-1 outer products for
8×9/2=36 distinct AGOP entries → highly overdetermined with B=256 and 64 projections.

Training protocol (strict D=20N)
---------------------------------
Training stops at exactly cfg.steps steps — no extension, no patience early-stop.
The final model state (at D=20N) is evaluated and reported.

Data budget (D=20N)
-------------------
"Token" = one scalar prediction target:
    tokens_per_step = batch_size × horizon × 1
    D = steps × tokens_per_step  ≈  20 × N

Shapes
------
10 non-extreme GRU shapes (depth 3..12), hidden found by downward scan + padding.

Unified correlation metrics
---------------------------
  Pearson(AOFE=agop_offdiag_energy,  test_mse)    — raw MSE, no log
  Pearson(AOFE_ratio=agop_offdiag_ratio, test_mse) — raw MSE, no log

Loss curves (for appendix)
---------------------------
Per-shape curves saved to curves/ with log-scale y for readability.
"""

from __future__ import annotations

import os
import math
import csv
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------
# Reproducibility helpers
# -----------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def symmetrize_(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


def agop_offdiag_metrics(agop: torch.Tensor) -> Tuple[float, float]:
    """
    AOFE (AGOP Off-diagonal Frobenius Energy):
      offdiag_energy = ||AGOP||_F^2 - ||diag(AGOP)||_2^2
      offdiag_ratio  = offdiag_energy / ||AGOP||_F^2
    """
    agop = agop.float()
    fro2  = float((agop * agop).sum().item()) + 1e-12
    diag  = torch.diag(agop)
    diag2 = float((diag * diag).sum().item())
    offdiag = max(fro2 - diag2, 0.0)
    return offdiag, offdiag / fro2


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
    return float(np.dot(x, y) / den)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum   = np.cumsum(counts)
    starts = csum - counts + 1
    avg    = (starts + csum) / 2.0
    return avg[inv]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    return pearson_corr(rx, ry)


# -----------------------
# NARMA-30 generator
# -----------------------

def generate_narma30(
    *,
    total_len: int,
    order: int = 30,
    warmup: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a NARMA-30 sequence.
    Returns:
      u: [total_len, 1] exogenous drive in [0, 0.5]
      y: [total_len, 1] target series
    """
    rng = np.random.default_rng(int(seed))
    n_total = int(total_len) + int(warmup) + int(order) + 1
    u = rng.uniform(0.0, 0.5, size=n_total).astype(np.float64)
    y = np.zeros(n_total, dtype=np.float64)

    for t in range(order - 1, n_total - 1):
        window_sum = y[t - order + 1:t + 1].sum()
        y[t + 1] = (
            0.2 * y[t]
            + 0.04 * y[t] * window_sum
            + 1.5 * u[t - order + 1] * u[t]
            + 0.001
        )
        if not np.isfinite(y[t + 1]):
            raise FloatingPointError(f"Non-finite NARMA state at step {t+1}.")

    start = int(warmup) + int(order)
    u = u[start:start + int(total_len)].astype(np.float32).reshape(-1, 1)
    y = y[start:start + int(total_len)].astype(np.float32).reshape(-1, 1)
    return u, y


# -----------------------
# Dataset
# -----------------------

class NARMAHorizonDataset(torch.utils.data.Dataset):
    """
    Multi-step-ahead prediction for NARMA-30.

    Input:  T consecutive steps × 2 channels (u_t, y_t)  [T, 2]
    Output: next H target values y                       [H, 1]
    """
    def __init__(
        self,
        *,
        inputs: np.ndarray,   # shape [L, 2], float32
        targets: np.ndarray,  # shape [L, 1], float32
        seq_len: int,
        horizon: int,
        size: int,
        seed: int,
    ):
        super().__init__()
        self.inputs  = np.asarray(inputs, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.size    = int(size)

        L = int(self.inputs.shape[0])
        max_start = L - (self.seq_len + self.horizon)
        if max_start < 0:
            raise ValueError(
                f"Series too short: need at least {self.seq_len + self.horizon} steps, got {L}."
            )
        if self.size > max_start + 1:
            raise ValueError(
                f"Requested size={self.size} > unique windows={max_start + 1}. "
                "Increase series length or decrease size."
            )
        rng = np.random.default_rng(int(seed))
        self.starts = rng.integers(0, max_start + 1, size=self.size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        x = torch.from_numpy(self.inputs[s : s + self.seq_len].copy())  # [T, 2]
        y = torch.from_numpy(
            self.targets[s + self.seq_len : s + self.seq_len + self.horizon].copy()
        )  # [H, 1]
        return x, y


# -----------------------
# Model: stacked GRU with horizon head
# -----------------------

class TinyRNNRegressor(nn.Module):
    """
    Encoder-decoder GRU for multi-step-ahead prediction.

    Encoder: stacked GRU processes T-step 3-D input sequence.
    Decoder: linear head maps the last layer's final hidden state to H×3 predictions.

    This bottleneck design (hidden state = compressed trajectory representation)
    makes the model's capacity directly visible in the test MSE, creating
    clear shape-dependent performance differences at D=20N.
    """
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        depth: int,
        horizon: int,
        rnn_type: str = "gru",
        dropout: float = 0.0,
        pad_params: int = 0,
    ):
        super().__init__()
        self.input_size  = int(input_size)
        self.output_size = int(output_size)
        self.hidden_size = int(hidden_size)
        self.depth       = int(depth)
        self.horizon     = int(horizon)
        self.rnn_type    = str(rnn_type).lower()

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_size, hidden_size=self.hidden_size,
                num_layers=self.depth, batch_first=True,
                dropout=dropout if self.depth > 1 else 0.0, bidirectional=False,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.input_size, hidden_size=self.hidden_size,
                num_layers=self.depth, batch_first=True,
                dropout=dropout if self.depth > 1 else 0.0, bidirectional=False,
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=self.input_size, hidden_size=self.hidden_size,
                num_layers=self.depth, nonlinearity="tanh", batch_first=True,
                dropout=dropout if self.depth > 1 else 0.0, bidirectional=False,
            )
        else:
            raise ValueError("rnn_type must be one of: rnn, gru, lstm")

        # Horizon head: last-layer final hidden → H × output_size predictions
        self.head = nn.Linear(self.hidden_size, self.horizon * self.output_size, bias=True)

        self._pad_params = None
        if pad_params > 0:
            self._pad_params = nn.Parameter(torch.zeros(int(pad_params)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_size]
        returns: [B, horizon, output_size]
        """
        _, h_n = self.rnn(x)
        if self.rnn_type == "lstm":
            h_n = h_n[0]              # extract hidden state from (h, c) tuple
        h_last = h_n[-1]              # [B, hidden_size] — last layer, final step
        out = self.head(h_last)       # [B, horizon * output_size]
        return out.view(x.shape[0], self.horizon, self.output_size)  # [B, H, C]


# -----------------------
# AGOP estimation (wrt input sequence, output-space AGOP)
# -----------------------

def estimate_agop_wrt_inputs(
    model: TinyRNNRegressor,
    x: torch.Tensor,
    *,
    proj_samples: int = 64,
    max_agop_dim: int = 4096,
) -> torch.Tensor:
    """
    AGOP = E[J J^T],  J = d(y_flat)/d(x_in_flat).
    y_flat ∈ R^{H×C}  (horizon × input_size, e.g. 8×3=24).
    AGOP ∈ R^{H*C × H*C}  — fixed dimension across all model shapes.

    With B=256, proj_samples=64: ~16384 rank-1 outer products for
    a 24×24=576-element AGOP → ~55× overdetermined (excellent estimation).

    cuDNN must be disabled for JVP to work with stacked RNNs.
    model.train() is required for cuDNN RNN backward compatibility.
    """
    device = x.device
    model.train()   # cuDNN RNN requires train mode for JVP
    B, T, C = x.shape
    D_out = model.horizon * model.output_size

    if D_out > max_agop_dim:
        raise ValueError(f"AGOP dim H*C={D_out} > max_agop_dim={max_agop_dim}.")

    x0   = x.detach()
    agop = torch.zeros((D_out, D_out), device=device, dtype=torch.float32)

    def fwd(x_in: torch.Tensor) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            B_ = x_in.shape[0]
            return model(x_in).view(B_, -1)   # [B, H*C]

    for _ in range(int(proj_samples)):
        u = torch.randn_like(x0)
        _, Ju = torch.autograd.functional.jvp(fwd, (x0,), (u,), create_graph=False, strict=False)
        Ju = Ju.float()
        Ju = torch.nan_to_num(Ju, nan=0.0, posinf=0.0, neginf=0.0)
        agop = agop + (Ju.T @ Ju) / float(B)

    agop = agop / float(proj_samples)
    return symmetrize_(agop).detach()


# -----------------------
# Training
# -----------------------

@dataclass
class TrainCfg:
    lr: float = 3e-4
    weight_decay: float = 0.0
    steps: int = 0
    warmup_steps: int = 200
    batch_size: int = 64
    grad_clip: float = 1.0
    eval_every: int = 500

    data_ratio: float = 20.0
    input_size: int = 2          # NARMA inputs: (u_t, y_t)
    output_size: int = 1         # predict future y only
    seq_len: int = 80            # input context window length
    horizon: int = 8             # multi-step prediction horizon (H)
    train_size: int = 0
    val_size: int = 2000
    test_size: int = 2000

    narma_order: int = 30
    narma_warmup: int = 200
    train_series_len: int = 0
    val_series_len: int = 0
    test_series_len: int = 0

    target_params: int = 1_000_000
    depth_list: List[int] = None
    rnn_type: str = "gru"
    dropout: float = 0.0
    hidden_step: int = 16
    min_hidden: int = 112
    max_hidden: int = 1024
    max_padding_ratio: float = 0.15
    max_train_factor: float = 3.0
    fit_patience: int = 8

    agop_batch: int = 256
    agop_proj_samples: int = 64
    max_agop_dim: int = 4096

    seed: int = 0


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    t = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate_mse(
    model: TinyRNNRegressor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_se     = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)                                   # [B, H, C]
        se    = F.mse_loss(y_hat, y, reduction="sum")
        total_se     += float(se.item())
        total_tokens += int(y.numel())
    return total_se / max(1, total_tokens)


def train_one_model(
    model: TinyRNNRegressor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cfg: TrainCfg,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Train until validation MSE plateaus after the D≈20N budget is reached.
    Report the best validation-state checkpoint to stay near the fitted regime.
    """
    model.to(device)
    model.train()

    opt        = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_iter = iter(train_loader)

    t0        = time.time()
    history: List[Dict[str, float]] = []
    min_steps = int(cfg.steps)
    max_steps = max(min_steps, int(math.ceil(cfg.max_train_factor * cfg.steps)))
    best_val = float("inf")
    best_state = None
    stale_evals = 0

    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, max_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        y_hat = model(x)                                   # [B, H, C]
        loss  = F.mse_loss(y_hat, y, reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == max_steps:
            train_mse = evaluate_mse(model, train_loader, device, max_batches=50)
            val_mse   = evaluate_mse(model, val_loader,   device, max_batches=None)
            test_mse  = evaluate_mse(model, test_loader,  device, max_batches=None)
            model.train()    # restore after eval (cuDNN RNN)
            history.append({
                "step":      int(step + 1),
                "lr":        float(lr),
                "train_mse": float(train_mse),
                "val_mse":   float(val_mse),
                "test_mse":  float(test_mse),
            })
            dt  = time.time() - t0
            gap = test_mse - train_mse
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  test_mse={test_mse:.6f}  "
                f"rmse={math.sqrt(test_mse):.6f}  gap={gap:+.6f}  time={dt:.1f}s"
            )
            if val_mse + 1e-12 < best_val:
                best_val = float(val_mse)
                stale_evals = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_evals += 1
            if (step + 1) >= min_steps and stale_evals >= int(cfg.fit_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse = evaluate_mse(model, train_loader, device, max_batches=None)
    val_mse   = evaluate_mse(model, val_loader,   device, max_batches=None)
    test_mse  = evaluate_mse(model, test_loader,  device, max_batches=None)

    if test_mse > 2.0 * train_mse and train_mse < 1e-4:
        print(f"  [WARNING] test/train MSE ratio={test_mse/max(train_mse, 1e-30):.2f} (>2×), possible overfitting")

    return {
        "train_mse": float(train_mse),
        "val_mse":   float(val_mse),
        "test_mse":  float(test_mse),
        "test_rmse": float(math.sqrt(test_mse)),
        "steps_run": int(history[-1]["step"]) if history else 0,
    }, history


# -----------------------
# Shape / parameter matching
# -----------------------

def build_rnn_model(
    *,
    depth: int,
    hidden_size: int,
    cfg: TrainCfg,
    pad_to_target: bool = True,
) -> TinyRNNRegressor:
    tmp    = TinyRNNRegressor(
        input_size=cfg.input_size, output_size=cfg.output_size, hidden_size=hidden_size, depth=depth,
        horizon=cfg.horizon, rnn_type=cfg.rnn_type, dropout=cfg.dropout, pad_params=0,
    )
    active = count_params(tmp)
    pad    = 0
    if pad_to_target:
        if active > cfg.target_params:
            raise ValueError(
                f"Active params {active} > target {cfg.target_params} "
                f"for depth={depth}, hidden={hidden_size}."
            )
        pad = int(cfg.target_params - active)
    return TinyRNNRegressor(
        input_size=cfg.input_size, output_size=cfg.output_size, hidden_size=hidden_size, depth=depth,
        horizon=cfg.horizon, rnn_type=cfg.rnn_type, dropout=cfg.dropout, pad_params=pad,
    )


def find_hidden_for_target_params(*, depth: int, cfg: TrainCfg) -> Tuple[int, int]:
    """
    Largest hidden_size (multiple of hidden_step) such that active params <= target_params.

    With 3-D input, GRU params ≈ 3(3+h)h + 6h²(L-1) + horizon*input_size*h, giving
    diverse hidden sizes across depths (256 at L=3 → 112 at L=12 for N=1M).
    If min_hidden is infeasible at large depth, we lower hidden and warn.
    """
    step = int(cfg.hidden_step)

    def round_step(h: int) -> int:
        return max(step, int((h // step) * step))

    h_max      = round_step(cfg.max_hidden)
    h_pref_lo  = round_step(max(cfg.min_hidden, step))

    def active_params(h: int) -> int:
        m = TinyRNNRegressor(
            input_size=cfg.input_size, output_size=cfg.output_size, hidden_size=h, depth=depth,
            horizon=cfg.horizon, rnn_type=cfg.rnn_type, dropout=cfg.dropout, pad_params=0,
        )
        return count_params(m)

    if active_params(h_max) <= cfg.target_params:
        return int(h_max), int(active_params(h_max))

    best_h: Optional[int] = None
    for hv in range(h_max, step - 1, -step):
        a = active_params(hv)
        if a <= cfg.target_params:
            best_h = hv
            break

    if best_h is None:
        raise ValueError(
            f"No hidden in [{step}, {h_max}] fits target_params={cfg.target_params} "
            f"at depth={depth}. Raise --target_params, reduce --depth, or raise --max_hidden."
        )

    if best_h < h_pref_lo:
        print(
            f"  [WARNING] depth={depth}: --min_hidden={cfg.min_hidden} infeasible "
            f"(GRU params ~ O(L*H^2)); using hidden={best_h}."
        )

    candidates: List[Tuple[int, int, int]] = []
    for h in [max(step, best_h - step), best_h, min(h_max, best_h + step)]:
        hv = round_step(h)
        if hv < step:
            continue
        a = active_params(hv)
        if a <= cfg.target_params:
            candidates.append((abs(cfg.target_params - a), hv, a))
    candidates.sort(key=lambda t: t[0])
    _, h_best, a_best = candidates[0]
    return int(h_best), int(a_best)


# -----------------------
# Plotting
# -----------------------

def scatter_plot(x, y, xlabel, ylabel, title, outpath, depths=None, r=None, r_label="Pearson r"):
    """Linear-scale scatter plot with optional Pearson r annotation."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y)
    if depths is not None:
        for i, d in enumerate(depths):
            ax.annotate(f"d{d}", (x[i], y[i]), fontsize=8, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if r is not None:
        ax.text(0.05, 0.95, f"{r_label} = {r:.3f}",
                transform=ax.transAxes, verticalalignment="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.85))
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_curve(history: List[Dict[str, float]], out_dir: str, stem: str) -> None:
    """Save per-shape training curve (log-scale y for appendix readability)."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    png_path = os.path.join(out_dir, f"{stem}.png")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    steps      = [row["step"]      for row in history]
    train_vals = [row["train_mse"] for row in history]
    val_vals   = [row["val_mse"]   for row in history]
    test_vals  = [row["test_mse"]  for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_vals, label="train_mse")
    plt.plot(steps, val_vals,   label="val_mse")
    plt.plot(steps, test_vals,  label="test_mse")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE  (log scale)")
    plt.title(stem)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./results_rnn_mg_shape_sweep")
    parser.add_argument("--device",  type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",    type=int, default=0)

    parser.add_argument("--target_params", type=int, default=1_000_000)
    parser.add_argument("--depth_list",    type=str, default="3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--rnn_type",      type=str, default="gru", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--hidden_step",   type=int,   default=16)
    parser.add_argument("--min_hidden",    type=int,   default=112,
                        help="Preferred min hidden size (may be lowered for deep stacks at fixed N).")
    parser.add_argument("--max_hidden",    type=int,   default=1024)

    parser.add_argument("--input_size",  type=int, default=2,
                        help="Number of input channels (NARMA uses u_t and y_t).")
    parser.add_argument("--seq_len",     type=int, default=80,
                        help="Input context window length (number of timesteps).")
    parser.add_argument("--horizon",     type=int, default=8,
                        help="Multi-step prediction horizon H.")
    parser.add_argument("--train_size",  type=int, default=0,
                        help="0 = auto-compute from D=data_ratio*N.")
    parser.add_argument("--val_size",    type=int, default=2000)
    parser.add_argument("--test_size",   type=int, default=2000)

    parser.add_argument("--narma_order",    type=int,   default=30)
    parser.add_argument("--narma_warmup",   type=int,   default=200)
    parser.add_argument("--train_series_len", type=int, default=0)
    parser.add_argument("--val_series_len",   type=int, default=0)
    parser.add_argument("--test_series_len",  type=int, default=0)

    parser.add_argument("--data_ratio",   type=float, default=20.0)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--steps",        type=int,   default=0,
                        help="0 = auto-compute from D=data_ratio×N.")
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int,   default=200)
    parser.add_argument("--eval_every",   type=int,   default=500)
    parser.add_argument("--grad_clip",    type=float, default=1.0)

    parser.add_argument("--agop_batch",        type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=64)
    parser.add_argument("--max_agop_dim",      type=int, default=4096)
    parser.add_argument("--max_padding_ratio", type=float, default=0.15)
    parser.add_argument("--max_train_factor",  type=float, default=3.0)
    parser.add_argument("--fit_patience",      type=int,   default=8)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device    = torch.device(args.device)
    curve_dir = os.path.join(args.out_dir, "curves")
    set_global_seed(args.seed)

    depths = [int(x) for x in args.depth_list.split(",") if x.strip()]
    if len(depths) != 10:
        raise ValueError(f"depth_list must contain exactly 10 shapes, got {len(depths)}.")

    # Auto-compute train_size and steps from D=data_ratio*N
    tokens_per_step = int(args.batch_size) * int(args.horizon)
    if args.train_size <= 0:
        args.train_size = int(math.ceil(
            args.data_ratio * args.target_params / float(args.horizon)
        ))
    if args.train_series_len <= 0:
        args.train_series_len = args.train_size + args.seq_len + args.horizon + 1024
    if args.val_series_len <= 0:
        args.val_series_len = args.val_size + args.seq_len + args.horizon + 512
    if args.test_series_len <= 0:
        args.test_series_len = args.test_size + args.seq_len + args.horizon + 512

    cfg = TrainCfg(
        lr=args.lr, weight_decay=args.weight_decay, steps=args.steps,
        warmup_steps=args.warmup_steps, batch_size=args.batch_size, grad_clip=args.grad_clip,
        eval_every=args.eval_every, data_ratio=args.data_ratio,
        input_size=args.input_size, seq_len=args.seq_len, horizon=args.horizon,
        train_size=args.train_size, val_size=args.val_size, test_size=args.test_size,
        narma_order=args.narma_order, narma_warmup=args.narma_warmup,
        train_series_len=args.train_series_len, val_series_len=args.val_series_len, test_series_len=args.test_series_len,
        target_params=args.target_params, depth_list=depths, rnn_type=args.rnn_type,
        dropout=args.dropout, hidden_step=args.hidden_step,
        min_hidden=args.min_hidden, max_hidden=args.max_hidden,
        max_padding_ratio=args.max_padding_ratio,
        max_train_factor=args.max_train_factor,
        fit_patience=args.fit_patience,
        agop_batch=args.agop_batch, agop_proj_samples=args.agop_proj_samples,
        max_agop_dim=args.max_agop_dim, seed=args.seed,
    )

    if cfg.steps <= 0:
        cfg.steps = int(math.ceil(cfg.data_ratio * cfg.target_params / float(tokens_per_step)))

    total_train_tokens  = cfg.steps * tokens_per_step
    unique_train_tokens = cfg.train_size * cfg.horizon
    approx_epochs       = cfg.steps * cfg.batch_size / cfg.train_size
    agop_dim            = cfg.horizon

    print("========== Budget (RNN / NARMA-30 multi-step) ==========")
    print(f"target_params N     = {cfg.target_params:,}")
    print(f"rnn_type            = {cfg.rnn_type}")
    print(f"input_size          = {cfg.input_size}  (u_t, y_t)")
    print(f"seq_len             = {cfg.seq_len}  (input context steps)")
    print(f"horizon H           = {cfg.horizon}  (steps ahead)")
    print(f"AGOP dim            = H*C = {agop_dim}×{agop_dim}")
    print(f"train_size          = {cfg.train_size:,}  windows")
    print(f"tokens_per_step     = batch×H = {cfg.batch_size}×{cfg.horizon} = {tokens_per_step:,}")
    print(f"base steps          = {cfg.steps:,}")
    print(f"approx epochs       = {approx_epochs:.2f}")
    print(f"D (total tokens)    = {total_train_tokens:,}   D/N = {total_train_tokens/cfg.target_params:.1f}×")
    print(f"unique tokens       = {unique_train_tokens:,}   {unique_train_tokens/cfg.target_params:.1f}×N")
    print("=========================================================")

    print("\nGenerating NARMA-30 sequences ...")
    train_u, train_y = generate_narma30(
        total_len=cfg.train_series_len, order=cfg.narma_order, warmup=cfg.narma_warmup, seed=cfg.seed + 123
    )
    val_u, val_y = generate_narma30(
        total_len=cfg.val_series_len, order=cfg.narma_order, warmup=cfg.narma_warmup, seed=cfg.seed + 456
    )
    test_u, test_y = generate_narma30(
        total_len=cfg.test_series_len, order=cfg.narma_order, warmup=cfg.narma_warmup, seed=cfg.seed + 789
    )

    mu_u  = train_u.mean(axis=0, keepdims=True)
    std_u = train_u.std(axis=0, keepdims=True) + 1e-8
    mu_y  = train_y.mean(axis=0, keepdims=True)
    std_y = train_y.std(axis=0, keepdims=True) + 1e-8

    def norm_inputs(u_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        return np.concatenate([(u_arr - mu_u) / std_u, (y_arr - mu_y) / std_y], axis=1).astype(np.float32)

    train_inputs = norm_inputs(train_u, train_y)
    val_inputs   = norm_inputs(val_u,   val_y)
    test_inputs  = norm_inputs(test_u,  test_y)
    train_targets = ((train_y - mu_y) / std_y).astype(np.float32)
    val_targets   = ((val_y - mu_y) / std_y).astype(np.float32)
    test_targets  = ((test_y - mu_y) / std_y).astype(np.float32)

    train_ds = NARMAHorizonDataset(
        inputs=train_inputs, targets=train_targets, seq_len=cfg.seq_len, horizon=cfg.horizon,
        size=cfg.train_size, seed=cfg.seed + 1,
    )
    val_ds = NARMAHorizonDataset(
        inputs=val_inputs, targets=val_targets, seq_len=cfg.seq_len, horizon=cfg.horizon,
        size=cfg.val_size, seed=cfg.seed + 2,
    )
    test_ds = NARMAHorizonDataset(
        inputs=test_inputs, targets=test_targets, seq_len=cfg.seq_len, horizon=cfg.horizon,
        size=cfg.test_size,  seed=cfg.seed + 2,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Fixed AGOP batch from test set
    x_agop_list = []
    for x, _y in test_loader:
        x_agop_list.append(x)
        if sum(t.shape[0] for t in x_agop_list) >= cfg.agop_batch:
            break
    x_agop = torch.cat(x_agop_list, dim=0)[: cfg.agop_batch].to(device)

    results: List[Dict[str, float]] = []

    print("\n========== Shape sweep ==========")
    for i, depth in enumerate(cfg.depth_list):
        hidden, active = find_hidden_for_target_params(depth=depth, cfg=cfg)
        model  = build_rnn_model(depth=depth, hidden_size=hidden, cfg=cfg)
        total  = count_params(model)
        pad    = total - active
        pad_ratio = pad / max(1, total)
        if pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] depth={depth}: padding_ratio={pad_ratio:.3f} exceeds max_padding_ratio={cfg.max_padding_ratio:.3f}")
            continue

        print(f"\n[{i+1:02d}/{len(cfg.depth_list)}] depth={depth:2d}  hidden={hidden:4d}  "
              f"active={active:,}  pad={pad:,}  total={total:,}  agop_dim={agop_dim}")

        metrics, history = train_one_model(model, train_loader, val_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"rnn_depth{depth}_hidden{hidden}")

        agop = estimate_agop_wrt_inputs(
            model, x_agop, proj_samples=cfg.agop_proj_samples, max_agop_dim=cfg.max_agop_dim,
        )
        off_e, off_r = agop_offdiag_metrics(agop)

        row: Dict[str, float] = dict(metrics)
        row.update({
            "depth":               int(depth),
            "hidden_size":         int(hidden),
            "active_params":       int(active),
            "pad_params":          int(pad),
            "total_params":        int(total),
            "padding_ratio":       float(pad_ratio),
            "agop_dim":            int(agop.shape[0]),
            "agop_offdiag_energy": float(off_e),
            "agop_offdiag_ratio":  float(off_r),
        })
        results.append(row)

        del model, agop
        torch.cuda.empty_cache()

    # ---------- Unified correlation metrics ----------
    print("\n========== Correlations (AOFE hypothesis) ==========")
    test_mse   = np.array([r["test_mse"]            for r in results], dtype=np.float64)
    off_energy = np.array([r["agop_offdiag_energy"]  for r in results], dtype=np.float64)
    off_ratio  = np.array([r["agop_offdiag_ratio"]   for r in results], dtype=np.float64)
    depths_arr = [int(r["depth"]) for r in results]

    p_aofe       = pearson_corr(off_energy, test_mse)
    p_aofe_ratio = pearson_corr(off_ratio,  test_mse)
    s_aofe       = spearman_corr(off_energy, test_mse)
    s_aofe_ratio = spearman_corr(off_ratio,  test_mse)

    print("Unified AOFE metrics (raw test_mse, no log):")
    print(f"  Pearson (AOFE=offdiag_energy,     test_mse) = {p_aofe:.4f}   Spearman = {s_aofe:.4f}")
    print(f"  Pearson (AOFE_ratio=offdiag_ratio, test_mse) = {p_aofe_ratio:.4f}   Spearman = {s_aofe_ratio:.4f}")

    # ---------- Save results ----------
    csv_path = os.path.join(args.out_dir, "results.csv")
    npy_path = os.path.join(args.out_dir, "results.npy")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    np.save(npy_path, results, allow_pickle=True)
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")

    # ---------- Scatter plots (with Pearson r annotation) ----------
    scatter_plot(
        off_energy, test_mse,
        xlabel="AOFE  (AGOP off-diagonal energy)",
        ylabel="Test MSE (per scalar)",
        title=f"Lorenz-63 RNN: test MSE vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_energy.png"),
        depths=depths_arr,
        r=p_aofe, r_label="Pearson r (AOFE, loss)",
    )
    scatter_plot(
        off_ratio, test_mse,
        xlabel="AOFE ratio  (AGOP off-diagonal ratio)",
        ylabel="Test MSE (per scalar)",
        title=f"Lorenz-63 RNN: test MSE vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_ratio.png"),
        depths=depths_arr,
        r=p_aofe_ratio, r_label="Pearson r (AOFE_ratio, loss)",
    )
    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
