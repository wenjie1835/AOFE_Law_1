#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rnn_shape_sweep_mackeyglass_superposition_agop.py
=================================================

Goal
----
Fixed-parameter GRU shape sweep on Mackey–Glass time-series next-step prediction.
Tests the AOFE hypothesis: under fixed N, different (depth, hidden_size) shapes reach
similar loss, mediated by AOFE (AGOP Off-diagonal Frobenius Energy).

Task
----
Mackey–Glass DDE (Euler discretization):
    x_{t+1} = x_t + dt * ( beta * x_{t-τ}/(1+x_{t-τ}^n) - gamma * x_t )

Windows of length T:
    input  : [x_t, ..., x_{t+T-1}]   shape [T, 1]
    target : [x_{t+1}, ..., x_{t+T}]  shape [T, 1]  (next-step prediction)

Output-space AGOP:
    J = d(y_hat_flat)/d(x_in),  y_hat_flat ∈ R^T
    AGOP = E_data[J J^T] ∈ R^{T×T}  — fixed dimension across shapes.
Estimated via JVP random projections.

Training protocol (strict D=20N)
---------------------------------
Training stops at exactly cfg.steps steps — no extension, no patience early-stop.
The final model state (at D=20N) is evaluated and reported.

Data budget (D=20N)
-------------------
"Token" = one scalar next-step supervision target:
    tokens_per_step = batch_size × seq_len
    D = steps × tokens_per_step  ≈  20 × N

train_size (number of unique windows) is set so that:
    train_size × seq_len  ≈  20 × N  (each window seen ~1 time on average)

Shapes
------
10 non-extreme GRU shapes (depth 2..12), width found by binary search.
min_hidden=64 prevents too-narrow shapes; gradient clipping prevents explosion.

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
# Mackey–Glass generator
# -----------------------

def generate_mackey_glass(
    *,
    total_len: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    dt: float = 1.0,
    warmup: int = 1000,
    x0: float = 1.2,
    init_noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate Mackey–Glass time series via Euler discretization.
    Returns 1D array of length total_len (after warmup).
    """
    total_len = int(total_len)
    tau       = int(tau)
    warmup    = int(warmup)
    rng = np.random.default_rng(int(seed))

    sim_len = warmup + total_len
    x = np.zeros(tau + sim_len + 1, dtype=np.float64)
    hist = x0 + init_noise * rng.standard_normal(size=(tau + 1,))
    hist = np.clip(hist, 1e-6, None)
    x[: tau + 1] = hist

    for t in range(tau, tau + sim_len):
        x_tau  = x[t - tau]
        growth = beta * x_tau / (1.0 + (x_tau ** n))
        decay  = gamma * x[t]
        x[t + 1] = x[t] + dt * (growth - decay)
        if not np.isfinite(x[t + 1]):
            raise FloatingPointError("Non-finite value in Mackey–Glass simulation.")
        x[t + 1] = max(x[t + 1], 1e-6)

    return x[tau + warmup : tau + warmup + total_len].astype(np.float32)


# -----------------------
# Dataset
# -----------------------

class MackeyGlassWindowDataset(torch.utils.data.Dataset):
    """Fixed sampling of non-overlapping windows from the time series."""
    def __init__(self, *, series: np.ndarray, seq_len: int, size: int, seed: int):
        super().__init__()
        series = np.asarray(series, dtype=np.float32)
        self.series  = series
        self.seq_len = int(seq_len)
        self.size    = int(size)

        L = int(series.shape[0])
        if L <= self.seq_len + 1:
            raise ValueError(f"Series length {L} must be > seq_len+1={self.seq_len+1}.")

        max_start    = L - (self.seq_len + 1)
        support_size = max_start + 1
        if self.size > support_size:
            raise ValueError(
                f"Requested size={self.size} exceeds unique support {support_size}. "
                "Increase series length or decrease train_size/test_size."
            )
        g = np.random.default_rng(int(seed))
        self.starts = g.permutation(support_size)[: self.size]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        x = self.series[s : s + self.seq_len]
        y = self.series[s + 1 : s + 1 + self.seq_len]
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y).unsqueeze(-1)


# -----------------------
# Model: stacked GRU
# -----------------------

class TinyRNNRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        depth: int,
        rnn_type: str = "gru",
        dropout: float = 0.0,
        pad_params: int = 0,
    ):
        super().__init__()
        self.input_size  = int(input_size)
        self.hidden_size = int(hidden_size)
        self.depth       = int(depth)
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

        self.head = nn.Linear(self.hidden_size, 1, bias=False)

        self._pad_params = None
        if pad_params > 0:
            self._pad_params = nn.Parameter(torch.zeros(int(pad_params)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.head(h)


# -----------------------
# AGOP estimation (wrt input sequence)
# -----------------------

def estimate_agop_wrt_inputs(
    model: TinyRNNRegressor,
    x: torch.Tensor,
    *,
    proj_samples: int = 16,
    max_agop_dim: int = 4096,
) -> torch.Tensor:
    """
    AGOP = E[J J^T],  J = d(y_hat_flat)/d(x_in),  y_hat_flat ∈ R^T.
    AGOP ∈ R^{T×T}  — fixed dimension across model shapes.
    cuDNN must be disabled for JVP to work with RNNs.
    """
    device = x.device
    model.train()  # cuDNN RNN requires train mode for JVP backward
    B, T, C = x.shape
    assert C == 1
    if T > max_agop_dim:
        raise ValueError(f"AGOP dim T={T} > max_agop_dim={max_agop_dim}.")

    x0   = x.detach()
    agop = torch.zeros((T, T), device=device, dtype=torch.float32)

    def fwd(x_in: torch.Tensor) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            return model(x_in).squeeze(-1)  # [B, T]

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
    eval_every: int = 1000

    data_ratio: float = 20.0
    seq_len: int = 128
    train_size: int = 0
    test_size: int = 1000

    tau: int = 17
    beta: float = 0.2
    gamma: float = 0.1
    n: float = 10.0
    dt: float = 1.0
    warmup: int = 1000
    x0: float = 1.2
    init_noise: float = 0.05
    train_series_len: int = 0
    test_series_len: int = 0

    target_params: int = 1_000_000
    depth_list: List[int] = None
    rnn_type: str = "gru"
    dropout: float = 0.0
    hidden_step: int = 16
    min_hidden: int = 64
    max_hidden: int = 1024

    agop_batch: int = 256
    agop_proj_samples: int = 16
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
        y_hat = model(x)
        se    = F.mse_loss(y_hat, y, reduction="sum")
        total_se     += float(se.item())
        total_tokens += int(y.numel())
    return total_se / max(1, total_tokens)


def train_one_model(
    model: TinyRNNRegressor,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cfg: TrainCfg,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Train for exactly cfg.steps steps (strict D=20N budget).
    No early stopping, no checkpoint restoration — the final model state is reported.
    """
    model.to(device)
    model.train()

    opt        = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_iter = iter(train_loader)

    t0        = time.time()
    history: List[Dict[str, float]] = []
    max_steps = int(cfg.steps)  # strict D=20N budget — no extension

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

        y_hat = model(x)
        loss  = F.mse_loss(y_hat, y, reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == max_steps:
            train_mse = evaluate_mse(model, train_loader, device, max_batches=50)
            test_mse  = evaluate_mse(model, test_loader,  device, max_batches=None)
            model.train()   # restore train mode for cuDNN RNN
            history.append({
                "step":      int(step + 1),
                "lr":        float(lr),
                "train_mse": float(train_mse),
                "test_mse":  float(test_mse),
            })
            dt  = time.time() - t0
            gap = test_mse - train_mse
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_mse={train_mse:.6f}  test_mse={test_mse:.6f}  "
                f"rmse={math.sqrt(test_mse):.6f}  gap={gap:+.6f}  time={dt:.1f}s"
            )

    # Final evaluation at end of strict D=20N budget
    train_mse = evaluate_mse(model, train_loader, device, max_batches=None)
    test_mse  = evaluate_mse(model, test_loader,  device, max_batches=None)

    if test_mse > 2.0 * train_mse and train_mse < 1e-4:
        print(f"  [WARNING] test/train MSE ratio={test_mse/max(train_mse, 1e-30):.2f} (>2×), possible overfitting")

    return {
        "train_mse": float(train_mse),
        "test_mse":  float(test_mse),
        "test_rmse": float(math.sqrt(test_mse)),
        "steps_run": int(history[-1]["step"]) if history else 0,
    }, history


# -----------------------
# Shape/parameter matching
# -----------------------

def build_rnn_model(
    *,
    depth: int,
    hidden_size: int,
    cfg: TrainCfg,
    pad_to_target: bool = True,
) -> TinyRNNRegressor:
    tmp    = TinyRNNRegressor(input_size=1, hidden_size=hidden_size, depth=depth,
                               rnn_type=cfg.rnn_type, dropout=cfg.dropout, pad_params=0)
    active = count_params(tmp)
    pad    = 0
    if pad_to_target:
        if active > cfg.target_params:
            raise ValueError(f"Active params {active} > target {cfg.target_params} "
                             f"for depth={depth}, hidden={hidden_size}.")
        pad = int(cfg.target_params - active)
    return TinyRNNRegressor(input_size=1, hidden_size=hidden_size, depth=depth,
                             rnn_type=cfg.rnn_type, dropout=cfg.dropout, pad_params=pad)


def find_hidden_for_target_params(*, depth: int, cfg: TrainCfg) -> Tuple[int, int]:
    step = int(cfg.hidden_step)

    def to_valid(h: int) -> int:
        h = max(cfg.min_hidden, h)
        return int((h // step) * step)

    h_min = to_valid(cfg.min_hidden)
    h_max = to_valid(cfg.max_hidden)

    def active_params(h: int) -> int:
        m = TinyRNNRegressor(input_size=1, hidden_size=h, depth=depth,
                              rnn_type=cfg.rnn_type, dropout=cfg.dropout, pad_params=0)
        return count_params(m)

    if active_params(h_min) > cfg.target_params:
        raise ValueError(f"hidden={h_min} already exceeds target_params={cfg.target_params} at depth={depth}.")
    if active_params(h_max) <= cfg.target_params:
        return h_max, active_params(h_max)

    lo, hi   = h_min, h_max
    best_h   = h_min
    best_a   = active_params(best_h)
    while lo <= hi:
        mid = to_valid((lo + hi) // 2)
        a   = active_params(mid)
        if a <= cfg.target_params:
            best_h, best_a = mid, a
            lo = mid + step
        else:
            hi = mid - step

    candidates = []
    for h in [max(h_min, best_h - step), best_h, min(h_max, best_h + step)]:
        a = active_params(h)
        if a <= cfg.target_params:
            candidates.append((abs(cfg.target_params - a), h, a))
    candidates.sort(key=lambda t: t[0])
    _, h_best, a_best = candidates[0]
    return int(h_best), int(a_best)


# -----------------------
# Plotting
# -----------------------

def scatter_plot(x, y, xlabel, ylabel, title, outpath, depths=None, r=None, r_label="Pearson r"):
    """Linear-scale scatter plot (no log on y-axis, per AOFE hypothesis requirements)."""
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
    test_vals  = [row["test_mse"]  for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_vals, label="train_mse")
    plt.plot(steps, test_vals,  label="test_mse")
    plt.yscale("log")   # log scale for appendix readability (NOT used in correlation)
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
    parser.add_argument("--depth_list",    type=str, default="2,3,4,5,6,7,8,9,10,12")
    parser.add_argument("--rnn_type",      type=str, default="gru", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--hidden_step",   type=int,   default=16)
    parser.add_argument("--min_hidden",    type=int,   default=64)
    parser.add_argument("--max_hidden",    type=int,   default=1024)

    parser.add_argument("--seq_len",    type=int, default=128)
    parser.add_argument("--train_size", type=int, default=0)
    parser.add_argument("--test_size",  type=int, default=1000)
    parser.add_argument("--train_series_len", type=int, default=0)
    parser.add_argument("--test_series_len",  type=int, default=0)

    parser.add_argument("--tau",        type=int,   default=17)
    parser.add_argument("--beta",       type=float, default=0.2)
    parser.add_argument("--gamma",      type=float, default=0.1)
    parser.add_argument("--n",          type=float, default=10.0)
    parser.add_argument("--dt",         type=float, default=1.0)
    parser.add_argument("--warmup",     type=int,   default=1000)
    parser.add_argument("--x0",         type=float, default=1.2)
    parser.add_argument("--init_noise", type=float, default=0.05)

    parser.add_argument("--data_ratio",   type=float, default=20.0)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--steps",        type=int,   default=0,
                        help="0 = auto-compute from D=data_ratio×N.")
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int,   default=200)
    parser.add_argument("--eval_every",   type=int,   default=1000)
    parser.add_argument("--grad_clip",    type=float, default=1.0)

    parser.add_argument("--agop_batch",        type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=16)
    parser.add_argument("--max_agop_dim",      type=int, default=4096)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device    = torch.device(args.device)
    curve_dir = os.path.join(args.out_dir, "curves")
    set_global_seed(args.seed)

    depths = [int(x) for x in args.depth_list.split(",") if x.strip()]
    if len(depths) != 10:
        raise ValueError(f"depth_list must contain exactly 10 shapes, got {len(depths)}.")

    if args.train_size <= 0:
        args.train_size = int(math.ceil(args.data_ratio * args.target_params / float(args.seq_len)))
    if args.train_series_len <= 0:
        args.train_series_len = args.train_size + args.seq_len + 1024
    if args.test_series_len <= 0:
        args.test_series_len = max(args.test_size + args.seq_len + 64, 4096)

    cfg = TrainCfg(
        lr=args.lr, weight_decay=args.weight_decay, steps=args.steps,
        warmup_steps=args.warmup_steps, batch_size=args.batch_size, grad_clip=args.grad_clip,
        eval_every=args.eval_every, data_ratio=args.data_ratio, seq_len=args.seq_len,
        train_size=args.train_size, test_size=args.test_size,
        tau=args.tau, beta=args.beta, gamma=args.gamma, n=args.n, dt=args.dt,
        warmup=args.warmup, x0=args.x0, init_noise=args.init_noise,
        train_series_len=args.train_series_len, test_series_len=args.test_series_len,
        target_params=args.target_params, depth_list=depths, rnn_type=args.rnn_type,
        dropout=args.dropout, hidden_step=args.hidden_step,
        min_hidden=args.min_hidden, max_hidden=args.max_hidden,
        agop_batch=args.agop_batch, agop_proj_samples=args.agop_proj_samples,
        max_agop_dim=args.max_agop_dim, seed=args.seed,
    )

    tokens_per_step = cfg.batch_size * cfg.seq_len
    if cfg.steps <= 0:
        cfg.steps = int(math.ceil(cfg.data_ratio * cfg.target_params / float(tokens_per_step)))

    total_train_tokens  = cfg.steps * tokens_per_step
    unique_train_tokens = cfg.train_size * cfg.seq_len
    approx_epochs       = cfg.steps * cfg.batch_size / cfg.train_size

    print("========== Budget (RNN / Mackey-Glass) ==========")
    print(f"target_params N     = {cfg.target_params:,}")
    print(f"rnn_type            = {cfg.rnn_type}")
    print(f"train_size          = {cfg.train_size:,}  windows")
    print(f"tokens_per_step     = batch×seq_len = {cfg.batch_size}×{cfg.seq_len} = {tokens_per_step:,}")
    print(f"base steps          = {cfg.steps:,}")
    print(f"approx epochs       = {approx_epochs:.2f}")
    print(f"D (total tokens)    = {total_train_tokens:,}   D/N = {total_train_tokens/cfg.target_params:.1f}×")
    print(f"unique tokens       = {unique_train_tokens:,}   {unique_train_tokens/cfg.target_params:.1f}×N")
    print("=================================================")

    # Generate series
    total_len = cfg.train_series_len + cfg.test_series_len
    series    = generate_mackey_glass(
        total_len=total_len, tau=cfg.tau, beta=cfg.beta, gamma=cfg.gamma,
        n=cfg.n, dt=cfg.dt, warmup=cfg.warmup, x0=cfg.x0, init_noise=cfg.init_noise,
        seed=cfg.seed + 123,
    )
    train_series = series[: cfg.train_series_len]
    test_series  = series[cfg.train_series_len : cfg.train_series_len + cfg.test_series_len]

    mu  = float(train_series.mean())
    std = float(train_series.std() + 1e-8)
    train_series = (train_series - mu) / std
    test_series  = (test_series  - mu) / std
    print(f"Series normalized: train mean={mu:.4f}, std={std:.4f}")

    train_ds = MackeyGlassWindowDataset(
        series=train_series, seq_len=cfg.seq_len, size=cfg.train_size, seed=cfg.seed + 1)
    test_ds  = MackeyGlassWindowDataset(
        series=test_series,  seq_len=cfg.seq_len, size=cfg.test_size,  seed=cfg.seed + 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True,  num_workers=0)
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

        print(f"\n[{i+1:02d}/{len(cfg.depth_list)}] depth={depth:2d}  hidden={hidden:4d}  "
              f"active={active:,}  pad={pad:,}  total={total:,}  agop_dim(T)={cfg.seq_len}")

        metrics, history = train_one_model(model, train_loader, test_loader, cfg, device)
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
            "padding_ratio":       float(pad / max(1, total)),
            "agop_dim":            int(agop.shape[0]),
            "agop_offdiag_energy": float(off_e),
            "agop_offdiag_ratio":  float(off_r),
        })
        results.append(row)

        del model, agop
        torch.cuda.empty_cache()

    # ---------- Unified correlation metrics (no log) ----------
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

    # ---------- Scatter plots (linear y-axis) ----------
    scatter_plot(
        off_energy, test_mse,
        xlabel="AOFE  (AGOP off-diagonal energy)",
        ylabel="Test MSE (per token)",
        title=f"Mackey-Glass: test MSE vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_energy.png"),
        depths=depths_arr,
    )
    scatter_plot(
        off_ratio, test_mse,
        xlabel="AOFE ratio  (AGOP off-diagonal ratio)",
        ylabel="Test MSE (per token)",
        title=f"Mackey-Glass: test MSE vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_ratio.png"),
        depths=depths_arr,
    )
    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
