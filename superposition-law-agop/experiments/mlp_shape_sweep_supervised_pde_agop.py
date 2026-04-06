#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp_shape_sweep_supervised_pde_agop.py
=====================================

Goal
----
Fixed-parameter MLP shape sweep on nonlinear regression via a frozen random
teacher MLP.  Tests the hypothesis: under fixed N, different (depth, width)
shapes reach similar loss, mediated by AOFE (AGOP Off-diagonal Frobenius Energy).

Task (Teacher-Student Nonlinear Regression)
-------------------------------------------
A fixed random 3-layer teacher MLP generates (x, y) pairs:
    teacher: Linear(in_dim → H) → GELU → Linear(H → H) → GELU → Linear(H → out_dim)
    x ~ N(0, I_{in_dim}),  y = teacher(x)   (both normalized for stable training)

Output-space AGOP:
    J = d(ŷ_student)/d(x)  ∈ R^{out_dim × in_dim}
    AGOP = E_data[J J^T] ∈ R^{out_dim × out_dim}   — fixed dimension across all shapes.

Why teacher-student instead of a linear PDE?
--------------------------------------------
For smooth linear PDEs (e.g., heat equation), the target function's Jacobian
structure is analytically fixed — all converged models approximate the same smooth
function and therefore produce nearly identical AGOP structure (AOFE ratio varies
by < 0.03 across shapes).  A nonlinear teacher with hidden nonlinearities forces
different student architectures to learn genuinely different feature representations,
producing meaningful AOFE variation that can correlate with test MSE.

Data budget
-----------
"Token" = one scalar supervised target:
    tokens_per_step = batch_size × out_dim
    D = steps × tokens_per_step  ≈  20 × N   (Chinchilla D=20N)

Training protocol (strict D=20N)
---------------------------------
Training stops at exactly cfg.steps steps — no extension, no patience early-stop.
The final model state (at D=20N) is evaluated and reported, matching Chinchilla
compute-optimal evaluation.

Shapes
------
10 non-extreme MLP shapes (depth 3..12), each with Pre-LN residual blocks.
Width found by binary search to keep active_params ≤ target_params (pad remainder).
Min width=192 ensures no shape is too narrow.

Unified correlation metrics
---------------------------
  Pearson(AOFE=agop_offdiag_energy,  test_mse)   — raw MSE, no log
  Pearson(AOFE_ratio=agop_offdiag_ratio, test_mse) — raw MSE, no log

Loss curves (for appendix)
---------------------------
Per-shape loss curves saved to curves/ with log-scale y-axis for readability
(log scale is applied only to the curve images, NOT to the correlation computation).
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


def _apply_act(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu":  return F.gelu(x)
    if activation == "silu":  return F.silu(x)
    if activation == "tanh":  return torch.tanh(x)
    if activation == "relu":  return F.relu(x)
    raise ValueError(f"Unknown activation: {activation}")


class _ResBlock(nn.Module):
    """Pre-LN residual block: x → x + fc(act(ln(x))). Prevents vanishing gradients for deep MLPs."""
    def __init__(self, width: int, activation: str, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc = nn.Linear(width, width, bias=True)
        self.activation = activation
        self._dp = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _apply_act(self.ln(x), self.activation)
        if self._dp > 0:
            h = F.dropout(h, p=self._dp, training=self.training)
        return x + self.fc(h)


def agop_offdiag_metrics(agop: torch.Tensor) -> Tuple[float, float]:
    """
    AOFE (AGOP Off-diagonal Frobenius Energy):
      offdiag_energy = ||AGOP||_F^2 - ||diag(AGOP)||_2^2
      offdiag_ratio  = offdiag_energy / ||AGOP||_F^2
    """
    agop = agop.float()
    fro2 = float((agop * agop).sum().item()) + 1e-12
    diag = torch.diag(agop)
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
    csum = np.cumsum(counts)
    starts = csum - counts + 1
    avg = (starts + csum) / 2.0
    return avg[inv]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    return pearson_corr(rx, ry)


# -----------------------
# Dataset: teacher-student nonlinear regression
# -----------------------

class TeacherStudentDataset(torch.utils.data.Dataset):
    """
    Multi-output nonlinear regression via a frozen random teacher MLP.

    Teacher:  Linear(in_dim → H) → GELU → Linear(H → H) → GELU → Linear(H → out_dim)
    Data:     x ~ N(0, I_{in_dim}),  y = teacher(x)

    Both inputs and outputs are normalized per-dimension (zero mean, unit std)
    at dataset construction time for numerical stability.

    This creates a rich nonlinear regression target where different student
    depths/widths learn genuinely different effective feature representations,
    causing the output-space AGOP E[J J^T] (out_dim × out_dim) to vary
    across shapes — enabling a clean test of the AOFE hypothesis.

    The teacher weights are drawn once from a fixed seed and never updated.
    """
    def __init__(
        self,
        *,
        size: int,
        in_dim: int,
        out_dim: int,
        teacher_hidden: int,
        teacher_seed: int,
        data_seed: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.size    = int(size)
        self.in_dim  = int(in_dim)
        self.out_dim = int(out_dim)

        # Fixed random teacher (4-layer, never trained — used only for data generation).
        # Larger hidden size + extra layer creates a richer target function so that
        # different student shapes genuinely differ in which features they can represent,
        # producing shape-dependent Jacobian structure (and thus AGOP variation).
        g_t = torch.Generator()
        g_t.manual_seed(int(teacher_seed))
        s1 = 1.0 / math.sqrt(in_dim)
        s2 = 1.0 / math.sqrt(teacher_hidden)
        W1 = (torch.randn(in_dim,         teacher_hidden, generator=g_t) * s1).to(dtype)
        W2 = (torch.randn(teacher_hidden, teacher_hidden, generator=g_t) * s2).to(dtype)
        W3 = (torch.randn(teacher_hidden, teacher_hidden, generator=g_t) * s2).to(dtype)
        W4 = (torch.randn(teacher_hidden, out_dim,        generator=g_t) * s2).to(dtype)

        # Generate raw inputs
        g_d = torch.Generator()
        g_d.manual_seed(int(data_seed))
        x_raw = torch.randn(self.size, in_dim, generator=g_d, dtype=dtype)

        # Teacher forward pass (no biases — keeps symmetry simple)
        h1    = F.gelu(x_raw @ W1)
        h2    = F.gelu(h1 @ W2)
        h3    = F.gelu(h2 @ W3)
        y_raw = h3 @ W4  # [size, out_dim]

        # Normalize inputs per-dim
        x_mean = x_raw.mean(0)
        x_std  = x_raw.std(0).clamp(min=1e-6)
        self.x = (x_raw - x_mean) / x_std

        # Normalize outputs per-dim
        y_mean = y_raw.mean(0)
        y_std  = y_raw.std(0).clamp(min=1e-6)
        self.y = (y_raw - y_mean) / y_std

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# -----------------------
# Model: MLP with Pre-LN residual blocks
# -----------------------

class TinyMLP(nn.Module):
    """
    Pre-LN Residual MLP:
        entry:   Linear(in_dim → width) + act
        hidden:  (depth-1) × _ResBlock  [x = x + fc(act(ln(x)))]
        head:    Linear(width → out_dim)

    pad_params: unused parameters to equalize total params across shapes.
    """
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        depth: int,
        width: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        pad_params: int = 0,
    ):
        super().__init__()
        assert depth >= 1
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.depth = int(depth)
        self.width = int(width)
        self.activation = str(activation)

        _apply_act(torch.zeros(1), activation)

        self.entry = nn.Linear(in_dim, width, bias=True)
        self.res_blocks = nn.ModuleList([
            _ResBlock(width, activation, dropout) for _ in range(depth - 1)
        ])
        self.head = nn.Linear(width, out_dim, bias=True)

        self._pad = None
        if pad_params > 0:
            self._pad = nn.Parameter(torch.zeros(int(pad_params)), requires_grad=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = _apply_act(self.entry(z), self.activation)
        for blk in self.res_blocks:
            x = blk(x)
        return self.head(x)


# -----------------------
# AGOP estimation (wrt inputs z)
# -----------------------

def estimate_agop_wrt_inputs(
    model: TinyMLP,
    z: torch.Tensor,
    *,
    proj_samples: int = 64,
) -> torch.Tensor:
    """
    Input-space AGOP: G = E_data[J^T J] ∈ R^{d_in × d_in},  J = d(y)/d(z).

    This is the theoretically correct metric for measuring INPUT-FEATURE superposition
    (cf. the Neural Feature Ansatz: NFM ≈ G = E[J^T J]).

    G_ij = E[<∂f/∂z_i, ∂f/∂z_j>] measures how much input dimensions i and j are
    co-used by the network.  Different student shapes with the same parameter count
    develop genuinely different input-coupling structures:
      • Wide shallow networks: more neurons per layer → can represent input features
        more orthogonally → lower off-diagonal G → lower AOFE.
      • Narrow deep networks: bottleneck forces features to share neurons →
        higher input-feature interference → higher AOFE.

    Estimated via VJP (backward-mode) random OUTPUT projections v ~ N(0, I_{d_out}):
      J^T v = ∂(v^T f)/∂z  (per-sample gradient from a single backward pass)
      G ≈ (1/proj_samples) Σ_r E_z[(J^T v_r)(J^T v_r)^T]

    With B=256, proj_samples=64: 256*64=16384 rank-1 updates for
    d_in*(d_in+1)/2 = 64*65/2 = 2080 unique AGOP entries → ~7.9× overdetermined.
    """
    device = z.device
    model.eval()
    B, d_in = z.shape
    with torch.no_grad():
        y0    = model(z)
        d_out = y0.shape[1]

    agop = torch.zeros((d_in, d_in), device=device, dtype=torch.float32)

    for _ in range(int(proj_samples)):
        v    = torch.randn(B, d_out, device=device)   # random output direction
        z_in = z.detach().requires_grad_(True)
        y    = model(z_in)                             # [B, d_out]
        # scalar = Σ_{b,k} v_bk y_bk  →  ∂scalar/∂z_in[b] = J(z_b)^T v_b  (per sample)
        loss = (v * y).sum()
        loss.backward()
        with torch.no_grad():
            g = z_in.grad.float()                      # [B, d_in]: J^T v per sample
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            agop = agop + (g.T @ g) / float(B)

    agop = agop / float(proj_samples)
    return symmetrize_(agop).detach()


# -----------------------
# Training
# -----------------------

@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 0
    data_ratio: float = 20.0
    warmup_steps: int = 0
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 1000

    in_dim: int = 64
    out_dim: int = 64
    teacher_hidden: int = 256
    train_size: int = 0
    val_size: int = 2000
    test_size: int = 2000

    target_params: int = 1_000_000
    depth_list: List[int] = None
    width_multiple: int = 16
    min_width: int = 256
    max_width: int = 2048
    activation: str = "gelu"
    dropout: float = 0.0
    max_padding_ratio: float = 0.15
    max_train_factor: float = 3.0
    fit_patience: int = 8

    agop_batch: int = 256
    agop_proj_samples: int = 64

    seed: int = 0


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    t = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate_mse(
    model: TinyMLP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    sse = 0.0
    n   = 0
    for i, (z, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        z = z.to(device)
        y = y.to(device)
        pred = model(z)
        err  = (pred - y).float()
        sse += float((err * err).sum().item())
        n   += int(y.numel())
    return sse / max(1, n)


def train_one_model(
    model: TinyMLP,
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_iter = iter(train_loader)

    t0       = time.time()
    history: List[Dict[str, float]] = []
    min_steps = int(cfg.steps)
    max_steps = max(min_steps, int(math.ceil(cfg.max_train_factor * cfg.steps)))
    best_val = float("inf")
    best_state = None
    stale_evals = 0

    for step in range(max_steps):
        try:
            z, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            z, y = next(train_iter)

        z = z.to(device)
        y = y.to(device)

        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, max_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        pred = model(z)
        loss = F.mse_loss(pred, y, reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == max_steps:
            train_mse = evaluate_mse(model, train_loader, device, max_batches=50)
            val_mse   = evaluate_mse(model, val_loader,   device, max_batches=None)
            test_mse  = evaluate_mse(model, test_loader,  device, max_batches=None)
            history.append({
                "step":      int(step + 1),
                "lr":        float(lr),
                "train_mse": float(train_mse),
                "val_mse":   float(val_mse),
                "test_mse":  float(test_mse),
            })
            dt = time.time() - t0
            gap = test_mse - train_mse
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_mse={train_mse:.6e}  val_mse={val_mse:.6e}  test_mse={test_mse:.6e}  "
                f"gap={gap:+.3e}  time={dt:.1f}s"
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

    gap = test_mse - train_mse
    if gap > 0.5 * test_mse:
        print(f"  [WARNING] test_mse / train_mse = {test_mse/max(train_mse, 1e-30):.2f} (>2×), possible overfitting")

    return {
        "train_mse": float(train_mse),
        "val_mse":   float(val_mse),
        "test_mse":  float(test_mse),
        "steps_run": int(history[-1]["step"]) if history else 0,
    }, history


# -----------------------
# Shape/parameter matching
# -----------------------

def build_mlp_model(
    *,
    depth: int,
    width: int,
    cfg: TrainCfg,
    pad_to_target: bool = True,
) -> TinyMLP:
    in_dim  = cfg.in_dim
    out_dim = cfg.out_dim
    tmp     = TinyMLP(in_dim=in_dim, out_dim=out_dim, depth=depth, width=width,
                      activation=cfg.activation, dropout=cfg.dropout, pad_params=0)
    active = count_params(tmp)
    pad    = 0
    if pad_to_target:
        if active > cfg.target_params:
            raise ValueError(f"Active params {active} > target {cfg.target_params} for depth={depth}, width={width}.")
        pad = int(cfg.target_params - active)
    return TinyMLP(in_dim=in_dim, out_dim=out_dim, depth=depth, width=width,
                   activation=cfg.activation, dropout=cfg.dropout, pad_params=pad)


def find_width_for_target_params(*, depth: int, cfg: TrainCfg) -> Tuple[int, int]:
    mul = int(cfg.width_multiple)

    def to_valid(w: int) -> int:
        return max(mul, (w // mul) * mul)

    w_min = to_valid(max(cfg.min_width, mul))
    w_max = to_valid(max(cfg.max_width, mul))

    in_dim  = cfg.in_dim
    out_dim = cfg.out_dim

    def active_params(w: int) -> int:
        m = TinyMLP(in_dim=in_dim, out_dim=out_dim, depth=depth, width=w,
                    activation=cfg.activation, dropout=cfg.dropout, pad_params=0)
        return count_params(m)

    if active_params(w_min) > cfg.target_params:
        raise ValueError(f"width={w_min} already exceeds target_params={cfg.target_params} at depth={depth}.")
    if active_params(w_max) <= cfg.target_params:
        return w_max, active_params(w_max)

    lo, hi   = w_min, w_max
    best_w   = w_min
    best_a   = active_params(best_w)
    while lo <= hi:
        mid = to_valid((lo + hi) // 2)
        a   = active_params(mid)
        if a <= cfg.target_params:
            best_w, best_a = mid, a
            lo = mid + mul
        else:
            hi = mid - mul

    candidates = []
    for w in [max(w_min, best_w - mul), best_w, min(w_max, best_w + mul)]:
        a = active_params(w)
        if a <= cfg.target_params:
            candidates.append((abs(cfg.target_params - a), w, a))
    candidates.sort(key=lambda t: t[0])
    _, w, a = candidates[0]
    return int(w), int(a)


# -----------------------
# Plotting
# -----------------------

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def save_curve(history: List[Dict[str, float]], out_dir: str, stem: str) -> None:
    """Save per-shape training curve to CSV and PNG (log-scale y for readability in appendix)."""
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
    plt.yscale("log")   # log scale for appendix readability (not used in correlation)
    plt.xlabel("step")
    plt.ylabel("MSE  (log scale)")
    plt.title(stem)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def scatter_plot(x, y, xlabel, ylabel, title, outpath, depths=None, r=None, r_label="Pearson r"):
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


# -----------------------
# Main
# -----------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",   type=str, default="./results_mlp_pde_shape_sweep")

    p.add_argument("--target_params",  type=int, default=1_000_000)
    p.add_argument("--depth_list",     type=str, default="3,4,5,6,7,8,9,10,11,12")
    p.add_argument("--min_width",      type=int, default=256)
    p.add_argument("--max_width",      type=int, default=2048)
    p.add_argument("--width_multiple", type=int, default=16)

    p.add_argument("--in_dim",         type=int, default=64,
                   help="Input dimension for student and data generation.")
    p.add_argument("--out_dim",        type=int, default=64,
                   help="Output dimension for student and data generation.")
    p.add_argument("--teacher_hidden", type=int, default=256,
                   help="Hidden size of the random 4-layer teacher MLP.")
    p.add_argument("--train_size",     type=int, default=0,
                   help="0 = auto-match unique_targets ≈ data_ratio×N.")
    p.add_argument("--val_size",       type=int, default=2000)
    p.add_argument("--test_size",      type=int, default=2000)

    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--steps",        type=int,   default=0,
                   help="0 = auto-compute from D=data_ratio×N.")
    p.add_argument("--data_ratio",   type=float, default=20.0)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--warmup_steps", type=int,   default=0)
    p.add_argument("--eval_every",   type=int,   default=0,
                   help="Eval/log interval. 0 = auto (steps // 100).")
    p.add_argument("--grad_clip",    type=float, default=1.0)

    p.add_argument("--activation",   type=str,   default="gelu",
                   choices=["gelu", "silu", "tanh", "relu"])
    p.add_argument("--dropout",      type=float, default=0.0)

    p.add_argument("--agop_batch",        type=int, default=256)
    p.add_argument("--agop_proj_samples", type=int, default=64)
    p.add_argument("--max_padding_ratio", type=float, default=0.15)
    p.add_argument("--max_train_factor",  type=float, default=3.0)
    p.add_argument("--fit_patience",      type=int,   default=8)

    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    cfg = TrainCfg()
    cfg.target_params  = int(args.target_params)
    cfg.depth_list     = parse_int_list(args.depth_list)
    cfg.min_width      = int(args.min_width)
    cfg.max_width      = int(args.max_width)
    cfg.width_multiple = int(args.width_multiple)

    cfg.in_dim         = int(args.in_dim)
    cfg.out_dim        = int(args.out_dim)
    cfg.teacher_hidden = int(args.teacher_hidden)
    cfg.train_size     = int(args.train_size)
    cfg.val_size       = int(args.val_size)
    cfg.test_size      = int(args.test_size)

    cfg.lr           = float(args.lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.steps        = int(args.steps)
    cfg.data_ratio   = float(args.data_ratio)
    cfg.batch_size   = int(args.batch_size)
    cfg.warmup_steps = int(args.warmup_steps)
    cfg.eval_every   = int(args.eval_every)
    cfg.grad_clip    = float(args.grad_clip)
    cfg.activation   = str(args.activation)
    cfg.dropout      = float(args.dropout)

    cfg.agop_batch        = int(args.agop_batch)
    cfg.agop_proj_samples = int(args.agop_proj_samples)
    cfg.max_padding_ratio = float(args.max_padding_ratio)
    cfg.max_train_factor  = float(args.max_train_factor)
    cfg.fit_patience      = int(args.fit_patience)
    cfg.seed              = int(args.seed)

    set_global_seed(cfg.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    curve_dir = os.path.join(args.out_dir, "curves")

    if len(cfg.depth_list) != 10:
        raise ValueError(f"depth_list must contain exactly 10 shapes, got {len(cfg.depth_list)}.")

    # Auto train_size: unique scalars ≈ D = 20N
    if cfg.train_size <= 0:
        cfg.train_size = int(math.ceil(cfg.data_ratio * cfg.target_params / float(cfg.out_dim)))

    # Auto steps: D = steps × batch × out_dim ≈ 20N
    tokens_per_step = int(cfg.batch_size) * int(cfg.out_dim)
    if cfg.steps <= 0:
        cfg.steps = int(math.ceil(cfg.data_ratio * cfg.target_params / float(tokens_per_step)))
    if cfg.warmup_steps <= 0:
        cfg.warmup_steps = max(50, int(0.1 * cfg.steps))
    if cfg.eval_every <= 0:
        cfg.eval_every = max(50, cfg.steps // 100)

    D_actual       = int(cfg.steps) * tokens_per_step
    unique_tokens  = int(cfg.train_size) * int(cfg.out_dim)
    approx_epochs  = float(cfg.steps) * float(cfg.batch_size) / float(cfg.train_size)

    print("========== Budget (MLP / teacher-student regression) ==========")
    print(f"target_params N     = {cfg.target_params:,}")
    print(f"in_dim / out_dim    = {cfg.in_dim} / {cfg.out_dim}")
    print(f"AGOP (input-space)  = J^T J ∈ R^{{{cfg.in_dim}×{cfg.in_dim}}} "
          f"({cfg.in_dim*(cfg.in_dim-1)//2} unique off-diag entries)")
    print(f"teacher_hidden      = {cfg.teacher_hidden}  (4-layer teacher)")
    print(f"train_size          = {cfg.train_size:,}  samples")
    print(f"tokens_per_step     = batch×out_dim = {cfg.batch_size}×{cfg.out_dim} = {tokens_per_step:,}")
    print(f"base steps          = {cfg.steps:,}")
    print(f"approx epochs       = {approx_epochs:.1f}")
    print(f"D (base)            = {D_actual:,}   D/N = {D_actual/cfg.target_params:.1f}×")
    print(f"unique targets      = {unique_tokens:,}   {unique_tokens/cfg.target_params:.1f}×N")
    print("===============================================================")

    teacher_seed = cfg.seed + 99999   # fixed teacher — same for all shapes and runs
    train_ds = TeacherStudentDataset(
        size=cfg.train_size, in_dim=cfg.in_dim, out_dim=cfg.out_dim,
        teacher_hidden=cfg.teacher_hidden,
        teacher_seed=teacher_seed, data_seed=cfg.seed + 123,
    )
    test_ds = TeacherStudentDataset(
        size=cfg.test_size, in_dim=cfg.in_dim, out_dim=cfg.out_dim,
        teacher_hidden=cfg.teacher_hidden,
        teacher_seed=teacher_seed, data_seed=cfg.seed + 456,
    )
    val_ds = TeacherStudentDataset(
        size=cfg.val_size, in_dim=cfg.in_dim, out_dim=cfg.out_dim,
        teacher_hidden=cfg.teacher_hidden,
        teacher_seed=teacher_seed, data_seed=cfg.seed + 789,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)

    agop_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.agop_batch, shuffle=True, drop_last=True, num_workers=0)
    agop_z0, _ = next(iter(agop_loader))

    results: List[Dict[str, float]] = []

    print("\n========== Shape sweep ==========")
    for i, depth in enumerate(cfg.depth_list):
        width, active = find_width_for_target_params(depth=depth, cfg=cfg)
        model  = build_mlp_model(depth=depth, width=width, cfg=cfg, pad_to_target=True)
        total  = count_params(model)
        pad    = int(total - active)
        pad_ratio = pad / max(1, total)
        if pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] depth={depth}: padding_ratio={pad_ratio:.3f} exceeds max_padding_ratio={cfg.max_padding_ratio:.3f}")
            continue

        print(f"\n[{i+1:02d}/{len(cfg.depth_list)}] depth={depth:2d}  width={width:4d}  "
              f"active={active:,}  pad={pad:,}  total={total:,}")

        metrics, history = train_one_model(model, train_loader, val_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"mlp_depth{depth}_width{width}")

        z    = agop_z0.to(device)
        agop = estimate_agop_wrt_inputs(model.to(device), z, proj_samples=cfg.agop_proj_samples)
        # agop ∈ R^{in_dim × in_dim} (input-space J^T J)
        off_e, off_r = agop_offdiag_metrics(agop)

        row = {
            "depth":               int(depth),
            "width":               int(width),
            "active_params":       int(active),
            "pad_params":          int(pad),
            "total_params":        int(total),
            "padding_ratio":       float(pad / max(1, total)),
            "train_mse":           float(metrics["train_mse"]),
            "val_mse":             float(metrics["val_mse"]),
            "test_mse":            float(metrics["test_mse"]),
            "steps_run":           int(metrics["steps_run"]),
            "agop_offdiag_energy": float(off_e),
            "agop_offdiag_ratio":  float(off_r),
        }
        results.append(row)

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
    print(f"  Pearson (AOFE=offdiag_energy,  test_mse)  = {p_aofe:.4f}   Spearman = {s_aofe:.4f}")
    print(f"  Pearson (AOFE_ratio=offdiag_ratio, test_mse) = {p_aofe_ratio:.4f}   Spearman = {s_aofe_ratio:.4f}")

    # ---------- Save results ----------
    csv_path = os.path.join(args.out_dir, "results.csv")
    npy_path = os.path.join(args.out_dir, "results.npy")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    np.save(npy_path, results, allow_pickle=True)
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")

    # ---------- Scatter plots (linear y-axis, with Pearson r annotation) ----------
    scatter_plot(
        off_energy, test_mse,
        xlabel="AOFE  (input-space AGOP off-diagonal energy)",
        ylabel="Test MSE (per scalar)",
        title=f"Teacher-Student MLP: test MSE vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_energy.png"),
        depths=depths_arr,
        r=p_aofe, r_label="Pearson r (AOFE, loss)",
    )
    scatter_plot(
        off_ratio, test_mse,
        xlabel="AOFE ratio  (input-space AGOP off-diagonal ratio)",
        ylabel="Test MSE (per scalar)",
        title=f"Teacher-Student MLP: test MSE vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_ratio.png"),
        depths=depths_arr,
        r=p_aofe_ratio, r_label="Pearson r (AOFE_ratio, loss)",
    )
    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
