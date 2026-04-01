#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp_shape_sweep_supervised_pde_agop.py
=====================================

Goal
----
Fixed-parameter MLP shape sweep on supervised PDE (analytic heat equation operator learning).
Tests the hypothesis: under fixed N, different (depth, width) shapes reach similar loss,
mediated by AOFE (AGOP Off-diagonal Frobenius Energy).

Task (Supervised PDE — Heat Equation)
--------------------------------------
1D Heat equation with Dirichlet BC, analytic solution:
    u_t = α u_xx ,  x∈[0,1], t∈[0, t_max]
    u(x,0) = Σ_{k=1..K} a_k sin(kπx)
    u(x,t) = Σ a_k exp(-α(kπ)²t) sin(kπx)

We learn the solution operator slice:
    input  z = [a_1,...,a_K, t] ∈ R^{K+1}
    output y = [u(x_1,t),...,u(x_M,t)] ∈ R^M   (fixed spatial grid)

Output-space AGOP:
    J = d(y)/d(z)  ∈ R^{M×(K+1)}
    AGOP = E_data[J J^T] ∈ R^{M×M}   — fixed dimension across all shapes.

Data budget
-----------
"Token" = one scalar supervised target:
    tokens_per_step = batch_size × M (out_grid)
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
# Dataset: supervised PDE (analytic heat equation)
# -----------------------

class HeatEquationOperatorDataset(torch.utils.data.Dataset):
    """
    Supervised heat equation operator dataset.
    Each sample:
      z = [a_1..a_K, t]  (K Fourier coefficients + time)
      y = u(x_grid, t)   (M fixed spatial grid values)

    Coefficients and times are fixed at construction time (reproducible).
    y is computed analytically on-the-fly at __getitem__.
    """
    def __init__(
        self,
        *,
        size: int,
        modes: int,
        out_grid: int,
        alpha: float,
        t_max: float,
        seed: int,
        coeff_std_decay: float = 1.0,
        coeff_scale: float = 1.0,
        include_endpoints: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.size     = int(size)
        self.modes    = int(modes)
        self.out_grid = int(out_grid)
        self.alpha    = float(alpha)
        self.t_max    = float(t_max)
        self.dtype    = dtype

        if include_endpoints:
            x = torch.linspace(0.0, 1.0, self.out_grid, dtype=dtype)
        else:
            x = torch.linspace(0.0, 1.0, self.out_grid + 2, dtype=dtype)[1:-1]
        self.x_grid = x

        k = torch.arange(1, self.modes + 1, dtype=dtype).unsqueeze(1)
        x_row = self.x_grid.unsqueeze(0)
        self.basis = torch.sin(math.pi * k * x_row)
        self.lam   = self.alpha * (math.pi * k.squeeze(1)) ** 2

        g = torch.Generator()
        g.manual_seed(int(seed))
        kk  = torch.arange(1, self.modes + 1, dtype=dtype)
        std = (coeff_scale / (kk ** float(coeff_std_decay))).to(dtype)
        self.coeffs = torch.randn((self.size, self.modes), generator=g, dtype=dtype) * std.unsqueeze(0)
        self.times  = torch.rand((self.size, 1), generator=g, dtype=dtype) * self.t_max

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.coeffs[idx]
        t = self.times[idx].squeeze(0)
        decay  = torch.exp(-self.lam * t)
        scaled = a * decay
        y = scaled @ self.basis
        z = torch.cat([a, t.view(1)], dim=0)
        return z, y


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
    proj_samples: int = 16,
) -> torch.Tensor:
    """
    AGOP = E_data[J J^T],  J = d(y)/d(z)
    y ∈ R^M, z ∈ R^{d_in}  →  AGOP ∈ R^{M×M}  (fixed across shapes)
    Estimated via forward-mode JVP random projections.
    """
    device = z.device
    model.eval()
    B, d_in = z.shape
    with torch.no_grad():
        y0 = model(z)
        M  = y0.shape[1]

    z0   = z.detach()
    agop = torch.zeros((M, M), device=device, dtype=torch.float32)

    def fwd(z_in: torch.Tensor) -> torch.Tensor:
        return model(z_in)

    for _ in range(int(proj_samples)):
        u = torch.randn_like(z0)
        _, Ju = torch.autograd.functional.jvp(fwd, (z0,), (u,), create_graph=False, strict=False)
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
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 0
    data_ratio: float = 20.0
    warmup_steps: int = 0
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 1000

    modes: int = 32
    out_grid: int = 128
    alpha: float = 0.1
    t_max: float = 1.0
    train_size: int = 0
    test_size: int = 1000
    coeff_std_decay: float = 1.0
    coeff_scale: float = 1.0
    include_endpoints: bool = True

    target_params: int = 1_000_000
    depth_list: List[int] = None
    width_multiple: int = 16
    min_width: int = 192
    max_width: int = 2048
    activation: str = "gelu"
    dropout: float = 0.0

    agop_batch: int = 256
    agop_proj_samples: int = 16

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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_iter = iter(train_loader)

    t0       = time.time()
    history: List[Dict[str, float]] = []
    max_steps = int(cfg.steps)  # strict D=20N budget — no extension

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
            test_mse  = evaluate_mse(model, test_loader,  device, max_batches=None)
            history.append({
                "step":      int(step + 1),
                "lr":        float(lr),
                "train_mse": float(train_mse),
                "test_mse":  float(test_mse),
            })
            dt = time.time() - t0
            gap = test_mse - train_mse
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_mse={train_mse:.6e}  test_mse={test_mse:.6e}  "
                f"gap={gap:+.3e}  time={dt:.1f}s"
            )

    # Final evaluation at end of strict D=20N budget
    train_mse = evaluate_mse(model, train_loader, device, max_batches=None)
    test_mse  = evaluate_mse(model, test_loader,  device, max_batches=None)

    gap = test_mse - train_mse
    if gap > 0.5 * test_mse:
        print(f"  [WARNING] test_mse / train_mse = {test_mse/max(train_mse, 1e-30):.2f} (>2×), possible overfitting")

    return {
        "train_mse": float(train_mse),
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
    in_dim  = cfg.modes + 1
    out_dim = cfg.out_grid
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

    in_dim  = cfg.modes + 1
    out_dim = cfg.out_grid

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
    test_vals  = [row["test_mse"]  for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_vals, label="train_mse")
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
    p.add_argument("--min_width",      type=int, default=192)
    p.add_argument("--max_width",      type=int, default=2048)
    p.add_argument("--width_multiple", type=int, default=16)

    p.add_argument("--modes",            type=int,   default=16,
                   help="Number of Fourier modes (input dim = modes+1). Default 16.")
    p.add_argument("--out_grid",         type=int,   default=32,
                   help="Spatial output resolution. Default 32 (aligns tokens/step with Transformer seq_len=32).")
    p.add_argument("--alpha",            type=float, default=0.1)
    p.add_argument("--t_max",            type=float, default=1.0)
    p.add_argument("--train_size",       type=int,   default=0,
                   help="0 = auto-match unique_targets ≈ data_ratio×N.")
    p.add_argument("--test_size",        type=int,   default=1000)
    p.add_argument("--coeff_std_decay",  type=float, default=1.0)
    p.add_argument("--coeff_scale",      type=float, default=1.0)
    p.add_argument("--include_endpoints", action="store_true")
    p.add_argument("--no_endpoints",     action="store_true")

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
    p.add_argument("--agop_proj_samples", type=int, default=16)

    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    cfg = TrainCfg()
    cfg.target_params  = int(args.target_params)
    cfg.depth_list     = parse_int_list(args.depth_list)
    cfg.min_width      = int(args.min_width)
    cfg.max_width      = int(args.max_width)
    cfg.width_multiple = int(args.width_multiple)

    cfg.modes           = int(args.modes)
    cfg.out_grid        = int(args.out_grid)
    cfg.alpha           = float(args.alpha)
    cfg.t_max           = float(args.t_max)
    cfg.train_size      = int(args.train_size)
    cfg.test_size       = int(args.test_size)
    cfg.coeff_std_decay = float(args.coeff_std_decay)
    cfg.coeff_scale     = float(args.coeff_scale)
    cfg.include_endpoints = True
    if args.no_endpoints:      cfg.include_endpoints = False
    if args.include_endpoints: cfg.include_endpoints = True

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
    cfg.seed              = int(args.seed)

    set_global_seed(cfg.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    curve_dir = os.path.join(args.out_dir, "curves")

    if len(cfg.depth_list) != 10:
        raise ValueError(f"depth_list must contain exactly 10 shapes, got {len(cfg.depth_list)}.")

    # Auto train_size: unique scalars ≈ D = 20N
    if cfg.train_size <= 0:
        cfg.train_size = int(math.ceil(cfg.data_ratio * cfg.target_params / float(cfg.out_grid)))

    # Auto steps: D = steps × batch × out_grid ≈ 20N
    tokens_per_step = int(cfg.batch_size) * int(cfg.out_grid)
    if cfg.steps <= 0:
        cfg.steps = int(math.ceil(cfg.data_ratio * cfg.target_params / float(tokens_per_step)))
    if cfg.warmup_steps <= 0:
        cfg.warmup_steps = max(50, int(0.1 * cfg.steps))
    if cfg.eval_every <= 0:
        cfg.eval_every = max(50, cfg.steps // 100)

    D_actual       = int(cfg.steps) * tokens_per_step
    unique_tokens  = int(cfg.train_size) * int(cfg.out_grid)
    approx_epochs  = float(cfg.steps) * float(cfg.batch_size) / float(cfg.train_size)

    print("========== Budget (MLP / supervised PDE) ==========")
    print(f"target_params N     = {cfg.target_params:,}")
    print(f"train_size          = {cfg.train_size:,}  samples")
    print(f"tokens_per_step     = batch×out_grid = {cfg.batch_size}×{cfg.out_grid} = {tokens_per_step:,}")
    print(f"base steps          = {cfg.steps:,}")
    print(f"approx epochs       = {approx_epochs:.1f}")
    print(f"D (base)            = {D_actual:,}   D/N = {D_actual/cfg.target_params:.1f}×")
    print(f"unique targets      = {unique_tokens:,}   {unique_tokens/cfg.target_params:.1f}×N")
    print("====================================================")

    train_ds = HeatEquationOperatorDataset(
        size=cfg.train_size, modes=cfg.modes, out_grid=cfg.out_grid,
        alpha=cfg.alpha, t_max=cfg.t_max, seed=cfg.seed + 123,
        coeff_std_decay=cfg.coeff_std_decay, coeff_scale=cfg.coeff_scale,
        include_endpoints=cfg.include_endpoints,
    )
    test_ds = HeatEquationOperatorDataset(
        size=cfg.test_size, modes=cfg.modes, out_grid=cfg.out_grid,
        alpha=cfg.alpha, t_max=cfg.t_max, seed=cfg.seed + 456,
        coeff_std_decay=cfg.coeff_std_decay, coeff_scale=cfg.coeff_scale,
        include_endpoints=cfg.include_endpoints,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
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

        print(f"\n[{i+1:02d}/{len(cfg.depth_list)}] depth={depth:2d}  width={width:4d}  "
              f"active={active:,}  pad={pad:,}  total={total:,}")

        metrics, history = train_one_model(model, train_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"mlp_depth{depth}_width{width}")

        z    = agop_z0.to(device)
        agop = estimate_agop_wrt_inputs(model.to(device), z, proj_samples=cfg.agop_proj_samples)
        off_e, off_r = agop_offdiag_metrics(agop)

        row = {
            "depth":               int(depth),
            "width":               int(width),
            "active_params":       int(active),
            "pad_params":          int(pad),
            "total_params":        int(total),
            "padding_ratio":       float(pad / max(1, total)),
            "train_mse":           float(metrics["train_mse"]),
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
        xlabel="AOFE  (AGOP off-diagonal energy)",
        ylabel="Test MSE (per scalar)",
        title=f"PDE-MLP: test MSE vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_energy.png"),
        depths=depths_arr,
        r=p_aofe, r_label="Pearson r (AOFE, loss)",
    )
    scatter_plot(
        off_ratio, test_mse,
        xlabel="AOFE ratio  (AGOP off-diagonal ratio)",
        ylabel="Test MSE (per scalar)",
        title=f"PDE-MLP: test MSE vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_ratio.png"),
        depths=depths_arr,
        r=p_aofe_ratio, r_label="Pearson r (AOFE_ratio, loss)",
    )
    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
