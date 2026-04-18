#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rnn_shape_sweep_mackeyglass_superposition_agop.py
=================================================

Goal
----
Fixed-parameter GRU shape sweep using teacher-student regression. Tests the
AOFE hypothesis: under fixed N, different (depth, hidden_size) shapes reach
different fitted MSE, mediated by AOFE (AGOP Off-diagonal Frobenius Energy).

Task: Teacher-Student Feature Regression
-----------------------------------------
A fixed, randomly-initialised 2-layer GRU (hidden=64, input=1, frozen)
maps random 80-step scalar sequences to 64-dimensional features via a
linear output head. The student (varying depth/hidden) must match the
teacher's 64-dim output from the same random sequence.

Why teacher-student (replacing VAR-64 forecasting)
---------------------------------------------------
VAR-64 multi-variate forecasting causes non-monotonic depth-vs-MSE behaviour:
some intermediate shapes (e.g. depth=5) fail catastrophically (val_mse ≈ 1.5)
while shallower and deeper shapes converge. This creates outliers that destroy
the AOFE-loss correlation. Root cause: deep GRUs overfit specific trajectory
patterns even with independent-series training.

Teacher-student is a FIXED supervised regression task (deterministic teacher).
No temporal dynamics to learn — the bottleneck is purely the hidden state
capacity relative to the 64 teacher features. This is identical in spirit
to the MLP experiment (r=0.984) and gives monotonic MSE-vs-shape scaling.

Width bottleneck mechanism
--------------------------
  • Teacher output: 64-dim features at last step h_T ∈ R^{64}
  • Student: depth-layer GRU → last hidden h_T → head = Linear(hidden, 64)
  • Wide student  (depth=3,  hidden≈255): hidden/64 ≈ 4.0 → features fit
    in distinct h_T subspaces → low AOFE + low MSE
  • Narrow student (depth=12, hidden≈119): hidden/64 ≈ 1.9 → features
    must share subspaces → high AOFE + high MSE

Output-space AGOP (fixed 64×64 across all shapes)
---------------------------------------------------
  J = d(head(h_T) ∈ R^{64})/d(x_in ∈ R^{T×1})
  AGOP = E_data[J J^T] ∈ R^{64×64}   — fixed, independent of hidden size

  Estimated via JVP with cuDNN disabled (required for stacked GRU backprop).

Training protocol (strict D=20N)
---------------------------------
supervised_per_sample = TEACHER_OUT = 64.
train_size = 20N / 64 ≈ 312 500 unique random sequences (≈ 1 epoch).
Steps auto-computed from D = 20N supervised outputs. Best val_mse state
checkpointed for fitted (not overfitting) regime.
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

from experiments.compositional_generators import generate_compositional_sequences


# ---------------------
# Constants
# ---------------------

TEACHER_OUT = 64   # teacher output dimension → AGOP ∈ R^{64×64}
SEQ_LEN     = 80   # sequence length (steps per sequence)
INP_DIM     = 1    # input dimension per step (scalar sequences)


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
# Teacher: 4-layer MLP (architecturally mismatched with GRU students)
# -----------------------

class TeacherMLP(nn.Module):
    """
    4-layer MLP: R^{SEQ_LEN} → R^{256} → R^{256} → R^{256} → R^{TEACHER_OUT}.
    Processes ALL 80 input values SIMULTANEOUSLY (not sequentially).
    Random Kaiming init (default PyTorch), ALL parameters frozen.

    Why MLP teacher instead of GRU teacher:
      A GRU teacher with hidden=64 produces near-trivially-replicable features:
      even the narrowest student (hidden=119 >> 64) achieves MSE ≈ 1e-5
      (effectively zero) — all shapes converge to the same solution, giving
      no MSE variation and therefore no AOFE correlation.

    Architectural mismatch: MLP sees ALL 80 inputs at once; GRU students
    process them ONE STEP AT A TIME with limited hidden state. Even the
    widest student (hidden=252) cannot fully replicate the MLP's global
    interactions within D=20N training steps → natural partial-fit regime:
      wide GRU  → rich hidden state → better approximation  → lower MSE
      narrow GRU → compressed state → partial approximation → higher MSE
    """
    def __init__(
        self,
        *,
        in_dim: int = SEQ_LEN,
        hidden: int = 256,
        out_dim: int = TEACHER_OUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True), nn.GELU(),
            nn.Linear(hidden, hidden, bias=True), nn.GELU(),
            nn.Linear(hidden, hidden, bias=True), nn.GELU(),
            nn.Linear(hidden, out_dim, bias=True),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, SEQ_LEN, 1] → flatten → [B, SEQ_LEN] → [B, TEACHER_OUT]"""
        B = x.shape[0]
        return self.net(x.view(B, -1))   # [B, TEACHER_OUT]


def build_teacher(*, seed: int, device: torch.device) -> TeacherMLP:
    set_global_seed(seed)
    teacher = TeacherMLP()
    teacher.eval()
    teacher.to(device)
    print(f"Teacher params (frozen): {count_params(teacher):,}")
    return teacher


@torch.no_grad()
def precompute_teacher_outputs(
    teacher: TeacherGRU,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Run teacher in batches and return [N, TEACHER_OUT] targets."""
    out_list = []
    for i in range(0, len(x), batch_size):
        xb = x[i: i + batch_size].to(device)
        yb = teacher(xb)   # [B, TEACHER_OUT]
        out_list.append(yb.cpu())
    return torch.cat(out_list, dim=0)   # [N, TEACHER_OUT]


def make_dataset(
    n: int, seed: int, teacher: TeacherMLP, device: torch.device,
    *,
    y_scale: Optional[float] = None,
) -> Tuple[torch.utils.data.TensorDataset, float]:
    """
    Generate compositional scalar sequences [n, SEQ_LEN, 1] with repeated motifs
    and pre-compute the frozen teacher's outputs [n, TEACHER_OUT].

    Outputs are NORMALISED to unit std (scalar normalisation over all
    elements of the training set) so the baseline MSE ≈ 1.0 regardless
    of the teacher's weight scale.  Pass y_scale from the training set
    to val/test so all splits use identical normalisation.

    Returns: (TensorDataset, y_scale)
    """
    x_t = generate_compositional_sequences(
        size=n,
        seq_len=SEQ_LEN,
        structure_seed=12345,
        sample_seed=int(seed),
        segment_len=10,
        latent_dim=8,
        motif_count=4,
        dtype=torch.float32,
    )
    y_t  = precompute_teacher_outputs(teacher, x_t, device, batch_size=512)
    if y_scale is None:
        y_scale = float(y_t.std().item())
        if y_scale < 1e-8:
            y_scale = 1.0
    y_t = y_t / y_scale
    return torch.utils.data.TensorDataset(x_t, y_t), y_scale


# -----------------------
# Student model: stacked GRU with teacher-matching head
# -----------------------

class TinyRNNRegressor(nn.Module):
    """
    Depth-layer GRU for teacher-student regression.

    forward(x: [B, T, 1]) → head(h_T) ∈ R^{B, TEACHER_OUT}

    Bottleneck: student's final hidden state h_T ∈ R^{B, hidden_size}
    must encode 64 teacher features. Wide models (large hidden_size)
    can represent features independently; narrow models must superpose.
    """
    def __init__(
        self,
        *,
        hidden_size: int,
        depth: int,
        rnn_type: str = "gru",
        dropout: float = 0.0,
        pad_params: int = 0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.depth       = int(depth)
        self.rnn_type    = str(rnn_type).lower()

        rnn_kwargs = dict(
            input_size=INP_DIM, hidden_size=self.hidden_size,
            num_layers=self.depth, batch_first=True,
            dropout=dropout if self.depth > 1 else 0.0,
            bidirectional=False,
        )
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(nonlinearity="tanh", **rnn_kwargs)
        else:
            raise ValueError("rnn_type must be one of: rnn, gru, lstm")

        self.head = nn.Linear(self.hidden_size, TEACHER_OUT, bias=True)

        self._pad_params = None
        if pad_params > 0:
            self._pad_params = nn.Parameter(torch.zeros(int(pad_params)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 1] → [B, TEACHER_OUT]"""
        _, h_n = self.rnn(x)
        if self.rnn_type == "lstm":
            h_n = h_n[0]   # extract hidden from (h, c)
        h_last = h_n[-1]   # [B, hidden_size] — last layer, final step
        return self.head(h_last)   # [B, TEACHER_OUT]


# -----------------------
# AGOP estimation (wrt input sequence)
# -----------------------

def estimate_agop_wrt_inputs(
    model: TinyRNNRegressor,
    x: torch.Tensor,
    *,
    proj_samples: int = 128,
    max_agop_dim: int = 4096,
) -> torch.Tensor:
    """
    AGOP = E[J J^T],  J = d(head(h_T) ∈ R^{64})/d(x_in ∈ R^{T×1}).
    AGOP ∈ R^{64×64} — fixed dimension across all model shapes.

    cuDNN must be disabled for JVP to work with stacked RNNs.
    model.train() is required for cuDNN RNN backward compatibility.
    """
    device = x.device
    model.train()   # cuDNN RNN requires train mode for JVP
    B = x.shape[0]
    D_out = TEACHER_OUT

    assert D_out <= max_agop_dim

    x0   = x.detach()
    agop = torch.zeros((D_out, D_out), device=device, dtype=torch.float32)

    def fwd(x_in: torch.Tensor) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            return model(x_in)   # [B, TEACHER_OUT]

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
    weight_decay: float = 1e-4
    steps: int = 0
    warmup_steps: int = 200
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 250

    data_ratio: float = 20.0
    train_size: int = 0
    val_size: int = 3000
    test_size: int = 3000

    target_params: int = 1_000_000
    depth_list: List[int] = None
    rnn_type: str = "gru"
    dropout: float = 0.0
    hidden_step: int = 8
    min_hidden: int = 64
    max_hidden: int = 1024
    max_padding_ratio: float = 0.20
    max_train_factor: float = 3.0
    fit_patience: int = 8

    agop_batch: int = 256
    agop_proj_samples: int = 128
    max_agop_dim: int = 4096

    seed: int = 0


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    t = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate_ts(
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
        x, y  = x.to(device), y.to(device)
        pred  = model(x)   # [B, TEACHER_OUT]
        se    = F.mse_loss(pred, y, reduction="sum")
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
    Train on teacher-student MSE until val_mse plateaus after D=20N budget.
    Checkpoint best val_mse state to stay in fitted regime.
    """
    model.to(device)
    model.train()

    opt        = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_iter = iter(train_loader)

    t0         = time.time()
    history: List[Dict[str, float]] = []
    min_steps  = int(cfg.steps)
    max_steps  = max(min_steps, int(math.ceil(cfg.max_train_factor * cfg.steps)))
    best_val   = float("inf")
    best_state = None
    stale_evals = 0

    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, max_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        pred = model(x)   # [B, TEACHER_OUT]
        loss = F.mse_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == max_steps:
            train_mse = evaluate_ts(model, train_loader, device, max_batches=30)
            val_mse   = evaluate_ts(model, val_loader,   device)
            test_mse  = evaluate_ts(model, test_loader,  device)
            history.append({
                "step":      int(step + 1),
                "lr":        float(lr),
                "train_mse": float(train_mse),
                "val_mse":   float(val_mse),
                "test_mse":  float(test_mse),
            })
            dt  = time.time() - t0
            gap = val_mse - train_mse
            rmse = math.sqrt(max(0.0, val_mse))
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
                f"test_mse={test_mse:.6f}  rmse={rmse:.6f}  "
                f"gap={gap:+.6f}  time={dt:.1f}s"
            )
            if val_mse + 1e-10 < best_val:
                best_val    = float(val_mse)
                stale_evals = 0
                best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_evals += 1
            if (step + 1) >= min_steps and stale_evals >= int(cfg.fit_patience):
                break
            model.train()   # cuDNN GRU requires train mode for backward

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse = evaluate_ts(model, train_loader, device)
    val_mse   = evaluate_ts(model, val_loader,   device)
    test_mse  = evaluate_ts(model, test_loader,  device)

    gap = val_mse - train_mse
    if gap > 0.5:
        print(f"  [WARNING] val_mse - train_mse = {gap:.5f} (>0.5), possible overfitting")

    return {
        "train_mse": float(train_mse),
        "val_mse":   float(val_mse),
        "test_mse":  float(test_mse),
        "steps_run": int(history[-1]["step"]) if history else 0,
    }, history


# -----------------------
# Shape/parameter matching
# -----------------------

def count_student_params(hidden: int, depth: int, rnn_type: str = "gru") -> int:
    m = TinyRNNRegressor(hidden_size=hidden, depth=depth, rnn_type=rnn_type, pad_params=0)
    return count_params(m)


def find_hidden_for_target_params(
    *,
    depth: int,
    cfg: TrainCfg,
) -> Tuple[int, int]:
    """Binary search for hidden_size that fits within target_params. Returns (hidden, active)."""
    lo, hi = cfg.min_hidden, cfg.max_hidden
    step   = cfg.hidden_step
    best_h, best_a = lo, count_student_params(lo, depth, cfg.rnn_type)

    if best_a > cfg.target_params:
        raise ValueError(
            f"min_hidden={lo} already exceeds target_params={cfg.target_params} at depth={depth}."
        )

    while lo <= hi:
        mid = ((lo + hi) // 2 // step) * step
        if mid < cfg.min_hidden:
            mid = cfg.min_hidden
        a = count_student_params(mid, depth, cfg.rnn_type)
        if a <= cfg.target_params:
            best_h, best_a = mid, a
            lo = mid + step
        else:
            hi = mid - step

    return int(best_h), int(best_a)


def build_student_model(
    *,
    hidden: int,
    depth: int,
    cfg: TrainCfg,
    active: int,
    pad_to_target: bool = True,
) -> TinyRNNRegressor:
    pad = int(cfg.target_params - active) if pad_to_target else 0
    return TinyRNNRegressor(
        hidden_size=hidden, depth=depth, rnn_type=cfg.rnn_type,
        dropout=cfg.dropout, pad_params=pad,
    )


# -----------------------
# Plotting
# -----------------------

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
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_curve(history: List[Dict[str, float]], out_dir: str, stem: str) -> None:
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
    plt.xlabel("step")
    plt.ylabel("MSE (teacher-student)")
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
    parser.add_argument("--out_dir",  type=str, default="./results_rnn_shape_sweep")
    parser.add_argument("--device",   type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",     type=int, default=0)

    parser.add_argument("--target_params", type=int,   default=1_000_000)
    parser.add_argument("--depth_list",    type=str,   default="3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--rnn_type",      type=str,   default="gru")
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--hidden_step",   type=int,   default=8)
    parser.add_argument("--min_hidden",    type=int,   default=64)
    parser.add_argument("--max_hidden",    type=int,   default=1024)

    parser.add_argument("--train_size",    type=int, default=0)
    parser.add_argument("--val_size",      type=int, default=3000)
    parser.add_argument("--test_size",     type=int, default=3000)

    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--steps",        type=int,   default=0)
    parser.add_argument("--data_ratio",   type=float, default=20.0)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int,   default=200)
    parser.add_argument("--eval_every",   type=int,   default=250)
    parser.add_argument("--grad_clip",    type=float, default=1.0)

    parser.add_argument("--agop_batch",        type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=128)
    parser.add_argument("--max_agop_dim",      type=int, default=4096)
    parser.add_argument("--max_padding_ratio", type=float, default=0.20)
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

    supervised_per_sample = TEACHER_OUT
    if args.train_size <= 0:
        args.train_size = int(math.ceil(
            args.data_ratio * args.target_params / float(supervised_per_sample)
        ))
    if args.steps <= 0:
        args.steps = int(math.ceil(
            args.data_ratio * args.target_params
            / float(args.batch_size * supervised_per_sample)
        ))

    cfg = TrainCfg(
        lr=args.lr, weight_decay=args.weight_decay, steps=args.steps,
        warmup_steps=args.warmup_steps, batch_size=args.batch_size,
        eval_every=args.eval_every, grad_clip=args.grad_clip,
        data_ratio=args.data_ratio,
        train_size=args.train_size, val_size=args.val_size, test_size=args.test_size,
        target_params=args.target_params, depth_list=depths,
        rnn_type=args.rnn_type, dropout=args.dropout,
        hidden_step=args.hidden_step,
        min_hidden=args.min_hidden, max_hidden=args.max_hidden,
        max_padding_ratio=args.max_padding_ratio,
        max_train_factor=args.max_train_factor,
        fit_patience=args.fit_patience,
        agop_batch=args.agop_batch,
        agop_proj_samples=args.agop_proj_samples,
        max_agop_dim=args.max_agop_dim,
        seed=args.seed,
    )

    total_train_tokens = cfg.steps * cfg.batch_size * supervised_per_sample
    unique_tokens      = cfg.train_size * supervised_per_sample
    approx_epochs      = cfg.steps * cfg.batch_size / max(1, cfg.train_size)

    print("========== Budget (RNN / Teacher-Student) ==========")
    print(f"target_params N     = {cfg.target_params:,}")
    print(f"Teacher: 2-layer GRU, hidden={TEACHER_OUT} (frozen)")
    print(f"TEACHER_OUT         = {TEACHER_OUT}  (regression target dimension)")
    print(f"seq_len             = {SEQ_LEN}  (compositional scalar steps with repeated motifs)")
    print(f"inp_dim             = {INP_DIM}  (1D scalar per step)")
    print(f"AGOP output dim     = {TEACHER_OUT}×{TEACHER_OUT}  "
          f"({TEACHER_OUT*(TEACHER_OUT-1)//2} unique off-diag entries)")
    print(f"proj_samples        = {cfg.agop_proj_samples}")
    print(f"train_size          = {cfg.train_size:,}  unique synthetic sequences")
    print(f"approx epochs       = {approx_epochs:.2f}")
    print(f"base steps          = {cfg.steps:,}")
    print(f"D (total tokens)    = {total_train_tokens:,}   D/N = {total_train_tokens/cfg.target_params:.1f}×")
    print(f"unique tokens       = {unique_tokens:,}   {unique_tokens/cfg.target_params:.1f}×N")
    print("=============================================================")

    # Build teacher (same for all student shapes)
    teacher = build_teacher(seed=args.seed + 999, device=device)

    print("\nPre-computing teacher outputs for train / val / test ...")
    t_pre = time.time()
    train_ds, y_scale = make_dataset(cfg.train_size, seed=args.seed + 1, teacher=teacher, device=device)
    val_ds,   _       = make_dataset(cfg.val_size,   seed=args.seed + 2, teacher=teacher, device=device, y_scale=y_scale)
    test_ds,  _       = make_dataset(cfg.test_size,  seed=args.seed + 3, teacher=teacher, device=device, y_scale=y_scale)
    print(f"  done in {time.time()-t_pre:.1f}s  (teacher output y_scale={y_scale:.4f})")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Fixed AGOP batch from test set
    x_agop = test_ds.tensors[0][: cfg.agop_batch].to(device)   # [256, 80, 1]

    results: List[Dict[str, float]] = []

    for depth in cfg.depth_list:
        hidden, active = find_hidden_for_target_params(depth=depth, cfg=cfg)
        model = build_student_model(depth=depth, hidden=hidden, cfg=cfg, active=active)
        total = count_params(model)
        pad   = total - active
        pad_ratio = pad / max(1, total)
        if pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] depth={depth}: padding_ratio={pad_ratio:.3f} "
                  f"exceeds max_padding_ratio={cfg.max_padding_ratio:.3f}")
            continue

        ratio = hidden / TEACHER_OUT
        print("\n" + "=" * 80)
        print(f"[RNN] depth={depth:3d}  hidden={hidden:5d}  "
              f"active={active:,}  pad={pad:,}  total={total:,}  "
              f"hidden/TEACHER_OUT={ratio:.2f}")
        print("=" * 80)

        set_global_seed(args.seed + depth)
        metrics, history = train_one_model(model, train_loader, val_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"rnn_depth{depth}_hidden{hidden}")

        agop = estimate_agop_wrt_inputs(
            model, x_agop,
            proj_samples=cfg.agop_proj_samples,
            max_agop_dim=cfg.max_agop_dim,
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
        print(f"  depth={depth}  test_mse={metrics['test_mse']:.6f}  "
              f"AOFE={off_e:.4f}  AOFE_ratio={off_r:.4f}")

        del model, agop
        torch.cuda.empty_cache()

    if not results:
        print("[ERROR] No valid shapes found. Exiting.")
        return

    csv_path = os.path.join(args.out_dir, "results.csv")
    npy_path = os.path.join(args.out_dir, "results.npy")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    np.save(npy_path, results, allow_pickle=True)

    test_mse   = np.array([r["test_mse"]             for r in results])
    off_energy = np.array([r["agop_offdiag_energy"]   for r in results])
    off_ratio  = np.array([r["agop_offdiag_ratio"]    for r in results])
    depths_arr = [int(r["depth"]) for r in results]

    p_aofe       = pearson_corr(off_energy, test_mse)
    p_aofe_ratio = pearson_corr(off_ratio,  test_mse)
    s_aofe       = spearman_corr(off_energy, test_mse)
    s_aofe_ratio = spearman_corr(off_ratio,  test_mse)

    print("\n" + "-" * 80)
    print("Unified AOFE metrics (raw test_mse, no log):")
    print(f"  Pearson (AOFE=offdiag_energy,     test_mse) = {p_aofe:.4f}   Spearman = {s_aofe:.4f}")
    print(f"  Pearson (AOFE_ratio=offdiag_ratio, test_mse) = {p_aofe_ratio:.4f}   Spearman = {s_aofe_ratio:.4f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")
    print("-" * 80)

    scatter_plot(
        off_energy, test_mse,
        xlabel="AOFE  (AGOP off-diagonal energy)",
        ylabel="Test MSE (teacher-student)",
        title=f"Teacher-Student RNN: test MSE vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_energy.png"),
        depths=depths_arr, r=p_aofe, r_label="Pearson r (AOFE, loss)",
    )
    scatter_plot(
        off_ratio, test_mse,
        xlabel="AOFE ratio  (AGOP off-diagonal ratio)",
        ylabel="Test MSE (teacher-student)",
        title=f"Teacher-Student RNN: test MSE vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_ratio.png"),
        depths=depths_arr, r=p_aofe_ratio, r_label="Pearson r (AOFE_ratio, loss)",
    )


if __name__ == "__main__":
    main()
