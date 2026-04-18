#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transformer_shape_agop.py
=========================

Goal
----
Fixed-parameter decoder-only Transformer (TinyGPT) shape sweep using
teacher-student regression. Tests the AOFE hypothesis: under fixed N,
different (depth, d_model) shapes reach different fitted MSE, mediated by
AOFE (AGOP Off-diagonal Frobenius Energy).

Task: Teacher-Student Feature Regression
-----------------------------------------
A fixed, randomly-initialised 2-layer TinyGPT (d_model=64, n_heads=8,
d_ff=256, seq_len=80, frozen) maps random 80-token scalar sequences to
64-dimensional output features. The student (varying depth/d_model) must
match the teacher's last-position 64-dim output.

Why teacher-student (replacing ICL regression)
-----------------------------------------------
ICL regression requires the model to INFER random per-task weights w from
context. This demands learning a general algorithm, which requires >> D=20N
gradient steps (verified: MSE stuck at 0.50 baseline after 7000 steps).

Teacher-student is a FIXED supervised regression task (same teacher for all
sequences). The student sees a deterministic (x → teacher(x)) mapping and
converges within D=20N steps — identical to the MLP experiment that yielded
r=0.984.

Width bottleneck (direct analogue of MLP/CNN teacher-student)
-------------------------------------------------------------
  • Teacher output: 64-dim complex features from 80 random scalar tokens
  • Wide student  (depth=3,  d_model≈160): d_model/64 ≈ 2.5 → features fit
    in distinct subspaces → low AOFE + low MSE
  • Narrow student (depth=12, d_model≈80):  d_model/64 ≈ 1.25 → features
    share subspaces → high AOFE + high MSE

Output-space AGOP (fixed 64×64 across all shapes)
---------------------------------------------------
  J = d(ŷ ∈ R^{64})/d(e_flat ∈ R^{T×d_model}),   T=80
  AGOP = E_data[J J^T] ∈ R^{64×64}   — fixed, independent of d_model

  64×64 AGOP: 2016 unique off-diagonal entries.
  With B=256, proj_samples=128: 32,768 rank-1 updates → 16× overdetermined.

Training protocol (strict D=20N)
---------------------------------
supervised_per_sample = TEACHER_OUT = 64 (64-dim prediction per sequence).
train_size = D / 64 = 20N / 64 ≈ 312 500 sequences (≈ 1 epoch).
Steps are auto-computed from D = 20N supervised outputs. Best val_mse state
is checkpointed to ensure fitted (not overfitting) regime.
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
SEQ_LEN     = 80   # sequence length (tokens per sequence)
INP_DIM     = 1    # token input dimension (scalar sequences)


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
# Model: TinyGPT (decoder-only, continuous-input mode)
# -----------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head  = d_model // n_heads
        self.dropout = float(dropout)
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask   = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        out = self.proj(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff, bias=False)
        self.fc2     = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model, elementwise_affine=True)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model, elementwise_affine=True)
        self.mlp  = MLPBlock(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Decoder-only transformer for teacher-student regression.

    Accepts continuous 1-D scalar sequences:
      inp_emb  = nn.Linear(inp_dim, d_model)   maps token scalar → d_model
      head_out = nn.Linear(d_model, out_dim)   maps last hidden state → out_dim

    forward(x: [B, T, inp_dim]) → output[:, -1, :] ∈ R^{B, out_dim}
    """

    def __init__(
        self,
        *,
        inp_dim: int = INP_DIM,
        out_dim: int = TEACHER_OUT,
        seq_len: int,
        depth: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        pad_params: int = 0,
    ):
        super().__init__()
        self.inp_dim  = int(inp_dim)
        self.out_dim  = int(out_dim)
        self.seq_len  = int(seq_len)
        self.depth    = int(depth)
        self.d_model  = int(d_model)

        self.inp_emb  = nn.Linear(inp_dim, d_model, bias=True)
        self.pos_emb  = nn.Embedding(seq_len, d_model)
        self.drop     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_f    = nn.LayerNorm(d_model, elementwise_affine=True)
        self.head_out = nn.Linear(d_model, out_dim, bias=True)

        self._pad_params = None
        if pad_params > 0:
            self._pad_params = nn.Parameter(torch.zeros(int(pad_params)), requires_grad=True)

    def forward_from_embeddings(self, e: torch.Tensor) -> torch.Tensor:
        """e: [B, T, d_model] → [B, T, out_dim]"""
        x = self.drop(e)
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)
        return self.head_out(h)   # [B, T, out_dim]

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """x_in: [B, T, inp_dim] → [B, T, out_dim]"""
        B, T, _ = x_in.shape
        assert T == self.seq_len, f"seq_len mismatch: expected {self.seq_len}, got {T}"
        pos = torch.arange(T, device=x_in.device).unsqueeze(0).expand(B, T)
        e   = self.inp_emb(x_in) + self.pos_emb(pos)
        return self.forward_from_embeddings(e)


# -----------------------
# Teacher: 4-layer MLP (architecturally mismatched with Transformer students)
# -----------------------

class TeacherMLP(nn.Module):
    """
    4-layer MLP: R^{SEQ_LEN} → R^{256} → R^{256} → R^{256} → R^{TEACHER_OUT}.
    Processes ALL 80 input values SIMULTANEOUSLY (not through attention).
    Random Kaiming init (default PyTorch), ALL parameters frozen.

    Why MLP teacher instead of TinyGPT teacher:
      A 2-layer TinyGPT teacher (d_model=64) achieves near-trivial student MSE:
      even the narrowest student (d_model=80) achieves MSE ≈ 1e-4 because
      d_model=80 >> teacher_out=64 → student can freely represent all features.

    Architectural mismatch: MLP sees ALL 80 tokens simultaneously as a flat
    vector; Transformer students process them through d_model-dimensional
    representations. Wide student (d_model=160) has more "channels" per token
    → richer local+global representation → better approximation → lower MSE.
    Narrow student (d_model=80) → tight bottleneck → higher MSE.
    The mismatch ensures partial-fit regime for all shapes.
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
    teacher: TeacherMLP,
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
    Generate compositional scalar sequences [n, SEQ_LEN, 1] with repeated
    latent motifs and pre-compute the frozen teacher's outputs [n, TEACHER_OUT].

    Outputs are NORMALISED to unit std so the baseline MSE ≈ 1.0 regardless
    of the teacher's weight scale. Pass y_scale from the training set to
    val/test so all splits use identical normalisation.

    Returns: (TensorDataset, y_scale)
    """
    x_t = generate_compositional_sequences(
        size=n,
        seq_len=SEQ_LEN,
        structure_seed=22345,
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
# AGOP estimation (wrt input embeddings)
# -----------------------

def estimate_agop_wrt_embeddings(
    model: TinyGPT,
    x_in: torch.Tensor,
    *,
    proj_samples: int = 128,
    max_agop_dim: int = 4096,
) -> torch.Tensor:
    """
    AGOP = E_data[J J^T],  J = d(ŷ ∈ R^{64})/d(e_flat ∈ R^{T×d_model}).
    ŷ = model output at LAST position (pos -1).
    AGOP ∈ R^{64×64} — fixed across all (depth, d_model) shapes.

    Estimated via JVP with random Gaussian perturbations (forward-mode AD).
    """
    device = x_in.device
    model.eval()

    D_out = model.out_dim   # 64
    assert D_out <= max_agop_dim

    with torch.no_grad():
        B, T, _ = x_in.shape
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        e   = (model.inp_emb(x_in) + model.pos_emb(pos)).detach()

    agop = torch.zeros((D_out, D_out), device=device, dtype=torch.float32)

    def fwd(e_in: torch.Tensor) -> torch.Tensor:
        out = model.forward_from_embeddings(e_in)   # [B, T, D_out]
        return out[:, -1, :]                         # [B, D_out] — last position

    for _ in range(int(proj_samples)):
        u = torch.randn_like(e)
        _, Ju = torch.autograd.functional.jvp(fwd, (e,), (u,), create_graph=False, strict=False)
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
    data_ratio: float = 20.0
    warmup_steps: int = 300
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 250

    train_size: int = 0
    val_size:   int = 3000
    test_size:  int = 3000

    target_params: int = 1_000_000
    depth_list: List[int] = None
    head_dim: int = 8
    dropout: float = 0.0
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
    model: TinyGPT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """Average per-element MSE against teacher targets."""
    model.eval()
    total_loss    = 0.0
    total_samples = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y  = x.to(device), y.to(device)
        pred  = model(x)[:, -1, :]                    # [B, TEACHER_OUT]
        loss  = F.mse_loss(pred, y, reduction="sum")
        total_loss    += float(loss.item())
        total_samples += int(y.numel())
    return total_loss / max(1, total_samples)


def train_one_model(
    model: TinyGPT,
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

    t0        = time.time()
    history: List[Dict[str, float]] = []
    min_steps = int(cfg.steps)
    max_steps = max(min_steps, int(math.ceil(cfg.max_train_factor * cfg.steps)))
    best_val  = float("inf")
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

        pred = model(x)[:, -1, :]          # [B, TEACHER_OUT]
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
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_mse={train_mse:.5f}  val_mse={val_mse:.5f}  "
                f"test_mse={test_mse:.5f}  gap={gap:+.5f}  time={dt:.1f}s"
            )
            if val_mse + 1e-8 < best_val:
                best_val    = float(val_mse)
                stale_evals = 0
                best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_evals += 1
            if (step + 1) >= min_steps and stale_evals >= int(cfg.fit_patience):
                break
            model.train()

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

def build_student_model(
    *,
    depth: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    cfg: TrainCfg,
    pad_to_target: bool = True,
) -> TinyGPT:
    tmp = TinyGPT(
        inp_dim=INP_DIM, out_dim=TEACHER_OUT, seq_len=SEQ_LEN,
        depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=cfg.dropout, pad_params=0,
    )
    active = count_params(tmp)
    pad    = 0
    if pad_to_target:
        if active > cfg.target_params:
            raise ValueError(
                f"Active params {active} > target {cfg.target_params} "
                f"for depth={depth}, d_model={d_model}."
            )
        pad = int(cfg.target_params - active)
    return TinyGPT(
        inp_dim=INP_DIM, out_dim=TEACHER_OUT, seq_len=SEQ_LEN,
        depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=cfg.dropout, pad_params=pad,
    )


def find_d_model_for_target_params(
    *,
    depth: int,
    cfg: TrainCfg,
    d_model_min: int = 48,
    d_model_max: int = 512,
) -> Tuple[int, int, int, int]:
    """Returns (d_model, n_heads, d_ff, active_params)."""
    hd = int(cfg.head_dim)

    def to_valid(d: int) -> int:
        return max(hd, (d // hd) * hd)

    d_model_min = to_valid(max(d_model_min, hd))
    d_model_max = to_valid(max(d_model_max, hd))

    def active_params(d_model: int) -> int:
        n_heads = max(1, d_model // hd)
        d_ff    = 4 * d_model
        m = TinyGPT(
            inp_dim=INP_DIM, out_dim=TEACHER_OUT, seq_len=SEQ_LEN,
            depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=cfg.dropout, pad_params=0,
        )
        return count_params(m)

    if active_params(d_model_min) > cfg.target_params:
        raise ValueError(
            f"d_model={d_model_min} already exceeds target_params={cfg.target_params} "
            f"at depth={depth}."
        )
    if active_params(d_model_max) <= cfg.target_params:
        d = d_model_max
        return d, max(1, d // hd), 4 * d, active_params(d)

    lo, hi   = d_model_min, d_model_max
    best_d   = d_model_min
    best_a   = active_params(best_d)
    while lo <= hi:
        mid = to_valid((lo + hi) // 2)
        a   = active_params(mid)
        if a <= cfg.target_params:
            best_d, best_a = mid, a
            lo = mid + hd
        else:
            hi = mid - hd

    candidates = []
    for d in [max(d_model_min, best_d - hd), best_d, min(d_model_max, best_d + hd)]:
        a = active_params(d)
        if a <= cfg.target_params:
            candidates.append((abs(cfg.target_params - a), d, a))
    candidates.sort(key=lambda t: t[0])
    _, d_best, a_best = candidates[0]
    n_heads_best = max(1, d_best // hd)
    return int(d_best), int(n_heads_best), int(4 * d_best), int(a_best)


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
    parser.add_argument("--out_dir",  type=str, default="./results_transformer_shape_sweep")
    parser.add_argument("--device",   type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",     type=int, default=0)

    parser.add_argument("--target_params", type=int, default=1_000_000)
    parser.add_argument("--depth_list",    type=str, default="3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--head_dim",      type=int, default=8)
    parser.add_argument("--dropout",       type=float, default=0.0)

    parser.add_argument("--train_size",    type=int, default=0)
    parser.add_argument("--val_size",      type=int, default=3000)
    parser.add_argument("--test_size",     type=int, default=3000)

    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--steps",        type=int,   default=0)
    parser.add_argument("--data_ratio",   type=float, default=20.0)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int,   default=300)
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

    # supervised_per_sample = TEACHER_OUT = 64 output dimensions per sequence
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
        data_ratio=args.data_ratio, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, eval_every=args.eval_every, grad_clip=args.grad_clip,
        train_size=args.train_size, val_size=args.val_size, test_size=args.test_size,
        target_params=args.target_params, depth_list=depths, head_dim=args.head_dim,
        dropout=args.dropout, agop_batch=args.agop_batch,
        agop_proj_samples=args.agop_proj_samples, max_agop_dim=args.max_agop_dim,
        max_padding_ratio=args.max_padding_ratio,
        max_train_factor=args.max_train_factor,
        fit_patience=args.fit_patience,
        seed=args.seed,
    )

    total_train_tokens = cfg.steps * cfg.batch_size * supervised_per_sample
    unique_tokens      = cfg.train_size * supervised_per_sample
    approx_epochs      = cfg.steps * cfg.batch_size / max(1, cfg.train_size)

    print("========== Budget (Transformer / Teacher-Student) ==========")
    print(f"target_params N     = {cfg.target_params:,}")
    print(f"Teacher: 2-layer TinyGPT, d_model=64, n_heads=8 (frozen)")
    print(f"TEACHER_OUT         = {TEACHER_OUT}  (regression target dimension)")
    print(f"seq_len             = {SEQ_LEN}  (compositional scalar tokens with repeated motifs)")
    print(f"inp_dim             = {INP_DIM}  (1D scalar per token)")
    print(f"AGOP output dim     = {TEACHER_OUT}×{TEACHER_OUT}  "
          f"({TEACHER_OUT*(TEACHER_OUT-1)//2} unique off-diag entries)")
    print(f"proj_samples        = {cfg.agop_proj_samples}  "
          f"(→ {cfg.agop_proj_samples*cfg.agop_batch} rank-1 updates)")
    print(f"train_size          = {cfg.train_size:,}  unique synthetic sequences")
    print(f"approx epochs       = {approx_epochs:.2f}")
    print(f"base steps          = {cfg.steps:,}")
    print(f"D (total tokens)    = {total_train_tokens:,}   D/N = {total_train_tokens/cfg.target_params:.1f}×")
    print(f"unique tokens       = {unique_tokens:,}   {unique_tokens/cfg.target_params:.1f}×N")
    print("=============================================================")

    # Build teacher (same teacher for all student shapes)
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
        d_model, n_heads, d_ff, active = find_d_model_for_target_params(
            depth=depth, cfg=cfg, d_model_min=48, d_model_max=512)
        model = build_student_model(
            depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff, cfg=cfg)
        total = count_params(model)
        pad   = total - active
        pad_ratio = pad / max(1, total)
        if pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] depth={depth}: padding_ratio={pad_ratio:.3f} "
                  f"exceeds max_padding_ratio={cfg.max_padding_ratio:.3f}")
            continue

        ratio = d_model / TEACHER_OUT
        print("\n" + "=" * 80)
        print(f"[Transformer] depth={depth:3d}  d_model={d_model:4d}  n_heads={n_heads:3d}  "
              f"d_ff={d_ff:5d}  active={active:,}  pad={pad:,}  total={total:,}  "
              f"d_model/TEACHER_OUT={ratio:.2f}")
        print("=" * 80)

        set_global_seed(args.seed + depth)
        metrics, history = train_one_model(model, train_loader, val_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"transformer_depth{depth}_dmodel{d_model}")

        agop = estimate_agop_wrt_embeddings(
            model, x_agop, proj_samples=cfg.agop_proj_samples, max_agop_dim=cfg.max_agop_dim)
        off_e, off_r = agop_offdiag_metrics(agop)

        row: Dict[str, float] = dict(metrics)
        row.update({
            "depth":               int(depth),
            "d_model":             int(d_model),
            "n_heads":             int(n_heads),
            "d_ff":                int(d_ff),
            "active_params":       int(active),
            "pad_params":          int(pad),
            "total_params":        int(total),
            "padding_ratio":       float(pad_ratio),
            "agop_dim":            int(agop.shape[0]),
            "agop_offdiag_energy": float(off_e),
            "agop_offdiag_ratio":  float(off_r),
        })
        results.append(row)
        print(f"  depth={depth}  test_mse={metrics['test_mse']:.5f}  "
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
        title=f"Teacher-Student Transformer: test MSE vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_energy.png"),
        depths=depths_arr, r=p_aofe, r_label="Pearson r (AOFE, loss)",
    )
    scatter_plot(
        off_ratio, test_mse,
        xlabel="AOFE ratio  (AGOP off-diagonal ratio)",
        ylabel="Test MSE (teacher-student)",
        title=f"Teacher-Student Transformer: test MSE vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testmse_vs_aofe_ratio.png"),
        depths=depths_arr, r=p_aofe_ratio, r_label="Pearson r (AOFE_ratio, loss)",
    )


if __name__ == "__main__":
    main()
