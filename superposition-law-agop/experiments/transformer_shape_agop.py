
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transformer_shape_sweep_nexttoken_agop.py
========================================

Goal
----
Fixed-parameter *decoder-only* Transformer (TinyGPT) shape sweep on a next-token prediction task.
Tests the hypothesis: under fixed parameter count, different (depth, width) shapes reach
similar loss, and the AGOP off-diagonal energy ("coupling") acts as the mediating metric.

Key design choices
------------------
1) AGOP = E[ J J^T ]  (output-space, NOT input-space J^T J)
      e = token_embedding(tokens) + positional_embedding(positions)
      J = d(logits) / d(e)       shape: [T*V, T*d_model]
      J J^T                      shape: [T*V, T*V]  -- FIXED across model shapes

   This is better than J^T J for cross-shape comparison because:
     - J^T J ∈ R^{T*d_model × T*d_model} changes size with d_model
     - J J^T ∈ R^{T*V × T*V} is FIXED regardless of depth or width
   Estimated via JVP random projections (forward-mode AD):
     sample u ~ N(0,I) in input space, compute Ju via JVP, accumulate (Ju)^T(Ju)/B

2) Dataset: PeriodicPatternDataset
     - K random token patterns of length period P
     - each sample = length-(seq_len+1) subsequence from a pattern at random offset
   Scaled to be rich enough for non-trivial generalization (128 patterns, period 32).

3) Chinchilla-inspired training budget:
     total_tokens = steps × batch_size × seq_len ≈ 20 × target_params
     Default: 10000 × 128 × 32 = 40.96M ≈ 41 × 1M params (~2× Chinchilla for convergence)

4) Shape sweep under fixed parameter count:
     sweep depth L in [2, 4, 6, 8] (non-extreme), solve d_model (multiple of head_dim)
     to match target params, then pad remaining with unused parameters.

Outputs
-------
Writes to: ./results_transformer_shape_sweep/
  - results.csv, results.npy
  - scatter plots (nll vs agop, ppl vs agop)

Run (example)
-------------
python transformer_shape_sweep_nexttoken_agop.py \
  --target_params 1000000 \
  --depth_list 2,3,4,5,6,7,8,10,12,16 \
  --head_dim 16 \
  --train_size 20000 \
  --test_size 5000 \
  --seq_len 32 \
  --vocab_size 128 \
  --steps 10000 \
  --device cuda
"""

from __future__ import annotations

import os
import math
import csv
import time
import copy
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
# Dataset: small deterministic next-token prediction
# -----------------------

class PeriodicPatternDataset(torch.utils.data.Dataset):
    """
    Deterministic dataset that still allows a notion of generalization.

    We sample K random token patterns (length=period). Each sample is a length-(seq_len+1) slice
    from one pattern with random offset. Next token is deterministic.

    train/test sets are generated with independent RNG streams but share the same underlying patterns,
    so the test distribution is "same task", not random labels.

    If train_size is very small relative to model capacity, models can memorize (interpolate),
    which is a common regime where superposition appears.
    """
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        size: int,
        num_patterns: int,
        period: int,
        seed: int,
        patterns_seed: int,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.size = int(size)
        self.num_patterns = int(num_patterns)
        self.period = int(period)

        # Fixed underlying patterns
        g_pat = torch.Generator()
        g_pat.manual_seed(int(patterns_seed))
        patterns = torch.randint(
            low=0, high=vocab_size, size=(num_patterns, period), generator=g_pat, dtype=torch.long
        )
        self.patterns = patterns  # [K, P]

        support_size = num_patterns * period
        if size > support_size:
            raise ValueError(
                f"Requested size={size} exceeds unique support num_patterns×period={support_size}. "
                f"Increase num_patterns or decrease size."
            )

        # Sample unique (pattern_id, offset) pairs without replacement.
        g = torch.Generator()
        g.manual_seed(int(seed))
        perm = torch.randperm(support_size, generator=g)[:size]
        self.pattern_ids = torch.div(perm, period, rounding_mode="floor")
        self.offsets = perm % period

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pid = int(self.pattern_ids[idx])
        off = int(self.offsets[idx])
        pat = self.patterns[pid]  # [P]
        # Create length seq_len+1 by wrapping around period
        # tokens[t] = pat[(off + t) % period]
        t = torch.arange(self.seq_len + 1)
        toks = pat[(off + t) % self.period]
        x = toks[:-1].clone()  # [T]
        y = toks[1:].clone()   # [T]
        return x, y


# -----------------------
# Model: TinyGPT
# -----------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.dropout = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # causal mask buffer (built at runtime for given T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d]
        B, T, d = x.shape
        qkv = self.qkv(x)  # [B,T,3d]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, nh, T, dh]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # attn scores: [B, nh, T, T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # causal mask: allow i>=j
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = attn @ v  # [B, nh, T, dh]
        out = out.transpose(1, 2).contiguous().view(B, T, d)  # [B,T,d]
        out = self.proj(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.mlp = MLP(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        depth: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        pad_params: int = 0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.depth = int(depth)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=True)

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            # weight tying (head weight = token embedding weight)
            self.head.weight = self.tok_emb.weight

        self._pad_params = None
        if pad_params > 0:
            self._pad_params = nn.Parameter(torch.zeros(int(pad_params)), requires_grad=True)

    def forward_from_embeddings(self, e: torch.Tensor) -> torch.Tensor:
        # e: [B,T,d]
        x = self.drop(e)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B,T,V]
        return logits

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B,T]
        B, T = idx.shape
        assert T == self.seq_len
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        e = self.tok_emb(idx) + self.pos_emb(pos)
        return self.forward_from_embeddings(e)


# -----------------------
# AGOP estimation (wrt input embeddings)
# -----------------------

def estimate_agop_wrt_embeddings(
    model: TinyGPT,
    idx: torch.Tensor,
    *,
    proj_samples: int = 16,
    max_agop_dim: int = 8192,
) -> torch.Tensor:
    """
    Compute AGOP = E_data[ J J^T ] where J = d(logits)/d(embedding).

    J J^T lives in OUTPUT logit space: shape [T*V, T*V].
    This dimension is FIXED across model shapes (independent of d_model),
    which makes it directly comparable when sweeping depth/width at fixed param count.

    Estimated via JVP (Jacobian-vector product) random projections using forward-mode AD:
        For u_b ~ N(0, I) in input embedding space [B, T, d_model]:
            J_b u_b  →  [T*V]  (computed via torch.autograd.functional.jvp)
        AGOP ≈ (1/K) sum_k  (1/B) * (Ju)^T (Ju)  ∈ R^{T*V × T*V}

    NOTE: torch.autograd.functional.jvp requires PyTorch >= 1.9.
    """
    device = idx.device
    model.eval()
    B, T = idx.shape
    assert T == model.seq_len, "idx length must equal model.seq_len."
    V = model.vocab_size
    D_out = T * V

    if D_out > max_agop_dim:
        raise ValueError(
            f"AGOP output dimension T*V={D_out} > max_agop_dim={max_agop_dim}. "
            f"Reduce seq_len or vocab_size, or increase max_agop_dim."
        )

    with torch.no_grad():
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        e0 = model.tok_emb(idx) + model.pos_emb(pos)  # [B, T, d_model]
        e = e0.detach()

    agop = torch.zeros((D_out, D_out), device=device, dtype=torch.float32)

    def fwd(e_in: torch.Tensor) -> torch.Tensor:
        # maps [B, T, d_model] → [B, T*V]
        return model.forward_from_embeddings(e_in).reshape(B, D_out)

    for _ in range(int(proj_samples)):
        u = torch.randn_like(e)  # [B, T, d_model] — random tangent in input space
        # JVP via forward-mode AD: Ju[b] = J_b u_b,  shape [B, D_out]
        _, Ju = torch.autograd.functional.jvp(fwd, (e,), (u,), create_graph=False, strict=False)
        Ju = Ju.float()
        Ju = torch.nan_to_num(Ju, nan=0.0, posinf=0.0, neginf=0.0)
        agop = agop + (Ju.T @ Ju) / float(B)

    agop = agop / float(proj_samples)
    agop = symmetrize_(agop).detach()
    return agop


# -----------------------
# Training
# -----------------------

@dataclass
class TrainCfg:
    lr: float = 3e-4
    weight_decay: float = 0.0
    # Chinchilla-inspired: base_steps × batch_size × seq_len ≈ data_ratio × target_params
    steps: int = 0
    data_ratio: float = 20.0
    max_steps_multiplier: float = 4.0
    warmup_steps: int = 1000
    batch_size: int = 128
    grad_clip: float = 1.0

    eval_every: int = 1000
    fit_patience_evals: int = 5
    fit_rel_improve_tol: float = 1e-3

    # dataset — larger and richer than original to avoid trivial memorization
    vocab_size: int = 128   # was 64
    seq_len: int = 32       # was 16
    train_size: int = 0
    test_size: int = 5000   # was 2048
    num_patterns: int = 0
    period: int = 32        # was 16   (longer period = harder task)

    # model sweep — 10 non-extreme shapes spanning depth 2..16
    # head_dim=16 gives finer d_model granularity (steps of 16: 64,80,96,112,128,160,192…)
    # which avoids large parameter gaps between consecutive valid d_model values.
    # With head_dim=32 the grid is too coarse and some depths end up with >40% padding.
    target_params: int = 1_000_000
    depth_list: List[int] = None
    head_dim: int = 16  # was 32; finer step = better param matching across 10 shapes
    dropout: float = 0.0

    # agop — output-space J J^T, dim = T*V = 32*128 = 4096
    agop_batch: int = 256           # was 128
    agop_proj_samples: int = 16     # was 8 (more JVP samples for better estimate)
    max_agop_dim: int = 8192        # T*V = 4096 < 8192 (headroom for larger vocab/seq)

    # misc
    seed: int = 0


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    t = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate_lm(
    model: TinyGPT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """
    Returns average NLL per token.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)  # [B,T,V]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int(y.numel())
    return total_loss / max(1, total_tokens)


def train_one_model(
    model: TinyGPT,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cfg: TrainCfg,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_iter = iter(train_loader)

    t0 = time.time()
    history: List[Dict[str, float]] = []
    best_train_nll = float("inf")
    best_test_nll = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_eval_idx = -1
    max_steps = max(int(cfg.steps), int(math.ceil(cfg.steps * cfg.max_steps_multiplier)))

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

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == int(cfg.steps) or (step + 1) == max_steps:
            train_nll = evaluate_lm(model, train_loader, device, max_batches=50)
            test_nll = evaluate_lm(model, test_loader, device, max_batches=None)
            history.append({
                "step": int(step + 1),
                "lr": float(lr),
                "train_nll": float(train_nll),
                "test_nll": float(test_nll),
            })
            if train_nll < best_train_nll:
                best_train_nll = float(train_nll)
                best_state = copy.deepcopy(model.state_dict())
            best_test_nll = min(best_test_nll, test_nll)
            dt = time.time() - t0
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_nll={train_nll:.4f}  test_nll={test_nll:.4f}  "
                f"best_test_nll={best_test_nll:.4f}  ppl={math.exp(test_nll):.2f}  "
                f"time={dt:.1f}s"
            )
            if train_nll <= best_train_nll * (1.0 + cfg.fit_rel_improve_tol):
                best_eval_idx = len(history) - 1
            if (step + 1) >= int(cfg.steps) and (len(history) - 1 - best_eval_idx) >= int(cfg.fit_patience_evals):
                print(f"Early stop at step {step+1}: train NLL has plateaued after base budget.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_nll = evaluate_lm(model, train_loader, device, max_batches=None)
    test_nll = evaluate_lm(model, test_loader, device, max_batches=None)
    return {
        "train_nll": float(train_nll),
        "test_nll": float(test_nll),
        "best_train_nll": float(best_train_nll),
        "best_test_nll": float(best_test_nll),
        "test_ppl": float(math.exp(test_nll)),
        "steps_run": int(history[-1]["step"]) if history else 0,
    }, history


# -----------------------
# Shape/parameter matching
# -----------------------

def build_transformer_model(
    *,
    depth: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    cfg: TrainCfg,
    pad_to_target: bool = True,
) -> TinyGPT:
    tmp = TinyGPT(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        depth=depth,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=cfg.dropout,
        pad_params=0,
        tie_weights=True,
    )
    active = count_params(tmp)
    pad = 0
    if pad_to_target:
        if active > cfg.target_params:
            raise ValueError(
                f"Active params {active} exceed target {cfg.target_params} "
                f"for depth={depth}, d_model={d_model}."
            )
        pad = int(cfg.target_params - active)
    model = TinyGPT(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        depth=depth,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=cfg.dropout,
        pad_params=pad,
        tie_weights=True,
    )
    return model


def find_d_model_for_target_params(
    *,
    depth: int,
    cfg: TrainCfg,
    d_model_min: int = 64,
    d_model_max: int = 512,
) -> Tuple[int, int, int, int]:
    """
    Returns (d_model, n_heads, d_ff, active_params_without_padding)

    Constraints:
      - d_model must be a multiple of head_dim
      - d_model in [d_model_min, d_model_max]

    Note: the old AGOP-based d_model constraint (T*d_model <= max_agop_dim) has been
    removed because AGOP is now computed in output space [T*V, T*V], which is independent
    of d_model. The check T*V <= max_agop_dim is done once at the start of main().
    """
    # enforce multiples of head_dim
    hd = int(cfg.head_dim)
    def to_valid(d: int) -> int:
        return max(hd, (d // hd) * hd)

    d_model_min = to_valid(max(d_model_min, hd))
    d_model_max = to_valid(max(d_model_max, hd))

    if d_model_min > d_model_max:
        raise ValueError(f"d_model_min > d_model_max after constraints ({d_model_min} > {d_model_max}).")

    def active_params(d_model: int) -> int:
        n_heads = max(1, d_model // hd)
        d_ff = 4 * d_model
        m = TinyGPT(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            depth=depth,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=cfg.dropout,
            pad_params=0,
            tie_weights=True,
        )
        return count_params(m)

    # Feasibility checks
    if active_params(d_model_min) > cfg.target_params:
        raise ValueError(
            f"Even d_model={d_model_min} exceeds target_params={cfg.target_params} at depth={depth}. "
            f"Reduce depth or increase target_params."
        )
    if active_params(d_model_max) <= cfg.target_params:
        d = d_model_max
        n_heads = max(1, d // hd)
        d_ff = 4 * d
        return d, n_heads, d_ff, active_params(d)

    # Binary search on discrete grid of multiples of head_dim
    lo, hi = d_model_min, d_model_max
    best_d = d_model_min
    best_a = active_params(best_d)

    while lo <= hi:
        mid = to_valid((lo + hi) // 2)
        a = active_params(mid)
        if a <= cfg.target_params:
            best_d, best_a = mid, a
            lo = mid + hd
        else:
            hi = mid - hd

    # Check neighbors for closer absolute match
    candidates = []
    for d in [max(d_model_min, best_d - hd), best_d, min(d_model_max, best_d + hd)]:
        a = active_params(d)
        if a <= cfg.target_params:
            candidates.append((abs(cfg.target_params - a), d, a))
    candidates.sort(key=lambda t: t[0])
    _, d_best, a_best = candidates[0]
    n_heads_best = max(1, d_best // hd)
    d_ff_best = 4 * d_best
    return int(d_best), int(n_heads_best), int(d_ff_best), int(a_best)


# -----------------------
# Plotting
# -----------------------

def scatter_plot(x, y, xlabel, ylabel, title, outpath):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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

    steps = [row["step"] for row in history]
    train_vals = [row["train_nll"] for row in history]
    test_vals = [row["test_nll"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_vals, label="train_nll")
    plt.plot(steps, test_vals, label="test_nll")
    plt.xlabel("step")
    plt.ylabel("NLL")
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
    parser.add_argument("--out_dir", type=str, default="./results_transformer_shape_sweep")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--target_params", type=int, default=1_000_000)
    # 10 shapes: depth 2..16, width 64..192, all non-extreme (min depth=2, min d_model=64)
    # head_dim=16 gives d_model steps of 16, keeping padding ≤23% for all 10 depths.
    # Expected (depth→d_model): 2→192, 3→160, 4→128, 5→112, 6→112, 7→96,
    #                           8→96, 10→80, 12→80, 16→64
    parser.add_argument("--depth_list", type=str, default="2,3,4,5,6,7,8,9,10,12")
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--train_size", type=int, default=0)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--num_patterns", type=int, default=0)
    parser.add_argument("--period", type=int, default=32)

    # Chinchilla-inspired: 10000 × 128 × 32 = 40.96M ≈ 41 × 1M params
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--data_ratio", type=float, default=20.0)
    parser.add_argument("--max_steps_multiplier", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--fit_patience_evals", type=int, default=5)
    parser.add_argument("--fit_rel_improve_tol", type=float, default=1e-3)

    # AGOP: J J^T in output space [T*V, T*V]; T*V = 32*128 = 4096
    parser.add_argument("--agop_batch", type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=16)
    parser.add_argument("--max_agop_dim", type=int, default=8192)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    curve_dir = os.path.join(args.out_dir, "curves")

    set_global_seed(args.seed)

    depths = [int(x) for x in args.depth_list.split(",") if x.strip() != ""]
    if len(depths) != 10:
        raise ValueError(f"depth_list must contain exactly 10 shapes, got {len(depths)}.")
    if args.train_size <= 0:
        args.train_size = int(math.ceil(args.data_ratio * args.target_params / float(args.seq_len)))
    if args.num_patterns <= 0:
        args.num_patterns = int(math.ceil((args.train_size + args.test_size) / float(args.period)))
    if args.steps <= 0:
        args.steps = int(math.ceil(args.data_ratio * args.target_params / float(args.batch_size * args.seq_len)))
    cfg = TrainCfg(
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps=args.steps,
        data_ratio=args.data_ratio,
        max_steps_multiplier=args.max_steps_multiplier,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        fit_patience_evals=args.fit_patience_evals,
        fit_rel_improve_tol=args.fit_rel_improve_tol,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        train_size=args.train_size,
        test_size=args.test_size,
        num_patterns=args.num_patterns,
        period=args.period,
        target_params=args.target_params,
        depth_list=depths,
        head_dim=args.head_dim,
        dropout=args.dropout,
        agop_batch=args.agop_batch,
        agop_proj_samples=args.agop_proj_samples,
        max_agop_dim=args.max_agop_dim,
        seed=args.seed,
    )

    # Validate AGOP output dimension (T*V must fit in max_agop_dim)
    agop_out_dim = cfg.seq_len * cfg.vocab_size
    if agop_out_dim > cfg.max_agop_dim:
        raise ValueError(
            f"AGOP output dimension T*V = {cfg.seq_len}×{cfg.vocab_size} = {agop_out_dim} "
            f"> max_agop_dim={cfg.max_agop_dim}. "
            f"Reduce seq_len/vocab_size or increase --max_agop_dim."
        )
    print(f"AGOP output dim: T*V = {agop_out_dim}  (fixed across all model shapes)")
    total_train_tokens = cfg.steps * cfg.batch_size * cfg.seq_len
    print(
        f"Base budget: steps={cfg.steps} × batch={cfg.batch_size} × seq_len={cfg.seq_len}"
        f" = {total_train_tokens:,} tokens  ≈  {total_train_tokens / cfg.target_params:.1f} × N"
    )
    unique_tokens = cfg.train_size * cfg.seq_len
    print(
        f"Unique dataset tokens: train_size={cfg.train_size} × seq_len={cfg.seq_len}"
        f" = {unique_tokens:,}  ≈  {unique_tokens / cfg.target_params:.1f} × N"
    )

    # Build datasets
    patterns_seed = cfg.seed + 12345  # shared between train/test
    train_ds = PeriodicPatternDataset(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        size=cfg.train_size,
        num_patterns=cfg.num_patterns,
        period=cfg.period,
        seed=cfg.seed + 1,
        patterns_seed=patterns_seed,
    )
    test_ds = PeriodicPatternDataset(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        size=cfg.test_size,
        num_patterns=cfg.num_patterns,
        period=cfg.period,
        seed=cfg.seed + 2,
        patterns_seed=patterns_seed,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Fixed AGOP batch from test set
    x_agop = []
    for x, _y in test_loader:
        x_agop.append(x)
        if sum(t.shape[0] for t in x_agop) >= cfg.agop_batch:
            break
    x_agop = torch.cat(x_agop, dim=0)[:cfg.agop_batch].to(device)

    results: List[Dict[str, float]] = []

    for depth in cfg.depth_list:
        d_model, n_heads, d_ff, active = find_d_model_for_target_params(
            depth=depth, cfg=cfg, d_model_min=64, d_model_max=512
        )
        model = build_transformer_model(depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff, cfg=cfg)
        total = count_params(model)
        pad = total - active

        print("\n" + "=" * 80)
        print(
            f"[Transformer] depth={depth:3d}  d_model={d_model:4d}  n_heads={n_heads:3d}  "
            f"d_ff={d_ff:5d}  active_params={active}  pad={pad}  total={total}  "
            f"agop_dim(T*V)={cfg.seq_len * cfg.vocab_size}"
        )
        print("=" * 80)

        metrics, history = train_one_model(model, train_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"transformer_depth{depth}_dmodel{d_model}")

        agop = estimate_agop_wrt_embeddings(
            model,
            x_agop,
            proj_samples=cfg.agop_proj_samples,
            max_agop_dim=cfg.max_agop_dim,
        )
        off_e, off_r = agop_offdiag_metrics(agop)

        row: Dict[str, float] = dict(metrics)
        row.update({
            "depth": int(depth),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "d_ff": int(d_ff),
            "active_params": int(active),
            "pad_params": int(pad),
            "total_params": int(total),
            "padding_ratio": float(pad / max(1, total)),
            "agop_dim": int(agop.shape[0]),
            "agop_offdiag_energy": float(off_e),
            "agop_offdiag_ratio": float(off_r),
            "steps_run": int(metrics["steps_run"]),
        })
        results.append(row)

        del model, agop
        torch.cuda.empty_cache()

    # Save
    csv_path = os.path.join(args.out_dir, "results.csv")
    npy_path = os.path.join(args.out_dir, "results.npy")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    np.save(npy_path, results, allow_pickle=True)

    # Correlations + plots
    test_nll = np.array([r["test_nll"] for r in results])
    test_ppl = np.array([r["test_ppl"] for r in results])
    off_ratio = np.array([r["agop_offdiag_ratio"] for r in results])

    p_nll = pearson_corr(off_ratio, test_nll)
    s_nll = spearman_corr(off_ratio, test_nll)
    p_ppl = pearson_corr(off_ratio, test_ppl)
    s_ppl = spearman_corr(off_ratio, test_ppl)

    print("\n" + "-" * 80)
    print(f"Pearson(offdiag_ratio, test_nll)   = {p_nll:.4f}")
    print(f"Spearman(offdiag_ratio, test_nll)  = {s_nll:.4f}")
    print(f"Pearson(offdiag_ratio, test_ppl)   = {p_ppl:.4f}")
    print(f"Spearman(offdiag_ratio, test_ppl)  = {s_ppl:.4f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")
    print("-" * 80)

    scatter_plot(
        off_ratio,
        test_nll,
        xlabel="AGOP off-diagonal ratio (wrt embeddings)",
        ylabel="Test NLL (per token)",
        title=f"Next-token: test NLL vs AGOP offdiag ratio (fixed params={cfg.target_params})",
        outpath=os.path.join(args.out_dir, "scatter_testnll_vs_agop.png"),
    )
    scatter_plot(
        off_ratio,
        test_ppl,
        xlabel="AGOP off-diagonal ratio (wrt embeddings)",
        ylabel="Test perplexity",
        title=f"Next-token: test PPL vs AGOP offdiag ratio (fixed params={cfg.target_params})",
        outpath=os.path.join(args.out_dir, "scatter_testppl_vs_agop.png"),
    )


if __name__ == "__main__":
    main()
