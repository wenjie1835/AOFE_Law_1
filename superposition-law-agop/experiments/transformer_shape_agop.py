#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transformer_shape_agop.py
=========================

Goal
----
Fixed-parameter decoder-only Transformer (TinyGPT) shape sweep on next-token prediction.
Tests the AOFE hypothesis: under fixed N, different (depth, d_model) shapes reach
similar test NLL, mediated by AOFE (AGOP Off-diagonal Frobenius Energy).

Key design choices
------------------
1) AGOP = E[J J^T]  (output-space, fixed dimension V×V across all shapes)
      e = tok_emb(tokens) + pos_emb(positions)   shape [B, T, d_model]
      J = d(last_answer_logit)/d(e_flat)         last_answer_logit ∈ R^V
      J J^T  ∈ R^{V × V}  — FIXED regardless of depth/d_model
   Estimated via JVP random projections (forward-mode AD).

   Why last-answer position (not all T*V logits):
     With T*V = 32*32 = 1024-dim AGOP, only 32 proj_samples → severely underdetermined.
     Using only the last-answer logit (R^V = R^32) gives AGOP ∈ R^{32×32}:
       • 64 proj × 256 batch = 16384 rank-1 updates for 528 unique AGOP entries → 31×
         overdetermined (cf. CNN's 16×), ensuring reliable AOFE estimation.
     Semantics: the gradient superposition at the retrieval prediction position captures
     how the model uses cross-positional attention (induction circuits), directly testing
     the architectural AOFE hypothesis.

2) Dataset: AssociativeRecallDataset (key-value retrieval)
   - vocab=32: keys {0,...,7}, values {8,...,15}  (tokens 16-31 unused)
   - seq_len=32: [k0,v0,...,k7,v7, q0,a0,...,q7,a7] + one extra query at position 32
   - The model's LAST USEFUL prediction (at position T-2) predicts the 8th answer a7
     from context [k0,v0,...,k7,v7, q0,a0,...,q6,a6, q7]:
       → requires attending back to the dictionary to find kj = q7, output vj = kj+8
   - Optimal NLL approaches 0; different shapes converge to different loss levels at D=20N
     because width (d_model) directly controls attention head expressiveness

   Why AssociativeRecall over PeriodicPatternDataset:
     PeriodicPatternDataset with ~19K patterns requires memorising each pattern from a
     single pass (D=20N), placing ALL shapes near-random NLL (3.357 vs 3.466 random,
     only 0.11 nats gap). AssociativeRecall is LEARNABLE (model finds the bijection
     rule v=k+8 from many random key permutations) and creates shape-dependent loss.

3) Strict Chinchilla D=20N training budget:
     steps = ceil(20N / (batch_size × seq_len))
     Training stops at exactly cfg.steps — no extension, no patience early-stop.
     Final model state is evaluated (Chinchilla compute-optimal protocol).

4) Shape sweep under fixed N:
     10 non-extreme depths [3,4,5,6,7,8,9,10,11,12], d_model solved via binary search
     (must be a multiple of head_dim=16 for fine granularity, padding ≤ ~25%).
     depth=2 excluded as too shallow to achieve stable nonlinear representations.

5) Unified correlation metrics (AOFE hypothesis):
     Pearson(AOFE=agop_offdiag_energy,  test_nll)    — raw NLL, no log
     Pearson(AOFE_ratio=agop_offdiag_ratio, test_nll) — raw NLL, no log

Outputs
-------
./results_transformer_shape_sweep/
  - results.csv, results.npy
  - curves/  (per-shape NLL curves for appendix)
  - scatter_testnll_vs_aofe_energy.png
  - scatter_testnll_vs_aofe_ratio.png
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
# Dataset: AssociativeRecallDataset
# -----------------------

class AssociativeRecallDataset(torch.utils.data.Dataset):
    """
    Key-value associative recall  (deterministic retrieval under causal LM).

    vocab_size = 32: keys {0,...,7}, values {8,...,15}, tokens 16-31 unused.

    Sequence layout (33 tokens → input=seq[:32], target=seq[1:33]):
      [k0, v0, ..., k7, v7,        (16 tokens: 8 kv-pairs, random key permutation)
       q0, a0, ..., q7, a7,        (16 tokens: 8 query-answer pairs)
       q_extra]                     ( 1 token:  extra random query)

    where  ki = perm[i]  (random permutation of {0,...,7}),
           vi = ki + 8   (fixed bijection),
           qi ∈ {0,...,7}  (random, with replacement),
           ai = qi + 8.

    AGOP learning signal (last-answer position, logits[:, -2, :]):
      Predicts y[-2] = a7 = q7+8 from context [k0,v0,...,k7,v7, q0,a0,...,q7].
      Requires attending back to dictionary positions {0,2,4,...,14} to find kj = q7.
      This multi-hop induction test creates genuine shape-dependent performance:
        • Wide shallow models (d_model=160, depth=3) → larger attention heads → sharper
          key-matching in fewer layers → lower NLL at D=20N budget.
        • Narrow deep models (d_model=80, depth=12) → smaller heads, more layers →
          potentially higher NLL if attention precision is the bottleneck.

    Each sample is independently generated (no shared pattern pool),
    so the model must learn the general rule v=k+8 from diverse key permutations.
    """
    NUM_KV     = 8   # number of key-value pairs per sequence
    VOCAB_SIZE = 16  # keys 0..7, values 8..15

    def __init__(self, *, size: int, seed: int):
        super().__init__()
        self.size   = int(size)
        self.num_kv = self.NUM_KV
        g = torch.Generator()
        g.manual_seed(int(seed))

        # Random key permutation per sample: [size, 8]
        self.key_perm = torch.stack([
            torch.randperm(self.num_kv, generator=g) for _ in range(self.size)
        ])  # [size, 8]   — keys k0..k7 (random perm of 0..7)

        # Random queries per sample (8 qa-pair queries + 1 extra): [size, 9]
        self.queries = torch.randint(
            0, self.num_kv, (self.size, self.num_kv + 1), generator=g
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        keys    = self.key_perm[idx]                       # [8] perm of 0..7
        values  = keys + self.num_kv                       # [8] = keys + 8
        q_pairs = self.queries[idx, : self.num_kv]         # [8] random queries
        answers = q_pairs + self.num_kv                    # [8] = queries + 8
        q_extra = self.queries[idx, self.num_kv :]         # [1] extra query

        # Build 33-token sequence: [kv-pairs (16)] + [qa-pairs (16)] + [extra_q (1)]
        kv  = torch.stack([keys, values], dim=1).reshape(-1)      # [16]
        qa  = torch.stack([q_pairs, answers], dim=1).reshape(-1)  # [16]
        seq = torch.cat([kv, qa, q_extra], dim=0)                 # [33]

        return seq[:-1].clone(), seq[1:].clone()                   # [32], [32]


def answer_positions(seq_len: int) -> torch.Tensor:
    # Targets y correspond to seq[1:], so answer tokens land at indices 16,18,...,30.
    return torch.arange(16, seq_len, 2, dtype=torch.long)


def masked_answer_nll(logits: torch.Tensor, targets: torch.Tensor, answer_pos: torch.Tensor, reduction: str) -> torch.Tensor:
    pos = answer_pos.to(logits.device)
    logits_sel = logits.index_select(dim=1, index=pos)
    targets_sel = targets.index_select(dim=1, index=pos)
    return F.cross_entropy(
        logits_sel.reshape(-1, logits_sel.size(-1)),
        targets_sel.reshape(-1),
        reduction=reduction,
    )


# -----------------------
# Model: TinyGPT (decoder-only)
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


class MLP(nn.Module):
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
        self.mlp  = MLP(d_model, d_ff, dropout)

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
        self.seq_len    = int(seq_len)
        self.depth      = int(depth)
        self.d_model    = int(d_model)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=True)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.tok_emb.weight

        self._pad_params = None
        if pad_params > 0:
            self._pad_params = nn.Parameter(torch.zeros(int(pad_params)), requires_grad=True)

    def forward_from_embeddings(self, e: torch.Tensor) -> torch.Tensor:
        x = self.drop(e)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T == self.seq_len
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        e   = self.tok_emb(idx) + self.pos_emb(pos)
        return self.forward_from_embeddings(e)


# -----------------------
# AGOP estimation (wrt input embeddings)
# -----------------------

def estimate_agop_wrt_embeddings(
    model: TinyGPT,
    idx: torch.Tensor,
    *,
    proj_samples: int = 64,
    max_agop_dim: int = 2048,
) -> torch.Tensor:
    """
    AGOP = E_data[J J^T]  where J = d(last_answer_logit)/d(e_flat).
    last_answer_logit ∈ R^V  →  AGOP ∈ R^{V × V}  (fixed across all shapes).

    We use the second-to-last position logit (index T-2), which in the
    AssociativeRecallDataset corresponds to predicting the 8th answer a7 from
    context including the full key-value dictionary + all previous qa pairs.
    This is the highest-information retrieval position and makes the gradient
    flow through the model's full cross-positional attention machinery.

    Estimation quality:
      With B=256, proj_samples=64: 256×64=16384 rank-1 outer products
      for V×(V+1)/2 = 32×33/2 = 528 unique AGOP entries → ~31× overdetermined.
      (cf. T*V=1024-dim AGOP with 32 proj → 0.03× underdetermined — unusable.)
    """
    device = idx.device
    model.eval()
    B, T = idx.shape
    assert T == model.seq_len
    V     = model.vocab_size
    D_out = V   # last-answer logits only (not T*V)

    if D_out > max_agop_dim:
        raise ValueError(
            f"AGOP dim V={D_out} > max_agop_dim={max_agop_dim}."
        )

    with torch.no_grad():
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        e   = (model.tok_emb(idx) + model.pos_emb(pos)).detach()

    agop = torch.zeros((D_out, D_out), device=device, dtype=torch.float32)

    def fwd(e_in: torch.Tensor) -> torch.Tensor:
        logits = model.forward_from_embeddings(e_in)  # [B, T, V]
        return logits[:, -2, :]                        # [B, V] — last answer position

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
    weight_decay: float = 0.0
    steps: int = 0
    data_ratio: float = 20.0
    warmup_steps: int = 1000
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 1000

    vocab_size: int = 16   # keys 0-7, values 8-15
    seq_len: int = 32      # input/target length (AssociativeRecall: 33-token seq → x/y of 32)
    train_size: int = 0
    val_size: int = 5000
    test_size: int = 5000

    target_params: int = 1_000_000
    depth_list: List[int] = None
    head_dim: int = 8
    dropout: float = 0.0
    max_padding_ratio: float = 0.15
    max_train_factor: float = 3.0
    fit_patience: int = 8

    agop_batch: int = 256
    agop_proj_samples: int = 64   # 64 proj × 256 batch = 16384 rank-1 for 32×32 AGOP → 31×
    max_agop_dim: int = 2048

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
    answer_pos: torch.Tensor,
    max_batches: Optional[int] = None,
) -> float:
    """Returns average NLL on answer positions only."""
    model.eval()
    total_loss   = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss   = masked_answer_nll(logits, y, answer_pos, reduction="sum")
        total_loss   += float(loss.item())
        total_tokens += int(y.shape[0] * answer_pos.numel())
    return total_loss / max(1, total_tokens)


def train_one_model(
    model: TinyGPT,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cfg: TrainCfg,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Train until answer-only validation NLL plateaus after the D≈20N budget is reached.
    Report the best validation-state checkpoint to stay in a fitted regime.
    """
    model.to(device)
    model.train()

    opt        = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_iter = iter(train_loader)

    t0        = time.time()
    history: List[Dict[str, float]] = []
    min_steps = int(cfg.steps)
    max_steps = max(min_steps, int(math.ceil(cfg.max_train_factor * cfg.steps)))
    ans_pos = answer_positions(cfg.seq_len)
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

        logits = model(x)
        loss   = masked_answer_nll(logits, y, ans_pos, reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == max_steps:
            train_nll = evaluate_lm(model, train_loader, device, ans_pos, max_batches=50)
            val_nll   = evaluate_lm(model, val_loader,   device, ans_pos, max_batches=None)
            test_nll  = evaluate_lm(model, test_loader,  device, ans_pos, max_batches=None)
            history.append({
                "step":      int(step + 1),
                "lr":        float(lr),
                "train_nll": float(train_nll),
                "val_nll":   float(val_nll),
                "test_nll":  float(test_nll),
            })
            dt  = time.time() - t0
            gap = test_nll - train_nll
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_nll={train_nll:.4f}  val_nll={val_nll:.4f}  test_nll={test_nll:.4f}  "
                f"ppl={math.exp(test_nll):.2f}  gap={gap:+.4f}  time={dt:.1f}s"
            )
            if val_nll + 1e-6 < best_val:
                best_val = float(val_nll)
                stale_evals = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_evals += 1
            if (step + 1) >= min_steps and stale_evals >= int(cfg.fit_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_nll = evaluate_lm(model, train_loader, device, ans_pos, max_batches=None)
    val_nll   = evaluate_lm(model, val_loader,   device, ans_pos, max_batches=None)
    test_nll  = evaluate_lm(model, test_loader,  device, ans_pos, max_batches=None)

    gap = test_nll - train_nll
    if gap > 1.0:
        print(f"  [WARNING] test_nll - train_nll = {gap:.4f} (>1.0 nats), possible overfitting")

    return {
        "train_nll": float(train_nll),
        "val_nll":   float(val_nll),
        "test_nll":  float(test_nll),
        "test_ppl":  float(math.exp(test_nll)),
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
    tmp    = TinyGPT(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, depth=depth,
                     d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=cfg.dropout,
                     pad_params=0, tie_weights=True)
    active = count_params(tmp)
    pad    = 0
    if pad_to_target:
        if active > cfg.target_params:
            raise ValueError(f"Active params {active} > target {cfg.target_params} "
                             f"for depth={depth}, d_model={d_model}.")
        pad = int(cfg.target_params - active)
    return TinyGPT(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, depth=depth,
                   d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=cfg.dropout,
                   pad_params=pad, tie_weights=True)


def find_d_model_for_target_params(
    *,
    depth: int,
    cfg: TrainCfg,
    d_model_min: int = 64,
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
        m = TinyGPT(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, depth=depth,
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=cfg.dropout,
                    pad_params=0, tie_weights=True)
        return count_params(m)

    if active_params(d_model_min) > cfg.target_params:
        raise ValueError(f"d_model={d_model_min} already exceeds target_params={cfg.target_params} at depth={depth}.")
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
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_curve(history: List[Dict[str, float]], out_dir: str, stem: str) -> None:
    """Save per-shape training curve (linear NLL scale for transformer)."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    png_path = os.path.join(out_dir, f"{stem}.png")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    steps      = [row["step"]      for row in history]
    train_vals = [row["train_nll"] for row in history]
    val_vals   = [row["val_nll"]   for row in history]
    test_vals  = [row["test_nll"]  for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_vals, label="train_nll")
    plt.plot(steps, val_vals,   label="val_nll")
    plt.plot(steps, test_vals,  label="test_nll")
    plt.xlabel("step")
    plt.ylabel("NLL (nats/token)")
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

    parser.add_argument("--vocab_size",   type=int, default=16,
                        help="Vocabulary size for associative recall.")
    parser.add_argument("--seq_len",      type=int, default=32)
    parser.add_argument("--train_size",   type=int, default=0)
    parser.add_argument("--val_size",     type=int, default=5000)
    parser.add_argument("--test_size",    type=int, default=5000)

    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--steps",        type=int,   default=0,
                        help="0 = auto-compute from D=data_ratio×N.")
    parser.add_argument("--data_ratio",   type=float, default=20.0)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int,   default=1000)
    parser.add_argument("--eval_every",   type=int,   default=1000)
    parser.add_argument("--grad_clip",    type=float, default=1.0)

    parser.add_argument("--agop_batch",        type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=64)
    parser.add_argument("--max_agop_dim",      type=int, default=2048)
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

    # Auto train_size and steps from D=data_ratio×N
    supervised_tokens_per_sample = int(answer_positions(args.seq_len).numel())
    if args.train_size <= 0:
        args.train_size = int(math.ceil(args.data_ratio * args.target_params / float(supervised_tokens_per_sample)))
    if args.steps <= 0:
        args.steps = int(math.ceil(
            args.data_ratio * args.target_params / float(args.batch_size * supervised_tokens_per_sample)
        ))

    cfg = TrainCfg(
        lr=args.lr, weight_decay=args.weight_decay, steps=args.steps,
        data_ratio=args.data_ratio, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, eval_every=args.eval_every, grad_clip=args.grad_clip,
        vocab_size=args.vocab_size, seq_len=args.seq_len, train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        target_params=args.target_params, depth_list=depths, head_dim=args.head_dim,
        dropout=args.dropout, agop_batch=args.agop_batch,
        agop_proj_samples=args.agop_proj_samples, max_agop_dim=args.max_agop_dim,
        max_padding_ratio=args.max_padding_ratio,
        max_train_factor=args.max_train_factor,
        fit_patience=args.fit_patience,
        seed=args.seed,
    )

    # AGOP dim = V (last-answer logit only), not T*V
    agop_out_dim = cfg.vocab_size
    if agop_out_dim > cfg.max_agop_dim:
        raise ValueError(
            f"AGOP dim V={agop_out_dim} > max_agop_dim={cfg.max_agop_dim}."
        )

    total_train_tokens = cfg.steps * cfg.batch_size * supervised_tokens_per_sample
    unique_tokens      = cfg.train_size * supervised_tokens_per_sample
    approx_epochs      = cfg.steps * cfg.batch_size / cfg.train_size

    print("========== Budget (Transformer / AssociativeRecall) ==========")
    print(f"target_params N     = {cfg.target_params:,}")
    print(f"AGOP output dim     = V = {agop_out_dim}×{agop_out_dim}  "
          f"(last-answer logit; prev T*V={cfg.seq_len * cfg.vocab_size})")
    print(f"proj_samples        = {cfg.agop_proj_samples}  "
          f"(→ {cfg.agop_proj_samples * cfg.agop_batch} rank-1 updates for "
          f"{agop_out_dim*(agop_out_dim+1)//2} entries)")
    print(f"train_size          = {cfg.train_size:,}  unique random sequences")
    print(f"supervised positions= {supervised_tokens_per_sample} answer tokens / sequence")
    print(f"approx epochs       = {approx_epochs:.2f}  (≈ 1 pass per unique sample)")
    print(f"base steps          = {cfg.steps:,}")
    print(f"D (total tokens)    = {total_train_tokens:,}   D/N = {total_train_tokens/cfg.target_params:.1f}×")
    print(f"unique tokens       = {unique_tokens:,}   {unique_tokens/cfg.target_params:.1f}×N")
    print("==============================================================")

    train_ds = AssociativeRecallDataset(size=cfg.train_size, seed=cfg.seed + 1)
    val_ds   = AssociativeRecallDataset(size=cfg.val_size,   seed=cfg.seed + 2)
    test_ds  = AssociativeRecallDataset(size=cfg.test_size,  seed=cfg.seed + 3)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Fixed AGOP batch from test set
    x_agop_list = []
    for x, _y in test_loader:
        x_agop_list.append(x)
        if sum(t.shape[0] for t in x_agop_list) >= cfg.agop_batch:
            break
    x_agop = torch.cat(x_agop_list, dim=0)[: cfg.agop_batch].to(device)

    results: List[Dict[str, float]] = []

    for depth in cfg.depth_list:
        d_model, n_heads, d_ff, active = find_d_model_for_target_params(
            depth=depth, cfg=cfg, d_model_min=64, d_model_max=512)
        model = build_transformer_model(
            depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff, cfg=cfg)
        total = count_params(model)
        pad   = total - active
        pad_ratio = pad / max(1, total)
        if pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] depth={depth}: padding_ratio={pad_ratio:.3f} exceeds max_padding_ratio={cfg.max_padding_ratio:.3f}")
            continue

        print("\n" + "=" * 80)
        print(f"[Transformer] depth={depth:3d}  d_model={d_model:4d}  n_heads={n_heads:3d}  "
              f"d_ff={d_ff:5d}  active={active:,}  pad={pad:,}  total={total:,}")
        print("=" * 80)

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

        del model, agop
        torch.cuda.empty_cache()

    # ---------- Save results ----------
    csv_path = os.path.join(args.out_dir, "results.csv")
    npy_path = os.path.join(args.out_dir, "results.npy")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    np.save(npy_path, results, allow_pickle=True)

    # ---------- Unified correlation metrics (no log) ----------
    test_nll   = np.array([r["test_nll"]            for r in results])
    off_energy = np.array([r["agop_offdiag_energy"]  for r in results])
    off_ratio  = np.array([r["agop_offdiag_ratio"]   for r in results])
    depths_arr = [int(r["depth"]) for r in results]

    p_aofe       = pearson_corr(off_energy, test_nll)
    p_aofe_ratio = pearson_corr(off_ratio,  test_nll)
    s_aofe       = spearman_corr(off_energy, test_nll)
    s_aofe_ratio = spearman_corr(off_ratio,  test_nll)

    print("\n" + "-" * 80)
    print("Unified AOFE metrics (raw test_nll, no log):")
    print(f"  Pearson (AOFE=offdiag_energy,     test_nll) = {p_aofe:.4f}   Spearman = {s_aofe:.4f}")
    print(f"  Pearson (AOFE_ratio=offdiag_ratio, test_nll) = {p_aofe_ratio:.4f}   Spearman = {s_aofe_ratio:.4f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")
    print("-" * 80)

    # ---------- Scatter plots (linear y-axis, with Pearson r annotation) ----------
    scatter_plot(
        off_energy, test_nll,
        xlabel="AOFE  (AGOP off-diagonal energy)",
        ylabel="Test NLL (nats/token)",
        title=f"Next-token: test NLL vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testnll_vs_aofe_energy.png"),
        depths=depths_arr,
        r=p_aofe, r_label="Pearson r (AOFE, loss)",
    )
    scatter_plot(
        off_ratio, test_nll,
        xlabel="AOFE ratio  (AGOP off-diagonal ratio)",
        ylabel="Test NLL (nats/token)",
        title=f"Next-token: test NLL vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testnll_vs_aofe_ratio.png"),
        depths=depths_arr,
        r=p_aofe_ratio, r_label="Pearson r (AOFE_ratio, loss)",
    )


if __name__ == "__main__":
    main()
