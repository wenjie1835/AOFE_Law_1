#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transformer_ntp_shape_sweep.py
==============================

Goal
----
Chinchilla-style shape sweep for decoder-only Transformers on **next-token
prediction** (byte-level language modelling on WikiText-103).

For multiple fixed parameter budgets N, we sweep the depth-to-width ratio
(aspect ratio α = depth / d_model) and record:
  • test cross-entropy (nats / byte)  — the primary performance metric
  • AOFE_ratio = off-diagonal AGOP energy / total AGOP energy  — the coupling metric

The key prediction (AOFE hypothesis):
  1. AOFE_ratio is positively correlated with test cross-entropy across shapes
     for every N.
  2. Each N has an optimal aspect ratio α* that jointly minimises loss and
     AOFE_ratio — analogous to Chinchilla's optimal (N, D) pair.
  3. This α* provides actionable guidance for LLM shape initialisation.

Task: Byte-level next-token prediction (NTP)
--------------------------------------------
• Tokenisation: raw bytes, vocab_size = 256
• Dataset: WikiText-103-raw (HuggingFace), pre-downloaded to data/
• Training budget: D = data_ratio × N bytes  (default data_ratio = 60,
  roughly equivalent to Chinchilla's D = 20N BPE tokens at ~3 bytes/token)
• Loss: per-token cross-entropy in nats
• Eval: sequential non-overlapping windows on held-out test split

Output-space AGOP (fixed 64×64 across all shapes)
--------------------------------------------------
  J_P = P · J,  P ∈ R^{64×256} fixed random projection,
  J   = d(logits[-1] ∈ R^{256}) / d(e ∈ R^{T × d_model})
  AGOP = E_data[J_P J_P^T] ∈ R^{64 × 64}  — fixed regardless of d_model

  A 256×256 AGOP would always have AOFE_ratio ≈ 0.996 (no signal).
  Projecting to 64 dims gives the same 64×64 matrix as teacher-student for
  direct comparison.

  64×64 AGOP: 2 016 unique off-diagonal entries.
  With B=128, proj_samples=64: 8 192 rank-1 updates ≈ 4× overdetermined.

Usage
-----
  python experiments/transformer_ntp_shape_sweep.py \\
      --data_dir ./data \\
      --param_groups 300000,1000000,3000000 \\
      --depth_list 1,2,3,4,5,6,8,10,12,16,20,24 \\
      --out_dir ./results_ntp_shape_sweep \\
      --device cuda

  # Regenerate all plots from an existing CSV (after appending new N):
  python experiments/transformer_ntp_shape_sweep.py \\
      --plot_only --out_dir ./results/transformer_ntp_shape_sweep
"""

from __future__ import annotations

import os
import csv
import math
import time
import random
import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 256   # byte-level vocabulary
SEQ_LEN    = 256   # tokens (bytes) per sequence

# Output dimension for AGOP computation.
# We project vocab_size=256 logits down to AGOP_OUT=64 dims via a fixed
# random matrix before computing AGOP, for two reasons:
#   1. A 256×256 AGOP always has AOFE_ratio ≈ 1 - 1/256 ≈ 0.996 by construction
#      (255× more off-diagonal entries than diagonal) — no meaningful signal.
#   2. Matching the teacher-student experiment's 64×64 matrix makes metrics
#      directly comparable.
# The projection matrix is fixed (seed=42) across all shapes and N values.
AGOP_OUT = 64


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


# ─────────────────────────────────────────────────────────────────────────────
# AOFE metric
# ─────────────────────────────────────────────────────────────────────────────

def agop_offdiag_metrics(agop: torch.Tensor) -> Tuple[float, float]:
    """
    Returns (AOFE, AOFE_ratio):
      AOFE       = ||AGOP||_F^2 - ||diag(AGOP)||_2^2
      AOFE_ratio = AOFE / ||AGOP||_F^2
    """
    agop = agop.float()
    fro2  = float((agop * agop).sum().item()) + 1e-12
    diag2 = float((torch.diag(agop) ** 2).sum().item())
    return max(fro2 - diag2, 0.0), max(fro2 - diag2, 0.0) / fro2


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    def _rank(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
        return ranks
    return pearson_corr(_rank(x), _rank(y))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (WikiText-103, byte-level)
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load WikiText-103 train/val/test as byte arrays.

    Expects pre-downloaded binary files:
      data_dir/wikitext103_train.bin
      data_dir/wikitext103_validation.bin
      data_dir/wikitext103_test.bin

    If not found, tries the datasets library to download them.
    Falls back to a synthetic corpus.

    Returns (train_bytes, val_bytes, test_bytes) as uint8 numpy arrays.
    """
    splits = {}
    for split in ("train", "validation", "test"):
        path = Path(data_dir) / f"wikitext103_{split}.bin"
        if path.exists():
            splits[split] = np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()
            print(f"  [{split:10s}] loaded {len(splits[split])/1e6:.1f}M bytes from {path}")
        else:
            splits[split] = None

    if any(v is None for v in splits.values()):
        print("  Some splits missing; attempting download via datasets library ...")
        splits = _download_wikitext103(data_dir)

    return splits["train"], splits["validation"], splits["test"]


def _download_wikitext103(data_dir: str) -> Dict[str, np.ndarray]:
    """Download WikiText-103 via HuggingFace datasets and cache to disk."""
    os.makedirs(data_dir, exist_ok=True)
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets library not found; install with: pip install datasets"
        ) from exc

    result = {}
    for split in ("train", "validation", "test"):
        print(f"  Downloading WikiText-103 [{split}] ...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = "\n".join(ds["text"])
        data = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).copy()
        path = Path(data_dir) / f"wikitext103_{split}.bin"
        path.write_bytes(data.tobytes())
        print(f"    {split}: {len(data)/1e6:.1f}M bytes → {path}")
        result[split] = data
    return result


class RandomWindowDataset(torch.utils.data.Dataset):
    """Randomly sampled non-overlapping windows of length SEQ_LEN for training."""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        n_windows: int,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        max_start = len(data) - seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Corpus too short ({len(data)} bytes) for seq_len={seq_len}"
            )
        self.starts = rng.integers(0, max_start, size=n_windows)
        self.data   = data
        self.T      = seq_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        x = torch.from_numpy(self.data[s     : s + self.T    ].astype(np.int64))
        y = torch.from_numpy(self.data[s + 1 : s + self.T + 1].astype(np.int64))
        return x, y


class SequentialWindowDataset(torch.utils.data.Dataset):
    """Non-overlapping sequential windows for evaluation (deterministic)."""

    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.T    = seq_len
        self.n    = (len(data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.T
        x = torch.from_numpy(self.data[s     : s + self.T    ].astype(np.int64))
        y = torch.from_numpy(self.data[s + 1 : s + self.T + 1].astype(np.int64))
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Transformer model  (decoder-only, byte-level NTP)
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.dropout = dropout
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        scores   = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask     = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores   = scores.masked_fill(~mask, float("-inf"))
        attn     = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.proj(out)


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff, bias=False)
        self.fc2     = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout)
        return self.fc2(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = MLPBlock(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT_NTP(nn.Module):
    """
    Decoder-only Transformer for byte-level next-token prediction.

    forward(x: [B, T] int64) → logits [B, T, VOCAB_SIZE]

    Output-space AGOP is computed as:
      J = d(logits[-1] ∈ R^{VOCAB_SIZE}) / d(e ∈ R^{T × d_model})
      AGOP = E_data[J J^T] ∈ R^{VOCAB_SIZE × VOCAB_SIZE}

    Fixed-size (256×256) regardless of d_model — comparable across all shapes.
    """

    def __init__(
        self,
        *,
        depth:      int,
        d_model:    int,
        n_heads:    int,
        d_ff:       int,
        seq_len:    int  = SEQ_LEN,
        vocab_size: int  = VOCAB_SIZE,
        dropout:    float = 0.0,
        pad_params: int  = 0,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.depth      = depth

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len,    d_model)
        self.drop    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks  = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(depth)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._pad = None
        if pad_params > 0:
            self._pad = nn.Parameter(torch.zeros(pad_params), requires_grad=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        # Scale residual projections by 1/sqrt(depth) (GPT-2 style)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02 / math.sqrt(2 * self.depth))
            nn.init.normal_(blk.mlp.fc2.weight,   std=0.02 / math.sqrt(2 * self.depth))

    def forward_from_embeddings(self, e: torch.Tensor) -> torch.Tensor:
        """e: [B, T, d_model]  →  logits [B, T, vocab_size]"""
        x = self.drop(e)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] int64  →  logits [B, T, vocab_size]"""
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        e    = self.tok_emb(x) + self.pos_emb(pos)
        return self.forward_from_embeddings(e)


# ─────────────────────────────────────────────────────────────────────────────
# AGOP estimation (output-space, 256×256)
# ─────────────────────────────────────────────────────────────────────────────

def compute_wtw_aofe_ratio(model: "TinyGPT_NTP") -> Tuple[float, float]:
    """
    WtW AOFE ratio from the token-embedding matrix.

    W   = tok_emb.weight  ∈  R^{vocab × d_model}
    WtW = W^T W / vocab   ∈  R^{d_model × d_model}

    WtW[i, j] measures how correlated embedding dimensions i and j are across all
    vocab entries.  When d_model < vocab (deep narrow models), multiple bytes must
    share the same embedding directions → WtW has high off-diagonal energy → high
    AOFE_ratio.  This directly quantifies the superposition hypothesis:

        "As models become narrower (d_model decreases), byte representations
         must superpose → WtW AOFE_ratio increases → loss increases."

    Returns (wtw_aofe, wtw_aofe_ratio).
    """
    W = model.tok_emb.weight.detach().float()   # [vocab, d_model]
    WtW = (W.T @ W) / W.shape[0]               # [d_model, d_model]
    return agop_offdiag_metrics(WtW)


def estimate_agop_ntp(
    model:        TinyGPT_NTP,
    data:         np.ndarray,
    *,
    proj_samples: int = 64,
    batch_size:   int = 128,
    n_batches:    int = 4,
    seed:         int = 1,
    agop_out:     int = AGOP_OUT,
    device:       torch.device,
) -> torch.Tensor:
    """
    Estimate AGOP = E_data[J_P J_P^T] ∈ R^{agop_out × agop_out} via random JVPs.

    J_P = P · J,  where J = d(logits[-1] ∈ R^{VOCAB}) / d(e_flat ∈ R^{T×d_model})
    and P ∈ R^{agop_out × VOCAB} is a fixed random projection matrix (seed=42).

    Projecting to agop_out=64 before computing AGOP is necessary because a
    VOCAB×VOCAB (256×256) AGOP always has AOFE_ratio ≈ 1 - 1/256 ≈ 0.996 by
    sheer count of off-diagonal entries, giving no discriminative signal.
    The 64×64 projection preserves structure (Johnson-Lindenstrauss) and matches
    the teacher-student experiment's dimensionality for direct comparison.

    Using E_u[Ju (Ju)^T] = J E[uu^T] J^T = J J^T  (u ~ N(0, I)),
    the outer product of JVP outputs is an unbiased estimator of AGOP.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    T   = model.seq_len

    # Fixed projection matrix: same for all shapes and N values
    torch.manual_seed(42)
    proj = torch.randn(agop_out, model.vocab_size, device=device) / math.sqrt(agop_out)

    agop  = torch.zeros(agop_out, agop_out, device=device)
    count = 0

    for _ in range(n_batches):
        max_start = len(data) - T - 1
        if max_start <= 0:
            break
        starts = rng.integers(0, max_start, size=batch_size)
        x_np   = np.stack([data[s : s + T] for s in starts]).astype(np.int64)
        x      = torch.from_numpy(x_np).to(device)

        with torch.no_grad():
            pos = torch.arange(T, device=device).unsqueeze(0).expand(batch_size, T)
            e   = (model.tok_emb(x) + model.pos_emb(pos)).detach()

        def fwd(e_in: torch.Tensor) -> torch.Tensor:
            logits = model.forward_from_embeddings(e_in)  # [B, T, vocab]
            return logits[:, -1, :] @ proj.T              # [B, agop_out]

        for _ in range(proj_samples):
            u = torch.randn_like(e)
            _, Ju = torch.autograd.functional.jvp(
                fwd, (e,), (u,), create_graph=False
            )
            Ju   = torch.nan_to_num(Ju.float(), nan=0.0, posinf=0.0, neginf=0.0)
            agop = agop + (Ju.T @ Ju) / float(batch_size)

        count += proj_samples

    if count == 0:
        return agop
    return symmetrize(agop / float(count)).detach()


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainCfg:
    lr:                float = 3e-4
    weight_decay:      float = 1e-2
    # data_ratio = D / N  (in bytes).  Default 60 ≈ Chinchilla's 20 BPE-tokens
    # (since ~3 bytes/BPE token for English text).
    data_ratio:        float = 60.0
    warmup_steps:      int   = 300
    batch_size:        int   = 64
    grad_clip:         float = 1.0
    eval_every:        int   = 200
    # head_dim controls n_heads = d_model // head_dim
    head_dim:          int   = 4
    dropout:           float = 0.0
    max_padding_ratio: float = 0.20
    max_train_factor:  float = 1.5
    fit_patience:      int   = 10
    agop_batch:        int   = 128
    agop_proj_samples: int   = 64
    agop_n_batches:    int   = 4
    seed:              int   = 0
    d_model_min:       int   = 16
    d_model_max:       int   = 1024


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate_ntp(
    model:       TinyGPT_NTP,
    loader:      torch.utils.data.DataLoader,
    device:      torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """Per-token cross-entropy in nats."""
    model.eval()
    total_loss, total_n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)                            # [B, T, vocab]
        loss   = F.cross_entropy(
            logits.view(-1, model.vocab_size), y.view(-1), reduction="sum"
        )
        total_loss += float(loss.item())
        total_n    += int(y.numel())
    return total_loss / max(1, total_n)


def train_one_model(
    model:        TinyGPT_NTP,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    test_loader:  torch.utils.data.DataLoader,
    base_steps:   int,
    cfg:          TrainCfg,
    device:       torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Train via AdamW + cosine-LR with early stopping on val cross-entropy.
    Returns (metrics_dict, history_list).
    """
    model.to(device).train()
    opt       = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loader_it = iter(train_loader)

    max_steps  = max(base_steps, int(math.ceil(cfg.max_train_factor * base_steps)))
    best_val   = float("inf")
    best_state: Optional[Dict] = None
    stale      = 0
    history: List[Dict[str, float]] = []
    t0 = time.time()

    for step in range(max_steps):
        try:
            x, y = next(loader_it)
        except StopIteration:
            loader_it = iter(train_loader)
            x, y = next(loader_it)

        x, y = x.to(device), y.to(device)
        lr   = cosine_lr(step, cfg.lr, cfg.warmup_steps, max_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        logits = model(x)                            # [B, T, vocab]
        loss   = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % cfg.eval_every == 0 or (step + 1) == max_steps:
            tr_ce  = evaluate_ntp(model, train_loader, device, max_batches=20)
            val_ce = evaluate_ntp(model, val_loader,   device)
            te_ce  = evaluate_ntp(model, test_loader,  device)
            history.append({
                "step": step + 1, "lr": lr,
                "train_ce": tr_ce, "val_ce": val_ce, "test_ce": te_ce,
            })
            elapsed = time.time() - t0
            print(
                f"    step {step+1:6d}/{max_steps}  lr={lr:.2e}  "
                f"train={tr_ce:.4f}  val={val_ce:.4f}  test={te_ce:.4f} nats  "
                f"t={elapsed:.0f}s"
            )
            if val_ce + 1e-6 < best_val:
                best_val   = val_ce
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if (step + 1) >= base_steps and stale >= cfg.fit_patience:
                print(f"    [early-stop] patience={cfg.fit_patience} at step {step+1}")
                break
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    tr_ce  = evaluate_ntp(model, train_loader, device)
    val_ce = evaluate_ntp(model, val_loader,   device)
    te_ce  = evaluate_ntp(model, test_loader,  device)
    return {
        "train_ce": tr_ce, "val_ce": val_ce, "test_ce": te_ce,
        "steps_run": history[-1]["step"] if history else 0,
    }, history


# ─────────────────────────────────────────────────────────────────────────────
# Shape / parameter matching
# ─────────────────────────────────────────────────────────────────────────────

def find_d_model_for_target_params(
    *,
    depth:         int,
    target_params: int,
    cfg:           TrainCfg,
    seq_len:       int = SEQ_LEN,
    vocab_size:    int = VOCAB_SIZE,
) -> Tuple[int, int, int, int]:
    """Binary-search for the largest d_model (multiple of head_dim) such that
    active_params ≤ target_params.

    Returns (d_model, n_heads, d_ff, active_params).
    Raises ValueError if even d_model_min exceeds target.
    """
    hd = cfg.head_dim
    lo = max(hd, (cfg.d_model_min // hd) * hd)
    hi = max(lo,  (cfg.d_model_max // hd) * hd)

    def n_active(d: int) -> int:
        nh  = max(1, d // hd)
        dff = 4 * d
        m   = TinyGPT_NTP(
            depth=depth, d_model=d, n_heads=nh, d_ff=dff,
            seq_len=seq_len, vocab_size=vocab_size,
            dropout=cfg.dropout, pad_params=0,
        )
        return count_params(m)

    if n_active(lo) > target_params:
        raise ValueError(
            f"depth={depth}: d_model={lo} already exceeds "
            f"target_params={target_params:,}"
        )
    if n_active(hi) <= target_params:
        return hi, max(1, hi // hd), 4 * hi, n_active(hi)

    best_d, best_a = lo, n_active(lo)
    while lo <= hi:
        mid = ((lo + hi) // 2 // hd) * hd
        if mid < lo:
            break
        a = n_active(mid)
        if a <= target_params:
            best_d, best_a = mid, a
            lo = mid + hd
        else:
            hi = mid - hd

    candidates = []
    for d in [max(cfg.d_model_min, best_d - hd), best_d,
              min(cfg.d_model_max, best_d + hd)]:
        d = max(hd, (d // hd) * hd)
        a = n_active(d)
        if a <= target_params:
            candidates.append((abs(target_params - a), d, a))
    candidates.sort(key=lambda t: t[0])
    _, d_best, a_best = candidates[0]
    return d_best, max(1, d_best // hd), 4 * d_best, a_best


def build_student(
    *,
    depth:         int,
    d_model:       int,
    n_heads:       int,
    d_ff:          int,
    target_params: int,
    cfg:           TrainCfg,
    seq_len:       int = SEQ_LEN,
    vocab_size:    int = VOCAB_SIZE,
) -> TinyGPT_NTP:
    tmp    = TinyGPT_NTP(
        depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        seq_len=seq_len, vocab_size=vocab_size,
        dropout=cfg.dropout, pad_params=0,
    )
    active = count_params(tmp)
    if active > target_params:
        raise ValueError(f"active {active:,} > target {target_params:,}")
    pad = int(target_params - active)
    return TinyGPT_NTP(
        depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        seq_len=seq_len, vocab_size=vocab_size,
        dropout=cfg.dropout, pad_params=pad,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-N shape sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_shape_sweep_for_n(
    *,
    target_params: int,
    depths:        List[int],
    cfg:           TrainCfg,
    train_data:    np.ndarray,
    val_data:      np.ndarray,
    test_data:     np.ndarray,
    device:        torch.device,
    out_dir:       str,
    global_seed:   int,
) -> List[Dict]:
    """
    Run the full depth sweep for a single N value.
    Returns a list of result dicts (one per valid depth).
    """
    curve_dir = os.path.join(out_dir, f"curves_N{target_params}")
    os.makedirs(curve_dir, exist_ok=True)

    # Chinchilla-scaled data budget:  D = data_ratio × N bytes
    D         = int(cfg.data_ratio * target_params)
    n_windows = D // SEQ_LEN           # number of training windows
    base_steps = max(200, n_windows // cfg.batch_size)

    print(f"\n{'='*72}")
    print(f"N = {target_params:,}   D = {D:,} bytes   windows={n_windows:,}   "
          f"base_steps={base_steps:,}")
    print(f"{'='*72}")

    # Fixed eval loaders (sequential windows on val / test)
    val_loader  = torch.utils.data.DataLoader(
        SequentialWindowDataset(val_data, SEQ_LEN),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        SequentialWindowDataset(test_data, SEQ_LEN),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    results: List[Dict] = []

    for depth in depths:
        print(f"\n  ── depth={depth} ──")

        # 1. Find d_model
        try:
            d_model, n_heads, d_ff, active = find_d_model_for_target_params(
                depth=depth, target_params=target_params, cfg=cfg,
            )
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        pad_ratio = (target_params - active) / target_params
        alpha     = depth / d_model
        print(
            f"  d_model={d_model}  n_heads={n_heads}  d_ff={d_ff}  "
            f"active={active:,}  pad={pad_ratio:.1%}  α={alpha:.4f}"
        )
        if pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] padding ratio {pad_ratio:.1%} > {cfg.max_padding_ratio:.0%}")
            continue

        # 2. Build model
        set_seed(global_seed)
        model = build_student(
            depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            target_params=target_params, cfg=cfg,
        )
        print(f"  Total params (padded): {count_params(model):,}")

        # 3. Build train DataLoader (random windows, seeded per shape)
        shape_seed   = global_seed + depth * 1000
        train_dataset = RandomWindowDataset(
            train_data, SEQ_LEN, n_windows, seed=shape_seed,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
        )

        # 4. Train
        t_start = time.time()
        metrics, history = train_one_model(
            model, train_loader, val_loader, test_loader,
            base_steps, cfg, device,
        )
        elapsed = time.time() - t_start

        # 5. AGOP
        agop = estimate_agop_ntp(
            model, train_data,
            proj_samples=cfg.agop_proj_samples,
            batch_size=cfg.agop_batch,
            n_batches=cfg.agop_n_batches,
            seed=global_seed + 42,
            device=device,
        )
        aofe, aofe_ratio = agop_offdiag_metrics(agop)
        _, wtw_aofe_ratio = compute_wtw_aofe_ratio(model)

        row = {
            "target_n":   target_params,
            "depth":      depth,
            "d_model":    d_model,
            "n_heads":    n_heads,
            "d_ff":       d_ff,
            "active_n":   active,
            "pad_ratio":  round(pad_ratio, 4),
            "alpha":      round(alpha, 4),
            "train_ce":   round(metrics["train_ce"], 6),
            "val_ce":     round(metrics["val_ce"],   6),
            "test_ce":    round(metrics["test_ce"],  6),
            "aofe":          round(aofe,          6),
            "aofe_ratio":    round(aofe_ratio,    6),
            "wtw_aofe_ratio": round(wtw_aofe_ratio, 6),
            "steps_run":  metrics["steps_run"],
            "elapsed_s":  round(elapsed, 1),
        }
        results.append(row)
        print(
            f"  → test_ce={metrics['test_ce']:.4f} nats  "
            f"AOFE_ratio={aofe_ratio:.4f}  α={alpha:.4f}  t={elapsed:.0f}s"
        )

        # Save per-shape learning curve
        curve_path = os.path.join(curve_dir, f"depth{depth:04d}_d{d_model}.csv")
        with open(curve_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys() if history else [])
            w.writeheader(); w.writerows(history)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_n_results(
    results:       List[Dict],
    target_params: int,
    out_dir:       str,
) -> None:
    """Four-panel plot for a single N: loss and AOFE_ratio vs. depth and α."""
    if not results:
        return
    depths     = [r["depth"]      for r in results]
    alphas     = [r["alpha"]      for r in results]
    test_ces   = [r["test_ce"]    for r in results]
    aofe_rats  = [r["aofe_ratio"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"N = {target_params:,}  (byte-level NTP, WikiText-103)",
        fontsize=13, fontweight="bold",
    )

    # --- Loss vs depth ---
    ax = axes[0]
    ax.plot(depths, test_ces, "o-", color="tab:blue", lw=2, ms=6)
    best_idx = int(np.argmin(test_ces))
    ax.plot(depths[best_idx], test_ces[best_idx], "*", color="red", ms=14,
            zorder=5, label=f"best depth={depths[best_idx]}")
    ax.set_xlabel("Depth", fontsize=11)
    ax.set_ylabel("Test cross-entropy (nats/byte)", fontsize=11)
    ax.set_title("Loss vs. depth", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    # --- AOFE_ratio vs depth (twin axes with loss) ---
    ax2 = axes[0].twinx()
    ax2.plot(depths, aofe_rats, "s--", color="tab:orange", lw=1.5, ms=5,
             alpha=0.7, label="AOFE_ratio")
    ax2.set_ylabel("AOFE_ratio", color="tab:orange", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # --- Loss vs aspect ratio α ---
    ax = axes[1]
    sorted_by_alpha = sorted(zip(alphas, test_ces, aofe_rats))
    sa, sc, sr = zip(*sorted_by_alpha)
    ax.plot(sa, sc, "o-", color="tab:blue", lw=2, ms=6)
    best_alpha_idx = int(np.argmin(sc))
    ax.plot(sa[best_alpha_idx], sc[best_alpha_idx], "*", color="red", ms=14,
            zorder=5, label=f"α*={sa[best_alpha_idx]:.4f}")
    ax.set_xlabel("Aspect ratio α = depth / d_model", fontsize=11)
    ax.set_ylabel("Test cross-entropy (nats/byte)", fontsize=11)
    ax.set_title("Loss vs. α  (shape)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    ax3 = axes[1].twinx()
    ax3.plot(sa, sr, "s--", color="tab:orange", lw=1.5, ms=5, alpha=0.7,
             label="AOFE_ratio")
    ax3.set_ylabel("AOFE_ratio", color="tab:orange", fontsize=10)
    ax3.tick_params(axis="y", labelcolor="tab:orange")

    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"N{target_params}_ntp_depth_alpha.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_multi_n_summary(
    all_results: List[Dict],
    param_groups: List[int],
    out_dir: str,
) -> None:
    """
    Summary two-panel figure(s). Left: scatter of a coupling metric vs test CE; right: loss vs depth.

    We emit two PNGs: ``multi_N_ntp_summary.png`` uses **WtW AOFE_ratio** on the scatter x-axis
    (embedding Gram matrix; varies strongly with width / d_model). A companion file
    ``multi_N_ntp_summary_aofe_ratio.png`` uses **aofe_ratio** from the projected AGOP — the Jacobian
    coupling metric aligned with per-single-N plots. WtW is often preferred for this overview
    because it spans a wider numeric range across shapes than AGOP AOFE_ratio (~0.92--0.94 here),
    which makes the multi-N scatter easier to read; AGOP is kept as the primary metric elsewhere
    (twin axes on each ``N*_ntp_depth_alpha.png``).
    """
    if not all_results:
        return

    # Prefer the actual budgets present in the CSV / current result list so
    # plot_only always regenerates a complete summary after new N values are added.
    available_param_groups = sorted({int(r["target_n"]) for r in all_results})
    if param_groups:
        requested = [int(n) for n in param_groups]
        missing = [n for n in requested if n not in available_param_groups]
        if missing:
            print(
                f"[plot_multi_n_summary] Missing N in all_results, falling back to "
                f"available budgets: {available_param_groups}"
            )
            param_groups = available_param_groups
        else:
            extras = [n for n in available_param_groups if n not in requested]
            if extras:
                print(
                    f"[plot_multi_n_summary] Found extra budgets in results: {extras}; "
                    f"including all available budgets in summary."
                )
                param_groups = available_param_groups
    else:
        param_groups = available_param_groups

    try:
        cmap   = matplotlib.colormaps["tab10"]
    except AttributeError:
        cmap   = matplotlib.cm.get_cmap("tab10")   # matplotlib < 3.7 fallback
    colors = {n: cmap(i % 10) for i, n in enumerate(param_groups)}

    def _loss_vs_depth(ax: plt.Axes) -> None:
        for n in param_groups:
            rows = sorted(
                [r for r in all_results if r["target_n"] == n],
                key=lambda r: r["depth"],
            )
            if not rows:
                continue
            depths   = [r["depth"]   for r in rows]
            test_ces = [r["test_ce"] for r in rows]
            ax.plot(depths, test_ces, "o-", color=colors[n], lw=2, ms=5,
                    label=f"N={n/1e6:.1f}M")
            bi = int(np.argmin(test_ces))
            ax.plot(depths[bi], test_ces[bi], "*", color=colors[n], ms=14, zorder=5)
        ax.set_xlabel("Depth", fontsize=11)
        ax.set_ylabel("Test cross-entropy (nats/byte)", fontsize=11)
        ax.set_title("Loss vs. Depth  (★ = optimal)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    def _scatter_vs_loss(
        ax: plt.Axes,
        xkey: str,
        xlabel: str,
        title: str,
    ) -> None:
        for n in param_groups:
            rows = [r for r in all_results if r["target_n"] == n]
            if not rows:
                continue
            xs = [r[xkey] for r in rows]
            ys = [r["test_ce"] for r in rows]
            ax.scatter(xs, ys, color=colors[n], s=60, alpha=0.8, zorder=3,
                       label=f"N={n/1e6:.1f}M")
            bi = int(np.argmin(ys))
            ax.scatter(xs[bi], ys[bi], color=colors[n], s=200, marker="*", zorder=5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Test cross-entropy (nats/byte)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    # --- WtW (default multi-N overview) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "NTP Shape Sweep — Byte-level LM (WikiText-103)",
        fontsize=13, fontweight="bold",
    )
    _scatter_vs_loss(
        axes[0],
        xkey="wtw_aofe_ratio",
        xlabel="WtW AOFE_ratio (embedding superposition)",
        title="WtW AOFE_ratio vs. Loss  (every (N, shape) pair)",
    )
    _loss_vs_depth(axes[1])
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "multi_N_ntp_summary.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # --- Projected AGOP AOFE_ratio (same layout) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "NTP Shape Sweep — Byte-level LM (WikiText-103)",
        fontsize=13, fontweight="bold",
    )
    _scatter_vs_loss(
        axes[0],
        xkey="aofe_ratio",
        xlabel="AOFE_ratio (64×64 projected AGOP)",
        title="AOFE_ratio (AGOP) vs. Loss  (every (N, shape) pair)",
    )
    _loss_vs_depth(axes[1])
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "multi_N_ntp_summary_aofe_ratio.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # --- Additional: optimal α vs N (log-log) ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    opt_alphas, opt_ns = [], []
    for n in param_groups:
        rows = [r for r in all_results if r["target_n"] == n]
        if not rows:
            continue
        best = min(rows, key=lambda r: r["test_ce"])
        opt_alphas.append(best["alpha"])
        opt_ns.append(n)
        ax2.scatter(n, best["alpha"], color=colors[n], s=120, zorder=5,
                    label=f"N={n/1e6:.1f}M,  α*={best['alpha']:.4f},  "
                          f"depth*={best['depth']},  d_model*={best['d_model']}")
    if len(opt_ns) >= 2:
        ax2.plot(opt_ns, opt_alphas, "--", color="gray", lw=1.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Parameter budget N", fontsize=11)
    ax2.set_ylabel("Optimal aspect ratio α* = depth / d_model", fontsize=11)
    ax2.set_title("Optimal Shape vs. N  (Chinchilla-style frontier)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4, which="both")
    fig2.tight_layout()
    fig2.savefig(
        os.path.join(out_dir, "optimal_alpha_vs_N.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig2)

    # --- Additional: best-loss frontier vs WtW AOFE_ratio ---
    frontier = []
    for n in param_groups:
        rows = [r for r in all_results if r["target_n"] == n]
        if not rows:
            continue
        best = min(rows, key=lambda r: r["test_ce"])
        frontier.append(best)

    if len(frontier) >= 2:
        frontier = sorted(frontier, key=lambda r: r["target_n"])
        xs = np.array([r["wtw_aofe_ratio"] for r in frontier], dtype=np.float64)
        ys = np.array([r["test_ce"] for r in frontier], dtype=np.float64)
        ns = np.array([r["target_n"] for r in frontier], dtype=np.int64)

        # Scaling-law-style fit:  y = a * x^b
        log_coef = np.polyfit(np.log(xs), np.log(ys), deg=1)
        power_b = float(log_coef[0])
        power_a = float(np.exp(log_coef[1]))
        ys_hat = power_a * (xs ** power_b)
        ss_res = float(((ys - ys_hat) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum()) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        xfit = np.linspace(xs.min() * 0.98, xs.max() * 1.02, 200)
        yfit = power_a * (xfit ** power_b)

        fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
        fig3.suptitle(
            "Optimal Frontier: Best Loss vs. Best WtW AOFE_ratio",
            fontsize=13, fontweight="bold",
        )

        # Linear axes: easier to read actual values.
        ax = axes3[0]
        ax.scatter(xs, ys, s=90, color="tab:blue", zorder=4)
        ax.plot(xfit, yfit, "--", color="tab:red", lw=2.0,
                label=f"power law: CE = {power_a:.3f} · WtW^({power_b:.3f})")
        for x, y, n, row in zip(xs, ys, ns, frontier):
            ax.annotate(
                f"{n/1e6:.1f}M\n(d={row['depth']}, w={row['d_model']})",
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.85,
            )
        ax.set_xlabel("Best-shape WtW AOFE_ratio", fontsize=11)
        ax.set_ylabel("Best test cross-entropy (nats/byte)", fontsize=11)
        ax.set_title(f"Power-law fit on frontier  ($R^2$={r2:.3f})", fontsize=11)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)

        # Log-log panel: closest to the usual scaling-law presentation.
        ax = axes3[1]
        ax.scatter(xs, ys, s=90, color="tab:blue", zorder=4)
        ax.plot(xfit, yfit, "--", color="tab:red", lw=2.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Best-shape WtW AOFE_ratio (log)", fontsize=11)
        ax.set_ylabel("Best test cross-entropy (log)", fontsize=11)
        ax.set_title("Log-log view", fontsize=11)
        ax.grid(True, alpha=0.4, which="both")

        fig3.tight_layout()
        fig3.savefig(
            os.path.join(out_dir, "optimal_loss_vs_wtw_aofe_ratio.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig3)

        print(
            "[optimal frontier fit] "
            f"best_test_ce ≈ {power_a:.6f} * (best_wtw_aofe_ratio)^{power_b:.6f}   "
            f"(R^2={r2:.4f}, budgets={len(frontier)})"
        )


def print_summary_table(all_results: List[Dict], param_groups: List[int]) -> None:
    """Print a Chinchilla-style summary table to stdout."""
    print("\n" + "=" * 80)
    print("COMPLETE RESULTS: Transformer NTP × Shape Sweep")
    print("Task: Byte-level next-token prediction  (WikiText-103)")
    print("=" * 80)
    print(
        f"\n{'N':>8}  {'depth':>6}  {'d_model':>8}  {'α':>8}  "
        f"{'test_ce':>10}  {'AGOP_ratio':>12}  {'WtW_ratio':>10}"
    )
    print("-" * 78)

    for n in param_groups:
        rows = sorted(
            [r for r in all_results if r["target_n"] == n],
            key=lambda r: r["depth"],
        )
        if not rows:
            continue
        best_ce = min(r["test_ce"] for r in rows)
        for r in rows:
            marker = " ←" if r["test_ce"] == best_ce else ""
            print(
                f"  {n/1e6:>5.1f}M  {r['depth']:>6}  {r['d_model']:>8}  "
                f"{r['alpha']:>8.4f}  {r['test_ce']:>10.5f}  "
                f"{r['aofe_ratio']:>12.4f}  {r['wtw_aofe_ratio']:>10.4f}{marker}"
            )

        ces      = np.array([r["test_ce"]        for r in rows])
        aofes    = np.array([r["aofe_ratio"]      for r in rows])
        wtws     = np.array([r["wtw_aofe_ratio"]  for r in rows])
        pe_a  = pearson_corr(aofes, ces);  sp_a = spearman_corr(aofes, ces)
        pe_w  = pearson_corr(wtws,  ces);  sp_w = spearman_corr(wtws,  ces)
        best_row = min(rows, key=lambda r: r["test_ce"])
        print(
            f"        AGOP_ratio — Pearson={pe_a:.4f}  Spearman={sp_a:.4f}\n"
            f"        WtW_ratio  — Pearson={pe_w:.4f}  Spearman={sp_w:.4f}\n"
            f"        Optimal: depth*={best_row['depth']}  "
            f"d_model*={best_row['d_model']}  α*={best_row['alpha']:.4f}"
        )
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_results_csv(csv_path: str) -> List[Dict]:
    """
    Load results_ntp_shape_sweep.csv with numeric types.
    If duplicate (target_n, depth) rows exist, keeps the last occurrence.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    int_keys = {
        "target_n", "depth", "d_model", "n_heads", "d_ff", "active_n", "steps_run",
    }
    float_keys = {
        "pad_ratio", "alpha", "train_ce", "val_ce", "test_ce",
        "aofe", "aofe_ratio", "wtw_aofe_ratio", "elapsed_s",
    }
    parsed: List[Dict] = []
    for r in rows:
        d = dict(r)
        for k in int_keys:
            if k in d and d[k] != "":
                d[k] = int(float(d[k]))
        for k in float_keys:
            if k in d and d[k] != "":
                d[k] = float(d[k])
        parsed.append(d)
    # Dedupe (target_n, depth): last row wins
    by_key: Dict[Tuple[int, int], Dict] = {}
    for r in parsed:
        key = (int(r["target_n"]), int(r["depth"]))
        by_key[key] = r
    return sorted(by_key.values(), key=lambda x: (x["target_n"], x["depth"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NTP shape sweep: next-token prediction on WikiText-103 (byte-level).",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help=(
            "Skip training and corpus load; read results_ntp_shape_sweep.csv under "
            "--out_dir, then regenerate per-N plots, multi-N summaries, and the stdout table. "
            "Use after adding new runs so figures include all N in the CSV."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory with wikitext103_*.bin files (auto-downloads if missing).",
    )
    parser.add_argument(
        "--param_groups", type=str, default="300000,1000000,3000000",
        help="Comma-separated N values (model parameter budgets).",
    )
    parser.add_argument(
        "--depth_list", type=str, default="1,2,3,4,5,6,8,10,12,16,20,24",
        help="Comma-separated depth values to sweep.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results_ntp_shape_sweep",
        help="Output directory for CSVs and plots.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    # TrainCfg overrides
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-2)
    parser.add_argument("--data_ratio",    type=float, default=60.0,
                        help="D = data_ratio × N  bytes. Default 60 ≈ 20 BPE-tokens × 3.")
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--warmup_steps",  type=int,   default=300)
    parser.add_argument("--head_dim",      type=int,   default=4,
                        help="n_heads = d_model // head_dim.")
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--max_padding_ratio", type=float, default=0.20)
    parser.add_argument("--max_train_factor",  type=float, default=1.5)
    parser.add_argument("--fit_patience",  type=int,   default=10)
    parser.add_argument("--agop_batch",    type=int,   default=128)
    parser.add_argument("--agop_proj_samples", type=int, default=64)
    parser.add_argument("--agop_n_batches",    type=int, default=4)
    parser.add_argument("--d_model_min",   type=int,   default=16)
    parser.add_argument("--d_model_max",   type=int,   default=1024)
    parser.add_argument("--seed",          type=int,   default=0)
    args = parser.parse_args()

    if args.plot_only:
        csv_path = os.path.join(args.out_dir, "results_ntp_shape_sweep.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run training first or set --out_dir correctly."
            )
        all_results = load_results_csv(csv_path)
        param_groups = sorted({int(r["target_n"]) for r in all_results})
        print(f"[plot_only] Loaded {len(all_results)} rows from {csv_path}")
        print(f"[plot_only] N values: {param_groups}\n")
        for n in param_groups:
            rows = sorted(
                [r for r in all_results if r["target_n"] == n],
                key=lambda r: r["depth"],
            )
            plot_per_n_results(rows, n, args.out_dir)
        plot_multi_n_summary(all_results, param_groups, args.out_dir)
        print_summary_table(all_results, param_groups)
        print(f"\n[plot_only] Plots and table regenerated under: {args.out_dir}")
        print(f"[plot_only] CSV unchanged: {csv_path}")
        return

    # Build config from args
    cfg = TrainCfg(**{
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(TrainCfg)
        if hasattr(args, f.name)
    })

    param_groups  = [int(x) for x in args.param_groups.split(",")]
    depth_list    = [int(x) for x in args.depth_list.split(",")]
    device        = torch.device(args.device if torch.cuda.is_available()
                                 else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device : {device}")
    print(f"N      : {param_groups}")
    print(f"depths : {depth_list}")
    print(f"D=ratio×N: {cfg.data_ratio}×N bytes  (≈ {cfg.data_ratio/3:.0f}×N BPE-tokens)")
    print(f"Out    : {args.out_dir}\n")

    # Load corpus
    print("Loading WikiText-103 ...")
    train_data, val_data, test_data = load_corpus(args.data_dir)
    print(
        f"  train={len(train_data)/1e6:.1f}M  "
        f"val={len(val_data)/1e6:.1f}M  "
        f"test={len(test_data)/1e6:.1f}M bytes\n"
    )

    # ── Quick shape preview (no training) ────────────────────────────────────
    print("Shape preview (no training):")
    print(f"{'N':>10}  {'depth':>6}  {'d_model':>8}  {'active':>10}  "
          f"{'pad%':>6}  {'α':>8}")
    print("-" * 56)
    for n in param_groups:
        for depth in depth_list:
            try:
                d, nh, dff, active = find_d_model_for_target_params(
                    depth=depth, target_params=n, cfg=cfg,
                )
                pad = (n - active) / n
                alpha = depth / d
                flag = " [PAD>20%]" if pad > cfg.max_padding_ratio else ""
                print(
                    f"  {n:>9,}  {depth:>6}  {d:>8}  {active:>10,}  "
                    f"{pad:>5.1%}  {alpha:>8.4f}{flag}"
                )
            except ValueError as e:
                print(f"  {n:>9,}  {depth:>6}  {'SKIP':>8}  {str(e)}")
    print()

    # ── Main sweep ────────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    csv_path = os.path.join(args.out_dir, "results_ntp_shape_sweep.csv")
    fieldnames: Optional[List[str]] = None

    for n in param_groups:
        rows = run_shape_sweep_for_n(
            target_params=n,
            depths=depth_list,
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
            out_dir=args.out_dir,
            global_seed=cfg.seed,
        )
        all_results.extend(rows)

        # Plot per-N
        plot_per_n_results(rows, n, args.out_dir)

        # Append to CSV
        if rows:
            if fieldnames is None:
                fieldnames = list(rows[0].keys())
            mode = "a" if os.path.exists(csv_path) else "w"
            with open(csv_path, mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == "w":
                    w.writeheader()
                w.writerows(rows)

    # ── Final outputs ─────────────────────────────────────────────────────────
    if all_results:
        plot_multi_n_summary(all_results, param_groups, args.out_dir)
        print_summary_table(all_results, param_groups)

    print(f"\nAll results saved to: {args.out_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
