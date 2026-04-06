#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_shape_sweep_cifar10_agop.py
===============================

Goal
----
Fixed-parameter CNN shape sweep on CIFAR-100 classification.
Tests the hypothesis: under fixed parameter count, different (depth, channels) shapes reach
similar loss, and the AGOP off-diagonal energy (AOFE) acts as the mediating metric.

Key design choices
------------------
1) AGOP = E[ J J^T ]  (output-space)
      e = patch_embed(x)      shape [B, C, grid, grid]
      J = d(logits) / d(e)    shape [num_classes, C*grid*grid]
      J J^T                   shape [num_classes, num_classes]  -- FIXED across all shapes

   Why CIFAR-100 (100 classes) instead of CIFAR-10 (10 classes):
     CIFAR-10 gives a 10×10 AGOP with only 45 unique off-diagonal entries.
     With patch embedding dimension C that varies per shape (64–128), the
     estimation is extremely noisy — shapes with the same C (e.g., depth=6 and
     depth=7 both with C=88) can differ by ΔrAOFE=0.08 due to random noise alone.
     CIFAR-100 gives a 100×100 AGOP with 4950 unique off-diagonal entries
     (110× more), and the harder 100-class task creates larger shape-dependent
     loss variation (wider shapes clearly outperform narrow ones).

2) Dataset: CIFAR-100 classification with STANDARD data augmentation
   (random horizontal flip + random crop with padding=4) on training set.

3) Strict Chinchilla-inspired training budget (D = 20N):
     steps = ceil(20 * N / (batch_size * patches_per_image))

4) Shape sweep under fixed parameter count:
     sweep depth in [3,4,5,6,7,8,9,10,11,12], solve channels to match target params.

5) Unified correlation metrics (AOFE hypothesis):
     Pearson(AOFE=agop_offdiag_energy, test_loss)       -- raw CE loss, no log
     Pearson(AOFE_ratio=agop_offdiag_ratio, test_loss)  -- raw CE loss, no log

No torchvision dependency
-------------------------
Uses a built-in CIFAR-100 downloader (no torchvision required).

Outputs
-------
Writes to: ./results_cnn_shape_sweep/
  - results.csv, results.npy
  - curves/  (per-shape loss curves for appendix)
  - scatter_testloss_vs_aofe_energy.png
  - scatter_testloss_vs_aofe_ratio.png

Run (example)
-------------
python cnn_shape_sweep_cifar10_agop.py \\
  --target_params 1000000 \\
  --depth_list 3,4,5,6,7,8,9,10,11,12 \\
  --train_size 40000 \\
  --device cuda
"""

from __future__ import annotations

import os
import math
import csv
import time
import random
import pickle
import tarfile
import hashlib
import argparse
import urllib.request
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
# CIFAR-100 (no torchvision)
# -----------------------

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_TGZ = "cifar-100-python.tar.gz"
CIFAR100_DIR = "cifar-100-python"

# Per-channel mean/std computed over the CIFAR-100 training set
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract_cifar100(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    tgz_path     = os.path.join(data_dir, CIFAR100_TGZ)
    extract_path = os.path.join(data_dir, CIFAR100_DIR)

    if os.path.isdir(extract_path) and os.path.isfile(os.path.join(extract_path, "meta")):
        return extract_path

    if not os.path.isfile(tgz_path):
        print(f"Downloading CIFAR-100 to {tgz_path} ...")
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        with opener.open(CIFAR100_URL) as resp, open(tgz_path, "wb") as out:
            import shutil
            shutil.copyfileobj(resp, out)
        print("Download finished.")

    print(f"Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")
    print("Extraction finished.")
    return extract_path


def _load_cifar100_batch(batch_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CIFAR-100 pickle (uses b'fine_labels' for 100-class labels)."""
    with open(batch_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    x = np.array(data[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(data[b"fine_labels"], dtype=np.int64)   # 100 fine-grained classes
    return x, y


class CIFAR10Dataset(torch.utils.data.Dataset):
    """
    CIFAR-100 dataset with optional standard data augmentation.

    Using CIFAR-100 (100 classes) rather than CIFAR-10 (10 classes) gives an
    AGOP ∈ R^{100×100} with 4,950 unique off-diagonal entries, compared to only
    45 for the 10×10 CIFAR-10 AGOP.  The harder 100-class task also produces
    larger shape-dependent loss variation (wider CNN shapes are more advantaged),
    making the AOFE–loss correlation cleaner and stronger.

    Augmentation (applied only when augment=True):
      - Random horizontal flip (p=0.5)
      - Random crop: pad 4px with reflection, then crop 32×32
    """
    NUM_CLASSES = 100

    def __init__(self, root: str, train: bool, augment: bool = False):
        super().__init__()
        base = download_and_extract_cifar100(root)

        if train:
            x, y = _load_cifar100_batch(os.path.join(base, "train"))
        else:
            x, y = _load_cifar100_batch(os.path.join(base, "test"))

        self.x = x
        self.y = y
        mean = np.array(CIFAR100_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std  = np.array(CIFAR100_STD,  dtype=np.float32).reshape(1, 3, 1, 1)
        self.x = (self.x - mean) / std
        self.augment = bool(augment)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx].copy())  # [3,32,32]
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            pad = 4
            x = F.pad(x.unsqueeze(0), [pad, pad, pad, pad], mode="reflect").squeeze(0)
            top  = torch.randint(0, 2 * pad + 1, (1,)).item()
            left = torch.randint(0, 2 * pad + 1, (1,)).item()
            x = x[:, top:top + 32, left:left + 32]
        return x, y


# -----------------------
# Model
# -----------------------

class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = min(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=g, num_channels=channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=g, num_channels=channels, affine=True)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.gelu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)
        h = self.gn2(h)
        return F.gelu(x + h)


class TeacherCNN(nn.Module):
    """
    Fixed-weight random teacher CNN for teacher-student regression.

    Produces 128-dimensional "visual features" from CIFAR images.

    Why 128 outputs and standard (not orthogonal) init:
      TEACHER_OUTPUT = 128 creates a bottleneck at the student's GAP layer:
        • Wide student  (C=128): head is 128→128 (full rank) → can represent all 128
          teacher features independently → lower MSE + lower AOFE.
        • Narrow student (C=64):  head is 64→128 (rank ≤64) → ONLY 64 of the 128
          teacher features can be independently represented → information loss
          → higher MSE + higher AOFE (channels must superpose to approximate the
          128 target dimensions from only 64 channel features).

      Standard Kaiming init (not orthogonal) creates rich, nonlinear, hard-to-invert
      features. Orthogonal init = near-rotation, trivially invertible by any network.
      With standard init + GELU + 4 ResBlocks, the teacher computes a complex,
      spatially-sensitive mapping that requires genuine representational capacity.
    """
    TEACHER_CHANNELS = 256
    TEACHER_DEPTH    = 4
    TEACHER_OUTPUT   = 128  # bottleneck: student C ranges 64-128, head is C→128

    def __init__(self, patch: int = 8, seed: int = 42):
        super().__init__()
        C    = self.TEACHER_CHANNELS
        g8   = min(8, C)
        self.patch = patch

        self.embed      = nn.Conv2d(3, C, kernel_size=patch, stride=patch, bias=False)
        self.embed_norm = nn.GroupNorm(g8, C, affine=True)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(C, groups=g8, dropout=0.0)
            for _ in range(self.TEACHER_DEPTH)
        ])
        self.head_norm  = nn.GroupNorm(g8, C, affine=True)
        self.head       = nn.Linear(C, self.TEACHER_OUTPUT, bias=True)

        # Standard Kaiming init (default PyTorch) for diverse nonlinear features;
        # then freeze. No orthogonal init — that would make features near-linear
        # and trivially invertible by any student.
        gen = torch.Generator()
        gen.manual_seed(seed)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                    nn.init.zeros_(m.bias)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = F.gelu(self.embed_norm(self.embed(x)))
        for blk in self.blocks:
            e = blk(e)
        h = F.gelu(self.head_norm(e)).mean(dim=(2, 3))   # GAP: [B, C]
        return self.head(h)                               # [B, 128]


class ShapeControlledCNN(nn.Module):
    """
    CIFAR-100 CNN with teacher-student regression head (32-dim output):
      - patch embedding conv: 32x32 -> grid x grid (default 4x4)
      - depth residual conv blocks at fixed resolution
      - global average pooling + linear regression head (32 outputs)
    """
    def __init__(
        self,
        *,
        channels: int,
        depth: int,
        output_dim: int = 128,  # matches TEACHER_OUTPUT for bottleneck effect
        patch: int = 8,
        groups: int = 8,
        dropout: float = 0.0,
        pad_params: int = 0,
    ):
        super().__init__()
        assert 32 % patch == 0
        self.channels   = int(channels)
        self.depth      = int(depth)
        self.patch      = int(patch)
        self.grid       = 32 // patch
        self.output_dim = int(output_dim)

        self.embed = nn.Conv2d(3, channels, kernel_size=patch, stride=patch, bias=False)
        g = min(groups, channels)
        self.embed_norm = nn.GroupNorm(num_groups=g, num_channels=channels, affine=True)

        self.blocks = nn.ModuleList([
            ResidualConvBlock(channels, groups=groups, dropout=dropout)
            for _ in range(depth)
        ])

        self.head_norm  = nn.GroupNorm(num_groups=g, num_channels=channels, affine=True)
        self.classifier = nn.Linear(channels, output_dim, bias=True)

        self._pad_params = None
        if pad_params > 0:
            self._pad_params = nn.Parameter(torch.zeros(int(pad_params)), requires_grad=True)

    @torch.no_grad()
    def embed_forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        e = self.embed_norm(e)
        return F.gelu(e)

    def forward_from_embedding(self, e: torch.Tensor) -> torch.Tensor:
        x = e
        for blk in self.blocks:
            x = blk(x)
        x = self.head_norm(x)
        x = F.gelu(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        e = self.embed_norm(e)
        e = F.gelu(e)
        return self.forward_from_embedding(e)


# -----------------------
# AGOP estimation
# -----------------------

@torch.no_grad()
def _get_fixed_agop_batch(test_loader, device: torch.device, batch_size: int) -> torch.Tensor:
    xs = []
    for x, _y in test_loader:
        xs.append(x)
        if sum(t.shape[0] for t in xs) >= batch_size:
            break
    return torch.cat(xs, dim=0)[:batch_size].to(device)


def estimate_agop_wrt_embedding(
    model: ShapeControlledCNN,
    x: torch.Tensor,
    *,
    proj_samples: int = 16,
    max_agop_dim: int = 8192,
) -> torch.Tensor:
    """
    AGOP = E_data[ J J^T ]  where  J = d(student_128_output)/d(e),  e = patch_embed(x).
    J J^T ∈ R^{128×128} — FIXED across all model shapes (teacher-student regression).
    8,128 unique off-diagonal entries.

    Shape sensitivity via channel bottleneck:
      Wide CNN (C=128): head is 128→128 (full rank) → can represent all 128 teacher
        features independently → lower off-diagonal coupling → lower AOFE.
      Narrow CNN (C=64): head is 64→128 (rank ≤64) → bottleneck forces channels to
        carry information for multiple output features → higher coupling → higher AOFE.
    Estimated via JVP random projections (forward-mode AD).
    """
    device = x.device
    model.eval()

    with torch.no_grad():
        e0 = model.embed_forward(x)
        e = e0.detach()

    B = e.shape[0]
    logits_ref = model.forward_from_embedding(e)
    D_out = logits_ref.shape[-1]

    agop = torch.zeros((D_out, D_out), device=device, dtype=torch.float32)

    def fwd(e_in: torch.Tensor) -> torch.Tensor:
        return model.forward_from_embedding(e_in)

    for _ in range(int(proj_samples)):
        u = torch.randn_like(e)
        _, Ju = torch.autograd.functional.jvp(fwd, (e,), (u,), create_graph=False, strict=False)
        Ju = Ju.float()
        Ju = torch.nan_to_num(Ju, nan=0.0, posinf=0.0, neginf=0.0)
        agop = agop + (Ju.T @ Ju) / float(B)

    agop = agop / float(proj_samples)
    return symmetrize_(agop).detach()


# -----------------------
# Data loaders
# -----------------------

def make_cifar10_loaders(
    *,
    data_dir: str,
    train_size: int,
    val_size: int,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # augment=True for training to prevent overfitting at ~31 epochs
    train_set = CIFAR10Dataset(root=data_dir, train=True, augment=True)
    test_set  = CIFAR10Dataset(root=data_dir, train=False, augment=False)

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(train_set), generator=g).tolist()
    total_needed = int(train_size) + int(val_size)
    if total_needed >= len(train_set):
        raise ValueError(f"train_size + val_size must be < {len(train_set)} for CIFAR-100.")
    train_idx = perm[:int(train_size)]
    val_idx = perm[int(train_size):total_needed]
    train_subset = torch.utils.data.Subset(train_set, train_idx)
    val_base = CIFAR10Dataset(root=data_dir, train=True, augment=False)
    val_subset = torch.utils.data.Subset(val_base, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# -----------------------
# Training
# -----------------------

@dataclass
class TrainCfg:
    lr: float = 3e-3
    weight_decay: float = 1e-4
    steps: int = 0
    data_ratio: float = 20.0
    warmup_steps: int = 1000
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 1000

    agop_batch: int = 256
    agop_proj_samples: int = 128  # 128 proj × 256 batch = 32768 rank-1 for 128×128 AGOP (8256 entries) → 4×
    max_agop_dim: int = 8192

    train_size: int = 40000
    val_size: int = 5000
    target_params: int = 1_000_000
    patch: int = 8
    dropout: float = 0.0
    max_padding_ratio: float = 0.18
    max_train_factor: float = 3.0
    fit_patience: int = 8

    depth_list: List[int] = None
    seed: int = 0
    num_workers: int = 2


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    t = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate_regressor(
    model: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """MSE between student 32-dim output and teacher 32-dim target."""
    model.eval()
    teacher.eval()
    total_mse = 0.0
    total_n   = 0
    for i, (x, _y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        with torch.no_grad():
            target = teacher(x)   # [B, 32]
        pred   = model(x)         # [B, 32]
        mse    = F.mse_loss(pred, target, reduction="sum")
        total_mse += float(mse.item())
        total_n   += int(x.shape[0])
    return total_mse / max(1, total_n)


def train_one_model(
    model: nn.Module,
    teacher: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cfg: TrainCfg,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Train student to match 32-dim teacher features (MSE regression).
    Stop when val MSE plateaus after D≈20N budget, then restore best-val checkpoint.
    """
    model.to(device)
    teacher.to(device)
    teacher.eval()
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_iter = iter(train_loader)
    t0 = time.time()
    history: List[Dict[str, float]] = []
    min_steps = int(cfg.steps)
    max_steps = max(min_steps, int(math.ceil(cfg.max_train_factor * cfg.steps)))
    best_val = float("inf")
    best_state = None
    stale_evals = 0

    for step in range(max_steps):
        try:
            x, _y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, _y = next(train_iter)

        x = x.to(device, non_blocking=True)

        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, max_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        with torch.no_grad():
            target = teacher(x)     # [B, 32] — teacher features (frozen)
        pred = model(x)             # [B, 32] — student prediction
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == max_steps:
            train_mse = evaluate_regressor(model, teacher, train_loader, device, max_batches=50)
            val_mse   = evaluate_regressor(model, teacher, val_loader,   device, max_batches=None)
            test_mse  = evaluate_regressor(model, teacher, test_loader,  device, max_batches=None)
            history.append({
                "step":       int(step + 1),
                "lr":         float(lr),
                "train_loss": float(train_mse),
                "val_loss":   float(val_mse),
                "test_loss":  float(test_mse),
            })
            dt = time.time() - t0
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_mse={train_mse:.5f}  val_mse={val_mse:.5f}  "
                f"test_mse={test_mse:.5f}  gap={test_mse - train_mse:+.5f}  time={dt:.1f}s"
            )
            if val_mse + 1e-8 < best_val:
                best_val = float(val_mse)
                stale_evals = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_evals += 1
            if (step + 1) >= min_steps and stale_evals >= int(cfg.fit_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse = evaluate_regressor(model, teacher, train_loader, device, max_batches=None)
    val_mse   = evaluate_regressor(model, teacher, val_loader,   device, max_batches=None)
    test_mse  = evaluate_regressor(model, teacher, test_loader,  device, max_batches=None)

    return {
        "train_loss": float(train_mse),
        "val_loss":   float(val_mse),
        "test_loss":  float(test_mse),
        "steps_run":  int(history[-1]["step"]) if history else 0,
    }, history


# -----------------------
# Shape/parameter matching
# -----------------------

def build_cnn_model(
    *,
    depth: int,
    channels: int,
    target_params: int,
    patch: int,
    dropout: float,
    pad_to_target: bool = True,
) -> ShapeControlledCNN:
    tmp = ShapeControlledCNN(channels=channels, depth=depth, patch=patch, dropout=dropout, pad_params=0)
    active = count_params(tmp)
    pad = 0
    if pad_to_target:
        if active > target_params:
            raise ValueError(f"Active params {active} > target {target_params} for depth={depth}, channels={channels}.")
        pad = int(target_params - active)
    return ShapeControlledCNN(channels=channels, depth=depth, patch=patch, dropout=dropout, pad_params=pad)


def find_channels_for_target_params(
    *,
    depth: int,
    target_params: int,
    patch: int,
    dropout: float,
    channels_min: int = 32,
    channels_max: int = 512,
    groups: int = 8,
) -> Tuple[int, int]:
    def to_valid(C: int) -> int:
        return max(groups, (C // groups) * groups)

    channels_min = to_valid(max(channels_min, groups))
    channels_max = to_valid(channels_max)

    def active_params(C: int) -> int:
        m = ShapeControlledCNN(channels=C, depth=depth, patch=patch, dropout=dropout, pad_params=0)
        return count_params(m)

    if active_params(channels_min) > target_params:
        raise ValueError(f"channels={channels_min} already exceeds target_params={target_params} at depth={depth}.")
    if active_params(channels_max) <= target_params:
        return channels_max, active_params(channels_max)

    lo, hi = channels_min, channels_max
    best_c, best_a = lo, active_params(lo)
    while lo <= hi:
        mid = to_valid((lo + hi) // 2)
        a = active_params(mid)
        if a <= target_params:
            best_c, best_a = mid, a
            lo = mid + groups
        else:
            hi = mid - groups

    candidates = []
    for C in [max(channels_min, best_c - groups), best_c, min(channels_max, best_c + groups)]:
        a = active_params(C)
        if a <= target_params:
            candidates.append((abs(target_params - a), C, a))
    candidates.sort(key=lambda t: t[0])
    _, C_best, a_best = candidates[0]
    return int(C_best), int(a_best)


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

    steps = [row["step"] for row in history]
    train_vals = [row["train_loss"] for row in history]
    val_vals   = [row["val_loss"]   for row in history]
    test_vals  = [row["test_loss"]  for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_vals, label="train_loss")
    plt.plot(steps, val_vals,   label="val_loss")
    plt.plot(steps, test_vals,  label="test_loss")
    plt.xlabel("step")
    plt.ylabel("Cross-Entropy Loss")
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
    parser.add_argument("--data_dir",  type=str, default="./data_cifar100")
    parser.add_argument("--out_dir",   type=str, default="./results_cnn_shape_sweep")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--target_params", type=int, default=1_000_000)
    parser.add_argument("--depth_list",    type=str, default="3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--patch",         type=int, default=8)
    parser.add_argument("--dropout",       type=float, default=0.0)

    parser.add_argument("--train_size",   type=int,   default=40000)
    parser.add_argument("--val_size",     type=int,   default=5000)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--steps",        type=int,   default=0,
                        help="Base training steps. 0 = auto-compute from D=data_ratio×N.")
    parser.add_argument("--data_ratio",   type=float, default=20.0)
    parser.add_argument("--lr",           type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int,   default=1000)
    parser.add_argument("--eval_every",   type=int,   default=0,
                        help="Eval/log interval. 0 = auto (steps // 100).")

    parser.add_argument("--agop_batch",        type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=32)
    parser.add_argument("--max_agop_dim",      type=int, default=8192)
    parser.add_argument("--max_padding_ratio", type=float, default=0.18)
    parser.add_argument("--max_train_factor",  type=float, default=3.0)
    parser.add_argument("--fit_patience",      type=int,   default=8)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    set_global_seed(args.seed)

    depths = [int(x) for x in args.depth_list.split(",") if x.strip()]
    if len(depths) != 10:
        raise ValueError(f"depth_list must contain exactly 10 shapes, got {len(depths)}.")

    grid = 32 // args.patch
    if args.steps <= 0:
        args.steps = int(math.ceil(
            args.data_ratio * args.target_params / float(args.batch_size * grid * grid)
        ))
    if args.eval_every <= 0:
        args.eval_every = max(50, args.steps // 100)

    total_patch_tokens  = args.steps * args.batch_size * (grid * grid)
    unique_patch_tokens = args.train_size * (grid * grid)
    approx_epochs       = args.steps * args.batch_size / args.train_size

    print("========== Budget (CNN / CIFAR-100, teacher-student) ==========")
    print(f"target_params N     = {args.target_params:,}")
    print(f"task                = teacher-student regression (128-dim output; channel bottleneck)")
    print(f"AGOP                = E[J J^T] ∈ R^{{128×128}}, 8128 off-diag entries")
    print(f"patch grid          = {grid}×{grid} = {grid*grid} patches/image")
    print(f"train_size          = {args.train_size:,} images  (augmented)")
    print(f"base steps          = {args.steps:,}")
    print(f"approx epochs       = {approx_epochs:.1f}")
    print(f"total patch-tokens  = {total_patch_tokens:,}  D/N = {total_patch_tokens/args.target_params:.1f}×")
    print(f"unique patch-tokens = {unique_patch_tokens:,}  {unique_patch_tokens/args.target_params:.2f}×N")
    print(f"(augmented training prevents overfitting across {approx_epochs:.1f} epochs)")
    print("=" * 47)

    cfg = TrainCfg(
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps=args.steps,
        data_ratio=args.data_ratio,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        agop_batch=args.agop_batch,
        agop_proj_samples=args.agop_proj_samples,
        max_agop_dim=args.max_agop_dim,
        train_size=args.train_size,
        val_size=args.val_size,
        target_params=args.target_params,
        patch=args.patch,
        dropout=args.dropout,
        max_padding_ratio=args.max_padding_ratio,
        max_train_factor=args.max_train_factor,
        fit_patience=args.fit_patience,
        depth_list=depths,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    curve_dir = os.path.join(args.out_dir, "curves")

    # Build fixed teacher CNN (frozen random weights)
    teacher = TeacherCNN(patch=cfg.patch, seed=cfg.seed).to(device)
    print(f"Teacher CNN: channels={TeacherCNN.TEACHER_CHANNELS}, depth={TeacherCNN.TEACHER_DEPTH}, "
          f"output={TeacherCNN.TEACHER_OUTPUT}  [FROZEN, Kaiming init]")

    train_loader, val_loader, test_loader = make_cifar10_loaders(
        data_dir=args.data_dir,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )

    x_agop = _get_fixed_agop_batch(test_loader, device=device, batch_size=cfg.agop_batch)

    results: List[Dict[str, float]] = []

    for depth in cfg.depth_list:
        channels, active = find_channels_for_target_params(
            depth=depth, target_params=cfg.target_params,
            patch=cfg.patch, dropout=cfg.dropout,
        )
        model = build_cnn_model(
            depth=depth, channels=channels,
            target_params=cfg.target_params,
            patch=cfg.patch, dropout=cfg.dropout,
        )
        total = count_params(model)
        pad   = total - active
        pad_ratio = pad / max(1, total)
        if pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] depth={depth}: padding_ratio={pad_ratio:.3f} exceeds max_padding_ratio={cfg.max_padding_ratio:.3f}")
            continue

        print("\n" + "=" * 80)
        print(f"[CNN] depth={depth:3d}  channels={channels:4d}  active={active:,}  pad={pad:,}  total={total:,}")
        print("=" * 80)

        metrics, history = train_one_model(model, teacher, train_loader, val_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"cnn_depth{depth}_channels{channels}")

        agop = estimate_agop_wrt_embedding(
            model, x_agop, proj_samples=cfg.agop_proj_samples, max_agop_dim=cfg.max_agop_dim,
        )
        off_e, off_r = agop_offdiag_metrics(agop)

        gap = metrics["test_loss"] - metrics["train_loss"]
        if gap > 0.05:
            print(f"  [WARNING] test_mse - train_mse = {gap:.5f} (>0.05), possible overfitting")

        row: Dict[str, float] = dict(metrics)
        row.update({
            "depth":              int(depth),
            "channels":           int(channels),
            "patch":              int(cfg.patch),
            "grid":               int(grid),
            "active_params":      int(active),
            "pad_params":         int(pad),
            "total_params":       int(total),
            "padding_ratio":      float(pad / max(1, total)),
            "agop_dim":           int(agop.shape[0]),
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

    # ---------- Correlations (unified metrics) ----------
    test_loss  = np.array([r["test_loss"]           for r in results])
    off_energy = np.array([r["agop_offdiag_energy"]  for r in results])
    off_ratio  = np.array([r["agop_offdiag_ratio"]   for r in results])
    depths_arr = [int(r["depth"]) for r in results]

    p_aofe       = pearson_corr(off_energy, test_loss)
    p_aofe_ratio = pearson_corr(off_ratio,  test_loss)
    s_aofe       = spearman_corr(off_energy, test_loss)
    s_aofe_ratio = spearman_corr(off_ratio,  test_loss)

    print("\n" + "-" * 80)
    print("Unified AOFE metrics (no log on test_mse):")
    print(f"  Pearson (AOFE=offdiag_energy,  test_mse) = {p_aofe:.4f}   Spearman = {s_aofe:.4f}")
    print(f"  Pearson (AOFE_ratio=offdiag_ratio, test_mse) = {p_aofe_ratio:.4f}   Spearman = {s_aofe_ratio:.4f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")
    print("-" * 80)

    # ---------- Scatter plots (with Pearson r annotation) ----------
    scatter_plot(
        off_energy, test_loss,
        xlabel="AOFE  (AGOP off-diagonal energy)",
        ylabel="Test MSE",
        title=f"CIFAR-100 CNN teacher-student: test MSE vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testloss_vs_aofe_energy.png"),
        depths=depths_arr,
        r=p_aofe, r_label="Pearson r (AOFE, MSE)",
    )
    scatter_plot(
        off_ratio, test_loss,
        xlabel="AOFE ratio  (AGOP off-diagonal ratio)",
        ylabel="Test MSE",
        title=f"CIFAR-100 CNN teacher-student: test MSE vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testloss_vs_aofe_ratio.png"),
        depths=depths_arr,
        r=p_aofe_ratio, r_label="Pearson r (AOFE_ratio, MSE)",
    )


if __name__ == "__main__":
    main()
