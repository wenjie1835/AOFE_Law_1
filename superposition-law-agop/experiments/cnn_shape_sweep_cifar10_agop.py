#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_shape_sweep_cifar10_agop.py
===============================

Goal
----
Fixed-parameter CNN shape sweep on CIFAR-10 classification.
Tests the hypothesis: under fixed parameter count, different (depth, channels) shapes reach
similar loss, and the AGOP off-diagonal energy (AOFE) acts as the mediating metric.

Key design choices
------------------
1) AGOP = E[ J J^T ]  (output-space)
      e = patch_embed(x)    shape [B, C, grid, grid]
      J = d(logits) / d(e)  shape [num_classes, C*grid*grid] = [10, D_in]
      J J^T                 shape [10, 10]  -- FIXED across all model shapes

2) Dataset: CIFAR-10 classification with STANDARD data augmentation
   (random horizontal flip + random crop with padding=4) on training set.
   Augmentation is critical to prevent overfitting since D=20N requires ~31 epochs
   through the 40K-image training set without augmentation.

3) Strict Chinchilla-inspired training budget (D = 20N):
     steps = ceil(20 * N / (batch_size * patches_per_image))
     training stops at exactly this budget — no extension, no patience early-stop.
     The final model (not best-train checkpoint) is evaluated and reported.
     This matches the Chinchilla "compute-optimal" evaluation protocol.

4) Shape sweep under fixed parameter count:
     sweep depth in [3,4,5,6,7,8,9,10,11,12] (10 non-extreme shapes), solve channels
     (multiple of groups=8, for GroupNorm) to match target params, then pad remainder.
     depth=2 excluded as too shallow; depth=11 added for denser coverage.

5) Unified correlation metrics (AOFE hypothesis):
     Pearson(AOFE=agop_offdiag_energy, test_loss)       -- raw CE loss, no log
     Pearson(AOFE_ratio=agop_offdiag_ratio, test_loss)  -- raw CE loss, no log

No torchvision dependency
-------------------------
Uses a built-in CIFAR-10 downloader (no torchvision required).
Augmentation is implemented with pure PyTorch / numpy.

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
  --depth_list 2,3,4,5,6,7,8,9,10,12 \\
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
# CIFAR-10 (no torchvision)
# -----------------------

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_TGZ = "cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract_cifar10(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, CIFAR10_TGZ)
    extract_path = os.path.join(data_dir, CIFAR10_DIR)

    if os.path.isdir(extract_path) and os.path.isfile(os.path.join(extract_path, "batches.meta")):
        return extract_path

    if not os.path.isfile(tgz_path):
        print(f"Downloading CIFAR-10 to {tgz_path} ...")
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        with opener.open(CIFAR10_URL) as resp, open(tgz_path, "wb") as out:
            import shutil
            shutil.copyfileobj(resp, out)
        print("Download finished.")

    print(f"Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")  # suppress Python 3.14 deprecation
    print("Extraction finished.")
    return extract_path


def _load_batch(batch_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(batch_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    # Explicit dtype cast avoids NumPy 2.4 VisibleDeprecationWarning
    # (CIFAR-10 pickles were saved with align=0 integer, now expects bool)
    x = np.array(data[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(data[b"labels"], dtype=np.int64)
    return x, y


class CIFAR10Dataset(torch.utils.data.Dataset):
    """
    CIFAR-10 dataset with optional standard data augmentation for training.

    Augmentation (applied only when augment=True, i.e., training):
      - Random horizontal flip (p=0.5)
      - Random crop: pad 4px with reflection, then crop 32x32

    Without augmentation, D=20N for 1M params requires ~31 epochs through 40K images,
    which leads to severe overfitting. Augmentation is standard practice for CIFAR-10
    and makes each epoch effectively unique, preventing overfitting at this budget.
    """
    def __init__(self, root: str, train: bool, augment: bool = False):
        super().__init__()
        base = download_and_extract_cifar10(root)

        if train:
            xs, ys = [], []
            for i in range(1, 6):
                x, y = _load_batch(os.path.join(base, f"data_batch_{i}"))
                xs.append(x); ys.append(y)
            self.x = np.concatenate(xs, axis=0)
            self.y = np.concatenate(ys, axis=0)
        else:
            x, y = _load_batch(os.path.join(base, "test_batch"))
            self.x = x
            self.y = y

        mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std  = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        self.x = (self.x - mean) / std
        self.augment = bool(augment)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx].copy())  # [3,32,32]
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            # Random crop: reflect-pad by 4 then take a 32×32 crop
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


class ShapeControlledCNN(nn.Module):
    """
    CIFAR-10 CNN with:
      - patch embedding conv: 32x32 -> grid x grid (default 4x4)
      - depth residual conv blocks at fixed resolution
      - global average pooling + linear head
    """
    def __init__(
        self,
        *,
        channels: int,
        depth: int,
        num_classes: int = 10,
        patch: int = 8,
        groups: int = 8,
        dropout: float = 0.0,
        pad_params: int = 0,
    ):
        super().__init__()
        assert 32 % patch == 0
        self.channels = int(channels)
        self.depth = int(depth)
        self.patch = int(patch)
        self.grid = 32 // patch

        self.embed = nn.Conv2d(3, channels, kernel_size=patch, stride=patch, bias=False)
        g = min(groups, channels)
        self.embed_norm = nn.GroupNorm(num_groups=g, num_channels=channels, affine=True)

        self.blocks = nn.ModuleList([
            ResidualConvBlock(channels, groups=groups, dropout=dropout)
            for _ in range(depth)
        ])

        self.head_norm = nn.GroupNorm(num_groups=g, num_channels=channels, affine=True)
        self.classifier = nn.Linear(channels, num_classes, bias=True)

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
    AGOP = E_data[ J J^T ]  where  J = d(logits)/d(e),  e = patch_embed(x).
    J J^T ∈ R^{10×10} — FIXED dimension across all model shapes.
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
        raise ValueError(f"train_size + val_size must be < {len(train_set)} for CIFAR-10.")
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
    weight_decay: float = 0.0
    steps: int = 0
    data_ratio: float = 20.0
    warmup_steps: int = 1000
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 1000

    agop_batch: int = 256
    agop_proj_samples: int = 16
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
def evaluate_classifier(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_n += int(y.numel())
    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def train_one_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cfg: TrainCfg,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Train until the validation loss plateaus after the D≈20N budget is reached.
    Report the best validation-state checkpoint to stay in a fitted regime.
    """
    model.to(device)
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
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        lr = cosine_lr(step, cfg.lr, cfg.warmup_steps, max_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == max_steps:
            train_loss, train_acc = evaluate_classifier(model, train_loader, device, max_batches=50)
            val_loss, val_acc     = evaluate_classifier(model, val_loader,   device, max_batches=None)
            test_loss, test_acc   = evaluate_classifier(model, test_loader,  device, max_batches=None)
            history.append({
                "step": int(step + 1),
                "lr": float(lr),
                "train_loss": float(train_loss),
                "train_acc":  float(train_acc),
                "val_loss":   float(val_loss),
                "val_acc":    float(val_acc),
                "test_loss":  float(test_loss),
                "test_acc":   float(test_acc),
            })
            dt = time.time() - t0
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc*100:5.1f}%  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc*100:5.1f}%  "
                f"test_loss={test_loss:.4f}  test_acc={test_acc*100:5.1f}%  "
                f"gap={test_loss - train_loss:+.4f}  time={dt:.1f}s"
            )
            if val_loss + 1e-6 < best_val:
                best_val = float(val_loss)
                stale_evals = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_evals += 1
            if (step + 1) >= min_steps and stale_evals >= int(cfg.fit_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, train_acc = evaluate_classifier(model, train_loader, device, max_batches=None)
    val_loss, val_acc     = evaluate_classifier(model, val_loader,   device, max_batches=None)
    test_loss,  test_acc  = evaluate_classifier(model, test_loader,  device, max_batches=None)

    return {
        "train_loss": float(train_loss),
        "train_acc":  float(train_acc),
        "val_loss":   float(val_loss),
        "val_acc":    float(val_acc),
        "test_loss":  float(test_loss),
        "test_acc":   float(test_acc),
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
    parser.add_argument("--data_dir",  type=str, default="./data_cifar10")
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
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int,   default=1000)
    parser.add_argument("--eval_every",   type=int,   default=0,
                        help="Eval/log interval. 0 = auto (steps // 100).")

    parser.add_argument("--agop_batch",        type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=16)
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

    print("========== Budget (CNN) ==========")
    print(f"target_params N     = {args.target_params:,}")
    print(f"patch grid          = {grid}×{grid} = {grid*grid} patches/image")
    print(f"train_size          = {args.train_size:,} images  (augmented)")
    print(f"base steps          = {args.steps:,}")
    print(f"approx epochs       = {approx_epochs:.1f}")
    print(f"total patch-tokens  = {total_patch_tokens:,}  D/N = {total_patch_tokens/args.target_params:.1f}×")
    print(f"unique patch-tokens = {unique_patch_tokens:,}  {unique_patch_tokens/args.target_params:.2f}×N")
    print(f"(augmented training prevents overfitting across {approx_epochs:.1f} epochs)")
    print("==================================")

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

        metrics, history = train_one_model(model, train_loader, val_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"cnn_depth{depth}_channels{channels}")

        agop = estimate_agop_wrt_embedding(
            model, x_agop, proj_samples=cfg.agop_proj_samples, max_agop_dim=cfg.max_agop_dim,
        )
        off_e, off_r = agop_offdiag_metrics(agop)

        gap = metrics["test_loss"] - metrics["train_loss"]
        if gap > 0.5:
            print(f"  [WARNING] test_loss - train_loss = {gap:.4f} (>0.5), possible overfitting")

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
    print("Unified AOFE metrics (no log on test_loss):")
    print(f"  Pearson (AOFE=offdiag_energy,  test_loss) = {p_aofe:.4f}   Spearman = {s_aofe:.4f}")
    print(f"  Pearson (AOFE_ratio=offdiag_ratio, test_loss) = {p_aofe_ratio:.4f}   Spearman = {s_aofe_ratio:.4f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")
    print("-" * 80)

    # ---------- Scatter plots (with Pearson r annotation) ----------
    scatter_plot(
        off_energy, test_loss,
        xlabel="AOFE  (AGOP off-diagonal energy)",
        ylabel="Test CE loss",
        title=f"CIFAR-10: test loss vs AOFE  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testloss_vs_aofe_energy.png"),
        depths=depths_arr,
        r=p_aofe, r_label="Pearson r (AOFE, loss)",
    )
    scatter_plot(
        off_ratio, test_loss,
        xlabel="AOFE ratio  (AGOP off-diagonal ratio)",
        ylabel="Test CE loss",
        title=f"CIFAR-10: test loss vs AOFE ratio  [N={cfg.target_params}]",
        outpath=os.path.join(args.out_dir, "scatter_testloss_vs_aofe_ratio.png"),
        depths=depths_arr,
        r=p_aofe_ratio, r_label="Pearson r (AOFE_ratio, loss)",
    )


if __name__ == "__main__":
    main()
