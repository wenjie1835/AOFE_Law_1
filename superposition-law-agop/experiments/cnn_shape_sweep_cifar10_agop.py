
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_shape_sweep_cifar10_agop.py
===============================

Goal
----
Fixed-parameter CNN shape sweep on CIFAR-10 classification.
Tests the hypothesis: under fixed parameter count, different (depth, channels) shapes reach
similar loss, and the AGOP off-diagonal energy ("coupling") acts as the mediating metric.

Key design choices
------------------
1) AGOP = E[ J J^T ]  (output-space, NOT input-space J^T J)
      e = patch_embed(x)    shape [B, C, grid, grid]
      J = d(logits) / d(e)  shape [num_classes, C*grid*grid] = [10, D_in]
      J J^T                 shape [10, 10]  -- FIXED across all model shapes

   This is better than J^T J for cross-shape comparison because:
     - J^T J ∈ R^{D_in × D_in} changes size with channels  (D_in = C*grid²)
     - J J^T ∈ R^{10 × 10} is FIXED regardless of depth or channels
   Estimated via JVP random projections (forward-mode AD):
     u ~ N(0,I) in embedding space, Ju = J u via jvp, accumulate (Ju)^T(Ju)/B

2) Dataset: CIFAR-10 classification, using most of the 50K training images.

3) Chinchilla-inspired training budget:
     total_patch_tokens = steps × batch_size × patches_per_image ≈ 20 × target_params
     With patch=8 → grid=4 → 16 patches/image:
     10000 × 128 × 16 = 20.48M patch-tokens ≈ 20 × 1M params  (~Chinchilla ratio)

4) Shape sweep under fixed parameter count:
     sweep depth in [2,3,4,5,6,7,8,10,12,16] (10 non-extreme shapes), solve channels
     (multiple of groups=8, for GroupNorm) to match target params, then pad remainder.

No torchvision dependency
-------------------------
Uses a built-in CIFAR-10 downloader (no torchvision required).

Outputs
-------
Writes to: ./results_cnn_shape_sweep/
  - results.csv, results.npy
  - scatter plots (loss vs agop, acc vs agop)

Run (example)
-------------
python cnn_shape_sweep_cifar10_agop.py \
  --target_params 1000000 \
  --depth_list 2,3,4,5,6,7,8,10,12,16 \
  --train_size 40000 \
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
    Returns:
      offdiag_energy = ||AGOP - diag(AGOP)||_F^2
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
    """
    Returns path to extracted cifar-10-batches-py directory.
    """
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
        tar.extractall(path=data_dir)
    print("Extraction finished.")
    return extract_path


def _load_batch(batch_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(batch_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    x = data[b"data"]  # [N,3072], uint8
    y = np.array(data[b"labels"], dtype=np.int64)
    x = x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return x, y


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool):
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

        # Normalize with standard CIFAR-10 stats
        mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std  = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        self.x = (self.x - mean) / std

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx])  # [3,32,32]
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
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
        patch: int = 8,      # kernel=stride=patch => 32/patch = grid
        groups: int = 8,
        dropout: float = 0.0,
        pad_params: int = 0,
    ):
        super().__init__()
        assert 32 % patch == 0, "For CIFAR-10 (32x32), patch must divide 32."
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
        e = F.gelu(e)
        return e

    def forward_from_embedding(self, e: torch.Tensor) -> torch.Tensor:
        x = e
        for blk in self.blocks:
            x = blk(x)
        x = self.head_norm(x)
        x = F.gelu(x)
        x = x.mean(dim=(2, 3))  # [B, C]
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
    x = torch.cat(xs, dim=0)[:batch_size].to(device)
    return x


def estimate_agop_wrt_embedding(
    model: ShapeControlledCNN,
    x: torch.Tensor,
    *,
    proj_samples: int = 16,
    max_agop_dim: int = 8192,  # kept for API compat; J J^T is [10,10] so always satisfied
) -> torch.Tensor:
    """
    Compute AGOP = E_data[ J J^T ] where J = d(logits)/d(e), e = patch_embed(x).

    J J^T lives in OUTPUT class space: shape [num_classes, num_classes] = [10, 10].
    This dimension is FIXED across all model shapes (independent of channels/depth),
    making cross-shape AGOP comparison directly valid without any channel constraints.

    Estimated via JVP (forward-mode AD) random projections:
        u ~ N(0, I) in embedding space [B, C, grid, grid]
        Ju = J u  (shape [B, num_classes])  via torch.autograd.functional.jvp
        AGOP ≈ (1/K) sum_k (1/B) * Ju^T Ju  ∈ R^{num_classes × num_classes}

    NOTE: requires PyTorch >= 1.9 for torch.autograd.functional.jvp.
    """
    device = x.device
    model.eval()

    with torch.no_grad():
        e0 = model.embed_forward(x)  # [B, C, grid, grid]
        e = e0.detach()

    B = e.shape[0]
    logits_ref = model.forward_from_embedding(e)
    D_out = logits_ref.shape[-1]  # num_classes = 10

    agop = torch.zeros((D_out, D_out), device=device, dtype=torch.float32)

    def fwd(e_in: torch.Tensor) -> torch.Tensor:
        # maps [B, C, grid, grid] → [B, num_classes]
        return model.forward_from_embedding(e_in)

    for _ in range(int(proj_samples)):
        u = torch.randn_like(e)  # random tangent in embedding space [B, C, grid, grid]
        # JVP: Ju[b] = J_b u_b,  shape [B, num_classes]
        _, Ju = torch.autograd.functional.jvp(fwd, (e,), (u,), create_graph=False, strict=False)
        Ju = Ju.float()
        Ju = torch.nan_to_num(Ju, nan=0.0, posinf=0.0, neginf=0.0)
        agop = agop + (Ju.T @ Ju) / float(B)

    agop = agop / float(proj_samples)
    agop = symmetrize_(agop).detach()
    return agop


# -----------------------
# Data loaders
# -----------------------

def make_cifar10_loaders(
    *,
    data_dir: str,
    train_size: int,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_set = CIFAR10Dataset(root=data_dir, train=True)
    test_set  = CIFAR10Dataset(root=data_dir, train=False)

    # Fixed subset indices
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(train_set), generator=g).tolist()
    idx = perm[:int(train_size)]
    train_subset = torch.utils.data.Subset(train_set, idx)

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
    return train_loader, test_loader


# -----------------------
# Training
# -----------------------

@dataclass
class TrainCfg:
    # Optimization
    lr: float = 3e-3
    weight_decay: float = 0.0
    # Chinchilla-inspired: base_steps × batch_size × patches_per_image ≈ data_ratio × target_params
    steps: int = 0
    data_ratio: float = 20.0
    max_steps_multiplier: float = 4.0
    warmup_steps: int = 1000
    batch_size: int = 128
    grad_clip: float = 1.0
    fit_patience_evals: int = 5
    fit_rel_improve_tol: float = 1e-3

    # Eval
    eval_every: int = 1000
    agop_batch: int = 256
    agop_proj_samples: int = 16     # was 8; more JVP samples for stable J J^T estimate
    max_agop_dim: int = 8192        # J J^T is [10,10]; this limit is effectively unused

    # Data — use most of CIFAR-10 (50K training images available)
    train_size: int = 40000         # was 2000; 40K images avoids extreme memorization regime

    # Model sweep — 10 non-extreme shapes spanning depth 2..16
    target_params: int = 1_000_000  # was 2_000_000
    patch: int = 8
    dropout: float = 0.0

    # Sweep — non-extreme: min depth=2, min channels=32 enforced in find_channels_for_target_params
    depth_list: List[int] = None

    # misc
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
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_train_loss = float("inf")
    best_test_loss = float("inf")
    best_test_acc = 0.0
    best_eval_idx = -1

    max_steps = max(int(cfg.steps), int(math.ceil(cfg.steps * cfg.max_steps_multiplier)))

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

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == int(cfg.steps) or (step + 1) == max_steps:
            train_loss, train_acc = evaluate_classifier(model, train_loader, device, max_batches=50)
            test_loss, test_acc = evaluate_classifier(model, test_loader, device, max_batches=None)
            history.append({
                "step": int(step + 1),
                "lr": float(lr),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            })
            if train_loss < best_train_loss:
                best_train_loss = float(train_loss)
                best_state = copy.deepcopy(model.state_dict())
            best_test_loss = min(best_test_loss, test_loss)
            best_test_acc = max(best_test_acc, test_acc)
            dt = time.time() - t0
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.1f}%  "
                f"test_loss={test_loss:.4f} test_acc={test_acc*100:5.1f}%  "
                f"best_test_acc={best_test_acc*100:5.1f}%  "
                f"time={dt:.1f}s"
            )
            if train_loss <= best_train_loss * (1.0 + cfg.fit_rel_improve_tol):
                best_eval_idx = len(history) - 1
            if (step + 1) >= int(cfg.steps) and (len(history) - 1 - best_eval_idx) >= int(cfg.fit_patience_evals):
                print(f"Early stop at step {step+1}: train loss has plateaued after base budget.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, train_acc = evaluate_classifier(model, train_loader, device, max_batches=None)
    test_loss, test_acc = evaluate_classifier(model, test_loader, device, max_batches=None)

    return {
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "best_train_loss": float(best_train_loss),
        "best_test_loss": float(best_test_loss),
        "best_test_acc": float(best_test_acc),
        "steps_run": int(history[-1]["step"]) if history else 0,
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
    tmp = ShapeControlledCNN(
        channels=channels,
        depth=depth,
        patch=patch,
        dropout=dropout,
        pad_params=0,
    )
    active = count_params(tmp)
    pad = 0
    if pad_to_target:
        if active > target_params:
            raise ValueError(
                f"Active params {active} exceed target {target_params} for depth={depth}, channels={channels}."
            )
        pad = int(target_params - active)
    model = ShapeControlledCNN(
        channels=channels,
        depth=depth,
        patch=patch,
        dropout=dropout,
        pad_params=pad,
    )
    return model


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
    """
    Returns (best_channels, active_params_without_padding).

    channels is always a multiple of `groups` (required for GroupNorm: num_groups must
    divide num_channels).

    Note: the old AGOP-based channels_max constraint (C*grid² <= max_agop_dim) has been
    removed because AGOP is now computed in output space [num_classes, num_classes] = [10,10],
    which is independent of channels. The check is done once at the start of main().
    """
    def to_valid(C: int) -> int:
        return max(groups, (C // groups) * groups)

    channels_min = to_valid(max(channels_min, groups))
    channels_max = to_valid(channels_max)

    if channels_min > channels_max:
        raise ValueError(
            f"channels_min > channels_max after groups constraint ({channels_min} > {channels_max})."
        )

    def active_params(C: int) -> int:
        m = ShapeControlledCNN(channels=C, depth=depth, patch=patch, dropout=dropout, pad_params=0)
        return count_params(m)

    # Feasibility
    if active_params(channels_min) > target_params:
        raise ValueError(
            f"Even channels={channels_min} exceeds target_params={target_params} at depth={depth}. "
            f"Reduce depth or increase target_params."
        )
    if active_params(channels_max) <= target_params:
        return channels_max, active_params(channels_max)

    # Binary search on discrete grid of multiples of groups
    lo, hi = channels_min, channels_max
    best_c = lo
    best_a = active_params(lo)
    while lo <= hi:
        mid = to_valid((lo + hi) // 2)
        a = active_params(mid)
        if a <= target_params:
            best_c, best_a = mid, a
            lo = mid + groups
        else:
            hi = mid - groups

    # Check neighbors for closer absolute match
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

def scatter_plot(x, y, xlabel, ylabel, title, outpath):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_curve(history: List[Dict[str, float]], out_dir: str, stem: str, train_key: str, test_key: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    png_path = os.path.join(out_dir, f"{stem}.png")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    steps = [row["step"] for row in history]
    train_vals = [row[train_key] for row in history]
    test_vals = [row[test_key] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_vals, label=train_key)
    plt.plot(steps, test_vals, label=test_key)
    plt.xlabel("step")
    plt.ylabel("loss")
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
    parser.add_argument("--data_dir", type=str, default="./data_cifar10")
    parser.add_argument("--out_dir", type=str, default="./results_cnn_shape_sweep")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--target_params", type=int, default=1_000_000)
    # 10 shapes: depth 2..16, all non-extreme (min depth=2, min channels=32)
    # Expected (depth→channels, target_params=1M, patch=8): approx
    #   2→160, 3→136, 4→120, 5→104, 6→96, 7→88, 8→80, 10→72, 12→64, 16→56
    # all multiples of groups=8, padding ≤25%
    parser.add_argument("--depth_list", type=str, default="2,3,4,5,6,7,8,9,10,12")
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Use most of CIFAR-10 (50K available); 40K × 16 patches = 640K patches
    parser.add_argument("--train_size", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=128)
    # Chinchilla: 10000 × 128 × 16 patches = 20.48M ≈ 20 × 1M params
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--data_ratio", type=float, default=20.0)
    parser.add_argument("--max_steps_multiplier", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--fit_patience_evals", type=int, default=5)
    parser.add_argument("--fit_rel_improve_tol", type=float, default=1e-3)

    # AGOP: J J^T in output space [num_classes, num_classes] = [10, 10]
    parser.add_argument("--agop_batch", type=int, default=256)
    parser.add_argument("--agop_proj_samples", type=int, default=16)
    parser.add_argument("--max_agop_dim", type=int, default=8192)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    set_global_seed(args.seed)

    depths = [int(x) for x in args.depth_list.split(",") if x.strip() != ""]
    if len(depths) != 10:
        raise ValueError(f"depth_list must contain exactly 10 shapes, got {len(depths)}.")

    # Validate AGOP output dimension (num_classes=10, fixed regardless of model shape)
    grid = 32 // args.patch
    if args.steps <= 0:
        args.steps = int(math.ceil(args.data_ratio * args.target_params / float(args.batch_size * grid * grid)))
    print(f"AGOP output dim: num_classes = 10  (fixed across all model shapes)")
    total_patch_tokens = args.steps * args.batch_size * (grid * grid)
    print(
        f"Base budget: steps={args.steps} × batch={args.batch_size} × patches/image={grid*grid}"
        f" = {total_patch_tokens:,} patch-tokens  ≈  {total_patch_tokens / args.target_params:.1f} × N"
    )
    unique_patch_tokens = args.train_size * (grid * grid)
    print(
        f"Unique dataset tokens: train_size={args.train_size} × patches/image={grid*grid}"
        f" = {unique_patch_tokens:,}  ≈  {unique_patch_tokens / args.target_params:.1f} × N"
    )
    if unique_patch_tokens < args.data_ratio * args.target_params:
        print("Warning: CIFAR-10 unique patch tokens are below D=20N; the script enforces the training-compute ratio and reports the unique-data shortfall.")

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
        agop_batch=args.agop_batch,
        agop_proj_samples=args.agop_proj_samples,
        max_agop_dim=args.max_agop_dim,
        train_size=args.train_size,
        target_params=args.target_params,
        patch=args.patch,
        dropout=args.dropout,
        depth_list=depths,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    curve_dir = os.path.join(args.out_dir, "curves")

    train_loader, test_loader = make_cifar10_loaders(
        data_dir=args.data_dir,
        train_size=cfg.train_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )

    results: List[Dict[str, float]] = []

    # Fixed AGOP batch from test set
    x_agop = _get_fixed_agop_batch(test_loader, device=device, batch_size=cfg.agop_batch)

    for depth in cfg.depth_list:
        channels, active = find_channels_for_target_params(
            depth=depth,
            target_params=cfg.target_params,
            patch=cfg.patch,
            dropout=cfg.dropout,
        )

        model = build_cnn_model(
            depth=depth,
            channels=channels,
            target_params=cfg.target_params,
            patch=cfg.patch,
            dropout=cfg.dropout,
            pad_to_target=True,
        )
        total = count_params(model)
        pad = total - active

        print("\n" + "=" * 80)
        print(
            f"[CNN] depth={depth:3d}  channels={channels:4d}  active_params={active}  "
            f"pad={pad}  total={total}  agop_dim(num_classes)=10"
        )
        print("=" * 80)

        metrics, history = train_one_model(model, train_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"cnn_depth{depth}_channels{channels}", "train_loss", "test_loss")

        agop = estimate_agop_wrt_embedding(
            model,
            x_agop,
            proj_samples=cfg.agop_proj_samples,
            max_agop_dim=cfg.max_agop_dim,
        )
        off_e, off_r = agop_offdiag_metrics(agop)

        row: Dict[str, float] = dict(metrics)
        row.update({
            "depth": int(depth),
            "channels": int(channels),
            "patch": int(cfg.patch),
            "grid": int(32 // cfg.patch),
            "active_params": int(active),
            "pad_params": int(pad),
            "total_params": int(total),
            "padding_ratio": float(pad / max(1, total)),
            "agop_dim": int(agop.shape[0]),
            "agop_offdiag_energy": float(off_e),
            "agop_offdiag_ratio": float(off_r),
        })
        results.append(row)

        del model, agop
        torch.cuda.empty_cache()

    # Save results
    csv_path = os.path.join(args.out_dir, "results.csv")
    npy_path = os.path.join(args.out_dir, "results.npy")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    np.save(npy_path, results, allow_pickle=True)

    # Correlations + plots
    test_loss = np.array([r["test_loss"] for r in results])
    test_acc = np.array([r["test_acc"] for r in results])
    off_ratio = np.array([r["agop_offdiag_ratio"] for r in results])

    p_loss = pearson_corr(off_ratio, test_loss)
    s_loss = spearman_corr(off_ratio, test_loss)
    p_acc = pearson_corr(off_ratio, test_acc)
    s_acc = spearman_corr(off_ratio, test_acc)

    print("\n" + "-" * 80)
    print(f"Pearson(offdiag_ratio, test_loss)   = {p_loss:.4f}")
    print(f"Spearman(offdiag_ratio, test_loss)  = {s_loss:.4f}")
    print(f"Pearson(offdiag_ratio, test_acc)    = {p_acc:.4f}")
    print(f"Spearman(offdiag_ratio, test_acc)   = {s_acc:.4f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")
    print("-" * 80)

    scatter_plot(
        off_ratio,
        test_loss,
        xlabel="AGOP off-diagonal ratio (wrt embedding)",
        ylabel="Test CE loss",
        title=f"CIFAR-10: test loss vs AGOP offdiag ratio (fixed params={cfg.target_params})",
        outpath=os.path.join(args.out_dir, "scatter_testloss_vs_agop.png"),
    )
    scatter_plot(
        off_ratio,
        test_acc,
        xlabel="AGOP off-diagonal ratio (wrt embedding)",
        ylabel="Test accuracy",
        title=f"CIFAR-10: test acc vs AGOP offdiag ratio (fixed params={cfg.target_params})",
        outpath=os.path.join(args.out_dir, "scatter_testacc_vs_agop.png"),
    )


if __name__ == "__main__":
    main()
