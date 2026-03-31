
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp_shape_sweep_supervised_pde_agop.py
=====================================

Goal
----
用“监督版 PDE（解析解）”测试你在 Transformer 里提出的假设：

- 固定总参数量 N，只改变 MLP 的形状（深度 depth / 宽度 width）
- 训练预算按 scaling law 口径控制数据量：D ≈ 20 × N
- 计算 AGOP 的非对角线能量（off-diagonal energy）作为“耦合程度”，观察其与 loss 的相关性
- 验证“深/宽可通过耦合程度互补”的猜想（不同形状在相同 N、相同 D 下能否达到相近 loss）

Task (Supervised PDE)
---------------------
选择 1D 热方程 (Heat Equation) 的解析解来生成监督数据，避免 PINN 的训练不稳定：

    u_t = α u_xx ,  x∈[0,1], t∈[0, t_max]
    u(0,t)=u(1,t)=0 (Dirichlet)

初值用有限 Fourier-sine 展开：
    u(x,0) = Σ_{k=1..K} a_k sin(kπx)

解析解：
    u(x,t) = Σ a_k exp(-α (kπ)^2 t) sin(kπx)

我们学习“解算子”的一个切片：
    input  z = [a_1,...,a_K, t] ∈ R^{K+1}
    output y = [u(x_1,t),...,u(x_M,t)] ∈ R^M   (固定的空间网格 x_grid)

这样输出维度 M 固定，可以像你 Transformer 代码一样做“输出空间 AGOP”：
    J = d(y)/d(z)   ∈ R^{M×(K+1)}
    AGOP = E_data[ J J^T ] ∈ R^{M×M}   —— 尺寸固定，跨形状可比
用 JVP 随机投影近似（与 Transformer 脚本同套路）。

Data budget ("tokens")
----------------------
把每个标量监督目标当作一个 token（类比 next-token 里的每个位置）：
    tokens_per_step = batch_size × M
    D = steps × batch_size × M  ≈ 20 × N

默认会自动计算 steps 以满足 D≈20N（可在命令行覆盖）。

Shapes
------
默认设计 10 个“不过于极端”的 MLP 形状：
    depth_list = 3..12  (共 10 个深度)
每个 depth 用二分搜索寻找 width（width 为 16 的倍数）使 active_params <= target_params，
剩余参数用 pad_params 补齐到恰好 target_params。

Outputs
-------
写入目录 ./results_mlp_pde_shape_sweep/
  - results.csv / results.npy
  - scatter: test_mse vs coupling (offdiag energy / offdiag ratio)

Run
---
python mlp_shape_sweep_supervised_pde_agop.py --device cuda

"""

from __future__ import annotations

import copy
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
    """Functional activation dispatch — avoids sharing stateful nn.Module instances."""
    if activation == "gelu":  return F.gelu(x)
    if activation == "silu":  return F.silu(x)
    if activation == "tanh":  return torch.tanh(x)
    if activation == "relu":  return F.relu(x)
    raise ValueError(f"Unknown activation: {activation}")


class _ResBlock(nn.Module):
    """
    Pre-LN residual block: x → x + fc(act(ln(x)))

    Why Pre-LN + residual?
    - The skip connection provides a direct gradient path from output to early layers,
      preventing vanishing gradients in deep MLPs.
    - LayerNorm before the non-linearity keeps activations well-scaled at every depth.
    - Each block only needs to learn the *residual correction*, which is a much easier
      optimization target than learning the full transformation from scratch.
    """
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
    返回：
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
    每个样本：
      z = [a_1..a_K, t]   (K 个 Fourier 系数 + 时间)
      y = u(x_grid, t)   (M 个固定空间网格点上的解)

    为了可复现，coeffs 和 times 在 __init__ 一次性采样固定下来。
    y 在 __getitem__ 根据解析解即时计算（很快），不需要预生成大表。
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
        self.size = int(size)
        self.modes = int(modes)
        self.out_grid = int(out_grid)
        self.alpha = float(alpha)
        self.t_max = float(t_max)
        self.dtype = dtype

        # 固定空间网格
        if include_endpoints:
            x = torch.linspace(0.0, 1.0, self.out_grid, dtype=dtype)
        else:
            # 去掉端点，避免永远为 0 的 target（可选）
            x = torch.linspace(0.0, 1.0, self.out_grid + 2, dtype=dtype)[1:-1]
        self.x_grid = x  # [M]

        # 预计算 basis: sin(kπx), 以及 lambda_k = α (kπ)^2
        k = torch.arange(1, self.modes + 1, dtype=dtype).unsqueeze(1)  # [K,1]
        x_row = self.x_grid.unsqueeze(0)                               # [1,M]
        self.basis = torch.sin(math.pi * k * x_row)                    # [K,M]
        self.lam = self.alpha * (math.pi * k.squeeze(1)) ** 2          # [K]

        # 固定采样 coeffs 与 times（可复现）
        g = torch.Generator()
        g.manual_seed(int(seed))

        # a_k ~ N(0, (scale/k^decay)^2)
        kk = torch.arange(1, self.modes + 1, dtype=dtype)
        std = (coeff_scale / (kk ** float(coeff_std_decay))).to(dtype)  # [K]
        coeffs = torch.randn((self.size, self.modes), generator=g, dtype=dtype) * std.unsqueeze(0)
        times = torch.rand((self.size, 1), generator=g, dtype=dtype) * self.t_max

        self.coeffs = coeffs  # [N,K]
        self.times = times    # [N,1]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.coeffs[idx]        # [K]
        t = self.times[idx].squeeze(0)  # scalar

        # u(x,t) = Σ a_k exp(-λ_k t) sin(kπx)
        decay = torch.exp(-self.lam * t)   # [K]
        scaled = a * decay                  # [K]
        y = scaled @ self.basis             # [M]

        z = torch.cat([a, t.view(1)], dim=0)  # [K+1]
        return z, y


# -----------------------
# Model: MLP + pad_params
# -----------------------

class TinyMLP(nn.Module):
    """
    Pre-LN Residual MLP (use_residual=True, default) or plain MLP (use_residual=False).

    Residual architecture (default — fixes vanishing gradients for depth 8+):
        entry:   Linear(in_dim → width) + act
        hidden:  (depth-1) × _ResBlock  [x = x + fc(act(ln(x)))]
        head:    Linear(width → out_dim)

    Plain architecture (use_residual=False — original behavior):
        Linear(in_dim → width) + act
        (depth-1) × [Linear(width → width) + act]
        Linear(width → out_dim)

    pad_params: unused parameters to make total params exactly equal across shapes.

    Parameter overhead of residual vs plain:
        +2*width*(depth-1) for LayerNorm in residual blocks (~0.3% of 1M for typical configs).
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
        use_residual: bool = True,
    ):
        super().__init__()
        assert depth >= 1, "depth should be >= 1 (number of hidden layers)."
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.depth = int(depth)
        self.width = int(width)
        self.dropout = float(dropout)
        self.activation = str(activation)
        self.use_residual = bool(use_residual)

        # Validate activation
        _apply_act(torch.zeros(1), activation)

        if self.use_residual:
            # Pre-LN Residual MLP: direct gradient paths prevent vanishing gradients
            self.entry = nn.Linear(in_dim, width, bias=True)
            self.res_blocks = nn.ModuleList([
                _ResBlock(width, activation, dropout) for _ in range(depth - 1)
            ])
            self.head = nn.Linear(width, out_dim, bias=True)
            self.net = None
        else:
            # Plain MLP (original)
            if activation == "gelu":   act: nn.Module = nn.GELU()
            elif activation == "silu": act = nn.SiLU()
            elif activation == "tanh": act = nn.Tanh()
            elif activation == "relu": act = nn.ReLU()
            else: raise ValueError(f"Unknown activation: {activation}")

            layers: List[nn.Module] = []
            layers.append(nn.Linear(in_dim, width, bias=True))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(depth - 1):
                layers.append(nn.Linear(width, width, bias=True))
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(width, out_dim, bias=True))
            self.net = nn.Sequential(*layers)
            self.entry = None
            self.res_blocks = None
            self.head = None

        self.pad_params = int(pad_params)
        if self.pad_params > 0:
            self._pad = nn.Parameter(torch.zeros(self.pad_params, dtype=torch.float32), requires_grad=True)
        else:
            self._pad = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            x = _apply_act(self.entry(z), self.activation)
            for blk in self.res_blocks:
                x = blk(x)
            return self.head(x)
        else:
            return self.net(z)


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
    AGOP = E_data[ J J^T ],  J = d(y)/d(z)

    y ∈ R^M, z ∈ R^{d_in}
    => J J^T ∈ R^{M×M}  —— FIXED across model shapes

    用 forward-mode JVP 随机投影近似（与 Transformer 脚本一致）：
      u ~ N(0,I) in input space
      Ju = J u  ∈ R^M
      AGOP ≈ (1/K) Σ (1/B) (Ju)^T (Ju)
    """
    device = z.device
    model.eval()
    B, d_in = z.shape
    with torch.no_grad():
        y0 = model(z)
        M = y0.shape[1]

    z0 = z.detach()
    agop = torch.zeros((M, M), device=device, dtype=torch.float32)

    def fwd(z_in: torch.Tensor) -> torch.Tensor:
        # [B,d_in] -> [B,M]
        return model(z_in)

    for _ in range(int(proj_samples)):
        u = torch.randn_like(z0)
        _, Ju = torch.autograd.functional.jvp(fwd, (z0,), (u,), create_graph=False, strict=False)
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
    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    steps: int = 0
    data_ratio: float = 20.0
    max_steps_multiplier: float = 6.0
    warmup_steps: int = 0       # 0 => auto (10% of steps)
    batch_size: int = 128
    grad_clip: float = 1.0
    eval_every: int = 1000      # was 200; reduced to limit print overhead at 20k steps
    fit_patience_evals: int = 5
    fit_rel_improve_tol: float = 1e-3

    # PDE data
    modes: int = 32
    out_grid: int = 128
    alpha: float = 0.1
    t_max: float = 1.0
    train_size: int = 0
    test_size: int = 1000
    coeff_std_decay: float = 1.0
    coeff_scale: float = 1.0
    include_endpoints: bool = True

    # shape sweep
    target_params: int = 1_000_000
    depth_list: List[int] = None
    width_multiple: int = 16
    min_width: int = 192
    max_width: int = 2048
    activation: str = "gelu"
    dropout: float = 0.0
    # use_residual=True (default): Pre-LN residual blocks fix vanishing gradients
    # for deep MLPs (depth >= 8) while adding negligible parameter overhead (~0.3%).
    use_residual: bool = True

    # agop
    agop_batch: int = 256
    agop_proj_samples: int = 16

    # misc
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
    """
    返回每个标量输出的平均 MSE（类比 LM 的 per-token NLL）
    """
    model.eval()
    sse = 0.0
    n = 0
    for i, (z, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        z = z.to(device)
        y = y.to(device)
        pred = model(z)
        err = (pred - y).float()
        sse += float((err * err).sum().item())
        n += int(y.numel())
    return sse / max(1, n)


def train_one_model(
    model: TinyMLP,
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
    best_train_mse = float("inf")
    best_test_mse = float("inf")
    best_state: Optional[Dict] = None
    best_eval_idx = -1
    max_steps = max(int(cfg.steps), int(math.ceil(cfg.steps * cfg.max_steps_multiplier)))

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

        if (step + 1) % int(cfg.eval_every) == 0 or (step + 1) == int(cfg.steps) or (step + 1) == max_steps:
            train_mse = evaluate_mse(model, train_loader, device, max_batches=50)
            test_mse = evaluate_mse(model, test_loader, device, max_batches=None)
            history.append({
                "step": int(step + 1),
                "lr": float(lr),
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
            })
            if train_mse < best_train_mse:
                best_train_mse = float(train_mse)
                best_state = copy.deepcopy(model.state_dict())
            if test_mse < best_test_mse:
                best_test_mse = test_mse
            dt = time.time() - t0
            print(
                f"step {step+1:6d}/{max_steps}  lr={lr:.3e}  "
                f"train_mse={train_mse:.6e}  test_mse={test_mse:.6e}  "
                f"best_test_mse={best_test_mse:.6e}  "
                f"time={dt:.1f}s"
            )
            if train_mse <= best_train_mse * (1.0 + cfg.fit_rel_improve_tol):
                best_eval_idx = len(history) - 1
            if (step + 1) >= int(cfg.steps) and (len(history) - 1 - best_eval_idx) >= int(cfg.fit_patience_evals):
                print(f"Early stop at step {step+1}: train MSE has plateaued after base budget.")
                break

    # Restore best train-fit checkpoint so AGOP and reported metrics reflect a fitted state
    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse = evaluate_mse(model, train_loader, device, max_batches=None)
    test_mse = evaluate_mse(model, test_loader, device, max_batches=None)
    return {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "best_train_mse": float(best_train_mse),
        "best_test_mse": float(best_test_mse),
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
    in_dim = cfg.modes + 1
    out_dim = cfg.out_grid
    tmp = TinyMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        depth=depth,
        width=width,
        activation=cfg.activation,
        dropout=cfg.dropout,
        pad_params=0,
        use_residual=cfg.use_residual,
    )
    active = count_params(tmp)
    pad = 0
    if pad_to_target:
        if active > cfg.target_params:
            raise ValueError(
                f"Active params {active} exceed target {cfg.target_params} "
                f"for depth={depth}, width={width}."
            )
        pad = int(cfg.target_params - active)
    model = TinyMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        depth=depth,
        width=width,
        activation=cfg.activation,
        dropout=cfg.dropout,
        pad_params=pad,
        use_residual=cfg.use_residual,
    )
    return model


def find_width_for_target_params(
    *,
    depth: int,
    cfg: TrainCfg,
) -> Tuple[int, int]:
    """
    返回 (width, active_params_without_padding)

    约束：
      - width 是 cfg.width_multiple 的倍数
      - width ∈ [cfg.min_width, cfg.max_width]
      - active_params(width) <= target_params，并尽量接近 target_params（padding 少）
    """
    mul = int(cfg.width_multiple)

    def to_valid(w: int) -> int:
        return max(mul, (w // mul) * mul)

    w_min = to_valid(max(cfg.min_width, mul))
    w_max = to_valid(max(cfg.max_width, mul))

    if w_min > w_max:
        raise ValueError(f"min_width > max_width after constraints ({w_min} > {w_max}).")

    in_dim = cfg.modes + 1
    out_dim = cfg.out_grid

    def active_params(w: int) -> int:
        m = TinyMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            depth=depth,
            width=w,
            activation=cfg.activation,
            dropout=cfg.dropout,
            pad_params=0,
            use_residual=cfg.use_residual,
        )
        return count_params(m)

    # Feasibility
    if active_params(w_min) > cfg.target_params:
        raise ValueError(
            f"Even width={w_min} exceeds target_params={cfg.target_params} at depth={depth}. "
            f"Reduce depth, reduce min_width, or increase target_params."
        )
    if active_params(w_max) <= cfg.target_params:
        return w_max, active_params(w_max)

    # Binary search on discrete grid
    lo, hi = w_min, w_max
    best_w = w_min
    best_a = active_params(best_w)

    while lo <= hi:
        mid = to_valid((lo + hi) // 2)
        a = active_params(mid)
        if a <= cfg.target_params:
            best_w, best_a = mid, a
            lo = mid + mul
        else:
            hi = mid - mul

    # Check neighbors for closer match
    candidates = []
    for w in [max(w_min, best_w - mul), best_w, min(w_max, best_w + mul)]:
        a = active_params(w)
        if a <= cfg.target_params:
            candidates.append((abs(cfg.target_params - a), w, a))
    candidates.sort(key=lambda t: t[0])
    _, w, a = candidates[0]
    return int(w), int(a)


# -----------------------
# Main
# -----------------------

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def save_curve(history: List[Dict[str, float]], out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    png_path = os.path.join(out_dir, f"{stem}.png")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    steps = [row["step"] for row in history]
    train_vals = [row["train_mse"] for row in history]
    test_vals = [row["test_mse"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_vals, label="train_mse")
    plt.plot(steps, test_vals, label="test_mse")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.title(stem)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="./results_mlp_pde_shape_sweep")

    # params & shapes
    p.add_argument("--target_params", type=int, default=1_000_000)
    p.add_argument("--depth_list", type=str, default="3,4,5,6,7,8,9,10,11,12")
    p.add_argument("--min_width", type=int, default=192)
    p.add_argument("--max_width", type=int, default=2048)
    p.add_argument("--width_multiple", type=int, default=16)

    # data / pde
    p.add_argument("--modes", type=int, default=32)
    p.add_argument("--out_grid", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--t_max", type=float, default=1.0)
    p.add_argument("--train_size", type=int, default=0,
                   help="Number of training samples. Set 0 to auto-match unique_targets ≈ data_ratio×N.")
    p.add_argument("--test_size", type=int, default=1000)
    p.add_argument("--coeff_std_decay", type=float, default=1.0)
    p.add_argument("--coeff_scale", type=float, default=1.0)
    p.add_argument("--include_endpoints", action="store_true")
    p.add_argument("--no_endpoints", action="store_true")

    # optimization / budget
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=0,
                   help="Base training steps per model. Set 0 to auto-compute from D≈20N; training can extend until plateau.")
    p.add_argument("--data_ratio", type=float, default=20.0)
    p.add_argument("--max_steps_multiplier", type=float, default=6.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--fit_patience_evals", type=int, default=5)
    p.add_argument("--fit_rel_improve_tol", type=float, default=1e-3)

    # model
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu", "tanh", "relu"])
    p.add_argument("--dropout", type=float, default=0.0)
    # --no_residual: revert to plain MLP (for ablation; not recommended for depth >= 8)
    p.add_argument("--no_residual", action="store_true",
                   help="Disable Pre-LN residual blocks (plain MLP; may cause vanishing gradients for depth>=8)")

    # agop
    p.add_argument("--agop_batch", type=int, default=256)
    p.add_argument("--agop_proj_samples", type=int, default=16)

    # misc
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    cfg = TrainCfg()
    cfg.target_params = int(args.target_params)
    cfg.depth_list = parse_int_list(args.depth_list)
    cfg.min_width = int(args.min_width)
    cfg.max_width = int(args.max_width)
    cfg.width_multiple = int(args.width_multiple)

    cfg.modes = int(args.modes)
    cfg.out_grid = int(args.out_grid)
    cfg.alpha = float(args.alpha)
    cfg.t_max = float(args.t_max)
    cfg.train_size = int(args.train_size)
    cfg.test_size = int(args.test_size)
    cfg.coeff_std_decay = float(args.coeff_std_decay)
    cfg.coeff_scale = float(args.coeff_scale)
    cfg.include_endpoints = True
    if args.no_endpoints:
        cfg.include_endpoints = False
    if args.include_endpoints:
        cfg.include_endpoints = True

    cfg.lr = float(args.lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.steps = int(args.steps)
    cfg.data_ratio = float(args.data_ratio)
    cfg.max_steps_multiplier = float(args.max_steps_multiplier)
    cfg.batch_size = int(args.batch_size)
    cfg.warmup_steps = int(args.warmup_steps)
    cfg.eval_every = int(args.eval_every)
    cfg.grad_clip = float(args.grad_clip)
    cfg.fit_patience_evals = int(args.fit_patience_evals)
    cfg.fit_rel_improve_tol = float(args.fit_rel_improve_tol)

    cfg.activation = str(args.activation)
    cfg.dropout = float(args.dropout)
    cfg.use_residual = not args.no_residual

    cfg.agop_batch = int(args.agop_batch)
    cfg.agop_proj_samples = int(args.agop_proj_samples)

    cfg.seed = int(args.seed)

    set_global_seed(cfg.seed)
    device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    curve_dir = os.path.join(args.out_dir, "curves")
    if len(cfg.depth_list) != 10:
        raise ValueError(f"depth_list must contain exactly 10 shapes, got {len(cfg.depth_list)}.")

    if cfg.train_size <= 0:
        cfg.train_size = int(math.ceil(cfg.data_ratio * cfg.target_params / float(cfg.out_grid)))

    # Auto steps to match D ≈ 20N
    tokens_per_step = int(cfg.batch_size) * int(cfg.out_grid)
    if cfg.steps <= 0:
        D_target = cfg.data_ratio * int(cfg.target_params)
        cfg.steps = int(math.ceil(D_target / float(tokens_per_step)))
    if cfg.warmup_steps <= 0:
        cfg.warmup_steps = max(50, int(0.1 * cfg.steps))

    D_actual = int(cfg.steps) * int(cfg.batch_size) * int(cfg.out_grid)
    ratio = float(D_actual) / float(cfg.target_params)
    unique_tokens = int(cfg.train_size) * int(cfg.out_grid)
    approx_epochs = float(cfg.steps) * float(cfg.batch_size) / float(cfg.train_size)
    arch_str = "Pre-LN Residual MLP (use_residual=True)" if cfg.use_residual else "Plain MLP (use_residual=False)"
    print("========== Budget ==========")
    print(f"architecture        = {arch_str}")
    print(f"target_params N     = {cfg.target_params:,}")
    print(f"train_size          = {cfg.train_size:,}")
    print(f"tokens_per_step     = batch_size×out_grid = {cfg.batch_size}×{cfg.out_grid} = {tokens_per_step:,}")
    print(f"base steps          = {cfg.steps:,}")
    print(f"approx epochs       = steps×batch/train_size = {approx_epochs:.0f}")
    print(f"base total_tokens D = {D_actual:,}   D/N = {ratio:.1f}×")
    print(f"unique targets      = train_size×out_grid = {unique_tokens:,}   unique/N = {unique_tokens / cfg.target_params:.1f}×")
    print("============================")

    # Datasets / loaders
    train_ds = HeatEquationOperatorDataset(
        size=cfg.train_size,
        modes=cfg.modes,
        out_grid=cfg.out_grid,
        alpha=cfg.alpha,
        t_max=cfg.t_max,
        seed=cfg.seed + 123,
        coeff_std_decay=cfg.coeff_std_decay,
        coeff_scale=cfg.coeff_scale,
        include_endpoints=cfg.include_endpoints,
    )
    test_ds = HeatEquationOperatorDataset(
        size=cfg.test_size,
        modes=cfg.modes,
        out_grid=cfg.out_grid,
        alpha=cfg.alpha,
        t_max=cfg.t_max,
        seed=cfg.seed + 456,
        coeff_std_decay=cfg.coeff_std_decay,
        coeff_scale=cfg.coeff_scale,
        include_endpoints=cfg.include_endpoints,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Sweep
    results: List[Dict[str, float]] = []

    # Grab a fixed batch for AGOP from test set (more稳定)
    agop_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.agop_batch,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    agop_batch = next(iter(agop_loader))
    agop_z0, _ = agop_batch

    print("\n========== Shape sweep ==========")
    for i, depth in enumerate(cfg.depth_list):
        width, active = find_width_for_target_params(depth=depth, cfg=cfg)
        model = build_mlp_model(depth=depth, width=width, cfg=cfg, pad_to_target=True)
        total = count_params(model)
        pad = int(total - active)

        print(f"\n[{i+1:02d}/{len(cfg.depth_list)}] depth={depth:2d}  width={width:4d}  "
              f"active={active:,}  pad={pad:,}  total={total:,}")

        # Train
        metrics, history = train_one_model(model, train_loader, test_loader, cfg, device)
        save_curve(history, curve_dir, f"mlp_depth{depth}_width{width}")

        # AGOP (wrt inputs z) on a fixed batch
        z = agop_z0.to(device)
        agop = estimate_agop_wrt_inputs(model.to(device), z, proj_samples=cfg.agop_proj_samples)
        offdiag, offdiag_ratio = agop_offdiag_metrics(agop)

        row = {
            "depth": int(depth),
            "width": int(width),
            "active_params": int(active),
            "pad_params": int(pad),
            "total_params": int(total),
            "padding_ratio": float(pad / max(1, total)),
            "train_mse": float(metrics["train_mse"]),
            "test_mse": float(metrics["test_mse"]),
            "best_train_mse": float(metrics["best_train_mse"]),
            "best_test_mse": float(metrics["best_test_mse"]),
            "steps_run": int(metrics["steps_run"]),
            "agop_offdiag_energy": float(offdiag),
            "agop_offdiag_ratio": float(offdiag_ratio),
        }
        results.append(row)

    print("\n========== Correlations ==========")
    test_mse = np.array([r["test_mse"] for r in results], dtype=np.float64)
    best_test_mse = np.array([r["best_test_mse"] for r in results], dtype=np.float64)
    offE = np.array([r["agop_offdiag_energy"] for r in results], dtype=np.float64)
    offR = np.array([r["agop_offdiag_ratio"] for r in results], dtype=np.float64)

    # 在回归任务里，loss 常跨数量级，相关性更稳定的做法是用 log(loss)
    log_test = np.log(test_mse + 1e-18)
    log_best = np.log(best_test_mse + 1e-18)

    print(f"Pearson corr(offE, log_test_mse)      = {pearson_corr(offE, log_test):.4f}")
    print(f"Spearman corr(offE, log_test_mse)     = {spearman_corr(offE, log_test):.4f}")
    print(f"Pearson corr(offR, log_test_mse)      = {pearson_corr(offR, log_test):.4f}")
    print(f"Spearman corr(offR, log_test_mse)     = {spearman_corr(offR, log_test):.4f}")
    print(f"Pearson corr(offE, log_best_test_mse) = {pearson_corr(offE, log_best):.4f}")
    print(f"Spearman corr(offE, log_best_test_mse)= {spearman_corr(offE, log_best):.4f}")

    # Save CSV
    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # Save NPY
    npy_path = os.path.join(args.out_dir, "results.npy")
    np.save(npy_path, results, allow_pickle=True)
    print(f"Saved: {npy_path}")

    # Plots
    # Plots
    depths = [int(r["depth"]) for r in results]

    plt.figure(figsize=(6, 5))
    plt.yscale("log")
    plt.scatter(offE, test_mse)
    for i, d in enumerate(depths):
        plt.annotate(f"d{d}", (offE[i], test_mse[i]), fontsize=8, alpha=0.8)
    plt.xlabel("AGOP off-diagonal energy")
    plt.ylabel("test MSE (per-scalar)")
    plt.title("test MSE vs coupling (off-diagonal energy)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scatter_testmse_vs_offdiagE.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.yscale("log")
    plt.scatter(offR, test_mse)
    for i, d in enumerate(depths):
        plt.annotate(f"d{d}", (offR[i], test_mse[i]), fontsize=8, alpha=0.8)
    plt.xlabel("AGOP off-diagonal ratio")
    plt.ylabel("test MSE (per-scalar)")
    plt.title("test MSE vs coupling (off-diagonal ratio)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scatter_testmse_vs_offdiagR.png"), dpi=160)
    plt.close()

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
