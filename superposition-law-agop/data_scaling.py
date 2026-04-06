"""
Bottleneck data-scaling experiment + superposition metrics (WtW-based)
--------------------------------------------------------------------

This script is a cleaned-up + more statistically robust version of the
data-scaling experiment used in the LessWrong post:

"More findings on memorization and double descent"
https://www.lesswrong.com/posts/KzwB4ovzrZ8DYWgpw/more-findings-on-memorization-and-double-descent

Key improvements vs the minimal notebook implementation:
- Reproducible seeding: separate seeds for data generation and model init
- Fixed test set size (so test-loss variance doesn't depend on data_size)
- Keep *all* seeds (not only the best) for correlation analysis
- Adds two W^T W based superposition metrics:
    (1) off-diagonal energy ratio of W^T W (often saturates near 1 for large d)
    (2) weighted mean cos^2 between columns of W (recommended; not saturated)
- Produces:
    - mean±std curves vs data_size
    - scatter sup_metric vs test_loss
    - Pearson and Spearman correlations (also after de-trending by log(data_size))

Run:
    python data_scaling.py

Regenerate figures from saved data (no training):
    python data_scaling.py --plot-only ./results_bottleneck_scaling

Regenerate only the loss / AGOP off-diagonal energy figure (minimal npz):
    python data_scaling.py --plot-energy-only ./results_bottleneck_scaling

Writes summary.npz, loss_and_offdiag_energy_dual_axis_data.npz, and runs.npy under the results
directory so you can tweak plotting code and re-run with --plot-only or --plot-energy-only.

You can edit the EXPERIMENT CONFIG section at the bottom.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -----------------------
# Reproducibility helpers
# -----------------------

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a torch.Generator on the specified device (for deterministic data)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


# -----------------------
# Data generation
# -----------------------

@torch.no_grad()
def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize each row vector (safe for zero rows)."""
    denom = x.norm(dim=1, keepdim=True).clamp_min(eps)
    return x / denom


@torch.no_grad()
def generate_batch(
    num_data: int,
    num_dim: int,
    sparsity: float,
    *,
    normalize: bool = True,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate a sparse random batch with shape [num_data, num_dim].

    'sparsity' is the probability that an entry is zero.
    Non-zeros are uniform in [0,1).
    """
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"sparsity must be in [0,1], got {sparsity}")

    # Bernoulli mask: keep with prob (1 - sparsity)
    keep_prob = 1.0 - sparsity
    mask = (torch.rand((num_data, num_dim), device=device, generator=generator) <= keep_prob).to(dtype)
    feat = torch.rand((num_data, num_dim), device=device, generator=generator, dtype=dtype)
    x = feat * mask
    if normalize:
        x = l2_normalize_rows(x)
    return x


# -----------------------
# Model
# -----------------------

class SuperpositionNet(nn.Module):
    """
    Two-layer bottleneck model.

    If use_W_transpose=True, the effective linear map is W^T W (symmetric PSD).
    forward(x) = ReLU( x W^T W + b )
    """
    def __init__(self, input_size: int, hidden_size: int, *, bias: bool = True, use_W_transpose: bool = True):
        super().__init__()
        self.use_W_transpose = use_W_transpose
        self.bias_enabled = bias

        self.W1 = nn.Parameter(torch.empty((hidden_size, input_size)))
        nn.init.xavier_normal_(self.W1)

        if not use_W_transpose:
            self.W2 = nn.Parameter(torch.empty((input_size, hidden_size)))
            nn.init.xavier_normal_(self.W2)
        else:
            self.W2 = None

        if bias:
            # Important: initializing bias to non-zero can change qualitative behavior in these toy models.
            self.b = nn.Parameter(torch.zeros((input_size,)))
        else:
            self.b = None

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, input_size]
        h = x @ self.W1.T  # [N, hidden]
        if self.use_W_transpose:
            y = h @ self.W1  # [N, input]
        else:
            assert self.W2 is not None
            y = h @ self.W2.T  # [N, input]

        if self.b is not None:
            y = y + self.b

        return self.relu(y)


# -----------------------
# Loss
# -----------------------

def memorization_loss(pred: torch.Tensor, true: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
    """
    Weighted reconstruction loss used in the notebook:
        mean over samples of sum over dims of (true - |pred|)^2 * importance
    """
    # pred is already >=0 after ReLU, but keep abs() for compatibility with the original notebook.
    diff2 = (true - pred.abs()).pow(2)
    weighted = diff2 * importance  # broadcast importance [D] over batch
    return weighted.sum(dim=1).mean()


# -----------------------
# Superposition & AGOP metrics (WtW/AGOP-based)
# -----------------------

@torch.no_grad()
def wt_w(W1: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix G = W^T W where columns of W are feature directions in hidden space."""
    return W1.T @ W1


@torch.no_grad()
def offdiag_energy_from_matrix(M: torch.Tensor) -> float:
    """
    Off-diagonal energy (numerator only): ||M - diag(M)||_F^2.
    Raw magnitude, not normalized by ||M||_F^2.
    """
    diag = torch.diagonal(M)
    off = M - torch.diag(diag)
    return (off ** 2).sum().item()


@torch.no_grad()
def offdiag_energy_ratio_from_matrix(M: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Generic: off-diagonal energy ratio of a square matrix M.

        ratio = ||M - diag(M)||_F^2 / ||M||_F^2

    Note: For large d, this can saturate near 1 simply because there are O(d^2) off-diagonal entries.
    """
    diag = torch.diagonal(M)
    off = M - torch.diag(diag)
    num = (off ** 2).sum()
    den = (M ** 2).sum().clamp_min(eps)
    return (num / den).item()


@torch.no_grad()
def weighted_mean_cos2_from_gram(G: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Generic: weighted mean cos^2 implied by a Gram-like matrix G.

    Interpreting G_ij = <v_i, v_j> and d_i = ||v_i||^2 = G_ii:

        Sup = sum_{i!=j} G_ij^2 / sum_{i!=j} d_i d_j

    This equals the average cos^2(theta_ij) weighted by ||v_i||^2 ||v_j||^2.
    For PSD G (e.g., W^T W or AGOP), this is always in [0,1].
    """
    d = torch.diagonal(G)
    off_energy = (G ** 2).sum() - (d ** 2).sum()
    denom = (d.sum() ** 2 - (d ** 2).sum()).clamp_min(eps)
    return (off_energy / denom).item()


@torch.no_grad()
def offdiag_energy_ratio_from_WtW(W1: torch.Tensor, eps: float = 1e-12) -> float:
    """Convenience wrapper for G = W^T W."""
    return offdiag_energy_ratio_from_matrix(wt_w(W1), eps=eps)


@torch.no_grad()
def weighted_mean_cos2_from_WtW(W1: torch.Tensor, eps: float = 1e-12) -> float:
    """Convenience wrapper for G = W^T W."""
    return weighted_mean_cos2_from_gram(wt_w(W1), eps=eps)


@torch.no_grad()
def compute_agop_input_fast(
    model: "SuperpositionNet",
    x: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute the input-AGOP for this toy bottleneck model under the *use_W_transpose=True* setting.

    Paper (Radhakrishnan et al., Science 2024) defines AGOP as the average of gradient outer products.
    For multivariate outputs, gradients generalize to the Jacobian. Here we aggregate E[J J^T] (same
    D×D size as J^T J for square J); some references instead use E[J^T J].

    For our model:
        z = x @ G + b
        y = ReLU(z)
        where G = W^T W  (PSD, rank<=hidden)

    Jacobian wrt input x is:
        J(x) = diag(1[z>0]) @ G

    Then:
        J J^T = diag(1[z>0]) G G^T diag(1[z>0]) = diag(1[z>0]) G^2 diag(1[z>0])  (G symmetric)

    So (J J^T)_{ij} = 1[z_i>0] 1[z_j>0] (G^2)_{ij}.

    Averaging over data:
        AGOP = E[ J J^T ] = (G^2) ⊙ C
    where C_{ij} = E[ 1[z_i>0] 1[z_j>0] ] (joint ReLU activation/co-activation rate),
    ⊙ is elementwise product, and G^2 = G @ G.

    This needs the full D×D co-activation matrix C (not just per-coordinate p_k), accumulated as
    sum_b m_b^T m_b over minibatches, then divided by N.

    Complexity: O(D^3) for G^2 plus O(N D^2) to accumulate C; no per-sample Jacobian storage.

    NOTE: This closed-form is valid ONLY when model.use_W_transpose == True.
    """
    if not model.use_W_transpose:
        raise ValueError(
            "compute_agop_input_fast currently requires use_W_transpose=True. "
            "For separate W2, you'd need an autograd-based Jacobian/JVP implementation."
        )

    W1 = model.W1.detach()
    G = wt_w(W1)  # [D,D]
    b = model.b.detach() if model.b is not None else None

    N = x.shape[0]
    D = x.shape[1]

    G2 = G @ G  # [D,D] for (J J^T)_{ij} = m_i m_j (G^2)_{ij}

    # Co-activation C_{ij} = E[m_i m_j]; accumulate sum_b m_b^T m_b over the dataset
    coact_sum = torch.zeros((D, D), device=x.device, dtype=torch.float32)
    seen = 0

    for start in range(0, N, chunk_size):
        xb = x[start : start + chunk_size]  # [B,D]
        z = xb @ G  # [B,D]
        if b is not None:
            z = z + b
        active = (z > 0).to(torch.float32)  # [B,D]
        coact_sum += active.T @ active  # [D,D]
        seen += xb.shape[0]

    coact = coact_sum / max(1, seen)
    agop = G2 * coact  # E[J J^T] = (G^2) ⊙ coact

    # Small numerical cleanup: ensure symmetry
    agop = 0.5 * (agop + agop.T)

    # Prevent accidental NaNs (shouldn't happen, but safe)
    if torch.isnan(agop).any():
        agop = torch.nan_to_num(agop, nan=0.0, posinf=0.0, neginf=0.0)

    return agop


@torch.no_grad()
def pearson_corr_offdiag(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Pearson correlation between off-diagonal entries of two square matrices (flattened).
    Useful for checking the NFA-style proportionality between W^T W and AGOP. fileciteturn0file0L1-L18
    """
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be same-shape square matrices")

    d = A.shape[0]
    mask = ~torch.eye(d, device=A.device, dtype=torch.bool)
    a = A[mask].to(torch.float64)
    b = B[mask].to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.pow(2).sum() * b.pow(2).sum()).sqrt().clamp_min(eps)
    return float((a * b).sum() / denom)
# -----------------------
# Training
# -----------------------

@dataclass
class TrainConfig:
    lr: float = 5e-3
    weight_decay: float = 1e-2
    steps: int = 3000
    warmup_frac: float = 0.25  # fraction of steps for linear warmup
    use_scheduler: bool = True
    bias: bool = True
    use_W_transpose: bool = True


def train_memorization_model(
    x_train: torch.Tensor,
    importance: torch.Tensor,
    hidden_size: int,
    *,
    model_seed: int,
    device: torch.device,
    cfg: TrainConfig,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[SuperpositionNet, Dict[str, float]]:
    """
    Train the bottleneck model to reconstruct x_train.

    batch_size:
      - None: full-batch training (closest to the original notebook)
      - int: mini-batch training (faster for large datasets)
    """
    set_global_seed(model_seed)

    model = SuperpositionNet(
        input_size=x_train.shape[1],
        hidden_size=hidden_size,
        bias=cfg.bias,
        use_W_transpose=cfg.use_W_transpose,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_scheduler:
        warmup_steps = max(1, int(cfg.steps * cfg.warmup_frac))
        sched1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.steps - warmup_steps))
    else:
        sched1 = sched2 = None

    # If doing minibatch, create indices once on device for speed.
    N = x_train.shape[0]
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
        if batch_size > N:
            batch_size = N

    model.train()
    last_loss = None

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)

        if batch_size is None:
            xb = x_train
        else:
            idx = torch.randint(0, N, (batch_size,), device=device)
            xb = x_train[idx]

        out = model(xb)
        loss = memorization_loss(out, xb, importance)
        loss.backward()
        opt.step()

        if cfg.use_scheduler:
            if step < warmup_steps:
                sched1.step()
            else:
                sched2.step()

        last_loss = float(loss.detach().cpu().item())
        if verbose and (step % 500 == 0 or step == cfg.steps - 1):
            lr_now = opt.param_groups[0]["lr"]
            print(f"  step={step:5d}  loss={last_loss:.6f}  lr={lr_now:.3e}")

    # Final full-batch train loss (for comparability even if minibatch training)
    model.eval()
    with torch.no_grad():
        full_out = model(x_train)
        train_loss = float(memorization_loss(full_out, x_train, importance).cpu().item())

    stats = {
        "train_loss": train_loss,
        "last_step_loss": last_loss if last_loss is not None else float("nan"),
    }
    return model, stats


# -----------------------
# Experiment runner
# -----------------------

@dataclass
class ExperimentConfig:
    data_sizes: List[int]
    num_dim: int = 1000
    hidden_size: int = 2
    sparsity: float = 0.99
    normalize: bool = True

    # seeds
    data_seed: int = 12345         # seed for data generation
    test_seed: int = 54321         # seed for test generation (per data_size)
    model_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # evaluation
    test_size: int = 5000          # fixed test size (recommended)
    # training
    train_cfg: TrainConfig = field(default_factory=TrainConfig)
    batch_size: Optional[int] = None  # None = full-batch; set e.g. 2048 for speed

    # device
    device: Optional[str] = None   # "cuda" / "cpu" / None(auto)


@torch.no_grad()
def evaluate_test_loss(model: nn.Module, x_test: torch.Tensor, importance: torch.Tensor) -> float:
    model.eval()
    out = model(x_test)
    return float(memorization_loss(out, x_test, importance).cpu().item())


def run_data_scaling_experiment(exp: ExperimentConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      - a structured array of per-run results (rows = data_size x seed)
      - a dict of summary arrays keyed by metric name (mean/std per data_size)
    """
    device = torch.device(exp.device) if exp.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # optional speed tweak (kept conservative; comment out if you want strict determinism)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # can speed up matmuls on Ampere+
        torch.set_float32_matmul_precision("high")

    # importance weights
    importance = torch.ones((exp.num_dim,), device=device, dtype=torch.float32)

    rows = []

    for ds in exp.data_sizes:
        print(f"\n=== data_size={ds} ===")

        # Deterministic train & test data per data_size
        g_train = make_generator(exp.data_seed + ds, device)
        g_test = make_generator(exp.test_seed + ds, device)

        x_train = generate_batch(
            num_data=ds,
            num_dim=exp.num_dim,
            sparsity=exp.sparsity,
            normalize=exp.normalize,
            device=device,
            generator=g_train,
        )

        x_test = generate_batch(
            num_data=exp.test_size,
            num_dim=exp.num_dim,
            sparsity=exp.sparsity,
            normalize=exp.normalize,
            device=device,
            generator=g_test,
        )

        for seed in exp.model_seeds:
            print(f"  model_seed={seed}")
            model, train_stats = train_memorization_model(
                x_train=x_train,
                importance=importance,
                hidden_size=exp.hidden_size,
                model_seed=seed,
                device=device,
                cfg=exp.train_cfg,
                batch_size=exp.batch_size,
                verbose=False,
            )

            test_loss = evaluate_test_loss(model, x_test, importance)

            # Superposition metrics (from WᵀW)
            W1 = model.W1.detach()
            G = wt_w(W1)
            sup_offdiag = offdiag_energy_ratio_from_WtW(W1)
            sup_offdiag_energy = offdiag_energy_from_matrix(G)
            sup_cos2 = weighted_mean_cos2_from_WtW(W1)

            # AGOP (input-level) per Neural Feature Ansatz: NFM ~ (AGOP)^a. fileciteturn0file0L55-L78
            # We compute AGOP w.r.t. the *input* of this toy model, evaluated on the fixed test distribution
            # to make metrics comparable across data_size.
            agop = compute_agop_input_fast(model, x_test, chunk_size=4096)

            agop_offdiag = offdiag_energy_ratio_from_matrix(agop)
            agop_offdiag_energy = offdiag_energy_from_matrix(agop)
            agop_cos2 = weighted_mean_cos2_from_gram(agop)

            # Optional: check proportionality/correlation between WᵀW and AGOP off-diagonals
            wtw_agop_offdiag_pearson = pearson_corr_offdiag(G, agop)

            rows.append({
                "data_size": ds,
                "model_seed": seed,
                "train_loss": train_stats["train_loss"],
                "test_loss": test_loss,

                # WᵀW-based superposition proxies
                "sup_offdiag_energy_ratio": sup_offdiag,
                "sup_offdiag_energy": sup_offdiag_energy,
                "sup_weighted_mean_cos2": sup_cos2,

                # AGOP-based proxies (computed on fixed test distribution)
                "agop_offdiag_energy_ratio": agop_offdiag,
                "agop_offdiag_energy": agop_offdiag_energy,
                "agop_weighted_mean_cos2": agop_cos2,

                # correlation between WᵀW and AGOP off-diagonals (sanity check for NFA)
                "wtw_agop_offdiag_pearson": wtw_agop_offdiag_pearson,
            })

    # Convert to structured NumPy array for portability
    dtype = [
        ("data_size", "i8"),
        ("model_seed", "i8"),
        ("train_loss", "f8"),
        ("test_loss", "f8"),
        ("sup_offdiag_energy_ratio", "f8"),
        ("sup_offdiag_energy", "f8"),
        ("sup_weighted_mean_cos2", "f8"),
        ("agop_offdiag_energy_ratio", "f8"),
        ("agop_offdiag_energy", "f8"),
        ("agop_weighted_mean_cos2", "f8"),
        ("wtw_agop_offdiag_pearson", "f8"),
    ]
    arr = np.zeros(len(rows), dtype=dtype)
    for i, r in enumerate(rows):
        for k in arr.dtype.names:
            arr[k][i] = r[k]

    # Summaries per data_size
    unique_sizes = np.array(exp.data_sizes, dtype=np.int64)
    summary: Dict[str, np.ndarray] = {"data_size": unique_sizes}

    for metric in ["train_loss", "test_loss", "sup_offdiag_energy_ratio", "sup_offdiag_energy", "sup_weighted_mean_cos2", "agop_offdiag_energy_ratio", "agop_offdiag_energy", "agop_weighted_mean_cos2", "wtw_agop_offdiag_pearson"]:
        means = []
        stds = []
        for ds in unique_sizes:
            vals = arr[metric][arr["data_size"] == ds]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))
        summary[f"{metric}_mean"] = np.array(means)
        summary[f"{metric}_std"] = np.array(stds)

    return arr, summary


# -----------------------
# Analysis utilities
# -----------------------

def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    # Simple Spearman without SciPy: rank-transform then Pearson
    def rankdata(a: np.ndarray) -> np.ndarray:
        temp = a.argsort()
        ranks = np.empty_like(temp, dtype=np.float64)
        ranks[temp] = np.arange(len(a), dtype=np.float64)
        return ranks

    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    return pearsonr(rx, ry)


def detrend_by_log_datasize(arr: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Remove linear dependence on log10(data_size): y_res = y - (a*log10(ds) + b).
    """
    ds = np.asarray(arr["data_size"], dtype=np.float64)
    x = np.log10(ds)
    A = np.stack([x, np.ones_like(x)], axis=1)  # [N,2]
    # least squares
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coef
    return y - y_hat


# -----------------------
# Saved results (regenerate figures without re-running training)
# -----------------------

SUMMARY_NPZ_NAME = "summary.npz"
SUMMARY_NPY_LEGACY = "summary.npy"
RUNS_NPY_NAME = "runs.npy"
# Minimal arrays for the publication-style loss + AGOP off-diagonal energy figure only:
LOSS_AGOP_ENERGY_NPZ_NAME = "loss_and_offdiag_energy_dual_axis_data.npz"


def save_summary_npz(summary: Dict[str, np.ndarray], path: str) -> None:
    """Write summary dict to a compressed .npz (portable, no pickle)."""
    payload = {k: np.asarray(v) for k, v in summary.items()}
    np.savez_compressed(path, **payload)


def load_summary_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_summary_from_save_dir(save_dir: str) -> Dict[str, np.ndarray]:
    """
    Load aggregated curves: prefer summary.npz, fall back to legacy summary.npy (pickled dict).
    """
    npz_path = os.path.join(save_dir, SUMMARY_NPZ_NAME)
    npy_path = os.path.join(save_dir, SUMMARY_NPY_LEGACY)
    if os.path.isfile(npz_path):
        return load_summary_npz(npz_path)
    if os.path.isfile(npy_path):
        raw = np.load(npy_path, allow_pickle=True)
        d = raw.item()
        if not isinstance(d, dict):
            raise ValueError(f"Legacy {npy_path} does not contain a dict.")
        return d  # type: ignore[return-value]
    raise FileNotFoundError(
        f"No {SUMMARY_NPZ_NAME} or {SUMMARY_NPY_LEGACY} found in {save_dir}"
    )


def save_loss_agop_energy_plot_npz(summary: Dict[str, np.ndarray], path: str) -> None:
    """Subset of summary for loss + AGOP off-diagonal energy dual-axis plot only."""
    keys = (
        "data_size",
        "test_loss_mean",
        "test_loss_std",
        "agop_offdiag_energy_mean",
        "agop_offdiag_energy_std",
    )
    missing = [k for k in keys if k not in summary]
    if missing:
        raise KeyError(f"summary missing keys for plot npz: {missing}")
    np.savez_compressed(path, **{k: np.asarray(summary[k]) for k in keys})


def load_loss_agop_energy_plot_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def plot_summary(summary: Dict[str, np.ndarray], save_dir: Optional[str] = None) -> None:
    ds = summary["data_size"]

    def plot_with_error(y_mean, y_std, title, ylabel, fname):
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(ds, y_mean, yerr=y_std, fmt="o-", capsize=3)
        plt.xscale("log")
        plt.xlabel("data_size (log scale)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, fname), dpi=200)
        plt.show()

    plot_with_error(
        summary["test_loss_mean"], summary["test_loss_std"],
        "Test loss vs data size (mean ± std over seeds)",
        "test_loss",
        "test_loss_vs_data_size.png",
    )

    plot_with_error(
        summary["sup_weighted_mean_cos2_mean"], summary["sup_weighted_mean_cos2_std"],
        "Superposition (weighted mean cos²) vs data size (mean ± std over seeds)",
        "sup_weighted_mean_cos2",
        "sup_weighted_mean_cos2_vs_data_size.png",
    )

    plot_with_error(
        summary["sup_offdiag_energy_ratio_mean"], summary["sup_offdiag_energy_ratio_std"],
        "Superposition (off-diagonal energy ratio of WᵀW) vs data size (mean ± std over seeds)",
        "sup_offdiag_energy_ratio",
        "sup_offdiag_energy_ratio_vs_data_size.png",
    )

    plot_with_error(
        summary["agop_weighted_mean_cos2_mean"], summary["agop_weighted_mean_cos2_std"],
        "AGOP (weighted mean cos²) vs data size (mean ± std over seeds)",
        "agop_weighted_mean_cos2",
        "agop_weighted_mean_cos2_vs_data_size.png",
    )

    plot_with_error(
        summary["agop_offdiag_energy_ratio_mean"], summary["agop_offdiag_energy_ratio_std"],
        "AGOP (off-diagonal energy ratio) vs data size (mean ± std over seeds)",
        "agop_offdiag_energy_ratio",
        "agop_offdiag_energy_ratio_vs_data_size.png",
    )

    plot_with_error(
        summary["wtw_agop_offdiag_pearson_mean"], summary["wtw_agop_offdiag_pearson_std"],
        "Pearson corr(offdiag): WᵀW vs AGOP vs data size (mean ± std over seeds)",
        "wtw_agop_offdiag_pearson",
        "wtw_agop_offdiag_pearson_vs_data_size.png",
    )


def plot_loss_and_offdiag_dual_axis(summary: Dict[str, np.ndarray], save_dir: Optional[str] = None) -> None:
    """
    Single figure: x = data_size (log), left y = test_loss, right y = sup_offdiag & agop_offdiag.
    Shows true trend of loss and both off-diagonal energy ratios vs data size.
    """
    ds = summary["data_size"].astype(np.float64)

    fig, ax_left = plt.subplots(figsize=(9, 5))
    ax_left.set_xscale("log")
    ax_left.set_xlabel("data_size (log scale)")

    # Left axis: test loss
    ax_left.errorbar(
        ds,
        summary["test_loss_mean"],
        yerr=summary["test_loss_std"],
        fmt="o-",
        capsize=3,
        color="C0",
        label="test_loss",
    )
    ax_left.set_ylabel("test_loss", color="C0")
    ax_left.tick_params(axis="y", labelcolor="C0")
    ax_left.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Right axis: sup_offdiag and agop_offdiag
    ax_right = ax_left.twinx()
    ax_right.errorbar(
        ds,
        summary["sup_offdiag_energy_ratio_mean"],
        yerr=summary["sup_offdiag_energy_ratio_std"],
        fmt="s-",
        capsize=3,
        color="C1",
        label="sup_offdiag_energy_ratio",
    )
    ax_right.errorbar(
        ds,
        summary["agop_offdiag_energy_ratio_mean"],
        yerr=summary["agop_offdiag_energy_ratio_std"],
        fmt="^-",
        capsize=3,
        color="C2",
        label="agop_offdiag_energy_ratio",
    )
    ax_right.set_ylabel("off-diagonal energy ratio (sup / AGOP)")

    # Combined legend
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="center right")

    plt.title("Test loss (left) vs sup/AGOP off-diagonal energy ratio (right) vs data size")
    fig.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "loss_and_offdiag_dual_axis.png"), dpi=200)
    plt.show()


def plot_loss_and_offdiag_energy_dual_axis(summary: Dict[str, np.ndarray], save_dir: Optional[str] = None) -> None:
    """
    Single figure: x = dataset size (log), left y = test loss, right y = AOFE (AGOP off-diagonal
    Frobenius energy, not ratio). Styled for publication (serif). No figure title (caption in paper).

    ``summary`` may be the full experiment summary or the minimal dict loaded from
    ``loss_and_offdiag_energy_dual_axis_data.npz`` (see ``load_loss_agop_energy_plot_npz``).
    """
    required = ["agop_offdiag_energy_mean", "agop_offdiag_energy_std"]
    missing = [k for k in required if k not in summary]
    if missing:
        raise ValueError(
            f"summary is missing keys: {missing}. "
            "This usually means summary.npy was saved before the off-diagonal energy metrics were added. "
            "Re-run the full experiment: python data_scaling_v2.py"
        )
    ds = summary["data_size"].astype(np.float64)

    # Publication-style rc (local to this figure only; DejaVu Serif ships with matplotlib)
    pub_rc = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Bitstream Vera Serif", "Computer Modern Roman"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.25,
        "lines.markersize": 5,
        "mathtext.fontset": "cm",
    }

    with plt.rc_context(pub_rc):
        # ~2/3 text width for many two-column templates; adjust in manuscript if needed
        fig, ax_left = plt.subplots(figsize=(5.5, 3.4), layout="constrained")

        ax_left.set_xscale("log")
        ax_left.set_xlabel("Dataset size (log scale)")

        # Colorblind-friendly pair (Okabe–Ito blue + vermillion)
        c_loss = "#0072B2"
        c_agop = "#D55E00"

        ax_left.errorbar(
            ds,
            summary["test_loss_mean"],
            yerr=summary["test_loss_std"],
            fmt="o-",
            color=c_loss,
            ecolor=c_loss,
            elinewidth=0.9,
            capsize=2.5,
            label="Test loss",
            zorder=3,
        )
        ax_left.set_ylabel("Test loss", color=c_loss)
        ax_left.tick_params(axis="y", labelcolor=c_loss)
        ax_left.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.85)
        ax_left.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.55)

        ax_right = ax_left.twinx()
        ax_right.errorbar(
            ds,
            summary["agop_offdiag_energy_mean"],
            yerr=summary["agop_offdiag_energy_std"],
            fmt="s-",
            color=c_agop,
            ecolor=c_agop,
            elinewidth=0.9,
            capsize=2.5,
            label="AOFE",
            zorder=2,
        )
        ax_right.set_ylabel("AOFE", color=c_agop)
        ax_right.tick_params(axis="y", labelcolor=c_agop)
        # Full rectangle: twinx() leaves left/bottom/right by default; we had hidden top—show it again.
        ax_left.spines["top"].set_visible(True)
        ax_right.spines["top"].set_visible(True)

        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(
            lines_left + lines_right,
            labels_left + labels_right,
            loc="best",
            frameon=True,
            fancybox=False,
            edgecolor="0.4",
        )

        if save_dir:
            out = os.path.join(save_dir, "loss_and_offdiag_energy_dual_axis.png")
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
            fig.savefig(os.path.join(save_dir, "loss_and_offdiag_energy_dual_axis.pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.show()


def plot_scatter(arr: np.ndarray, x_key: str, y_key: str, title: str, save_path: Optional[str] = None) -> None:
    x = arr[x_key].astype(np.float64)
    y = arr[y_key].astype(np.float64)
    ds = arr["data_size"].astype(np.float64)

    plt.figure(figsize=(6.5, 5))
    sc = plt.scatter(x, y, c=np.log10(ds), s=35)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    cbar = plt.colorbar(sc)
    cbar.set_label("log10(data_size)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def print_correlations(arr: np.ndarray, large_data_start: int = 1000) -> None:
    """
    Print Pearson and Spearman correlations between offdiag energy ratio metrics and test_loss.
    Computed for (1) all data_sizes and (2) only large data regime (data_size >= large_data_start).
    """
    large_mask = arr["data_size"] >= large_data_start
    arr_large = arr[large_mask]

    y_all = arr["test_loss"].astype(np.float64)
    y_large = arr_large["test_loss"].astype(np.float64)

    metrics = ["sup_offdiag_energy_ratio", "agop_offdiag_energy_ratio", "sup_offdiag_energy", "agop_offdiag_energy"]

    for metric in metrics:
        x_all = arr[metric].astype(np.float64)
        x_large = arr_large[metric].astype(np.float64)

        p_all = pearsonr(x_all, y_all)
        s_all = spearmanr(x_all, y_all)
        p_large = pearsonr(x_large, y_large)
        s_large = spearmanr(x_large, y_large)

        print(f"\n--- {metric} vs test_loss ---")
        print(f"  [Global, n={len(arr)}]")
        print(f"    Pearson  r = {p_all:.4f}")
        print(f"    Spearman r = {s_all:.4f}")
        print(f"  [Large data regime, data_size >= {large_data_start}, n={int(large_mask.sum())}]")
        print(f"    Pearson  r = {p_large:.4f}")
        print(f"    Spearman r = {s_large:.4f}")


def run_all_figures(
    summary: Dict[str, np.ndarray],
    arr: Optional[np.ndarray],
    save_dir: str,
) -> None:
    """All matplotlib outputs used in main (curves, dual-axis, scatters, correlation printout)."""
    plot_summary(summary, save_dir=save_dir)
    plot_loss_and_offdiag_dual_axis(summary, save_dir=save_dir)
    plot_loss_and_offdiag_energy_dual_axis(summary, save_dir=save_dir)

    if arr is None:
        print(
            f"\nNote: {RUNS_NPY_NAME} not loaded — skipping scatter plots and correlation table."
        )
        return

    plot_scatter(
        arr,
        x_key="sup_weighted_mean_cos2",
        y_key="test_loss",
        title="Superposition (weighted mean cos²) vs test loss",
        save_path=None,
    )
    plot_scatter(
        arr,
        x_key="sup_offdiag_energy_ratio",
        y_key="test_loss",
        title="Superposition (off-diagonal energy ratio) vs test loss",
        save_path=None,
    )
    plot_scatter(
        arr,
        x_key="agop_weighted_mean_cos2",
        y_key="test_loss",
        title="AGOP (weighted mean cos²) vs test loss",
        save_path=None,
    )
    plot_scatter(
        arr,
        x_key="agop_offdiag_energy_ratio",
        y_key="test_loss",
        title="AGOP (off-diagonal energy ratio) vs test loss",
        save_path=None,
    )
    plot_scatter(
        arr,
        x_key="wtw_agop_offdiag_pearson",
        y_key="test_loss",
        title="Pearson corr(offdiag): WᵀW vs AGOP vs test loss",
        save_path=None,
    )
    print_correlations(arr, large_data_start=1000)


def main_plot_energy_only(save_dir: str) -> None:
    """Regenerate only the publication-style loss / AGOP off-diagonal energy dual-axis figure."""
    save_dir = os.path.abspath(save_dir)
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Not a directory: {save_dir}")
    path = os.path.join(save_dir, LOSS_AGOP_ENERGY_NPZ_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Run the full experiment once, or use --plot-only with {SUMMARY_NPZ_NAME}."
        )
    summary_subset = load_loss_agop_energy_plot_npz(path)
    plot_loss_and_offdiag_energy_dual_axis(summary_subset, save_dir=save_dir)


def main_plot_only(save_dir: str) -> None:
    """Reload saved arrays and regenerate figures (edit plotting code, then re-run this)."""
    save_dir = os.path.abspath(save_dir)
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Not a directory: {save_dir}")
    summary = load_summary_from_save_dir(save_dir)
    runs_path = os.path.join(save_dir, RUNS_NPY_NAME)
    arr: Optional[np.ndarray] = np.load(runs_path) if os.path.isfile(runs_path) else None
    run_all_figures(summary, arr, save_dir=save_dir)


# -----------------------
# EXPERIMENT CONFIG
# -----------------------

def main() -> None:
    # 1000-20000: 10 log-evenly-spaced points covering the large data regime
    _large_sizes = [int(round(x)) for x in np.logspace(np.log10(1000), np.log10(20000), 10).tolist()]
    # _large_sizes ≈ [1000, 1394, 1945, 2714, 3787, 5284, 7369, 10278, 14340, 20000]

    exp = ExperimentConfig(
        data_sizes=[3, 5, 8, 10, 15, 30, 50, 100, 200, 500] + _large_sizes,
        num_dim=1000,
        hidden_size=2,
        sparsity=0.99,
        normalize=True,

        # fixed test set size (important!)
        test_size=5000,

        # seeds
        data_seed=12345,
        test_seed=54321,
        model_seeds=(0, 1, 2, 3, 4),

        # training
        train_cfg=TrainConfig(
            lr=5e-3,
            weight_decay=1e-2,
            steps=3000,
            warmup_frac=0.25,
            use_scheduler=True,
            bias=True,
            use_W_transpose=True,
        ),

        # Mini-batch for large data sizes to avoid full-batch memory/speed issues.
        batch_size=2048,

        device=None,  # auto
    )

    arr, summary = run_data_scaling_experiment(exp)

    save_dir = "./results_bottleneck_scaling"
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Summary (first 5 rows) ===")
    # Print a few lines in a readable way
    for i in range(min(5, len(arr))):
        print({k: arr[k][i].item() if hasattr(arr[k][i], "item") else arr[k][i] for k in arr.dtype.names})

    # Plots
    run_all_figures(summary, arr, save_dir=save_dir)

    # Save numpy results (npz for clean reload; npy kept for backward compatibility)
    np.save(os.path.join(save_dir, RUNS_NPY_NAME), arr)
    save_summary_npz(summary, os.path.join(save_dir, SUMMARY_NPZ_NAME))
    save_loss_agop_energy_plot_npz(
        summary, os.path.join(save_dir, LOSS_AGOP_ENERGY_NPZ_NAME)
    )
    np.save(os.path.join(save_dir, SUMMARY_NPY_LEGACY), summary)
    abs_dir = os.path.abspath(save_dir)
    print(f"\nSaved numpy results to: {abs_dir}")
    print(
        f"  - {RUNS_NPY_NAME} (per-run rows for scatter / correlations)\n"
        f"  - {SUMMARY_NPZ_NAME} (mean±std curves; use with: python data_scaling.py --plot-only {abs_dir})\n"
        f"  - {LOSS_AGOP_ENERGY_NPZ_NAME} (minimal bundle for the loss/AGOP energy figure only)\n"
        f"  - {SUMMARY_NPY_LEGACY} (legacy pickle dict; prefer {SUMMARY_NPZ_NAME})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bottleneck data-scaling experiment and superposition metrics."
    )
    parser.add_argument(
        "--plot-only",
        metavar="SAVE_DIR",
        default=None,
        help=(
            "Regenerate all figures from saved data in SAVE_DIR (loads summary.npz or legacy summary.npy; "
            "optional runs.npy for scatters). No training."
        ),
    )
    parser.add_argument(
        "--plot-energy-only",
        metavar="SAVE_DIR",
        default=None,
        help=(
            f"Regenerate only loss_and_offdiag_energy_dual_axis (PNG/PDF) from {LOSS_AGOP_ENERGY_NPZ_NAME}."
        ),
    )
    args = parser.parse_args()
    if args.plot_only is not None and args.plot_energy_only is not None:
        parser.error("Use only one of --plot-only and --plot-energy-only.")
    if args.plot_energy_only is not None:
        main_plot_energy_only(args.plot_energy_only)
    elif args.plot_only is not None:
        main_plot_only(args.plot_only)
    else:
        main()
