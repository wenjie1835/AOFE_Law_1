from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _normalize_features(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def generate_compositional_vector_inputs(
    *,
    size: int,
    in_dim: int,
    structure_seed: int,
    sample_seed: int,
    latent_dim: int = 8,
    motif_count: int = 4,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate vector inputs with explicit feature reuse.

    Coordinates are partitioned into groups. Groups share a small motif bank, so
    many observed dimensions depend on the same latent causes. This keeps the
    task fully synthetic/controllable while introducing a meaningful incentive
    for representation reuse rather than pure feature disentanglement.
    """
    if motif_count <= 0:
        raise ValueError("motif_count must be positive")

    structure_g = _seeded_generator(structure_seed)
    sample_g = _seeded_generator(sample_seed)

    group_count = max(motif_count * 2, min(8, in_dim))
    chunk_sizes = [in_dim // group_count] * group_count
    for i in range(in_dim % group_count):
        chunk_sizes[i] += 1

    latent = torch.randn(size, latent_dim, generator=sample_g, dtype=dtype)
    low_rank = torch.randn(size, 3, generator=sample_g, dtype=dtype)

    motif_bank = [
        torch.randn(latent_dim, chunk, generator=structure_g, dtype=dtype) / math.sqrt(latent_dim)
        for chunk in chunk_sizes[:motif_count]
    ]
    local_bank = [
        torch.randn(latent_dim, chunk, generator=structure_g, dtype=dtype) / math.sqrt(latent_dim)
        for chunk in chunk_sizes
    ]
    bias_bank = [
        torch.randn(3, chunk, generator=structure_g, dtype=dtype) / math.sqrt(3.0)
        for chunk in chunk_sizes
    ]

    chunks = []
    for i, chunk in enumerate(chunk_sizes):
        motif = motif_bank[i % motif_count]
        local = local_bank[i]
        bias = bias_bank[i]
        shared = latent @ motif
        private = latent @ local
        low_rank_bias = low_rank @ bias
        chunk_x = torch.tanh(0.85 * shared + 0.35 * private + 0.20 * low_rank_bias)
        chunks.append(chunk_x[:, :chunk])

    x = torch.cat(chunks, dim=1)
    return _normalize_features(x[:, :in_dim])


def generate_compositional_sequences(
    *,
    size: int,
    seq_len: int,
    structure_seed: int,
    sample_seed: int,
    segment_len: int = 10,
    latent_dim: int = 8,
    motif_count: int = 4,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate 1D sequences with repeated motifs and shared latent causes.

    Each sequence is assembled from a small bank of motif templates placed
    across segments according to a repeated pattern family. This creates
    explicit long-range reuse while keeping the generator fully synthetic.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")

    structure_g = _seeded_generator(structure_seed)
    sample_g = _seeded_generator(sample_seed)

    num_segments = math.ceil(seq_len / segment_len)
    motif_bank = torch.randn(
        motif_count, latent_dim, segment_len, generator=structure_g, dtype=dtype
    ) / math.sqrt(latent_dim)
    trend_bank = torch.randn(
        motif_count, latent_dim, segment_len, generator=structure_g, dtype=dtype
    ) / math.sqrt(latent_dim)

    base_patterns = torch.tensor([
        [0, 1, 0, 2, 1, 0, 3, 1],
        [1, 2, 1, 0, 2, 1, 3, 2],
        [2, 0, 2, 1, 0, 2, 3, 0],
        [3, 1, 3, 0, 1, 3, 2, 1],
    ], dtype=torch.long)
    if num_segments > base_patterns.shape[1]:
        reps = math.ceil(num_segments / base_patterns.shape[1])
        base_patterns = base_patterns.repeat(1, reps)
    patterns = base_patterns[:, :num_segments]

    latent = torch.randn(size, latent_dim, generator=sample_g, dtype=dtype)
    pattern_ids = torch.randint(0, patterns.shape[0], (size,), generator=sample_g)
    drift = torch.randn(size, 2, generator=sample_g, dtype=dtype) * 0.15

    seq_chunks = []
    time_axis = torch.linspace(-1.0, 1.0, segment_len, dtype=dtype)
    for seg_idx in range(num_segments):
        motif_idx = patterns[pattern_ids, seg_idx]
        seg = torch.zeros(size, segment_len, dtype=dtype)
        for m in range(motif_count):
            mask = motif_idx == m
            if not torch.any(mask):
                continue
            z = latent[mask]
            shared = torch.tanh(z @ motif_bank[m])
            trend = 0.35 * torch.tanh(z @ trend_bank[m])
            drift_term = drift[mask, 0:1] * time_axis + drift[mask, 1:2]
            seg[mask] = shared + trend + drift_term
        if seg_idx > 0:
            seg = seg + 0.15 * seq_chunks[-1]
        seq_chunks.append(seg)

    seq = torch.cat(seq_chunks, dim=1)[:, :seq_len]
    seq = seq + 0.03 * torch.randn(seq.shape, generator=sample_g, dtype=dtype)
    seq = _normalize_features(seq)
    return seq.unsqueeze(-1)


@dataclass
class SceneConfig:
    image_size: int = 32
    patch: int = 8
    motif_count: int = 6
    layout_count: int = 6


class CompositionalSceneDataset(torch.utils.data.Dataset):
    """
    Deterministic procedural image dataset with repeated patch motifs.

    Each image is built from a small motif bank and a repeated layout template,
    so many patches reuse the same latent causes. This gives CNNs a controlled
    task where feature reuse is useful without introducing real-world label
    semantics or uncontrolled dataset biases.
    """

    def __init__(
        self,
        *,
        size: int,
        structure_seed: int,
        sample_seed: int,
        cfg: Optional[SceneConfig] = None,
    ):
        super().__init__()
        self.size = int(size)
        self.structure_seed = int(structure_seed)
        self.sample_seed = int(sample_seed)
        self.cfg = cfg or SceneConfig()

        if self.cfg.image_size % self.cfg.patch != 0:
            raise ValueError("image_size must be divisible by patch")
        self.grid = self.cfg.image_size // self.cfg.patch

        structure_g = _seeded_generator(self.structure_seed)
        motifs = torch.randn(
            self.cfg.motif_count,
            3,
            self.cfg.patch,
            self.cfg.patch,
            generator=structure_g,
            dtype=torch.float32,
        )
        motifs = torch.nn.functional.avg_pool2d(motifs, kernel_size=3, stride=1, padding=1)
        self.motif_bank = motifs

        base_layouts = torch.tensor([
            [
                [0, 1, 0, 2],
                [1, 3, 1, 4],
                [0, 1, 5, 2],
                [3, 4, 3, 5],
            ],
            [
                [2, 0, 2, 1],
                [4, 3, 4, 0],
                [2, 5, 2, 1],
                [4, 0, 4, 3],
            ],
            [
                [1, 2, 1, 3],
                [5, 0, 5, 2],
                [1, 3, 1, 4],
                [0, 2, 0, 5],
            ],
        ], dtype=torch.long)
        reps = math.ceil(self.cfg.layout_count / base_layouts.shape[0])
        self.layouts = base_layouts.repeat(reps, 1, 1)[: self.cfg.layout_count]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.sample_seed + idx)
        image = torch.zeros(3, self.cfg.image_size, self.cfg.image_size, dtype=torch.float32)

        layout = self.layouts[int(rng.integers(0, len(self.layouts)))]
        active = rng.choice(self.cfg.motif_count, size=min(4, self.cfg.motif_count), replace=False)
        channel_tint = torch.tensor(rng.normal(0.0, 0.25, size=(3,)), dtype=torch.float32)
        row_bias = torch.tensor(rng.normal(0.0, 0.15, size=(self.grid, 3)), dtype=torch.float32)
        col_bias = torch.tensor(rng.normal(0.0, 0.15, size=(self.grid, 3)), dtype=torch.float32)
        contrast = float(rng.uniform(0.8, 1.2))

        for r in range(self.grid):
            for c in range(self.grid):
                motif_id = int(layout[r, c].item()) % len(active)
                patch = self.motif_bank[int(active[motif_id])].clone()
                patch = contrast * patch
                patch = patch + channel_tint[:, None, None]
                patch = patch + row_bias[r][:, None, None] + col_bias[c][:, None, None]
                patch = patch + float(rng.normal(0.0, 0.02)) * torch.ones_like(patch)
                rs = r * self.cfg.patch
                cs = c * self.cfg.patch
                image[:, rs:rs + self.cfg.patch, cs:cs + self.cfg.patch] = patch

        image = image.clamp(-3.0, 3.0) / 1.5
        dummy = torch.tensor(0, dtype=torch.long)
        return image, dummy
