"""Synthetic embeddings with the structure real ones have: clusters on a
low-dimensional manifold plus isotropic noise, so compressors see realistic
anisotropy (a few directions carry most variance)."""

from __future__ import annotations

import numpy as np


def clustered(n: int, dim: int, clusters: int = 40, intrinsic: int = 24, noise: float = 0.15,
              seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(dim, intrinsic)))[0].T  # intrinsic x dim, orthonormal rows
    scales = np.linspace(2.0, 0.2, intrinsic)  # decaying variance across directions
    centres = (rng.normal(size=(clusters, intrinsic)) * scales) @ basis
    labels = rng.integers(clusters, size=n)
    latent = rng.normal(size=(n, intrinsic)) * scales * 0.5
    x = centres[labels] + latent @ basis + noise * rng.normal(size=(n, dim))
    return x.astype(np.float32)
