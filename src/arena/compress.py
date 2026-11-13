"""Embedding compressors. Each one fits on a corpus, then encodes vectors into a
compact form and scores queries against it without decompressing everything.

Scores are returned as "higher is better" so every method ranks the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1, n)


@dataclass
class Compressor:
    """A fitted encoder: ``encode`` maps float rows to a stored form, ``score`` ranks
    stored rows against float queries, and ``bytes_per_vector`` reports the stored size."""

    name: str
    encode: Callable[[np.ndarray], np.ndarray]
    score: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (queries, stored) -> (q, n)
    bytes_per_vector: int


def float32(dim: int) -> Compressor:
    def encode(x):
        return _unit(x).astype(np.float32)

    def score(q, s):
        return _unit(q).astype(np.float32) @ s.T

    return Compressor("float32", encode, score, dim * 4)


def int8(dim: int) -> Compressor:
    """Per-dimension symmetric scalar quantisation of unit vectors to int8."""
    scale_holder: dict[str, np.ndarray] = {}

    def fit(x):
        scale_holder["s"] = np.abs(_unit(x)).max(axis=0) + 1e-9

    def encode(x):
        return np.round(_unit(x) / scale_holder["s"] * 127).astype(np.int8)

    def score(q, s):
        qf = _unit(q) * scale_holder["s"]
        return (qf @ s.T.astype(np.float32)) / 127.0

    c = Compressor("int8", encode, score, dim)
    c.fit = fit  # type: ignore[attr-defined]
    return c


def binary(dim: int) -> Compressor:
    """One bit per dimension (sign after centring); scored by Hamming similarity."""
    centre: dict[str, np.ndarray] = {}

    def fit(x):
        centre["m"] = _unit(x).mean(axis=0)

    def encode(x):
        return np.packbits((_unit(x) - centre["m"]) > 0, axis=-1)

    def score(q, s):
        qb = np.packbits((_unit(q) - centre["m"]) > 0, axis=-1)
        # Hamming distance via popcount of XOR, then turn it into similarity.
        x = np.bitwise_xor(qb[:, None, :], s[None, :, :])
        ham = np.unpackbits(x, axis=-1).sum(axis=-1)
        return dim - 2.0 * ham

    c = Compressor("binary", encode, score, dim // 8)
    c.fit = fit  # type: ignore[attr-defined]
    return c


def pca(dim: int, k: int) -> Compressor:
    """Project onto the top-k principal components, then keep float32."""
    proj: dict[str, np.ndarray] = {}

    def fit(x):
        u = _unit(x)
        mu = u.mean(axis=0)
        _, _, vt = np.linalg.svd(u - mu, full_matrices=False)
        proj["mu"], proj["w"] = mu, vt[:k].T

    def encode(x):
        return ((_unit(x) - proj["mu"]) @ proj["w"]).astype(np.float32)

    def score(q, s):
        return ((_unit(q) - proj["mu"]) @ proj["w"]) @ s.T

    c = Compressor(f"pca-{k}", encode, score, k * 4)
    c.fit = fit  # type: ignore[attr-defined]
    return c


def truncate(dim: int, k: int) -> Compressor:
    """Keep the first k dimensions and renormalise (Matryoshka-style prefixes)."""

    def encode(x):
        return _unit(x[:, :k]).astype(np.float32)

    def score(q, s):
        return _unit(q[:, :k]).astype(np.float32) @ s.T

    return Compressor(f"truncate-{k}", encode, score, k * 4)


def fit(c: Compressor, corpus: np.ndarray) -> Compressor:
    """Fit a compressor if it has a fit step (int8, binary, pca), then return it."""
    if hasattr(c, "fit"):
        c.fit(corpus)  # type: ignore[attr-defined]
    return c
