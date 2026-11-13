"""Recall of compressed search against exact float32 search."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from arena.compress import Compressor, fit, float32


@dataclass(frozen=True)
class Report:
    name: str
    bytes_per_vector: int
    recall_at_k: float
    queries_per_second: float


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, idx], axis=1)
    return idx[rows, order]


def evaluate(compressors: list[Compressor], corpus: np.ndarray, queries: np.ndarray, k: int = 10,
             baseline: Compressor | None = None) -> list[Report]:
    """Score each compressor on recall@k against exact float32 cosine search."""
    dim = corpus.shape[1]
    base = baseline or float32(dim)
    truth = top_k(base.score(queries, base.encode(corpus)), k)
    reports = []
    for c in compressors:
        fit(c, corpus)
        stored = c.encode(corpus)
        q_enc = queries
        start = time.perf_counter()
        got = top_k(c.score(q_enc, stored), k)
        elapsed = time.perf_counter() - start
        hits = np.array([len(set(g) & set(t)) for g, t in zip(got, truth, strict=True)])
        reports.append(Report(c.name, c.bytes_per_vector, float(hits.sum() / (k * len(queries))),
                              len(queries) / max(elapsed, 1e-9)))
    return reports
