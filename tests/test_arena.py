import numpy as np
import pytest

from arena.compress import binary, float32, int8, pca, truncate
from arena.data import clustered
from arena.evaluate import evaluate, top_k


@pytest.fixture(scope="module")
def data():
    x = clustered(3000 + 100, 64, seed=3)
    return x[:3000], x[3000:]


def test_float32_is_exact(data):
    corpus, queries = data
    [r] = evaluate([float32(64)], corpus, queries, k=10)
    assert r.recall_at_k == 1.0 and r.bytes_per_vector == 256


def test_bytes_per_vector_reflect_storage():
    assert int8(384).bytes_per_vector == 384
    assert binary(384).bytes_per_vector == 48
    assert pca(384, 96).bytes_per_vector == 384
    assert truncate(384, 96).bytes_per_vector == 384


def test_int8_keeps_most_recall(data):
    corpus, queries = data
    [r] = evaluate([int8(64)], corpus, queries, k=10)
    assert r.recall_at_k > 0.95


def test_pca_recall_grows_with_components(data):
    corpus, queries = data
    small, large = evaluate([pca(64, 8), pca(64, 32)], corpus, queries, k=10)
    assert small.recall_at_k < large.recall_at_k


def test_binary_is_worst_but_better_than_chance(data):
    corpus, queries = data
    reports = {r.name: r.recall_at_k for r in evaluate([binary(64), int8(64)], corpus, queries, k=10)}
    assert 0.2 < reports["binary"] < reports["int8"]


def test_top_k_orders_descending():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k(scores, 3).tolist() == [[1, 3, 2]]


def test_clustered_data_is_anisotropic():
    x = clustered(2000, 64, seed=0)
    var = np.sort(np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2)[::-1]
    assert var[0] / var.sum() > 0.05
