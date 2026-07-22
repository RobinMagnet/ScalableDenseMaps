"""Nearest-neighbour utilities: batching, broadcasting, distances."""

import numpy as np
import pytest

from densemaps.numpy.nn_utils import knn_query, compute_sqdistmat
from densemaps.numpy.point_to_triangle import nn_query_precise_np


def test_knn_query_2d():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    Y = rng.standard_normal((10, 4))
    m = knn_query(X, Y)
    assert m.shape == (10,)
    # brute-force check
    ref = ((Y[:, None] - X[None]) ** 2).sum(-1).argmin(-1)
    assert np.array_equal(m, ref)


def test_knn_query_k_and_distance():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    Y = rng.standard_normal((10, 4))
    d, m = knn_query(X, Y, k=3, return_distance=True)
    assert d.shape == (10, 3) and m.shape == (10, 3)
    # nearest distance is the smallest
    assert (np.diff(d, axis=1) >= -1e-9).all()


def test_knn_query_batched():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 30, 4))
    Y = rng.standard_normal((3, 10, 4))
    m = knn_query(X, Y)
    assert m.shape == (3, 10)


@pytest.mark.parametrize("xb,yb", [(1, 3), (3, 1)])
def test_knn_query_broadcast(xb, yb):
    """A batch dim of 1 must broadcast against the other's batch (regression: X.shape==3)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((xb, 30, 4))
    Y = rng.standard_normal((yb, 10, 4))
    m = knn_query(X, Y)
    assert m.shape == (max(xb, yb), 10)


def test_knn_query_mismatched_batch_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        knn_query(rng.standard_normal((2, 30, 4)), rng.standard_normal((3, 10, 4)))


def test_compute_sqdistmat_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 5))
    Y = rng.standard_normal((9, 5))
    d = compute_sqdistmat(X, Y)
    ref = ((X[:, None] - Y[None]) ** 2).sum(-1)
    assert np.allclose(d, ref)


def test_compute_sqdistmat_batched():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 12, 5))
    Y = rng.standard_normal((4, 9, 5))
    assert compute_sqdistmat(X, Y).shape == (4, 12, 9)


def test_nn_query_precise_return_dist():
    """return_dist path must return three arrays (regression: np.linalg.norm dim= kwarg)."""
    rng = np.random.default_rng(0)
    V = rng.standard_normal((30, 4))
    n = 30
    F = np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n - 2)])
    P = rng.standard_normal((10, 4))
    face, bary, dists = nn_query_precise_np(V, F, P, return_dist=True, batch_size=5)
    assert face.shape == (10,) and bary.shape == (10, 3) and dists.shape == (10,)
    assert (dists >= 0).all()
