"""PreciseMap: n1 handling, stochasticity, sparsity."""

import numpy as np
import pytest
import scipy.sparse as sparse

from densemaps.numpy import maps as nm


def _fan_faces(n_faced):
    return np.array(
        [[i, (i + 1) % n_faced, (i + 2) % n_faced] for i in range(n_faced - 2)]
    )


def test_precise_map_keeps_trailing_unreferenced_vertices():
    """n1 must come from the embedding, not faces.max()+1 (regression for dropped verts)."""
    rng = np.random.default_rng(0)
    n_total, n_faced = 40, 30
    V = rng.standard_normal((n_total, 3))
    F = _fan_faces(n_faced)  # references only vertices 0..29
    P = rng.standard_normal((10, 3))

    M = nm.EmbPreciseMap(V, P, F)
    assert M.shape == (10, n_total)
    # pull_back of a function defined on all n_total vertices must work
    f = rng.standard_normal((n_total, 2))
    assert (M @ f).shape == (10, 2)


def test_precise_map_rows_are_stochastic_and_sparse():
    rng = np.random.default_rng(1)
    V = rng.standard_normal((30, 4))
    F = _fan_faces(30)
    P = rng.standard_normal((15, 3)) @ rng.standard_normal((3, 4))  # in the span
    M = nm.EmbPreciseMap(V, P, F)
    dense = M.to_dense()
    assert np.allclose(dense.sum(1), 1.0, atol=1e-8)
    # at most 3 non-zeros per row
    assert (np.count_nonzero(dense, axis=1) <= 3).all()


def test_precise_map_point_on_vertex_is_onehot():
    """A query embedding equal to a vertex embedding recovers a (near) one-hot row."""
    rng = np.random.default_rng(2)
    V = rng.standard_normal((30, 4))
    F = _fan_faces(30)
    # place queries exactly on vertices that belong to a face
    P = V[[0, 5, 10]]
    M = nm.EmbPreciseMap(V, P, F)
    dense = M.to_dense()
    assert np.allclose(dense.max(1), 1.0, atol=1e-6)


torch = pytest.importorskip("torch")
from densemaps.torch import maps as tm


def test_torch_precise_map_matches_numpy():
    rng = np.random.default_rng(3)
    V = rng.standard_normal((30, 4))
    F = _fan_faces(30)
    P = rng.standard_normal((12, 4))
    Mn = nm.EmbPreciseMap(V, P, F)
    Mt = tm.EmbPreciseMap(torch.tensor(V), torch.tensor(P), torch.tensor(F))
    assert tuple(Mt.shape) == Mn.shape
    assert isinstance(Mt.n1, int)
    f = rng.standard_normal((30, 2))
    assert np.allclose(Mn @ f, (Mt @ torch.tensor(f)).numpy(), atol=1e-10)


def test_torch_precise_map_composes():
    """PreciseMap @ P2PMap must work (regression for missing _to_sparse)."""
    rng = np.random.default_rng(4)
    V = rng.standard_normal((20, 3))
    F = _fan_faces(20)
    P = rng.standard_normal((8, 3))
    M = tm.EmbPreciseMap(torch.tensor(V), torch.tensor(P), torch.tensor(F))
    C = M @ tm.P2PMap(torch.randint(0, 5, (20,)), n1=5)
    assert tuple(C.shape) == (8, 5)
