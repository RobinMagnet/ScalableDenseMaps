"""Shape / dtype contract for every map class in both backends.

Covers pull_back over the (N1,), (N1,p), (B,N1,p) function shapes, shape/ndim, get_nn,
composition (regression for the dead-code composition branch), and the mT / reverse
round-trips.
"""

import numpy as np
import pytest

from densemaps.numpy import maps as nm

N1, N2, P = 25, 18, 4


def _faces(n_faced):
    return np.array([[i, (i + 1) % n_faced, (i + 2) % n_faced] for i in range(n_faced - 2)])


# ---------------------------------------------------------------- numpy backend


def _numpy_maps(rng):
    e1 = rng.standard_normal((N1, 5))
    e2 = rng.standard_normal((N2, 5))
    V = rng.standard_normal((N1, 5))
    F = _faces(N1)
    return {
        "P2PMap": nm.P2PMap(rng.integers(0, N1, N2), n1=N1),
        "SparseMap": nm.SparseMap(nm.P2PMap(rng.integers(0, N1, N2), n1=N1)._to_sparse()),
        "PreciseMap": nm.EmbPreciseMap(V, e2, F),
        "KernelDense": nm.EmbKernelDenseDistMap(e1, e2, blur=0.3),
    }


@pytest.fixture
def np_maps():
    return _numpy_maps(np.random.default_rng(0))


@pytest.mark.parametrize("name", ["P2PMap", "SparseMap", "PreciseMap", "KernelDense"])
def test_numpy_shape_and_ndim(np_maps, name):
    m = np_maps[name]
    assert tuple(m.shape) == (N2, N1)
    assert m.ndim == 2


def test_numpy_is_index_map_flag(np_maps):
    assert np_maps["P2PMap"].is_index_map is True
    for name in ["SparseMap", "PreciseMap", "KernelDense"]:
        assert np_maps[name].is_index_map is False
    # the raw index array is still (N2,)
    assert np_maps["P2PMap"].p2p_21.shape == (N2,)


@pytest.mark.parametrize("name", ["P2PMap", "SparseMap", "PreciseMap", "KernelDense"])
def test_numpy_pull_back_shapes(np_maps, name):
    rng = np.random.default_rng(1)
    m = np_maps[name]
    assert (m @ rng.standard_normal(N1)).shape == (N2,)
    assert (m @ rng.standard_normal((N1, P))).shape == (N2, P)


@pytest.mark.parametrize("name", ["P2PMap", "SparseMap", "PreciseMap", "KernelDense"])
def test_numpy_get_nn(np_maps, name):
    m = np_maps[name]
    nn = m.get_nn()
    assert nn.shape == (N2,)
    assert nn.max() < N1


@pytest.mark.parametrize("name", ["P2PMap", "SparseMap", "PreciseMap", "KernelDense"])
def test_numpy_compose_then_get_nn(np_maps, name):
    """(A @ B).get_nn() must work — A @ B always yields a SparseMap."""
    rng = np.random.default_rng(2)
    m = np_maps[name]
    other = nm.P2PMap(rng.integers(0, 6, N1), n1=6)  # (N1 -> 6)
    composed = m @ other
    assert composed.shape == (N2, 6)
    assert composed.get_nn().shape == (N2,)


def test_numpy_kernel_mT_and_reverse():
    rng = np.random.default_rng(3)
    m = nm.EmbKernelDenseDistMap(rng.standard_normal((N1, 5)), rng.standard_normal((N2, 5)), blur=0.3)
    # reverse is row-stochastic (N1, N2); mT is the literal transpose
    rev = m.reverse()
    assert rev.shape == (N1, N2)
    assert np.allclose(rev.to_dense().sum(1), 1.0)
    assert np.allclose(m.mT.to_dense(), m.to_dense().T)
    # mT.mT round trip
    assert np.allclose(m.mT.mT.to_dense(), m.to_dense())


# ---------------------------------------------------------------- torch backend

torch = pytest.importorskip("torch")
from densemaps.torch import maps as tm


def _torch_maps(rng):
    e1 = torch.tensor(rng.standard_normal((N1, 5)))
    e2 = torch.tensor(rng.standard_normal((N2, 5)))
    V = torch.tensor(rng.standard_normal((N1, 5)))
    F = torch.tensor(_faces(N1))
    return {
        "P2PMap": tm.P2PMap(torch.randint(0, N1, (N2,)), n1=N1),
        "SparseMap": tm.SparseMap(tm.P2PMap(torch.randint(0, N1, (N2,)), n1=N1)._to_sparse()),
        "PreciseMap": tm.EmbPreciseMap(V, e2, F),
        "KernelDense": tm.EmbKernelDenseDistMap(e1, e2, blur=0.3),
    }


@pytest.fixture
def th_maps():
    return _torch_maps(np.random.default_rng(0))


def test_torch_is_index_map_flag(th_maps):
    assert th_maps["P2PMap"].is_index_map is True
    for name in ["SparseMap", "PreciseMap", "KernelDense"]:
        assert th_maps[name].is_index_map is False
    assert tuple(th_maps["P2PMap"].shape) == (N2, N1)


@pytest.mark.parametrize("name", ["P2PMap", "SparseMap", "PreciseMap", "KernelDense"])
def test_torch_pull_back_shapes(th_maps, name):
    rng = np.random.default_rng(1)
    m = th_maps[name]
    assert tuple((m @ torch.tensor(rng.standard_normal(N1))).shape) == (N2,)
    assert tuple((m @ torch.tensor(rng.standard_normal((N1, P)))).shape) == (N2, P)


@pytest.mark.parametrize("name", ["P2PMap", "SparseMap", "PreciseMap", "KernelDense"])
def test_torch_get_nn(th_maps, name):
    m = th_maps[name]
    nn = m.get_nn()
    assert tuple(nn.shape) == (N2,)
    assert nn.dtype == torch.long


@pytest.mark.parametrize("name", ["P2PMap", "SparseMap", "PreciseMap", "KernelDense"])
def test_torch_compose_then_get_nn(th_maps, name):
    m = th_maps[name]
    other = tm.P2PMap(torch.randint(0, 6, (N1,)), n1=6)
    composed = m @ other
    assert tuple(composed.shape) == (N2, 6)
    assert tuple(composed.get_nn().shape) == (N2,)


def test_torch_batched_dense_compose_p2p():
    """Batched dense @ batched p2p must pair batches (regression for ~ bug)."""
    B = 3
    K = tm.KernelDenseDistMap(torch.randn(B, N2, N1))
    Pm = tm.P2PMap(torch.randint(0, 6, (B, N1)), n1=6)
    C = K @ Pm
    assert tuple(C.shape) == (B, N2, 6)
    ref = torch.stack([K.to_dense()[i] @ Pm._to_sparse()[i].to_dense() for i in range(B)])
    got = C._to_sparse()
    got = got.to_dense() if got.layout is torch.sparse_coo else got
    assert torch.allclose(got, ref, atol=1e-5)
