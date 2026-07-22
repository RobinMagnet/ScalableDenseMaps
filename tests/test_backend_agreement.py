"""NumPy and PyTorch backends must produce the same maps from the same inputs.

Run at float32 and float64 (regression for the float32-only blur buffer that produced
all-NaN results on float64 keops maps).
"""

import numpy as np
import pytest

from densemaps.numpy import maps as nm

torch = pytest.importorskip("torch")
from densemaps.torch import maps as tm

N1, N2 = 30, 20


@pytest.fixture(params=[np.float32, np.float64], ids=["float32", "float64"])
def dtype(request):
    return request.param


def _embeddings(dtype, dim=5):
    rng = np.random.default_rng(3)
    return (
        rng.standard_normal((N1, dim)).astype(dtype),
        rng.standard_normal((N2, dim)).astype(dtype),
    )


@pytest.mark.parametrize("dist_type", ["sqdist", "inner"])
def test_kernel_dense_agreement(dtype, dist_type):
    e1, e2 = _embeddings(dtype)
    f = np.random.default_rng(4).standard_normal((N1, 3)).astype(dtype)

    Kn = nm.EmbKernelDenseDistMap(e1, e2, blur=0.3, dist_type=dist_type)
    Kt = tm.EmbKernelDenseDistMap(torch.tensor(e1), torch.tensor(e2), blur=0.3, dist_type=dist_type)

    tol = 1e-4 if dtype == np.float32 else 1e-9
    assert np.allclose(Kn @ f, (Kt @ torch.tensor(f)).numpy(), atol=tol)
    assert np.array_equal(Kn.get_nn(), Kt.get_nn().numpy())
    assert np.allclose(Kn.to_dense(), Kt.to_dense().numpy(), atol=tol)
    assert not np.isnan((Kt @ torch.tensor(f)).numpy()).any()


def test_p2p_agreement(dtype):
    e1, e2 = _embeddings(dtype)
    f = np.random.default_rng(4).standard_normal((N1, 3)).astype(dtype)
    Pn = nm.EmbP2PMap(e1, e2)
    Pt = tm.EmbP2PMap(torch.tensor(e1), torch.tensor(e2))
    assert np.array_equal(Pn.get_nn(), Pt.get_nn().numpy())
    assert np.allclose(Pn @ f, (Pt @ torch.tensor(f)).numpy(), atol=1e-10)


@pytest.mark.parametrize("dist_type", ["sqdist", "inner"])
def test_scalable_kernel_matches_dense(dtype, dist_type):
    """KernelDistMap (keops) must match the dense EmbKernelDenseDistMap, at both dtypes."""
    pytest.importorskip("pykeops")
    e1, e2 = _embeddings(dtype)
    f = np.random.default_rng(4).standard_normal((N1, 3)).astype(dtype)

    K = tm.KernelDistMap(torch.tensor(e1), torch.tensor(e2), blur=0.3, dist_type=dist_type)
    D = tm.EmbKernelDenseDistMap(torch.tensor(e1), torch.tensor(e2), blur=0.3, dist_type=dist_type)

    out = K @ torch.tensor(f)
    assert not torch.isnan(out).any()
    tol = 1e-3 if dtype == np.float32 else 1e-6
    assert torch.allclose(out, D @ torch.tensor(f), atol=tol)
    assert torch.equal(K.get_nn(), D.get_nn())
