"""Correctness of the point-to-triangle projection.

The vectorized ``point_to_triangles_projection`` is checked against the scalar reference
``pointTriangleDistance`` (a separate, independently-written implementation), which is the
harness that originally exposed the region-4 distance bug.
"""

import numpy as np
import pytest

from densemaps.numpy.point_to_triangle import (
    point_to_triangles_projection,
    pointTriangleDistance,
    project_pc_to_triangles as project_np,
)

torch = pytest.importorskip("torch")
from densemaps.torch.point_to_triangle import project_pc_to_triangles as project_torch


@pytest.mark.parametrize("dim", [3, 5, 8])
def test_vectorized_matches_scalar_reference(dim):
    """Distances AND barycentric coords must match the scalar reference on every pair."""
    rng = np.random.default_rng(1)
    n_trials = 2000
    for _ in range(n_trials):
        tri = rng.standard_normal((1, 3, dim))
        pt = rng.standard_normal(dim) * 2

        d_vec, _, bary_vec = point_to_triangles_projection(tri, pt, return_bary=True)
        d_ref, _, bary_ref = pointTriangleDistance(tri[0], pt, return_bary=True)

        assert np.allclose(d_vec[0], d_ref, atol=1e-8)
        assert np.allclose(bary_vec[0], bary_ref, atol=1e-8)


def test_projection_lands_on_triangle():
    """Barycentric coordinates are valid (in [0,1], summing to 1)."""
    rng = np.random.default_rng(2)
    tri = rng.standard_normal((50, 3, 3))
    pt = rng.standard_normal(3)
    _, _, bary = point_to_triangles_projection(tri, pt, return_bary=True)
    assert np.allclose(bary.sum(-1), 1.0, atol=1e-8)
    assert (bary >= -1e-8).all() and (bary <= 1 + 1e-8).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(precompute_dmin=True),
        dict(precompute_dmin=False, batch_size=16),
        dict(precompute_dmin=False, batch_size=None),
    ],
)
def test_torch_matches_numpy_projected_distance(dtype, kwargs):
    """torch and numpy must agree on the *projected distance* (face ties are legitimate).

    Runs at float32 and float64 (regression for the float32-only buffer bug), and across
    all three projection modes (regression for the Deltamin-index bug and the NameError).
    """
    rng = np.random.default_rng(0)
    V = rng.standard_normal((150, 3)).astype(dtype)
    n = 150
    F = np.array([[i, (i + 1) % n, (i + 7) % n] for i in range(n)])
    P = rng.standard_normal((50, 3)).astype(dtype)

    fn, bn = project_np(V, F, P, precompute_dmin=True)
    ft, bt = project_torch(
        torch.tensor(V), torch.tensor(F), torch.tensor(P), **kwargs
    )
    ft = ft.numpy()
    bt = bt.numpy()

    # Reconstruct the projected points and compare distances to the query points.
    proj_n = (bn[..., None] * V[F[fn]]).sum(1)
    proj_t = (bt[..., None] * V[F[ft]]).sum(1)
    dn = np.linalg.norm(proj_n - P, axis=1)
    dt = np.linalg.norm(proj_t - P, axis=1)

    tol = 1e-4 if dtype == np.float32 else 1e-9
    assert np.abs(dn - dt).max() < tol
