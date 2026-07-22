"""Shared fixtures / helpers for the densemaps test suite."""

import numpy as np
import pytest


def make_triangle_mesh(n_vertices, dim=3, seed=0):
    """A random ``(vertices, faces)`` mesh in ``dim`` dimensions.

    The faces reference every vertex at least once (a simple fan-like triangulation),
    unless ``n_vertices`` is larger than the number of faced vertices, in which case the
    trailing vertices are intentionally left unreferenced (used to test ``n1`` handling).
    """
    rng = np.random.default_rng(seed)
    verts = rng.standard_normal((n_vertices, dim))
    n_faced = min(n_vertices, n_vertices - 0)
    faces = np.array(
        [[i, (i + 1) % n_faced, (i + 2) % n_faced] for i in range(n_faced - 2)]
    )
    return verts, faces


@pytest.fixture
def rng():
    return np.random.default_rng(0)
