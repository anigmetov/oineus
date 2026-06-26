"""Finite-difference gradient + optimizer-dispatch checks for the bit-packed VR
differentiable path (oineus.diff.vr_filtration(packed=True)). The packed builder
returns the same per-cell critical edges as the fat one, so the differentiable
distances -- and hence the gradients -- must match; and oineus.diff.TopologyOptimizer
must dispatch the packed type to TopologyOptimizerPacked_64.
"""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import oineus as oin
import oineus.diff as od
from oineus._dtype import REAL_DTYPE


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")


if REAL_DTYPE == np.float32:
    TORCH_DTYPE = torch.float32 if HAS_TORCH else None
    EPS = 1e-3
    ATOL = 1e-2
    RTOL = 1e-2
    GRAD_NONZERO_SQ = 1e-6
else:
    TORCH_DTYPE = torch.float64 if HAS_TORCH else None
    EPS = 1e-6
    ATOL = 1e-5
    RTOL = 1e-5
    GRAD_NONZERO_SQ = 1e-10


def _assert_grad_nonzero(*grads):
    for g in grads:
        assert float(np.sum(np.asarray(g) ** 2)) > GRAD_NONZERO_SQ, \
            "gradient is (numerically) zero -- test would pass trivially"


def _fd_grad(f, x_np, eps=EPS):
    g = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        base = x_np.copy()
        base[idx] += eps
        hi = float(f(base))
        base[idx] -= 2 * eps
        lo = float(f(base))
        g[idx] = (hi - lo) / (2 * eps)
    return g


def test_vr_packed_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    pts_np = rng.uniform(-1.0, 1.0, size=(5, 2)).astype(REAL_DTYPE)
    pts = torch.tensor(pts_np, dtype=TORCH_DTYPE, requires_grad=True)

    df = od.vr_filtration(pts, max_dim=1, max_diameter=10.0, packed=True, n_threads=1)
    assert type(df.under_fil).__name__ == "PackedSimplexFiltration_64"
    ((df.values ** 2).sum()).backward()
    grad_auto = pts.grad.detach().numpy()
    _assert_grad_nonzero(grad_auto)

    def f(x):
        t = torch.tensor(x, dtype=TORCH_DTYPE)
        d = od.vr_filtration(t, max_dim=1, max_diameter=10.0, packed=True, n_threads=1)
        return float((d.values ** 2).sum())

    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)


def test_vr_packed_gradient_matches_fat():
    # the packed and fat diff filtrations must produce identical differentiable values
    # (same critical edges, same cell order), so their gradients coincide exactly
    rng = np.random.default_rng(11)
    pts_np = rng.uniform(-1.0, 1.0, size=(6, 2)).astype(REAL_DTYPE)

    def grad(packed):
        pts = torch.tensor(pts_np, dtype=TORCH_DTYPE, requires_grad=True)
        d = od.vr_filtration(pts, max_dim=2, max_diameter=10.0, packed=packed, n_threads=1)
        ((d.values ** 2).sum()).backward()
        return pts.grad.detach().numpy()

    gp = grad(True)
    gf = grad(False)
    _assert_grad_nonzero(gp)
    np.testing.assert_allclose(gp, gf, atol=ATOL, rtol=RTOL)


def test_vr_packed_optimizer_dispatches():
    rng = np.random.default_rng(3)
    pts = torch.tensor(rng.uniform(-1.0, 1.0, size=(8, 2)), dtype=TORCH_DTYPE, requires_grad=True)
    df = od.vr_filtration(pts, max_dim=1, max_diameter=10.0, packed=True, n_threads=1)
    top = od.TopologyOptimizer(df)
    assert type(top.under_opt).__name__ == "TopologyOptimizerPacked_64"
    top.reduce_all()
    dgm = top.compute_diagram(include_inf_points=False)
    # a connected 8-point cloud has a non-trivial H0 (the merges)
    assert np.asarray(dgm.in_dimension(0)).shape[0] >= 1
