"""Finite-difference gradient checks for oineus.diff.

Each test builds a minimal differentiable object from a torch tensor,
computes a smooth scalar loss, runs backward, and compares against a
central finite difference. Sizes are tiny so the suite stays well under
a second per test.
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


# Tie test precision to the compiled Real. Central differences are
# O(eps^2) accurate; the tolerances below are loose enough to absorb
# float32 round-trip noise when Real=float.
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
    """Guard against a test silently passing with an all-zero gradient."""
    for g in grads:
        assert float(np.sum(np.asarray(g) ** 2)) > GRAD_NONZERO_SQ, \
            "gradient is (numerically) zero — test would pass trivially"


def _fd_grad(f, x_np, eps=EPS):
    """Central-difference gradient of scalar function ``f(x_np) -> float``."""
    g = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        base = x_np.copy()
        base[idx] += eps
        hi = float(f(base))
        base[idx] -= 2 * eps
        lo = float(f(base))
        g[idx] = (hi - lo) / (2 * eps)
    return g


# ---------------------------------------------------------------------------
# Grid-valued filtrations
# ---------------------------------------------------------------------------

def test_freudenthal_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    data_np = rng.uniform(-1.0, 1.0, size=(3, 3)).astype(REAL_DTYPE)
    data = torch.tensor(data_np, dtype=TORCH_DTYPE, requires_grad=True)

    df = od.freudenthal_filtration(data, negate=False, wrap=False, max_dim=2, n_threads=1)
    ((df.values ** 2).sum()).backward()
    grad_auto = data.grad.detach().numpy()
    _assert_grad_nonzero(grad_auto)

    def f(x):
        t = torch.tensor(x, dtype=TORCH_DTYPE)
        d = od.freudenthal_filtration(t, negate=False, wrap=False, max_dim=2, n_threads=1)
        return float((d.values ** 2).sum())

    np.testing.assert_allclose(grad_auto, _fd_grad(f, data_np), atol=ATOL, rtol=RTOL)


def test_cube_gradient_matches_finite_difference():
    """Mirror of the existing cube-only FD check, kept here for parity with the rest."""
    rng = np.random.default_rng(1)
    data_np = rng.uniform(-1.0, 1.0, size=(3, 3)).astype(REAL_DTYPE)
    data = torch.tensor(data_np, dtype=TORCH_DTYPE, requires_grad=True)

    df = od.cube_filtration(data, max_dim=2)
    ((df.values ** 2).sum()).backward()
    grad_auto = data.grad.detach().numpy()
    _assert_grad_nonzero(grad_auto)

    def f(x):
        t = torch.tensor(x, dtype=TORCH_DTYPE)
        return float((od.cube_filtration(t, max_dim=2).values ** 2).sum())

    np.testing.assert_allclose(grad_auto, _fd_grad(f, data_np), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Vietoris-Rips
# ---------------------------------------------------------------------------

def test_vr_from_points_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    pts_np = rng.uniform(-1.0, 1.0, size=(5, 2)).astype(REAL_DTYPE)
    pts = torch.tensor(pts_np, dtype=TORCH_DTYPE, requires_grad=True)

    df = od.vr_filtration(pts, max_dim=1, max_diameter=10.0, n_threads=1)
    ((df.values ** 2).sum()).backward()
    grad_auto = pts.grad.detach().numpy()
    _assert_grad_nonzero(grad_auto)

    def f(x):
        t = torch.tensor(x, dtype=TORCH_DTYPE)
        d = od.vr_filtration(t, max_dim=1, max_diameter=10.0, n_threads=1)
        return float((d.values ** 2).sum())

    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)


def test_vr_from_pwdists_gradient_matches_finite_difference():
    rng = np.random.default_rng(3)
    n = 4
    base = rng.uniform(0.3, 1.0, size=(n, n)).astype(REAL_DTYPE)
    d_np = (base + base.T) / 2
    np.fill_diagonal(d_np, 0.0)

    d = torch.tensor(d_np, dtype=TORCH_DTYPE, requires_grad=True)
    df = od.vr_filtration(d, from_pwdists=True, max_dim=1, max_diameter=10.0, n_threads=1)
    ((df.values ** 2).sum()).backward()
    grad_auto = d.grad.detach().numpy()
    _assert_grad_nonzero(grad_auto)

    def f(x):
        t = torch.tensor(x, dtype=TORCH_DTYPE)
        d = od.vr_filtration(t, from_pwdists=True, max_dim=1, max_diameter=10.0, n_threads=1)
        return float((d.values ** 2).sum())

    np.testing.assert_allclose(grad_auto, _fd_grad(f, d_np), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# min_filtration and mapping_cylinder_filtration on a hand-built segment
# ---------------------------------------------------------------------------

def _build_segment_diff_fil(values):
    """Turn a 3-element torch tensor into a DiffFiltration for a line segment.

    Simplices: vertex 0, vertex 1, edge [0,1], in that filtration order.
    Caller must supply strictly increasing values so the underlying C++
    Filtration's sort is a no-op and ``values[i]`` maps to sorted index ``i``.
    """
    vals = values.detach().numpy().astype(float).tolist()
    simps = [(0, [0], vals[0]), (1, [1], vals[1]), (2, [0, 1], vals[2])]
    under = oin.list_to_filtration(simps)
    return od.DiffFiltration(under, values)


def test_min_filtration_gradient_matches_finite_difference():
    v1_np = np.array([0.10, 0.20, 0.30], dtype=REAL_DTYPE)
    v2_np = np.array([0.15, 0.18, 0.35], dtype=REAL_DTYPE)
    v1 = torch.tensor(v1_np, dtype=TORCH_DTYPE, requires_grad=True)
    v2 = torch.tensor(v2_np, dtype=TORCH_DTYPE, requires_grad=True)

    df_min = od.min_filtration(_build_segment_diff_fil(v1), _build_segment_diff_fil(v2))
    ((df_min.values ** 2).sum()).backward()
    g1_auto = v1.grad.detach().numpy()
    g2_auto = v2.grad.detach().numpy()
    _assert_grad_nonzero(g1_auto, g2_auto)

    def f(a, b):
        ta = torch.tensor(a, dtype=TORCH_DTYPE)
        tb = torch.tensor(b, dtype=TORCH_DTYPE)
        df = od.min_filtration(_build_segment_diff_fil(ta), _build_segment_diff_fil(tb))
        return float((df.values ** 2).sum())

    np.testing.assert_allclose(g1_auto, _fd_grad(lambda a: f(a, v2_np), v1_np), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(g2_auto, _fd_grad(lambda b: f(v1_np, b), v2_np), atol=ATOL, rtol=RTOL)


def test_mapping_cylinder_gradient_matches_finite_difference():
    v_dom_np = np.array([0.10, 0.20, 0.30], dtype=REAL_DTYPE)
    v_cod_np = np.array([0.15, 0.25, 0.35], dtype=REAL_DTYPE)
    v_dom = torch.tensor(v_dom_np, dtype=TORCH_DTYPE, requires_grad=True)
    v_cod = torch.tensor(v_cod_np, dtype=TORCH_DTYPE, requires_grad=True)

    df_dom = _build_segment_diff_fil(v_dom)
    df_cod = _build_segment_diff_fil(v_cod)

    # Apex vertices must carry ids disjoint from the existing simplices.
    v_dom_id = df_dom.size() + df_cod.size()
    v_cod_id = v_dom_id + 1
    apex_dom = oin.Simplex([v_dom_id])
    apex_cod = oin.Simplex([v_cod_id])

    fil = od.mapping_cylinder_filtration(df_dom, df_cod, apex_dom, apex_cod)
    ((fil.values ** 2).sum()).backward()
    g_dom_auto = v_dom.grad.detach().numpy()
    g_cod_auto = v_cod.grad.detach().numpy()
    _assert_grad_nonzero(g_dom_auto, g_cod_auto)

    def f(a, b):
        ta = torch.tensor(a, dtype=TORCH_DTYPE)
        tb = torch.tensor(b, dtype=TORCH_DTYPE)
        df_a = _build_segment_diff_fil(ta)
        df_b = _build_segment_diff_fil(tb)
        va_id = df_a.size() + df_b.size()
        vb_id = va_id + 1
        return float((od.mapping_cylinder_filtration(
            df_a, df_b, oin.Simplex([va_id]), oin.Simplex([vb_id])
        ).values ** 2).sum())

    np.testing.assert_allclose(g_dom_auto, _fd_grad(lambda a: f(a, v_cod_np), v_dom_np), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(g_cod_auto, _fd_grad(lambda b: f(v_dom_np, b), v_cod_np), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# End-to-end: persistence_diagram through VR
# ---------------------------------------------------------------------------

def test_persistence_diagram_h0_gradient_matches_finite_difference():
    rng = np.random.default_rng(4)
    pts_np = rng.uniform(-1.0, 1.0, size=(5, 2)).astype(REAL_DTYPE)
    pts = torch.tensor(pts_np, dtype=TORCH_DTYPE, requires_grad=True)

    fil = od.vr_filtration(pts, max_dim=1, max_diameter=10.0, n_threads=1)
    dgms = od.persistence_diagram(fil, dualize=True)
    dgm0 = dgms[0]
    ((dgm0[:, 1] - dgm0[:, 0]) ** 2).sum().backward()
    grad_auto = pts.grad.detach().numpy()
    _assert_grad_nonzero(grad_auto)

    def f(x):
        t = torch.tensor(x, dtype=TORCH_DTYPE)
        fil = od.vr_filtration(t, max_dim=1, max_diameter=10.0, n_threads=1)
        dgms = od.persistence_diagram(fil, dualize=True)
        d0 = dgms[0]
        return float(((d0[:, 1] - d0[:, 0]) ** 2).sum())

    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)
