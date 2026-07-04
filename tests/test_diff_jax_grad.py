"""Finite-difference gradient checks for oineus.diff with JAX inputs.

Mirror of test_diff_grad.py: each test builds a differentiable object
from a jax array, computes a smooth scalar loss inside jax.grad, and
compares against a central finite difference. x64 is enabled at module
import so the whole file runs in float64 (float32 routing is covered by
test_diff_jax_f32.py). Skips entirely when jax is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

try:
    import diode
    HAS_DIODE = True
except ImportError:
    HAS_DIODE = False

import oineus as oin
import oineus.diff as od
from oineus.diff._backend import concrete_numpy


EPS = 1e-6
ATOL = 1e-5
RTOL = 1e-5
GRAD_NONZERO_SQ = 1e-10


def _assert_grad_nonzero(*grads):
    """Guard against a test silently passing with an all-zero gradient."""
    for g in grads:
        assert float(np.sum(np.asarray(g) ** 2)) > GRAD_NONZERO_SQ, \
            "gradient is (numerically) zero -- test would pass trivially"


def _fd_grad(f, x_np, eps=EPS):
    """Central-difference gradient of scalar function ``f(jax array) -> float``."""
    g = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        base = x_np.copy()
        base[idx] += eps
        hi = float(f(jnp.asarray(base)))
        base[idx] -= 2 * eps
        lo = float(f(jnp.asarray(base)))
        g[idx] = (hi - lo) / (2 * eps)
    return g


# ---------------------------------------------------------------------------
# float64 routing under x64
# ---------------------------------------------------------------------------

def test_x64_float64_routes_to_default_backend():
    data = jnp.asarray(np.linspace(0.0, 1.0, 9).reshape(3, 3))
    assert data.dtype == np.float64
    df = od.freudenthal_filtration(data, max_dim=2)
    # float64 arrays route to the default (top-module) float64 backend
    assert type(df.under_fil).__module__ == "oineus._oineus"
    assert str(df.values.dtype) == "float64"


# ---------------------------------------------------------------------------
# Grid-valued filtrations
# ---------------------------------------------------------------------------

def test_freudenthal_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    data_np = rng.uniform(-1.0, 1.0, size=(3, 3))

    def f(x):
        d = od.freudenthal_filtration(x, negate=False, wrap=False, max_dim=2, n_threads=1)
        return (d.values ** 2).sum()

    grad_auto = np.asarray(jax.grad(f)(jnp.asarray(data_np)))
    _assert_grad_nonzero(grad_auto)
    np.testing.assert_allclose(grad_auto, _fd_grad(f, data_np), atol=ATOL, rtol=RTOL)


def test_cube_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    data_np = rng.uniform(-1.0, 1.0, size=(3, 3))

    def f(x):
        return (od.cube_filtration(x, max_dim=2).values ** 2).sum()

    grad_auto = np.asarray(jax.grad(f)(jnp.asarray(data_np)))
    _assert_grad_nonzero(grad_auto)
    np.testing.assert_allclose(grad_auto, _fd_grad(f, data_np), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Vietoris-Rips
# ---------------------------------------------------------------------------

def test_vr_from_points_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    pts_np = rng.uniform(-1.0, 1.0, size=(5, 2))

    def f(x):
        d = od.vr_filtration(x, max_dim=1, max_diameter=10.0, n_threads=1)
        return (d.values ** 2).sum()

    grad_auto = np.asarray(jax.grad(f)(jnp.asarray(pts_np)))
    _assert_grad_nonzero(grad_auto)
    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)


def test_vr_from_pwdists_gradient_matches_finite_difference():
    rng = np.random.default_rng(3)
    n = 4
    base = rng.uniform(0.3, 1.0, size=(n, n))
    d_np = (base + base.T) / 2
    np.fill_diagonal(d_np, 0.0)

    def f(x):
        d = od.vr_filtration(x, from_pwdists=True, max_dim=1, max_diameter=10.0, n_threads=1)
        return (d.values ** 2).sum()

    grad_auto = np.asarray(jax.grad(f)(jnp.asarray(d_np)))
    _assert_grad_nonzero(grad_auto)
    np.testing.assert_allclose(grad_auto, _fd_grad(f, d_np), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# min_filtration and mapping_cylinder_filtration on a hand-built segment
# ---------------------------------------------------------------------------

def _build_segment_diff_fil(values):
    """Turn a 3-element jax array into a DiffFiltration for a line segment.

    Simplices: vertex 0, vertex 1, edge [0,1], in that filtration order.
    Caller must supply strictly increasing values so the underlying C++
    Filtration's sort is a no-op and ``values[i]`` maps to sorted index ``i``.
    """
    vals = concrete_numpy(values).astype(float).tolist()
    simps = [(0, [0], vals[0]), (1, [1], vals[1]), (2, [0, 1], vals[2])]
    under = oin.list_to_filtration(simps)
    return od.DiffFiltration(under, values)


def test_min_filtration_gradient_matches_finite_difference():
    v1_np = np.array([0.10, 0.20, 0.30])
    v2_np = np.array([0.15, 0.18, 0.35])

    def f(a, b):
        df = od.min_filtration(_build_segment_diff_fil(a), _build_segment_diff_fil(b))
        return (df.values ** 2).sum()

    g1_auto = np.asarray(jax.grad(f, argnums=0)(jnp.asarray(v1_np), jnp.asarray(v2_np)))
    g2_auto = np.asarray(jax.grad(f, argnums=1)(jnp.asarray(v1_np), jnp.asarray(v2_np)))
    _assert_grad_nonzero(g1_auto, g2_auto)

    np.testing.assert_allclose(
        g1_auto, _fd_grad(lambda a: f(a, jnp.asarray(v2_np)), v1_np), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        g2_auto, _fd_grad(lambda b: f(jnp.asarray(v1_np), b), v2_np), atol=ATOL, rtol=RTOL)


def test_mapping_cylinder_gradient_matches_finite_difference():
    v_dom_np = np.array([0.10, 0.20, 0.30])
    v_cod_np = np.array([0.15, 0.25, 0.35])

    def f(a, b):
        df_a = _build_segment_diff_fil(a)
        df_b = _build_segment_diff_fil(b)
        # Apex vertices must carry ids disjoint from the existing simplices.
        va_id = df_a.size() + df_b.size()
        vb_id = va_id + 1
        fil = od.mapping_cylinder_filtration(
            df_a, df_b, oin.Simplex([va_id]), oin.Simplex([vb_id]))
        return (fil.values ** 2).sum()

    g_dom = np.asarray(jax.grad(f, argnums=0)(jnp.asarray(v_dom_np), jnp.asarray(v_cod_np)))
    g_cod = np.asarray(jax.grad(f, argnums=1)(jnp.asarray(v_dom_np), jnp.asarray(v_cod_np)))
    _assert_grad_nonzero(g_dom, g_cod)

    np.testing.assert_allclose(
        g_dom, _fd_grad(lambda a: f(a, jnp.asarray(v_cod_np)), v_dom_np), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        g_cod, _fd_grad(lambda b: f(jnp.asarray(v_dom_np), b), v_cod_np), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# alpha / weak-alpha (diode-gated, same as the torch tests)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DIODE, reason="requires diode")
def test_alpha_gradient_matches_finite_difference():
    rng = np.random.default_rng(5)
    pts_np = rng.uniform(-1.0, 1.0, size=(8, 3))

    def f(x):
        return (od.alpha_filtration(x).values ** 2).sum()

    grad_auto = np.asarray(jax.grad(f)(jnp.asarray(pts_np)))
    _assert_grad_nonzero(grad_auto)
    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not HAS_DIODE, reason="requires diode")
def test_weak_alpha_gradient_matches_finite_difference():
    rng = np.random.default_rng(6)
    pts_np = rng.uniform(-1.0, 1.0, size=(8, 3))

    def f(x):
        return (od.weak_alpha_filtration(x).values ** 2).sum()

    grad_auto = np.asarray(jax.grad(f)(jnp.asarray(pts_np)))
    _assert_grad_nonzero(grad_auto)
    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# End-to-end: persistence_diagram through VR
# ---------------------------------------------------------------------------

def test_persistence_diagram_h0_gradient_matches_finite_difference():
    rng = np.random.default_rng(4)
    pts_np = rng.uniform(-1.0, 1.0, size=(5, 2))

    def f(x):
        fil = od.vr_filtration(x, max_dim=1, max_diameter=10.0, n_threads=1)
        dgms = od.persistence_diagram(fil, dualize=True)
        d0 = dgms[0]
        return ((d0[:, 1] - d0[:, 0]) ** 2).sum()

    grad_auto = np.asarray(jax.grad(f)(jnp.asarray(pts_np)))
    _assert_grad_nonzero(grad_auto)
    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)


def test_crit_sets_backward_runs_all_directions():
    """crit-sets is a modified gradient (not the loss gradient), so no FD
    check: mirror the torch smoke -- runs, finite, nonzero, all directions."""
    rng = np.random.default_rng(42)
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    pts_np = np.stack([np.cos(angles) + rng.normal(0, 0.1, 20),
                       np.sin(angles) + rng.normal(0, 0.1, 20),
                       rng.normal(0, 0.1, 20)], axis=1)

    for direction in ("death-up", "death-down", "birth-up", "birth-down"):
        def f(x):
            fil = od.vr_filtration(x, max_dim=2)
            d1 = od.persistence_diagram(fil, gradient_method="crit-sets").in_dimension(1)
            b = jax.lax.stop_gradient(d1[:, 0])
            d = jax.lax.stop_gradient(d1[:, 1])
            pers = d - b
            if direction == "death-up":
                target = d + 0.5 * jnp.abs(pers + 1.0)
                return ((d1[:, 1] - target) ** 2).sum()
            if direction == "death-down":
                target = b + 0.5 * pers
                return ((d1[:, 1] - target) ** 2).sum()
            if direction == "birth-up":
                target = b + 0.25 * pers
                return ((d1[:, 0] - target) ** 2).sum()
            target = b - 0.25 * jnp.abs(pers + 1.0)
            return ((d1[:, 0] - target) ** 2).sum()

        g = np.asarray(jax.grad(f)(jnp.asarray(pts_np)))
        assert np.isfinite(g).all(), direction
        assert (g ** 2).sum() > 0.0, direction
