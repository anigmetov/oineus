"""Torch-vs-jax gradient equality for oineus.diff.

Same seeded inputs, same losses, both frameworks in float64 (x64 enabled
at module import): the gradients must agree to numerical noise for the
dgm-loss method and for every critical-set conflict strategy. The C++
reduction and crit-sets machinery are deterministic, so the tolerance is
tight. Skips entirely unless both torch and jax are installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

try:
    import diode
    HAS_DIODE = True
except ImportError:
    HAS_DIODE = False

import oineus.diff as od


ATOL = 1e-9
RTOL = 1e-9


def _circle_pts(n=20, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles) + rng.normal(0, noise, n)
    y = np.sin(angles) + rng.normal(0, noise, n)
    z = rng.normal(0, noise, n)
    return np.stack([x, y, z], axis=1)


def _torch_grad(loss_fn, x_np):
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    loss_fn(x, torch).backward()
    return x.grad.detach().numpy()


def _jax_grad(loss_fn, x_np):
    return np.asarray(jax.grad(lambda x: loss_fn(x, jnp))(jnp.asarray(x_np)))


def _assert_same_grads(loss_fn, x_np):
    g_torch = _torch_grad(loss_fn, x_np)
    g_jax = _jax_grad(loss_fn, x_np)
    assert float((g_torch ** 2).sum()) > 1e-10, "trivial all-zero gradient"
    np.testing.assert_allclose(g_jax, g_torch, atol=ATOL, rtol=RTOL)


def _detach(t, backend):
    # framework-agnostic stop-gradient for building loss targets
    if backend is jnp:
        return jax.lax.stop_gradient(t)
    return t.detach()


# ---------------------------------------------------------------------------
# dgm-loss
# ---------------------------------------------------------------------------

def test_dgm_loss_h0_total_persistence_grads_equal():
    pts_np = np.random.default_rng(7).uniform(-1.0, 1.0, size=(6, 2))

    def loss(x, backend):
        fil = od.vr_filtration(x, max_dim=1, max_diameter=10.0, n_threads=1)
        d0 = od.persistence_diagram(fil, dualize=True)[0]
        return ((d0[:, 1] - d0[:, 0]) ** 2).sum()

    _assert_same_grads(loss, pts_np)


def test_dgm_loss_freudenthal_grads_equal():
    data_np = np.random.default_rng(8).uniform(-1.0, 1.0, size=(4, 4))

    def loss(x, backend):
        fil = od.freudenthal_filtration(x, max_dim=2, n_threads=1)
        dgms = od.persistence_diagram(fil)
        total = (dgms[0][:, 1] - dgms[0][:, 0]).sum()
        if dgms[1].shape[0] > 0:
            total = total + ((dgms[1][:, 1] - dgms[1][:, 0]) ** 2).sum()
        return total

    _assert_same_grads(loss, data_np)


def test_dgm_loss_h1_circle_grads_equal():
    pts_np = _circle_pts()

    def loss(x, backend):
        fil = od.vr_filtration(x, max_dim=2)
        d1 = od.persistence_diagram(fil)[1]
        return ((d1[:, 1] - d1[:, 0]) ** 2).sum()

    _assert_same_grads(loss, pts_np)


# ---------------------------------------------------------------------------
# crit-sets, every conflict strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["avg", "max", "sum", "fca"])
def test_crit_sets_strategy_grads_equal(strategy):
    pts_np = _circle_pts()

    def loss(x, backend):
        fil = od.vr_filtration(x, max_dim=2)
        d1 = od.persistence_diagram(
            fil, gradient_method="crit-sets",
            conflict_strategy=strategy).in_dimension(1)
        # push every death up by 0.5 -- many simultaneous singleton losses,
        # exercising the conflict-resolution path
        targets = _detach(d1[:, 1], backend) + 0.5
        return ((d1[:, 1] - targets) ** 2).sum()

    _assert_same_grads(loss, pts_np)


@pytest.mark.parametrize("direction", ["death-down", "birth-up", "birth-down"])
def test_crit_sets_other_directions_grads_equal(direction):
    pts_np = _circle_pts()

    def loss(x, backend):
        fil = od.vr_filtration(x, max_dim=2)
        d1 = od.persistence_diagram(
            fil, gradient_method="crit-sets").in_dimension(1)
        b = _detach(d1[:, 0], backend)
        d = _detach(d1[:, 1], backend)
        if direction == "death-down":
            target = b + 0.5 * (d - b)
            return ((d1[:, 1] - target) ** 2).sum()
        if direction == "birth-up":
            target = b + 0.25 * (d - b)
            return ((d1[:, 0] - target) ** 2).sum()
        target = b - 0.5
        return ((d1[:, 0] - target) ** 2).sum()

    _assert_same_grads(loss, pts_np)


# ---------------------------------------------------------------------------
# weak-alpha longest-edge ties: gradient must split evenly among tied edges
# ---------------------------------------------------------------------------

# Isosceles triangle: the triangle's longest edge is tied (|p0-p2|^2 ==
# |p1-p2|^2 == 26). torch.amax / jnp.max split the tie gradient evenly;
# eagerpy's torch max used to send it all to the first tied edge.
TIE_PTS = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 5.0]])
TIE_GRAD = np.array([[-7.0, -15.0], [7.0, -15.0], [0.0, 30.0]])


@pytest.mark.skipif(not HAS_DIODE, reason="requires diode")
def test_weak_alpha_tie_gradient_splits_evenly():
    def loss(x, backend):
        return od.weak_alpha_filtration(x).values.sum()

    g_torch = _torch_grad(loss, TIE_PTS)
    g_jax = _jax_grad(loss, TIE_PTS)
    np.testing.assert_allclose(g_torch, TIE_GRAD, atol=1e-12, rtol=0)
    np.testing.assert_allclose(g_jax, TIE_GRAD, atol=1e-12, rtol=0)


@pytest.mark.skipif(not HAS_DIODE, reason="requires diode")
def test_weak_alpha_tie_gradient_matches_finite_difference_torch():
    # central differences at a tie average the two one-sided derivatives,
    # which is exactly the even-split (amax) gradient
    def f(x_np):
        x = torch.tensor(x_np, dtype=torch.float64)
        return float(od.weak_alpha_filtration(x).values.sum())

    eps = 1e-6
    g_fd = np.zeros_like(TIE_PTS)
    for idx in np.ndindex(TIE_PTS.shape):
        hi = TIE_PTS.copy()
        hi[idx] += eps
        lo = TIE_PTS.copy()
        lo[idx] -= eps
        g_fd[idx] = (f(hi) - f(lo)) / (2 * eps)

    g_torch = _torch_grad(lambda x, backend: od.weak_alpha_filtration(x).values.sum(),
                          TIE_PTS)
    np.testing.assert_allclose(g_torch, g_fd, atol=1e-5, rtol=1e-5)
