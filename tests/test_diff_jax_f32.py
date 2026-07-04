"""Float32 (default-jax) routing checks for oineus.diff.

Without jax_enable_x64 -- deliberately NOT enabled here -- jax arrays are
float32, and the diff constructors must build genuine float32 filtrations
routed to the _f32 oineus backend when it is compiled in (falling back to
the float64 backend otherwise). Diagrams and gradients stay float32
either way. The float64/x64 counterpart lives in test_diff_jax_grad.py.
Skips entirely when jax is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import oineus.diff as od
from oineus._dtype import REAL_MODULES

F32_COMPILED = np.dtype("float32") in REAL_MODULES


def _expected_module():
    return "oineus._oineus._f32" if F32_COMPILED else "oineus._oineus"


def _rand_pts(n=5, d=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)


def test_default_jax_arrays_are_float32():
    # the premise of this file: without x64, jax arrays are float32
    assert jnp.asarray(np.zeros(3)).dtype == np.float32


def test_vr_float32_routing_and_dtypes():
    pts = jnp.asarray(_rand_pts())
    assert pts.dtype == np.float32
    fil = od.vr_filtration(pts, max_dim=1, max_diameter=10.0)
    assert type(fil.under_fil).__module__ == _expected_module()
    assert fil.values.dtype == np.float32
    dgm0 = od.persistence_diagram(fil, dualize=True)[0]
    assert dgm0.dtype == np.float32


def test_freudenthal_float32_routing():
    data = jnp.asarray(np.random.default_rng(1).uniform(
        -1.0, 1.0, size=(3, 3)).astype(np.float32))
    df = od.freudenthal_filtration(data, max_dim=2)
    assert type(df.under_fil).__module__ == _expected_module()
    assert df.values.dtype == np.float32


def test_dgm_loss_gradient_float32():
    pts = jnp.asarray(_rand_pts())

    def loss(x):
        fil = od.vr_filtration(x, max_dim=1, max_diameter=10.0)
        d0 = od.persistence_diagram(fil, dualize=True)[0]
        return ((d0[:, 1] - d0[:, 0]) ** 2).sum()

    g = jax.grad(loss)(pts)
    assert g.dtype == np.float32
    gn = np.asarray(g)
    assert np.isfinite(gn).all()
    assert (gn ** 2).sum() > 0


def test_crit_sets_gradient_float32():
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circ = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

    def loss(x):
        fil = od.vr_filtration(x, max_dim=2)
        d1 = od.persistence_diagram(fil, gradient_method="crit-sets")[1]
        tgt = jax.lax.stop_gradient(d1[:, 1]) + 0.5
        return ((d1[:, 1] - tgt) ** 2).sum()

    g = jax.grad(loss)(jnp.asarray(circ))
    assert g.dtype == np.float32
    gn = np.asarray(g)
    assert np.isfinite(gn).all()
    assert (gn ** 2).sum() > 0
