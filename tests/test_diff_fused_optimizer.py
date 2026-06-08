"""The TopologyOptimizer's default (parallel, crit-sets) path reduces via the
fused keep-working route: it builds the working RV columns straight from the
cached boundary and keeps them (no copy-back), reading R/V through
r_low/r_is_zero/v_col and computing U from the working V. These tests exercise
that path (n_threads > 1, which the n_threads=1 suites do not) and check it
against the classic path."""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import oineus as oin

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")


def _circle(n=20, seed=1):
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.uniform(0, 2 * np.pi, n))
    return (np.c_[np.cos(theta), np.sin(theta)]
            + 0.05 * rng.standard_normal((n, 2))).astype(np.float64)


def _under_fil(pts_np):
    import oineus.diff as od
    t = torch.tensor(pts_np, dtype=torch.float64)
    return od.vr_filtration(t, max_dim=2, max_diameter=10.0, n_threads=1).under_fil


def _srt(a):
    return a[np.lexsort(a.T)] if len(a) else a


def test_fused_optimizer_diagram_and_validity():
    # The parallel optimizer keeps R/V in working form; its diagram must match
    # the serial (classic) optimizer, and the homology decomposition must be a
    # valid D*V == R once materialized (sanity_check forces the materialize).
    under = _under_fil(_circle())
    D = under.boundary_matrix()

    def run(nt):
        opt = oin._oineus.TopologyOptimizer(under, with_crit_sets=True, n_threads=nt)
        opt.ensure_hom_reduced()
        opt.ensure_coh_reduced()
        dg = opt.compute_diagram(include_inf_points=False)
        dgms = [np.array(dg.in_dimension(k, as_numpy=True)) for k in range(3)]
        ok = opt.homology_decomposition_ref().sanity_check(D)
        return dgms, ok

    d_serial, ok_serial = run(1)
    d_par, ok_par = run(4)

    assert ok_serial and ok_par
    for k in range(3):
        a, b = _srt(d_serial[k]), _srt(d_par[k])
        assert a.shape == b.shape and np.allclose(a, b)


def test_fused_optimizer_crit_sets_backward_runs():
    # A crit-sets backward with n_threads > 1 drives the fused keep-working path
    # end to end: is_negative + increase/decrease death/birth through the
    # working-form accessors, and U computed from the working V. The gradient
    # must be finite and nonzero. (crit-sets is a big-step method, so its signal
    # is not the finite-difference loss gradient -- that is the dgm-loss path.)
    import oineus.diff as od
    pts_np = _circle()
    t = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
    fil = od.vr_filtration(t, max_dim=2, max_diameter=10.0, n_threads=1)
    dgms = od.persistence_diagram(fil, gradient_method="crit-sets",
                                  n_threads=4, dualize=False)
    if dgms[1].shape[0] == 0:
        pytest.skip("no H1 points")
    ((dgms[1][:, 1] - dgms[1][:, 0]) ** 2).sum().backward()
    g = t.grad.numpy()
    assert np.isfinite(g).all()
    assert float((g ** 2).sum()) > 1e-10
