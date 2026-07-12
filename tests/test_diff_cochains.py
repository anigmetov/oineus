"""Tests for birth/death cochain optimization (oineus.diff.cochains,
Weighill & Zhou arXiv:2603.25575)."""

import numpy as np
import pytest

import oineus as oin

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")

pytest.importorskip("scipy")
pytest.importorskip("dionysus")

if HAS_TORCH:
    import oineus.diff as od
    from oineus.diff.cochains import (
        _bar_and_cocycle,
        _beta_from_cocycle,
        _birth_cochain_coeffs,
        _complex_data,
        _death_cochain_coeffs,
        _prefix_count,
        _select_bar,
        _signed_coboundary,
        birth_death_cochains,
        death_content,
        persistence_content_loss,
    )


def noisy_circle(n=10, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    phi = np.sort(rng.uniform(0, 2 * np.pi, n))
    pts = np.column_stack([np.cos(phi), np.sin(phi)]) + noise * rng.standard_normal((n, 2))
    return torch.tensor(pts, dtype=torch.float64, requires_grad=True)


def test_eps_to_zero_recovers_singletons_cech():
    """With generic (distinct) simplex values -- full Cech -- both cochains
    collapse to the indicator of the single birth/death simplex."""
    fil = od.cech_filtration(noisy_circle())
    _, _, b_sid, d_sid = _select_bar(fil, 1, "longest")
    with pytest.warns(UserWarning):  # thresholds b/d +- 1e-9 graze the bar values
        bc, dc = birth_death_cochains(fil, dim=1, eps=1e-9, relative_eps=False)
    assert len(bc.sorted_ids) == 1 and bc.sorted_ids[0] == b_sid
    assert len(dc.sorted_ids) == 1 and dc.sorted_ids[0] == d_sid
    assert bc.weights[0] == pytest.approx(1.0)
    assert dc.weights[0] == pytest.approx(1.0)


def test_eps_to_zero_vr_death_ties():
    """VR has structural value ties at death (all triangles sharing the
    death edge enter at exactly d), so the eps->0 death support is the
    tied set, every member at value d; birth stays a singleton."""
    fil = od.vr_filtration(noisy_circle(), max_dim=2)
    under = fil.under_fil
    b, d, b_sid, _ = _select_bar(fil, 1, "longest")
    bc, dc = birth_death_cochains(fil, dim=1, eps=1e-9, relative_eps=False)
    assert len(bc.sorted_ids) == 1 and bc.sorted_ids[0] == b_sid
    for sid in dc.sorted_ids:
        assert under.cell_value_by_sorted_id(int(sid)) == pytest.approx(d, abs=1e-9)


def test_h0_death_cochain_is_death_edge():
    fil = od.vr_filtration(noisy_circle(seed=3), max_dim=2)
    _, d0, _, d_sid = _select_bar(fil, 0, "longest")
    bc, dc = birth_death_cochains(fil, dim=0, eps=1e-8, relative_eps=False)
    assert len(dc.sorted_ids) == 1 and dc.sorted_ids[0] == d_sid
    cost = death_content(fil, dim=0, bar="longest", eps=0.05)
    assert cost.item() == pytest.approx(d0, abs=0.05 * d0 + 1e-9)


def test_birth_cochain_invariants():
    """Vanishes on K, real cocycle on L, and ell2-minimal (orthogonal to
    the admissible correction space, checked via random corrections)."""
    fil = od.vr_filtration(noisy_circle(seed=1), max_dim=2)
    cdata, b, d, eps_abs, cocycle = _bar_and_cocycle(fil, 1, "longest", [0.1], True, 47)
    alpha, t_K, t_L = _birth_cochain_coeffs(cdata, 1, b, d, eps_abs[0], 47, cocycle)

    n_k_K = _prefix_count(cdata, 1, t_K)
    n_k_L = _prefix_count(cdata, 1, t_L)
    n_k1_L = _prefix_count(cdata, 2, t_L)
    assert np.abs(alpha[:n_k_K]).max(initial=0.0) < 1e-9

    delta = _signed_coboundary(cdata, 1, n_k_L, n_k1_L).astype(np.float64)
    assert np.abs(delta @ alpha).max(initial=0.0) < 1e-8

    # minimality: adding any admissible coboundary cannot shrink the norm
    rng = np.random.default_rng(0)
    from oineus.diff.cochains import _vertex_components
    import scipy.sparse as sp
    n_v_K = _prefix_count(cdata, 0, t_K)
    n_v_L = _prefix_count(cdata, 0, t_L)
    comp_K = _vertex_components(cdata, t_K)
    delta0_L = _signed_coboundary(cdata, 0, n_v_L, n_k_L).astype(np.float64)
    for _ in range(10):
        h = np.zeros(n_v_L)
        h[:n_v_K] = rng.standard_normal(comp_K.max() + 1)[comp_K]  # locally constant on K
        h[n_v_K:] = rng.standard_normal(n_v_L - n_v_K)
        perturbed = alpha + delta0_L @ h
        assert np.linalg.norm(perturbed) >= np.linalg.norm(alpha) - 1e-9


def test_death_cochain_representative_independent():
    """Cor 4.4: the death cochain does not depend on the chosen cocycle
    representative beta (perturb by a coboundary supported on K)."""
    fil = od.vr_filtration(noisy_circle(seed=2), max_dim=2)
    cdata, b, d, eps_abs, cocycle = _bar_and_cocycle(fil, 1, "longest", [0.1], True, 47)
    e = eps_abs[0]
    beta = _beta_from_cocycle(cdata, 1, d, e, cocycle)
    omega1, *_ = _death_cochain_coeffs(cdata, 1, d, e, beta)

    n_v_K = _prefix_count(cdata, 0, d - e)
    n_e_K = _prefix_count(cdata, 1, d - e)
    delta0_K = _signed_coboundary(cdata, 0, n_v_K, n_e_K).astype(np.float64)
    rng = np.random.default_rng(1)
    beta2 = beta + delta0_K @ rng.standard_normal(n_v_K)
    omega2, *_ = _death_cochain_coeffs(cdata, 1, d, e, beta2)
    assert np.allclose(omega1, omega2, atol=1e-8)


@pytest.mark.parametrize("edge_relaxed", [False, True])
def test_content_converges_to_persistence(edge_relaxed):
    fil = od.vr_filtration(noisy_circle(seed=4), max_dim=2)
    b, d, *_ = _select_bar(fil, 1, "longest")
    for eps in (0.2, 0.05, 0.01):
        loss = persistence_content_loss(fil, dim=1, eps=(eps,), relative_eps=True,
                                        edge_relaxed=edge_relaxed)
        eps_abs = eps * (d - b)
        assert abs(loss.item() - (d - b)) <= 2 * eps_abs + 1e-9


def test_lift_certificate_and_weights():
    fil = od.vr_filtration(noisy_circle(seed=5), max_dim=2)
    bc, dc = birth_death_cochains(fil, dim=1, eps=0.1)
    for c in (bc, dc):
        assert c.weights.sum() == pytest.approx(1.0)
        assert np.all(c.weights > 0)
        assert len(c.sorted_ids) == len(np.unique(c.sorted_ids))


def test_gradient_flows_and_optimization_improves():
    pts = noisy_circle(seed=6, noise=0.1)
    opt = torch.optim.SGD([pts], lr=0.02)

    def persistence():
        f = od.vr_filtration(pts, max_dim=2)
        b, d, *_ = _select_bar(f, 1, "longest")
        return d - b

    before = persistence()
    for _ in range(5):
        opt.zero_grad()
        f = od.vr_filtration(pts, max_dim=2)
        loss = -persistence_content_loss(f, dim=1, eps=(0.01, 0.05, 0.1))
        loss.backward()
        assert torch.isfinite(pts.grad).all()
        assert pts.grad.abs().sum() > 0
        opt.step()
    assert persistence() > before


def test_bar_spec_variants():
    fil = od.vr_filtration(noisy_circle(seed=7), max_dim=2)
    b, d, *_ = _select_bar(fil, 1, "longest")
    b0, d0, *_ = _select_bar(fil, 1, 0)
    assert (b0, d0) == (b, d)
    b1, d1, *_ = _select_bar(fil, 1, (b, d))
    assert (b1, d1) == (b, d)
    with pytest.raises(ValueError):
        _select_bar(fil, 1, (b + 100.0, d + 100.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
