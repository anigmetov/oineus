"""Tests for the differentiable full Cech filtration (oineus.diff.cech_filtration)."""

import numpy as np
import pytest

import oineus as oin

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")

if HAS_TORCH:
    import oineus.diff as od


def noisy_circle(n, seed, noise=0.05):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([np.cos(phi), np.sin(phi)]) + noise * rng.standard_normal((n, 2))
    return torch.tensor(pts, dtype=torch.float64, requires_grad=True)


def test_gradient_flows_to_points():
    pts = noisy_circle(16, 0)
    fil = od.cech_filtration(pts)
    dgm1 = od.persistence_diagram(fil)[1]
    assert len(dgm1) > 0
    i = torch.argmax(dgm1[:, 1] - dgm1[:, 0])
    loss = -(dgm1[i, 1] - dgm1[i, 0])
    loss.backward()
    assert pts.grad is not None
    assert torch.isfinite(pts.grad).all()
    assert pts.grad.abs().sum() > 0


def test_optimization_step_increases_persistence():
    pts = noisy_circle(14, 1)
    opt = torch.optim.SGD([pts], lr=0.05)

    def longest_h1():
        fil = od.cech_filtration(pts)
        dgm1 = od.persistence_diagram(fil)[1]
        i = torch.argmax(dgm1[:, 1] - dgm1[:, 0])
        return dgm1[i, 1] - dgm1[i, 0]

    before = longest_h1().item()
    for _ in range(5):
        opt.zero_grad()
        loss = -longest_h1()
        loss.backward()
        opt.step()
    after = longest_h1().item()
    assert after > before


@pytest.mark.parametrize("d,n,seed", [(2, 20, 2), (3, 14, 3)])
def test_dgms_equal_cech_delaunay(d, n, seed):
    """Strong oracle: full Cech and Cech-Delaunay diagrams agree in dims
    0..d-1 (Bauer-Edelsbrunner: both compute union-of-balls persistence).
    The dim-d Cech diagram is skeleton-truncated and excluded."""
    pytest.importorskip("diode")
    rng = np.random.default_rng(seed)
    pts_np = rng.random((n, d))
    pts = torch.tensor(pts_np, dtype=torch.float64)

    fil_cech = od.cech_filtration(pts.clone().requires_grad_(True))
    fil_cd = od.cech_delaunay_filtration(pts.clone().requires_grad_(True))

    for fil in (fil_cech, fil_cd):
        dcmp = oin.Decomposition(fil.under_fil, False)
        dcmp.reduce(oin.ReductionParams())
        fil.dgms = dcmp.diagram(fil=fil.under_fil, include_inf_points=True)

    for q in range(d):
        a = np.asarray(fil_cech.dgms.in_dimension(q))
        b = np.asarray(fil_cd.dgms.in_dimension(q))
        assert len(a) == len(b)
        assert oin.bottleneck_distance(a, b, delta=0.0) <= 1e-10


def test_values_match_nondiff():
    pts = noisy_circle(15, 4)
    fil = od.cech_filtration(pts)
    nondiff = oin.cech_filtration(pts.detach().numpy())
    diff_vals = fil.values.detach().numpy()
    nondiff_vals = np.array([s.value for s in nondiff.cells()])
    assert np.allclose(np.sort(diff_vals), np.sort(nondiff_vals), rtol=1e-12, atol=1e-14)
    # per-dim sorted alignment with the C++ filtration's sorted order
    under_vals = np.array([s.value for s in fil.under_fil.cells()])
    assert np.allclose(diff_vals, under_vals, rtol=1e-12, atol=1e-14)


def test_kind_and_reduction_default():
    from oineus.diff._reduction_policy import default_dualize_for_filtration
    pts = noisy_circle(8, 5)
    fil = od.cech_filtration(pts)
    assert fil.kind == oin.FiltrationKind.Cech
    assert default_dualize_for_filtration(fil) is False


def test_max_radius_truncation():
    pts = noisy_circle(12, 6)
    fil_full = od.cech_filtration(pts.clone().requires_grad_(True))
    fil_trunc = od.cech_filtration(pts.clone().requires_grad_(True), max_radius=0.5)
    assert fil_trunc.size() < fil_full.size()
    assert fil_trunc.values.max().item() <= 0.25 + 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
