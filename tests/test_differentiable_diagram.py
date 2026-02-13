#!/usr/bin/env python3
"""
Test script for differentiable persistence diagrams.

Tests:
1. Basic API - dict-like access, all dimensions computed at once
2. Death increase/decrease with dgm-loss and crit-sets
3. Birth increase/decrease with dgm-loss and crit-sets
4. Infinite points handling
5. Optimization loop
6. Comparison of dgm-loss vs crit-sets
"""

import numpy as np
import torch
import oineus as oin
import oineus.diff as oin_diff


def create_test_points():
    """
    Create a simple point cloud that produces meaningful persistence diagrams.
    Points arranged in a thickened circle for H1.
    """
    np.random.seed(42)
    n_points = 20
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radius = 1.0
    noise = 0.1

    x = radius * np.cos(angles) + np.random.normal(0, noise, n_points)
    y = radius * np.sin(angles) + np.random.normal(0, noise, n_points)
    z = np.random.normal(0, noise, n_points)

    pts = np.stack([x, y, z], axis=1)
    return torch.tensor(pts, dtype=torch.float64, requires_grad=True)


def test_basic_api():
    """Test that the basic API works - dict-like access, all dims at once."""
    print("=" * 60)
    print("TEST: Basic API")
    print("=" * 60)

    pts = create_test_points()

    # Create differentiable filtration
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    print(f"Created DiffFiltration with {fil.size()} simplices")
    print(f"fil.values type: {type(fil.values)}")

    # Get all diagrams at once
    dgms = oin_diff.persistence_diagram(fil, dualize=True)

    # Check dict-like interface
    print(f"Available dimensions: {list(dgms.keys())}")
    print(f"Number of dimensions: {len(dgms)}")

    for dim in dgms.keys():
        dgm = dgms[dim]
        print(f"  H{dim}: {len(dgm)} points, shape {dgm.shape}")

    # Check in_dimension alias
    dgm1 = dgms.in_dimension(1)
    assert torch.equal(dgm1, dgms[1])

    print("PASSED: Basic API works\n")
    return True


def test_infinite_points():
    """Test that infinite points are handled correctly."""
    print("=" * 60)
    print("TEST: Infinite Points")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)

    # Without infinite points
    dgms_finite = oin_diff.persistence_diagram(fil, include_inf_points=False)
    n_finite_h0 = len(dgms_finite[0])

    # With infinite points
    dgms_inf = oin_diff.persistence_diagram(fil, include_inf_points=True)
    n_with_inf_h0 = len(dgms_inf[0])

    print(f"H0 without inf: {n_finite_h0} points")
    print(f"H0 with inf: {n_with_inf_h0} points")

    # H0 should have more points when including infinite (the one component that survives)
    assert n_with_inf_h0 >= n_finite_h0, "Including inf should add at least one point"

    # Check that infinite death is inf
    if n_with_inf_h0 > n_finite_h0:
        dgm0 = dgms_inf[0]
        has_inf = torch.isinf(dgm0[:, 1]).any()
        print(f"Has infinite death: {has_inf}")
        assert has_inf, "Should have infinite death"

    # Test gradient on infinite point's birth
    pts2 = create_test_points()
    fil2 = oin_diff.vr_filtration(pts2, max_dim=2)
    dgms2 = oin_diff.persistence_diagram(fil2, include_inf_points=True)

    dgm0 = dgms2[0]
    # Loss on birth of infinite point (if any)
    inf_mask = torch.isinf(dgm0[:, 1])
    if inf_mask.any():
        inf_births = dgm0[inf_mask, 0]
        loss = inf_births.sum()  # Minimize total birth of infinite points
        loss.backward()
        assert pts2.grad is not None, "Should have gradients"
        print(f"Gradient norm from inf points: {pts2.grad.norm().item():.6f}")

    print("PASSED: Infinite points handled correctly\n")
    return True


def test_death_increase_dgm_loss():
    """Test death increase with dgm-loss gradient method."""
    print("=" * 60)
    print("TEST: Death Increase (dgm-loss)")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss")

    dgm1 = dgms[1]
    if len(dgm1) == 0:
        print("SKIPPED: No H1 points")
        return True

    target_death = dgm1[:, 1].max() + 0.5
    print(f"Current max death: {dgm1[:, 1].max().item():.4f}")
    print(f"Target death: {target_death:.4f}")

    loss = (dgm1[0, 1] - target_death) ** 2
    loss.backward()

    assert pts.grad is not None, "Gradients should exist"
    grad_norm = pts.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradients should be non-zero"

    print("PASSED: Death increase (dgm-loss)\n")
    return True


def test_death_decrease_dgm_loss():
    """Test death decrease with dgm-loss gradient method."""
    print("=" * 60)
    print("TEST: Death Decrease (dgm-loss)")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss")

    dgm1 = dgms[1]
    if len(dgm1) == 0:
        print("SKIPPED: No H1 points")
        return True

    current_death = dgm1[0, 1].item()
    current_birth = dgm1[0, 0].item()
    target_death = current_birth + (current_death - current_birth) * 0.5
    print(f"Current death: {current_death:.4f}")
    print(f"Target death: {target_death:.4f}")

    loss = (dgm1[0, 1] - target_death) ** 2
    loss.backward()

    assert pts.grad is not None
    grad_norm = pts.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0

    print("PASSED: Death decrease (dgm-loss)\n")
    return True


def test_death_increase_crit_sets():
    """Test death increase with crit-sets gradient method."""
    print("=" * 60)
    print("TEST: Death Increase (crit-sets)")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets", lr=1.0)

    dgm1 = dgms[1]
    if len(dgm1) == 0:
        print("SKIPPED: No H1 points")
        return True

    target_death = dgm1[:, 1].max() + 0.5
    loss = (dgm1[0, 1] - target_death) ** 2
    loss.backward()

    assert pts.grad is not None
    grad_norm = pts.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")

    n_affected = (pts.grad.abs() > 1e-10).any(dim=1).sum().item()
    print(f"Points with gradients: {n_affected}/{len(pts)}")

    print("PASSED: Death increase (crit-sets)\n")
    return True


def test_death_decrease_crit_sets():
    """Test death decrease with crit-sets gradient method."""
    print("=" * 60)
    print("TEST: Death Decrease (crit-sets)")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets", lr=1.0)

    dgm1 = dgms[1]
    if len(dgm1) == 0:
        print("SKIPPED: No H1 points")
        return True

    current_death = dgm1[0, 1].item()
    current_birth = dgm1[0, 0].item()
    target_death = current_birth + (current_death - current_birth) * 0.5

    loss = (dgm1[0, 1] - target_death) ** 2
    loss.backward()

    assert pts.grad is not None
    grad_norm = pts.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")

    n_affected = (pts.grad.abs() > 1e-10).any(dim=1).sum().item()
    print(f"Points with gradients: {n_affected}/{len(pts)}")

    print("PASSED: Death decrease (crit-sets)\n")
    return True


def test_birth_increase_dgm_loss():
    """Test birth increase with dgm-loss gradient method."""
    print("=" * 60)
    print("TEST: Birth Increase (dgm-loss)")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss")

    dgm1 = dgms[1]
    if len(dgm1) == 0:
        print("SKIPPED: No H1 points")
        return True

    # Find a point with non-zero birth
    non_zero_births = dgm1[:, 0] > 1e-6
    if not non_zero_births.any():
        print("SKIPPED: All births are zero")
        return True

    idx = non_zero_births.nonzero()[0].item()
    current_birth = dgm1[idx, 0].item()
    target_birth = current_birth + 0.2

    loss = (dgm1[idx, 0] - target_birth) ** 2
    loss.backward()

    assert pts.grad is not None
    grad_norm = pts.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")

    print("PASSED: Birth increase (dgm-loss)\n")
    return True


def test_birth_decrease_dgm_loss():
    """Test birth decrease with dgm-loss gradient method."""
    print("=" * 60)
    print("TEST: Birth Decrease (dgm-loss)")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss")

    dgm1 = dgms[1]
    if len(dgm1) == 0:
        print("SKIPPED: No H1 points")
        return True

    non_zero_births = dgm1[:, 0] > 1e-6
    if not non_zero_births.any():
        print("SKIPPED: All births are zero")
        return True

    idx = non_zero_births.nonzero()[0].item()
    current_birth = dgm1[idx, 0].item()
    target_birth = current_birth * 0.5

    loss = (dgm1[idx, 0] - target_birth) ** 2
    loss.backward()

    assert pts.grad is not None
    grad_norm = pts.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")

    print("PASSED: Birth decrease (dgm-loss)\n")
    return True


def test_birth_increase_crit_sets():
    """Test birth increase with crit-sets gradient method."""
    print("=" * 60)
    print("TEST: Birth Increase (crit-sets)")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets", lr=1.0)

    dgm1 = dgms[1]
    if len(dgm1) == 0:
        print("SKIPPED: No H1 points")
        return True

    non_zero_births = dgm1[:, 0] > 1e-6
    if not non_zero_births.any():
        print("SKIPPED: All births are zero")
        return True

    idx = non_zero_births.nonzero()[0].item()
    current_birth = dgm1[idx, 0].item()
    target_birth = current_birth + 0.2

    loss = (dgm1[idx, 0] - target_birth) ** 2
    loss.backward()

    assert pts.grad is not None
    grad_norm = pts.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")

    n_affected = (pts.grad.abs() > 1e-10).any(dim=1).sum().item()
    print(f"Points with gradients: {n_affected}/{len(pts)}")

    print("PASSED: Birth increase (crit-sets)\n")
    return True


def test_birth_decrease_crit_sets():
    """Test birth decrease with crit-sets gradient method."""
    print("=" * 60)
    print("TEST: Birth Decrease (crit-sets)")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets", lr=1.0)

    dgm1 = dgms[1]
    if len(dgm1) == 0:
        print("SKIPPED: No H1 points")
        return True

    non_zero_births = dgm1[:, 0] > 1e-6
    if not non_zero_births.any():
        print("SKIPPED: All births are zero")
        return True

    idx = non_zero_births.nonzero()[0].item()
    current_birth = dgm1[idx, 0].item()
    target_birth = current_birth * 0.5

    loss = (dgm1[idx, 0] - target_birth) ** 2
    loss.backward()

    assert pts.grad is not None
    grad_norm = pts.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.6f}")

    n_affected = (pts.grad.abs() > 1e-10).any(dim=1).sum().item()
    print(f"Points with gradients: {n_affected}/{len(pts)}")

    print("PASSED: Birth decrease (crit-sets)\n")
    return True


def test_optimization_loop():
    """Test a simple optimization loop."""
    print("=" * 60)
    print("TEST: Optimization Loop")
    print("=" * 60)

    pts = create_test_points()
    optimizer = torch.optim.SGD([pts], lr=0.01)

    initial_death = None

    for step in range(5):
        optimizer.zero_grad()

        fil = oin_diff.vr_filtration(pts, max_dim=2)
        dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets", lr=1.0)

        dgm1 = dgms[1]
        if len(dgm1) == 0:
            print(f"Step {step}: No H1 points, stopping")
            break

        persistence = dgm1[:, 1] - dgm1[:, 0]
        most_persistent_idx = persistence.argmax()
        current_death = dgm1[most_persistent_idx, 1]

        if initial_death is None:
            initial_death = current_death.item()

        target_death = current_death + 0.5
        loss = (current_death - target_death) ** 2

        print(f"Step {step}: death = {current_death.item():.4f}, loss = {loss.item():.6f}")

        loss.backward()
        optimizer.step()

    final_death = current_death.item()
    print(f"Initial death: {initial_death:.4f}, Final death: {final_death:.4f}")

    assert final_death > initial_death, "Death should have increased"

    print("PASSED: Optimization loop\n")
    return True


def test_compare_dgm_loss_vs_crit_sets():
    """Compare dgm-loss vs crit-sets: crit-sets should affect more points."""
    print("=" * 60)
    print("TEST: Compare dgm-loss vs crit-sets")
    print("=" * 60)

    # dgm-loss
    pts1 = create_test_points()
    fil1 = oin_diff.vr_filtration(pts1, max_dim=2)
    dgms1 = oin_diff.persistence_diagram(fil1, gradient_method="dgm-loss")

    dgm1_h1 = dgms1[1]
    if len(dgm1_h1) == 0:
        print("SKIPPED: No H1 points")
        return True

    target = dgm1_h1[0, 1] + 0.5
    loss1 = (dgm1_h1[0, 1] - target) ** 2
    loss1.backward()
    n_affected_dgm = (pts1.grad.abs() > 1e-10).any(dim=1).sum().item()

    # crit-sets
    pts2 = create_test_points()
    fil2 = oin_diff.vr_filtration(pts2, max_dim=2)
    dgms2 = oin_diff.persistence_diagram(fil2, gradient_method="crit-sets", lr=1.0)

    dgm2_h1 = dgms2[1]
    loss2 = (dgm2_h1[0, 1] - target) ** 2
    loss2.backward()
    n_affected_crit = (pts2.grad.abs() > 1e-10).any(dim=1).sum().item()

    print(f"dgm-loss: {n_affected_dgm} points affected")
    print(f"crit-sets: {n_affected_crit} points affected")

    assert n_affected_crit >= n_affected_dgm, \
        f"crit-sets should affect >= points than dgm-loss"

    print("PASSED: crit-sets affects at least as many points\n")
    return True


def test_dualize_flag():
    """Test that dualize flag works (homology vs cohomology)."""
    print("=" * 60)
    print("TEST: Dualize Flag")
    print("=" * 60)

    pts = create_test_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)

    # Homology (dualize=False)
    dgms_hom = oin_diff.persistence_diagram(fil, dualize=False)
    # Cohomology (dualize=True, default)
    dgms_coh = oin_diff.persistence_diagram(fil, dualize=True)

    print(f"Homology H1: {len(dgms_hom[1])} points")
    print(f"Cohomology H1: {len(dgms_coh[1])} points")

    # Both should give same number of points (different internal computation)
    assert len(dgms_hom[1]) == len(dgms_coh[1]), "Homology and cohomology should have same diagram size"

    print("PASSED: Dualize flag works\n")
    return True


def run_all_tests():
    """Run all tests."""
    tests = [
        test_basic_api,
        test_infinite_points,
        test_dualize_flag,
        test_death_increase_dgm_loss,
        test_death_decrease_dgm_loss,
        test_death_increase_crit_sets,
        test_death_decrease_crit_sets,
        test_birth_increase_dgm_loss,
        test_birth_decrease_dgm_loss,
        test_birth_increase_crit_sets,
        test_birth_decrease_crit_sets,
        test_optimization_loop,
        test_compare_dgm_loss_vs_crit_sets,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
