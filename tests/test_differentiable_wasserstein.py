"""Comprehensive tests for differentiable Wasserstein distance."""

import pytest
import torch
import numpy as np
import sys
import os

# Add build directory to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build/bindings/python'))

import oineus
import oineus.diff as oin_diff


class TestWassersteinCostBasics:
    """Basic functionality tests."""

    def test_cost_matches_nondifferentiable_distance(self):
        """Verify cost^(1/q) matches non-differentiable wasserstein_distance."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=torch.float64)

        q = 2.0
        delta = 0.01

        # Differentiable cost
        cost_diff = oin_diff.wasserstein_cost(
            dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta
        )

        # Non-differentiable distance
        dist_nondiff = oineus.wasserstein_distance(
            dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta
        )

        # cost = distance^q
        expected_cost = dist_nondiff ** q

        # Allow for approximation error from delta
        assert torch.allclose(
            cost_diff, torch.tensor(expected_cost, dtype=torch.float64),
            rtol=delta * 10
        )

    def test_identical_diagrams(self):
        """Cost should be zero for identical diagrams."""
        dgm = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm, dgm.clone())

        assert cost < 1e-10, f"Cost should be near zero, got {cost.item()}"

    def test_different_q_values(self):
        """Test with different Wasserstein power parameters."""
        dgm_a = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.1, 0.9]], dtype=torch.float64)

        for q in [1.0, 2.0, 3.0]:
            cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q)
            assert cost >= 0, f"Cost should be non-negative for q={q}"

    def test_different_internal_p(self):
        """Test with different internal L_p norms."""
        dgm_a = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.2, 0.8]], dtype=torch.float64)

        # L_1 norm
        cost_1 = oin_diff.wasserstein_cost(dgm_a, dgm_b, internal_p=1.0)

        # L_2 norm
        cost_2 = oin_diff.wasserstein_cost(dgm_a, dgm_b, internal_p=2.0)

        # L_inf norm
        cost_inf = oin_diff.wasserstein_cost(dgm_a, dgm_b, internal_p=float('inf'))

        assert cost_1 >= 0 and cost_2 >= 0 and cost_inf >= 0


class TestEmptyDiagrams:
    """Test handling of empty diagrams."""

    def test_both_empty(self):
        """Both diagrams empty."""
        dgm_a = torch.zeros((0, 2), dtype=torch.float64)
        dgm_b = torch.zeros((0, 2), dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b)
        assert cost == 0.0

    def test_one_empty(self):
        """One diagram empty, one non-empty."""
        dgm_a = torch.zeros((0, 2), dtype=torch.float64)
        dgm_b = torch.tensor([[0.0, 1.0], [0.5, 1.5]], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b)

        # Cost should equal the cost of matching dgm_b points to diagonal
        # For point (b, d), cost to diagonal = persistence^q / 2^q
        # With q=1 (default), cost = |d-b|/2 for each point
        expected = (1.0 / 2.0 + 1.0 / 2.0)  # Two points with persistence 1.0 each
        assert torch.allclose(cost, torch.tensor(expected, dtype=torch.float64), rtol=0.1)


class TestEssentialPoints:
    """Test handling of essential points (with infinity)."""

    def test_essential_points_same_cardinality(self):
        """Essential points with matching cardinalities."""
        dgm_a = torch.tensor([
            [0.0, 1.0],
            [0.5, float('inf')]  # Essential point (finite, +inf)
        ], dtype=torch.float64)
        dgm_b = torch.tensor([
            [0.1, 0.9],
            [0.6, float('inf')]  # Essential point (finite, +inf)
        ], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)
        assert cost >= 0

    def test_essential_points_different_cardinality_raises_error(self):
        """Essential points with different cardinalities should raise ValueError."""
        dgm_a = torch.tensor([
            [0.0, 1.0],
            [0.5, float('inf')]  # One essential point
        ], dtype=torch.float64)
        dgm_b = torch.tensor([
            [0.1, 0.9],
            [0.6, float('inf')],
            [0.7, float('inf')]  # Two essential points
        ], dtype=torch.float64)

        with pytest.raises(ValueError, match="Essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

    def test_ignore_inf_points_flag(self):
        """With ignore_inf_points=True, different cardinalities should work."""
        dgm_a = torch.tensor([
            [0.0, 1.0],
            [0.5, float('inf')]  # One essential point
        ], dtype=torch.float64)
        dgm_b = torch.tensor([
            [0.1, 0.9],
            [0.6, float('inf')],
            [0.7, float('inf')]  # Two essential points
        ], dtype=torch.float64)

        # Should not raise error with ignore_inf_points=True (default)
        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=True)
        assert cost >= 0

    def test_only_essential_points(self):
        """Diagrams with only essential points."""
        dgm_a = torch.tensor([
            [0.5, float('inf')],
            [1.0, float('inf')]
        ], dtype=torch.float64)
        dgm_b = torch.tensor([
            [0.6, float('inf')],
            [1.1, float('inf')]
        ], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

        # Cost should be sum of |0.6-0.5| + |1.1-1.0| = 0.1 + 0.1 = 0.2 for q=1
        assert torch.allclose(cost, torch.tensor(0.2, dtype=torch.float64), rtol=0.01)

    def test_all_four_essential_categories(self):
        """Test all four categories of essential points."""
        dgm_a = torch.tensor([
            [0.0, 1.0],  # Finite
            [0.5, float('inf')],   # (finite, +inf)
            [1.0, float('-inf')],  # (finite, -inf)
            [float('inf'), 2.0],   # (+inf, finite)
            [float('-inf'), 3.0]   # (-inf, finite)
        ], dtype=torch.float64)
        dgm_b = torch.tensor([
            [0.1, 0.9],  # Finite
            [0.6, float('inf')],   # (finite, +inf)
            [1.1, float('-inf')],  # (finite, -inf)
            [float('inf'), 2.1],   # (+inf, finite)
            [float('-inf'), 3.1]   # (-inf, finite)
        ], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)
        assert cost >= 0

    def test_mixed_essential_categories_raises_error(self):
        """Different essential categories should raise error."""
        # dgm_a has (finite, +inf) only
        dgm_a = torch.tensor([
            [0.0, 1.0],
            [0.5, float('inf')]  # (finite, +inf)
        ], dtype=torch.float64)

        # dgm_b has both (finite, +inf) and (finite, -inf)
        dgm_b = torch.tensor([
            [0.1, 0.9],
            [0.6, float('inf')],   # (finite, +inf)
            [0.7, float('-inf')]   # (finite, -inf)
        ], dtype=torch.float64)

        # Should raise error for (finite, -inf) mismatch (0 vs 1)
        with pytest.raises(ValueError, match="Essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

    def test_essential_posinf_vs_neginf(self):
        """Test (finite, +inf) vs (finite, -inf) mismatch."""
        dgm_a = torch.tensor([
            [0.5, float('inf')]   # (finite, +inf)
        ], dtype=torch.float64)

        dgm_b = torch.tensor([
            [0.5, float('-inf')]  # (finite, -inf)
        ], dtype=torch.float64)

        # Different categories - should raise error
        with pytest.raises(ValueError, match="Essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

    def test_essential_birth_vs_death_inf(self):
        """Test (+inf, finite) vs (finite, +inf) mismatch."""
        dgm_a = torch.tensor([
            [float('inf'), 2.0]   # (+inf, finite)
        ], dtype=torch.float64)

        dgm_b = torch.tensor([
            [2.0, float('inf')]   # (finite, +inf)
        ], dtype=torch.float64)

        # Different categories - should raise error
        with pytest.raises(ValueError, match="Essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

    def test_essential_multiple_category_mismatch(self):
        """Test complex mismatch: dgm_a has 2 categories, dgm_b has 3."""
        dgm_a = torch.tensor([
            [0.5, float('inf')],   # (finite, +inf)
            [float('inf'), 2.0]    # (+inf, finite)
        ], dtype=torch.float64)

        dgm_b = torch.tensor([
            [0.6, float('inf')],    # (finite, +inf)
            [1.0, float('-inf')],   # (finite, -inf)
            [float('inf'), 2.1]     # (+inf, finite)
        ], dtype=torch.float64)

        # Should raise error for (finite, -inf) category (0 vs 1)
        with pytest.raises(ValueError, match="Essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)


class TestGradients:
    """Test gradient computation and flow."""

    def test_gradient_flow_finite_to_finite(self):
        """Gradients should flow through finite-to-finite matches."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64, requires_grad=True)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b)
        cost.backward()

        assert dgm_a.grad is not None, "No gradient computed"
        assert not torch.all(dgm_a.grad == 0), "Gradient is all zeros"

    def test_diagonal_projection_detached(self):
        """
        Verify diagonal projections are detached.

        When a point matches to diagonal, gradient should point toward/away from diagonal.
        For a point (b, d), the diagonal projection is ((b+d)/2, (b+d)/2).
        The gradient components should have equal magnitude but opposite signs.
        """
        # Create two diagrams where points will match to diagonal
        # (they're far apart, so each matches to its own diagonal projection)
        dgm_a = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
        dgm_b = torch.tensor([[100.0, 101.0]], dtype=torch.float64)

        # Use W_2 with internal_p=2 for easier gradient computation
        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0, internal_p=2.0)
        cost.backward()

        # Extract gradients
        grad_b = dgm_a.grad[0, 0].item()  # gradient w.r.t. birth
        grad_d = dgm_a.grad[0, 1].item()  # gradient w.r.t. death

        # Gradients should have equal magnitude, opposite signs
        assert abs(abs(grad_b) - abs(grad_d)) < 1e-6, \
            f"Gradient magnitudes should be equal: |{grad_b}| vs |{grad_d}|"

        assert grad_b * grad_d < 0, \
            f"Gradients should have opposite signs: {grad_b} vs {grad_d}"

    def test_gradient_magnitude_analytical(self):
        """
        Verify gradient magnitude matches analytical computation.

        For point (b, d) matched to diagonal with L2 norm and q=2:
        Cost = ((b - (b+d)/2)^2 + (d - (b+d)/2)^2) = (b-d)^2/2

        Gradients:
        d(cost)/d(b) = b - d
        d(cost)/d(d) = d - b
        """
        b, d = 1.0, 2.0
        dgm_a = torch.tensor([[b, d]], dtype=torch.float64, requires_grad=True)
        dgm_b = torch.tensor([[100.0, 101.0]], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0, internal_p=2.0)
        cost.backward()

        # Analytical gradients
        expected_grad_b = b - d  # = -1.0
        expected_grad_d = d - b  # = 1.0

        grad_b = dgm_a.grad[0, 0].item()
        grad_d = dgm_a.grad[0, 1].item()

        assert abs(grad_b - expected_grad_b) < 1e-6, \
            f"Gradient w.r.t. birth mismatch: {grad_b} vs {expected_grad_b}"

        assert abs(grad_d - expected_grad_d) < 1e-6, \
            f"Gradient w.r.t. death mismatch: {grad_d} vs {expected_grad_d}"

    def test_gradient_with_multiple_points_to_diagonal(self):
        """Test gradients when multiple points match to diagonal."""
        dgm_a = torch.tensor([
            [1.0, 2.0],
            [3.0, 5.0]
        ], dtype=torch.float64, requires_grad=True)
        dgm_b = torch.tensor([[100.0, 101.0]], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0, internal_p=2.0)
        cost.backward()

        # Both points should have gradients pointing toward diagonal
        for i in range(2):
            grad_b = dgm_a.grad[i, 0].item()
            grad_d = dgm_a.grad[i, 1].item()

            # Equal magnitude, opposite signs
            assert abs(abs(grad_b) - abs(grad_d)) < 1e-6
            assert grad_b * grad_d < 0


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_very_small_diagrams(self):
        """Diagrams with very small persistence."""
        dgm_a = torch.tensor([[0.0, 1e-10]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.0, 2e-10]], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b)
        assert torch.isfinite(cost)

    def test_very_large_diagrams(self):
        """Diagrams with very large values."""
        dgm_a = torch.tensor([[0.0, 1e10]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.0, 1.1e10]], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b)
        assert torch.isfinite(cost)

    def test_mixed_device_types(self):
        """Test with different tensor dtypes."""
        dgm_a_f32 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        dgm_b_f32 = torch.tensor([[0.1, 0.9]], dtype=torch.float32)

        cost_f32 = oin_diff.wasserstein_cost(dgm_a_f32, dgm_b_f32)
        assert torch.isfinite(cost_f32)

        dgm_a_f64 = dgm_a_f32.to(torch.float64)
        dgm_b_f64 = dgm_b_f32.to(torch.float64)

        cost_f64 = oin_diff.wasserstein_cost(dgm_a_f64, dgm_b_f64)
        assert torch.isfinite(cost_f64)


class TestConsistency:
    """Test consistency with other distance functions."""

    def test_consistency_with_nondifferentiable_wasserstein_q1(self):
        """Differentiable cost should match non-differentiable distance for q=1."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 2.5]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8], [1.1, 2.3]], dtype=torch.float64)

        q = 1.0
        delta = 0.01

        # Differentiable cost
        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta)

        # Non-differentiable distance
        dist_nondiff = oineus.wasserstein_distance(
            dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta
        )

        # For q=1, cost should equal distance
        assert torch.allclose(
            cost_diff, torch.tensor(dist_nondiff, dtype=torch.float64),
            rtol=delta * 10
        ), f"Cost mismatch: {cost_diff.item()} vs {dist_nondiff}"

    def test_consistency_with_nondifferentiable_wasserstein_q2(self):
        """Differentiable cost^(1/q) should match non-differentiable distance for q=2."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 2.5]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8], [1.1, 2.3]], dtype=torch.float64)

        q = 2.0
        delta = 0.01

        # Differentiable cost
        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta)

        # Non-differentiable distance
        dist_nondiff = oineus.wasserstein_distance(
            dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta
        )

        # For q=2, cost^(1/2) should equal distance
        dist_from_cost = cost_diff ** (1.0 / q)

        assert torch.allclose(
            dist_from_cost, torch.tensor(dist_nondiff, dtype=torch.float64),
            rtol=delta * 10
        ), f"Distance mismatch: {dist_from_cost.item()} vs {dist_nondiff}"

    def test_consistency_with_nondifferentiable_wasserstein_q3(self):
        """Differentiable cost^(1/q) should match non-differentiable distance for q=3."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=torch.float64)

        q = 3.0
        delta = 0.01

        # Differentiable cost
        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta)

        # Non-differentiable distance
        dist_nondiff = oineus.wasserstein_distance(
            dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta
        )

        # For q=3, cost^(1/3) should equal distance
        dist_from_cost = cost_diff ** (1.0 / q)

        assert torch.allclose(
            dist_from_cost, torch.tensor(dist_nondiff, dtype=torch.float64),
            rtol=delta * 10
        ), f"Distance mismatch: {dist_from_cost.item()} vs {dist_nondiff}"

    def test_consistency_with_nondifferentiable_different_internal_p(self):
        """Test consistency with different internal_p values."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=torch.float64)

        q = 2.0
        delta = 0.01

        for internal_p in [1.0, 2.0, float('inf')]:
            # Differentiable cost
            cost_diff = oin_diff.wasserstein_cost(
                dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta, internal_p=internal_p
            )

            # Non-differentiable distance
            dist_nondiff = oineus.wasserstein_distance(
                dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta, internal_p=internal_p
            )

            # cost^(1/q) should equal distance
            dist_from_cost = cost_diff ** (1.0 / q)

            assert torch.allclose(
                dist_from_cost, torch.tensor(dist_nondiff, dtype=torch.float64),
                rtol=delta * 10
            ), f"Distance mismatch for internal_p={internal_p}: {dist_from_cost.item()} vs {dist_nondiff}"

    def test_consistency_with_essential_at_position_1(self):
        """Test with essential point at position 1 to check index handling."""
        # 3 finite points + 1 essential at position 1
        dgm_a = torch.tensor([
            [0.0, 1.0],           # finite
            [0.5, float('inf')],  # essential at position 1
            [1.0, 2.0],           # finite
            [1.5, 3.0]            # finite
        ], dtype=torch.float64)
        dgm_b = torch.tensor([
            [0.1, 0.9],           # finite
            [0.6, float('inf')],  # essential at position 1
            [1.1, 1.9],           # finite
            [1.6, 2.9]            # finite
        ], dtype=torch.float64)

        q = 1.0
        delta = 0.01

        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta, ignore_inf_points=False)
        dist_nondiff = oineus.wasserstein_distance(dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta)

        # For q=1, cost == distance
        assert torch.allclose(
            cost_diff, torch.tensor(dist_nondiff, dtype=torch.float64),
            rtol=delta * 10
        ), f"Distance mismatch with essential at pos 1: {cost_diff.item()} vs {dist_nondiff}"

    def test_consistency_with_essential_at_position_2(self):
        """Test with essential point at position 2 to check index handling."""
        # 3 finite points + 1 essential at position 2
        dgm_a = torch.tensor([
            [0.0, 1.0],           # finite
            [1.0, 2.0],           # finite
            [0.5, float('inf')],  # essential at position 2
            [1.5, 3.0]            # finite
        ], dtype=torch.float64)
        dgm_b = torch.tensor([
            [0.1, 0.9],           # finite
            [1.1, 1.9],           # finite
            [0.6, float('inf')],  # essential at position 2
            [1.6, 2.9]            # finite
        ], dtype=torch.float64)

        q = 1.0
        delta = 0.01

        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta, ignore_inf_points=False)
        dist_nondiff = oineus.wasserstein_distance(dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta)

        # For q=1, cost == distance
        assert torch.allclose(
            cost_diff, torch.tensor(dist_nondiff, dtype=torch.float64),
            rtol=delta * 10
        ), f"Distance mismatch with essential at pos 2: {cost_diff.item()} vs {dist_nondiff}"

    def test_consistency_with_essential_at_end(self):
        """Test with essential point at the end to check index handling."""
        # 3 finite points + 1 essential at the end
        dgm_a = torch.tensor([
            [0.0, 1.0],           # finite
            [1.0, 2.0],           # finite
            [1.5, 3.0],           # finite
            [0.5, float('inf')]   # essential at end
        ], dtype=torch.float64)
        dgm_b = torch.tensor([
            [0.1, 0.9],           # finite
            [1.1, 1.9],           # finite
            [1.6, 2.9],           # finite
            [0.6, float('inf')]   # essential at end
        ], dtype=torch.float64)

        q = 1.0
        delta = 0.01

        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta, ignore_inf_points=False)
        dist_nondiff = oineus.wasserstein_distance(dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta)

        # For q=1, cost == distance
        assert torch.allclose(
            cost_diff, torch.tensor(dist_nondiff, dtype=torch.float64),
            rtol=delta * 10
        ), f"Distance mismatch with essential at end: {cost_diff.item()} vs {dist_nondiff}"

    def test_consistency_with_large_random_diagrams(self):
        """Test with larger random diagrams (~50 points) plus essential points."""
        torch.manual_seed(42)  # For reproducibility
        np.random.seed(42)

        n_finite = 50

        # Generate random points from normal distribution
        points_a = torch.randn(n_finite, 2, dtype=torch.float64) * 0.5 + 1.0

        # Reflect below-diagonal points to ensure birth < death
        below_diag = points_a[:, 0] > points_a[:, 1]
        points_a[below_diag] = points_a[below_diag].flip(dims=[1])

        # Add small noise to get dgm_b
        points_b = points_a + torch.randn_like(points_a) * 0.05

        # Ensure birth < death after perturbation
        below_diag_b = points_b[:, 0] > points_b[:, 1]
        points_b[below_diag_b] = points_b[below_diag_b].flip(dims=[1])

        # Add essential points at different indices
        # Insert at positions 5, 20, 35
        essential_a = torch.tensor([
            [0.3, float('inf')],   # will be inserted at index 5
            [0.7, float('inf')],   # will be inserted at index 20
            [1.2, float('inf')]    # will be inserted at index 35
        ], dtype=torch.float64)

        essential_b = torch.tensor([
            [0.35, float('inf')],
            [0.75, float('inf')],
            [1.25, float('inf')]
        ], dtype=torch.float64)

        # Insert essential points
        dgm_a = torch.cat([
            points_a[:5],
            essential_a[0:1],
            points_a[5:20],
            essential_a[1:2],
            points_a[20:35],
            essential_a[2:3],
            points_a[35:]
        ], dim=0)

        dgm_b = torch.cat([
            points_b[:5],
            essential_b[0:1],
            points_b[5:20],
            essential_b[1:2],
            points_b[20:35],
            essential_b[2:3],
            points_b[35:]
        ], dim=0)

        q = 2.0
        delta = 0.01

        cost_diff = oin_diff.wasserstein_cost(
            dgm_a, dgm_b,
            wasserstein_q=q,
            wasserstein_delta=delta,
            ignore_inf_points=False
        )
        dist_nondiff = oineus.wasserstein_distance(
            dgm_a.numpy(), dgm_b.numpy(),
            q=q,
            delta=delta
        )

        # For q=2, cost = distance^2
        expected_cost = dist_nondiff ** q

        rel_error = abs(cost_diff.item() - expected_cost) / expected_cost
        assert rel_error < delta * 10, (
            f"Large diagram consistency failed: "
            f"cost={cost_diff.item():.6f}, "
            f"expected={expected_cost:.6f}, "
            f"distance={dist_nondiff:.6f}, "
            f"rel_error={rel_error:.6f}"
        )

    def test_consistency_with_sliced_wasserstein(self):
        """Both Wasserstein and sliced Wasserstein should give reasonable results."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=torch.float64)

        cost_wass = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=1.0)
        dist_sliced = oin_diff.sliced_wasserstein_distance(dgm_a, dgm_b, n_directions=100)

        # Both should be positive and finite
        assert cost_wass > 0 and torch.isfinite(cost_wass)
        assert dist_sliced > 0 and torch.isfinite(dist_sliced)

        # Wasserstein should be <= sliced Wasserstein (in general)
        # (This is not always true, but should hold approximately for these diagrams)
        # We just check they're in the same order of magnitude
        assert abs(np.log10(cost_wass.item()) - np.log10(dist_sliced.item())) < 2


class TestDifferentiability:
    """Test end-to-end differentiability."""

    def test_backward_pass(self):
        """Test that backward pass completes without error."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64, requires_grad=True)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=torch.float64)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0)
        cost.backward()

        assert dgm_a.grad is not None

    def test_gradient_descent_step(self):
        """Test that gradient descent reduces cost."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float64, requires_grad=True)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=torch.float64)

        initial_cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0)
        initial_cost.backward()

        # Take a small gradient descent step
        with torch.no_grad():
            dgm_a -= 0.01 * dgm_a.grad

        # Recompute cost (need to detach and create new tensor with grad)
        dgm_a_new = dgm_a.detach().clone().requires_grad_(True)
        new_cost = oin_diff.wasserstein_cost(dgm_a_new, dgm_b, wasserstein_q=2.0)

        # Cost should decrease
        assert new_cost < initial_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
