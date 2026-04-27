"""Comprehensive tests for differentiable Wasserstein distance."""

import pytest
import numpy as np
import sys
import os

from oineus._dtype import REAL_DTYPE

# Check if PyTorch is available (optional dependency)
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_DTYPE = torch.float32 if REAL_DTYPE == np.float32 else torch.float64
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DTYPE = None

# Tight: cost on identical diagrams should be effectively zero.
# Grad: precision floor for analytical gradient checks.
ABS_TIGHT = 1e-10 if REAL_DTYPE == np.float64 else 1e-5
GRAD_TOL = 1e-6 if REAL_DTYPE == np.float64 else 1e-3

# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

# NB: import oineus from whatever PYTHONPATH the test runner provides; do
# NOT hardcode a build directory here (the prior `sys.path.insert(0, ...)`
# pinned a stale Python-version-specific build dir and broke other configs).
import oineus
import oineus.diff as oin_diff


class TestWassersteinCostBasics:
    """Basic functionality tests."""

    def test_cost_matches_nondifferentiable_distance(self):
        """Verify cost^(1/q) matches non-differentiable wasserstein_distance."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=TORCH_DTYPE)

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
            cost_diff, torch.tensor(expected_cost, dtype=TORCH_DTYPE),
            rtol=delta * 10
        )

    def test_identical_diagrams(self):
        """Cost should be zero for identical diagrams."""
        dgm = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm, dgm.clone())

        assert cost < ABS_TIGHT, f"Cost should be near zero, got {cost.item()}"

    def test_different_q_values(self):
        """Test with different Wasserstein power parameters."""
        dgm_a = torch.tensor([[0.0, 1.0]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.1, 0.9]], dtype=TORCH_DTYPE)

        for q in [1.0, 2.0, 3.0]:
            cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q)
            assert cost >= 0, f"Cost should be non-negative for q={q}"

    def test_different_internal_p(self):
        """Test with different internal L_p norms."""
        dgm_a = torch.tensor([[0.0, 1.0]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.2, 0.8]], dtype=TORCH_DTYPE)

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
        dgm_a = torch.zeros((0, 2), dtype=TORCH_DTYPE)
        dgm_b = torch.zeros((0, 2), dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b)
        assert cost == 0.0

    def test_one_empty(self):
        """One diagram empty, one non-empty."""
        dgm_a = torch.zeros((0, 2), dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.0, 1.0], [0.5, 1.5]], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b)

        # Cost should equal the cost of matching dgm_b points to diagonal
        # For point (b, d), cost to diagonal = persistence^q / 2^q
        # With q=1 (default), cost = |d-b|/2 for each point
        expected = (1.0 / 2.0 + 1.0 / 2.0)  # Two points with persistence 1.0 each
        assert torch.allclose(cost, torch.tensor(expected, dtype=TORCH_DTYPE), rtol=0.1)


class TestEssentialPoints:
    """Test handling of essential points (with infinity)."""

    def test_essential_points_same_cardinality(self):
        """Essential points with matching cardinalities."""
        dgm_a = torch.tensor([
            [0.0, 1.0],
            [0.5, float('inf')]  # Essential point (finite, +inf)
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.1, 0.9],
            [0.6, float('inf')]  # Essential point (finite, +inf)
        ], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)
        assert cost >= 0

    def test_essential_points_different_cardinality_raises_error(self):
        """Essential points with different cardinalities should raise ValueError."""
        dgm_a = torch.tensor([
            [0.0, 1.0],
            [0.5, float('inf')]  # One essential point
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.1, 0.9],
            [0.6, float('inf')],
            [0.7, float('inf')]  # Two essential points
        ], dtype=TORCH_DTYPE)

        with pytest.raises(ValueError, match="essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

    def test_ignore_inf_points_flag(self):
        """With ignore_inf_points=True, different cardinalities should work."""
        dgm_a = torch.tensor([
            [0.0, 1.0],
            [0.5, float('inf')]  # One essential point
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.1, 0.9],
            [0.6, float('inf')],
            [0.7, float('inf')]  # Two essential points
        ], dtype=TORCH_DTYPE)

        # Should not raise error with ignore_inf_points=True (default)
        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=True)
        assert cost >= 0

    def test_only_essential_points(self):
        """Diagrams with only essential points."""
        dgm_a = torch.tensor([
            [0.5, float('inf')],
            [1.0, float('inf')]
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.6, float('inf')],
            [1.1, float('inf')]
        ], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

        # Cost should be sum of |0.6-0.5| + |1.1-1.0| = 0.1 + 0.1 = 0.2 for q=1
        assert torch.allclose(cost, torch.tensor(0.2, dtype=TORCH_DTYPE), rtol=0.01)

    def test_essential_points_q2(self):
        """Test essential points with q=2."""
        dgm_a = torch.tensor([
            [0.5, float('inf')],
            [1.0, float('inf')]
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.6, float('inf')],
            [1.1, float('inf')]
        ], dtype=TORCH_DTYPE)

        q = 2.0
        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, ignore_inf_points=False)
        dist_nondiff = oineus.wasserstein_distance(dgm_a.numpy(), dgm_b.numpy(), q=q, delta=0.01)

        # For q=2: cost = distance^2
        # distance^2 = (|0.6-0.5|^2 + |1.1-1.0|^2) = (0.1^2 + 0.1^2) = 0.02
        expected_cost = dist_nondiff ** q
        assert torch.allclose(cost_diff, torch.tensor(expected_cost, dtype=TORCH_DTYPE), rtol=0.01), (
            f"Essential q=2: cost={cost_diff.item()}, expected={expected_cost}, distance={dist_nondiff}"
        )

    def test_essential_points_q3(self):
        """Test essential points with q=3."""
        dgm_a = torch.tensor([
            [0.5, float('inf')],
            [1.0, float('inf')],
            [1.5, float('inf')]
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.7, float('inf')],
            [1.2, float('inf')],
            [1.6, float('inf')]
        ], dtype=TORCH_DTYPE)

        q = 3.0
        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, ignore_inf_points=False)
        dist_nondiff = oineus.wasserstein_distance(dgm_a.numpy(), dgm_b.numpy(), q=q, delta=0.01)

        # For q=3: cost = distance^3
        # distance^3 = (|0.7-0.5|^3 + |1.2-1.0|^3 + |1.6-1.5|^3)
        expected_cost = dist_nondiff ** q
        assert torch.allclose(cost_diff, torch.tensor(expected_cost, dtype=TORCH_DTYPE), rtol=0.01), (
            f"Essential q=3: cost={cost_diff.item()}, expected={expected_cost}, distance={dist_nondiff}"
        )

    def test_all_four_essential_categories(self):
        """Test all four categories of essential points."""
        dgm_a = torch.tensor([
            [0.0, 1.0],  # Finite
            [0.5, float('inf')],   # (finite, +inf)
            [1.0, float('-inf')],  # (finite, -inf)
            [float('inf'), 2.0],   # (+inf, finite)
            [float('-inf'), 3.0]   # (-inf, finite)
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.1, 0.9],  # Finite
            [0.6, float('inf')],   # (finite, +inf)
            [1.1, float('-inf')],  # (finite, -inf)
            [float('inf'), 2.1],   # (+inf, finite)
            [float('-inf'), 3.1]   # (-inf, finite)
        ], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)
        assert cost >= 0

    def test_mixed_essential_categories_raises_error(self):
        """Different essential categories should raise error."""
        # dgm_a has (finite, +inf) only
        dgm_a = torch.tensor([
            [0.0, 1.0],
            [0.5, float('inf')]  # (finite, +inf)
        ], dtype=TORCH_DTYPE)

        # dgm_b has both (finite, +inf) and (finite, -inf)
        dgm_b = torch.tensor([
            [0.1, 0.9],
            [0.6, float('inf')],   # (finite, +inf)
            [0.7, float('-inf')]   # (finite, -inf)
        ], dtype=TORCH_DTYPE)

        # Should raise error for (finite, -inf) mismatch (0 vs 1)
        with pytest.raises(ValueError, match="essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

    def test_essential_posinf_vs_neginf(self):
        """Test (finite, +inf) vs (finite, -inf) mismatch."""
        dgm_a = torch.tensor([
            [0.5, float('inf')]   # (finite, +inf)
        ], dtype=TORCH_DTYPE)

        dgm_b = torch.tensor([
            [0.5, float('-inf')]  # (finite, -inf)
        ], dtype=TORCH_DTYPE)

        # Different categories - should raise error
        with pytest.raises(ValueError, match="essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

    def test_essential_birth_vs_death_inf(self):
        """Test (+inf, finite) vs (finite, +inf) mismatch."""
        dgm_a = torch.tensor([
            [float('inf'), 2.0]   # (+inf, finite)
        ], dtype=TORCH_DTYPE)

        dgm_b = torch.tensor([
            [2.0, float('inf')]   # (finite, +inf)
        ], dtype=TORCH_DTYPE)

        # Different categories - should raise error
        with pytest.raises(ValueError, match="essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)

    def test_essential_multiple_category_mismatch(self):
        """Test complex mismatch: dgm_a has 2 categories, dgm_b has 3."""
        dgm_a = torch.tensor([
            [0.5, float('inf')],   # (finite, +inf)
            [float('inf'), 2.0]    # (+inf, finite)
        ], dtype=TORCH_DTYPE)

        dgm_b = torch.tensor([
            [0.6, float('inf')],    # (finite, +inf)
            [1.0, float('-inf')],   # (finite, -inf)
            [float('inf'), 2.1]     # (+inf, finite)
        ], dtype=TORCH_DTYPE)

        # Should raise error for (finite, -inf) category (0 vs 1)
        with pytest.raises(ValueError, match="essential point cardinalities must match"):
            oin_diff.wasserstein_cost(dgm_a, dgm_b, ignore_inf_points=False)


class TestGradients:
    """Test gradient computation and flow."""

    def test_gradient_flow_finite_to_finite(self):
        """Gradients should flow through finite-to-finite matches."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE, requires_grad=True)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=TORCH_DTYPE)

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
        dgm_a = torch.tensor([[1.0, 2.0]], dtype=TORCH_DTYPE, requires_grad=True)
        dgm_b = torch.tensor([[100.0, 101.0]], dtype=TORCH_DTYPE)

        # Use W_2 with internal_p=2 for easier gradient computation
        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0, internal_p=2.0)
        cost.backward()

        # Extract gradients
        grad_b = dgm_a.grad[0, 0].item()  # gradient w.r.t. birth
        grad_d = dgm_a.grad[0, 1].item()  # gradient w.r.t. death

        # Gradients should have equal magnitude, opposite signs
        assert abs(abs(grad_b) - abs(grad_d)) < GRAD_TOL, \
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
        dgm_a = torch.tensor([[b, d]], dtype=TORCH_DTYPE, requires_grad=True)
        dgm_b = torch.tensor([[100.0, 101.0]], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0, internal_p=2.0)
        cost.backward()

        # Analytical gradients
        expected_grad_b = b - d  # = -1.0
        expected_grad_d = d - b  # = 1.0

        grad_b = dgm_a.grad[0, 0].item()
        grad_d = dgm_a.grad[0, 1].item()

        assert abs(grad_b - expected_grad_b) < GRAD_TOL, \
            f"Gradient w.r.t. birth mismatch: {grad_b} vs {expected_grad_b}"

        assert abs(grad_d - expected_grad_d) < GRAD_TOL, \
            f"Gradient w.r.t. death mismatch: {grad_d} vs {expected_grad_d}"

    def test_gradient_with_multiple_points_to_diagonal(self):
        """Test gradients when multiple points match to diagonal."""
        dgm_a = torch.tensor([
            [1.0, 2.0],
            [3.0, 5.0]
        ], dtype=TORCH_DTYPE, requires_grad=True)
        dgm_b = torch.tensor([[100.0, 101.0]], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0, internal_p=2.0)
        cost.backward()

        # Both points should have gradients pointing toward diagonal
        for i in range(2):
            grad_b = dgm_a.grad[i, 0].item()
            grad_d = dgm_a.grad[i, 1].item()

            # Equal magnitude, opposite signs
            assert abs(abs(grad_b) - abs(grad_d)) < GRAD_TOL
            assert grad_b * grad_d < 0


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_very_small_diagrams(self):
        """Diagrams with very small persistence."""
        dgm_a = torch.tensor([[0.0, 1e-10]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.0, 2e-10]], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b)
        assert torch.isfinite(cost)

    def test_very_large_diagrams(self):
        """Diagrams with very large values."""
        dgm_a = torch.tensor([[0.0, 1e10]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.0, 1.1e10]], dtype=TORCH_DTYPE)

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
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 2.5]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8], [1.1, 2.3]], dtype=TORCH_DTYPE)

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
            cost_diff, torch.tensor(dist_nondiff, dtype=TORCH_DTYPE),
            rtol=delta * 10
        ), f"Cost mismatch: {cost_diff.item()} vs {dist_nondiff}"

    def test_consistency_with_nondifferentiable_wasserstein_q2(self):
        """Differentiable cost^(1/q) should match non-differentiable distance for q=2."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 2.5]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8], [1.1, 2.3]], dtype=TORCH_DTYPE)

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
            dist_from_cost, torch.tensor(dist_nondiff, dtype=TORCH_DTYPE),
            rtol=delta * 10
        ), f"Distance mismatch: {dist_from_cost.item()} vs {dist_nondiff}"

    def test_consistency_with_nondifferentiable_wasserstein_q3(self):
        """Differentiable cost^(1/q) should match non-differentiable distance for q=3."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=TORCH_DTYPE)

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
            dist_from_cost, torch.tensor(dist_nondiff, dtype=TORCH_DTYPE),
            rtol=delta * 10
        ), f"Distance mismatch: {dist_from_cost.item()} vs {dist_nondiff}"

    def test_consistency_with_nondifferentiable_different_internal_p(self):
        """Test consistency with different internal_p values."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=TORCH_DTYPE)

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
                dist_from_cost, torch.tensor(dist_nondiff, dtype=TORCH_DTYPE),
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
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.1, 0.9],           # finite
            [0.6, float('inf')],  # essential at position 1
            [1.1, 1.9],           # finite
            [1.6, 2.9]            # finite
        ], dtype=TORCH_DTYPE)

        q = 1.0
        delta = 0.01

        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta, ignore_inf_points=False)
        dist_nondiff = oineus.wasserstein_distance(dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta)

        # For q=1, cost == distance
        assert torch.allclose(
            cost_diff, torch.tensor(dist_nondiff, dtype=TORCH_DTYPE),
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
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.1, 0.9],           # finite
            [1.1, 1.9],           # finite
            [0.6, float('inf')],  # essential at position 2
            [1.6, 2.9]            # finite
        ], dtype=TORCH_DTYPE)

        q = 1.0
        delta = 0.01

        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta, ignore_inf_points=False)
        dist_nondiff = oineus.wasserstein_distance(dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta)

        # For q=1, cost == distance
        assert torch.allclose(
            cost_diff, torch.tensor(dist_nondiff, dtype=TORCH_DTYPE),
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
        ], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([
            [0.1, 0.9],           # finite
            [1.1, 1.9],           # finite
            [1.6, 2.9],           # finite
            [0.6, float('inf')]   # essential at end
        ], dtype=TORCH_DTYPE)

        q = 1.0
        delta = 0.01

        cost_diff = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=q, wasserstein_delta=delta, ignore_inf_points=False)
        dist_nondiff = oineus.wasserstein_distance(dgm_a.numpy(), dgm_b.numpy(), q=q, delta=delta)

        # For q=1, cost == distance
        assert torch.allclose(
            cost_diff, torch.tensor(dist_nondiff, dtype=TORCH_DTYPE),
            rtol=delta * 10
        ), f"Distance mismatch with essential at end: {cost_diff.item()} vs {dist_nondiff}"

    def test_consistency_with_large_random_diagrams(self):
        """Test with larger random diagrams (~50 points) plus essential points."""
        torch.manual_seed(42)  # For reproducibility
        np.random.seed(42)

        n_finite = 50

        # Generate random points from normal distribution
        points_a = torch.randn(n_finite, 2, dtype=TORCH_DTYPE) * 0.5 + 1.0

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
        ], dtype=TORCH_DTYPE)

        essential_b = torch.tensor([
            [0.35, float('inf')],
            [0.75, float('inf')],
            [1.25, float('inf')]
        ], dtype=TORCH_DTYPE)

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
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=TORCH_DTYPE)

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
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE, requires_grad=True)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=TORCH_DTYPE)

        cost = oin_diff.wasserstein_cost(dgm_a, dgm_b, wasserstein_q=2.0)
        cost.backward()

        assert dgm_a.grad is not None

    def test_gradient_descent_step(self):
        """Test that gradient descent reduces cost."""
        dgm_a = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE, requires_grad=True)
        dgm_b = torch.tensor([[0.1, 0.9], [0.6, 1.8]], dtype=TORCH_DTYPE)

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


class TestEssentialSortMatching:
    """Verify that the four essential families are paired by sorted-rank of the
    finite coordinate even when input order is scrambled.

    Layout: 14 points per side = 3 of each essential family + 2 finite. Within
    each family the finite-coord magnitudes are deliberately on different
    decade scales (essentials in 1, 10, 100, 1000; finite in 5000+) so that
    any incorrect cross-family pairing would yield a wildly wrong cost.

    Hand-derived pairings (sorted by finite coord, paired by rank):

      inf_death (axis 0, birth):
        a: idx9(0.1), idx1(0.5), idx5(0.9)
        b: idx1(0.2), idx7(0.6), idx10(0.8)
        |Δ|^2 = 0.01 + 0.01 + 0.01 = 0.03

      neg_inf_death (axis 0, birth):
        a: idx13(6.0), idx4(8.0), idx10(9.5)
        b: idx6(5.5),  idx13(7.5), idx0(10.0)
        |Δ|^2 = 0.25 + 0.25 + 0.25 = 0.75

      inf_birth (axis 1, death):
        a: idx6(60.0), idx0(75.0),  idx11(90.0)
        b: idx5(55.0), idx2(80.0),  idx12(95.0)
        |Δ|^2 = 25 + 25 + 25 = 75

      neg_inf_birth (axis 1, death):
        a: idx7(600.0), idx12(750.0), idx3(900.0)
        b: idx3(550.0), idx11(800.0), idx8(950.0)
        |Δ|^2 = 2500 + 2500 + 2500 = 7500

      finite (W2, internal_p=inf):
        a2=(5000,5100) ↔ b4=(5005,5105): max(|5|,|5|)^2 = 25
        a8=(8000,8100) ↔ b9=(8005,8105): max(|5|,|5|)^2 = 25
        total finite contribution = 50.

      grand total = 50 + 0.03 + 0.75 + 75 + 7500 = 7625.78
    """

    # --- shared input data: hard-coded, scrambled, no random ---
    @staticmethod
    def _make_dgm_a():
        INF = float("inf")
        return torch.tensor([
            [INF,    75.0],     # 0  inf_birth
            [0.5,    INF],      # 1  inf_death
            [5000.0, 5100.0],   # 2  finite
            [-INF,   900.0],    # 3  neg_inf_birth
            [8.0,    -INF],     # 4  neg_inf_death
            [0.9,    INF],      # 5  inf_death
            [INF,    60.0],     # 6  inf_birth
            [-INF,   600.0],    # 7  neg_inf_birth
            [8000.0, 8100.0],   # 8  finite
            [0.1,    INF],      # 9  inf_death
            [9.5,    -INF],     # 10 neg_inf_death
            [INF,    90.0],     # 11 inf_birth
            [-INF,   750.0],    # 12 neg_inf_birth
            [6.0,    -INF],     # 13 neg_inf_death
        ], dtype=TORCH_DTYPE)

    @staticmethod
    def _make_dgm_b():
        INF = float("inf")
        return torch.tensor([
            [10.0,   -INF],     # 0  neg_inf_death
            [0.2,    INF],      # 1  inf_death
            [INF,    80.0],     # 2  inf_birth
            [-INF,   550.0],    # 3  neg_inf_birth
            [5005.0, 5105.0],   # 4  finite
            [INF,    55.0],     # 5  inf_birth
            [5.5,    -INF],     # 6  neg_inf_death
            [0.6,    INF],      # 7  inf_death
            [-INF,   950.0],    # 8  neg_inf_birth
            [8005.0, 8105.0],   # 9  finite
            [0.8,    INF],      # 10 inf_death
            [-INF,   800.0],    # 11 neg_inf_birth
            [INF,    95.0],     # 12 inf_birth
            [7.5,    -INF],     # 13 neg_inf_death
        ], dtype=TORCH_DTYPE)

    # Index sets for each family on each side. Used to assert grad patterns.
    A_FAM = {
        "inf_death":     [1, 5, 9],
        "neg_inf_death": [4, 10, 13],
        "inf_birth":     [0, 6, 11],
        "neg_inf_birth": [3, 7, 12],
        "finite":        [2, 8],
    }
    A_FINITE_AXIS = {
        "inf_death":     0,
        "neg_inf_death": 0,
        "inf_birth":     1,
        "neg_inf_birth": 1,
    }

    # Finite-only sub-diagrams used to validate the finite contribution
    # without re-using wasserstein_cost on the full input (avoids circularity).
    @staticmethod
    def _make_finite_a():
        return torch.tensor([[5000.0, 5100.0], [8000.0, 8100.0]],
                            dtype=TORCH_DTYPE)

    @staticmethod
    def _make_finite_b():
        return torch.tensor([[5005.0, 5105.0], [8005.0, 8105.0]],
                            dtype=TORCH_DTYPE)

    # Per-family essential cost contributions (sum of |Δ|^q over rank-paired
    # finite coords). Hard-coded from the comment block above for q=2.
    EXPECTED_ESSENTIAL_COST_Q2 = {
        "inf_death":     0.03,
        "neg_inf_death": 0.75,
        "inf_birth":     75.0,
        "neg_inf_birth": 7500.0,
    }
    EXPECTED_FINITE_COST_Q2 = 50.0   # 25 + 25 under L_inf, q=2
    EXPECTED_TOTAL_COST_Q2  = 7625.78  # 50 + 0.03 + 0.75 + 75 + 7500

    DELTA = 0.001  # tight; finite distances are exact, essentials closed-form

    # ------------------------------------------------------------------
    def test_ignore_inf_points_drops_essentials(self):
        """ignore_inf_points=True must drop all 12 essentials per side and
        return the cost of matching just the two finite points."""
        dgm_a = self._make_dgm_a()
        dgm_b = self._make_dgm_b()

        cost = oin_diff.wasserstein_cost(
            dgm_a, dgm_b,
            wasserstein_q=2.0, wasserstein_delta=self.DELTA,
            ignore_inf_points=True,
        )

        # Hand-built finite-only diagrams should produce the same cost.
        finite_a = self._make_finite_a()
        finite_b = self._make_finite_b()
        finite_cost = oin_diff.wasserstein_cost(
            finite_a, finite_b,
            wasserstein_q=2.0, wasserstein_delta=self.DELTA,
            ignore_inf_points=True,
        )

        assert torch.isclose(
            cost, finite_cost, rtol=self.DELTA * 10
        ), f"cost={cost.item()} vs finite_cost={finite_cost.item()}"
        assert torch.isclose(
            cost,
            torch.tensor(self.EXPECTED_FINITE_COST_Q2, dtype=TORCH_DTYPE),
            rtol=self.DELTA * 10,
        ), f"cost={cost.item()} vs expected={self.EXPECTED_FINITE_COST_Q2}"

    # ------------------------------------------------------------------
    def test_essentials_paired_by_rank_value(self):
        """With ignore_inf_points=False, the cost must equal the
        finite-only contribution plus the per-family rank-paired sum-of-
        squared finite-coord differences."""
        dgm_a = self._make_dgm_a()
        dgm_b = self._make_dgm_b()

        cost = oin_diff.wasserstein_cost(
            dgm_a, dgm_b,
            wasserstein_q=2.0, wasserstein_delta=self.DELTA,
            ignore_inf_points=False,
        )

        # Build the expected total by hand from the per-family rule. We do
        # NOT call wasserstein_cost on the same essentials to avoid
        # circularity; the constants come from the comment block above.
        expected = self.EXPECTED_FINITE_COST_Q2 + sum(
            self.EXPECTED_ESSENTIAL_COST_Q2.values()
        )
        assert abs(expected - self.EXPECTED_TOTAL_COST_Q2) < 1e-9

        assert torch.isclose(
            cost,
            torch.tensor(self.EXPECTED_TOTAL_COST_Q2, dtype=TORCH_DTYPE),
            rtol=self.DELTA * 10,
        ), f"cost={cost.item()} vs expected={self.EXPECTED_TOTAL_COST_Q2}"

        # Also assert each family's contribution matches the per-pair sum
        # we expect by sorted-rank pairing — a stronger statement than just
        # the total. We compute the per-family contribution directly from
        # the input tensors here (no call to wasserstein_cost).
        per_family_expected = {}
        for name, axis in self.A_FINITE_AXIS.items():
            a_idx = self.A_FAM[name]
            # Identify the b-side indices for this family by matching the
            # infinite-axis sign pattern (independent of order).
            b_idx = []
            for i in range(dgm_b.shape[0]):
                bx, by = dgm_b[i, 0].item(), dgm_b[i, 1].item()
                if name == "inf_death" and np.isfinite(bx) and by == float("inf"):
                    b_idx.append(i)
                elif name == "neg_inf_death" and np.isfinite(bx) and by == float("-inf"):
                    b_idx.append(i)
                elif name == "inf_birth" and bx == float("inf") and np.isfinite(by):
                    b_idx.append(i)
                elif name == "neg_inf_birth" and bx == float("-inf") and np.isfinite(by):
                    b_idx.append(i)
            assert len(a_idx) == len(b_idx) == 3, name
            a_coords = sorted(dgm_a[i, axis].item() for i in a_idx)
            b_coords = sorted(dgm_b[i, axis].item() for i in b_idx)
            per_family_expected[name] = sum(
                (a - b) ** 2 for a, b in zip(a_coords, b_coords)
            )
        # Check the per-family cost we computed matches the hard-coded
        # constants used to build the total (sanity on our table). The
        # per-family v is summed from tensor .item()s, so it carries
        # REAL_DTYPE roundoff -- compare with the float-precision floor.
        for k, v in per_family_expected.items():
            assert abs(v - self.EXPECTED_ESSENTIAL_COST_Q2[k]) < ABS_TIGHT, (
                k, v, self.EXPECTED_ESSENTIAL_COST_Q2[k]
            )

    # ------------------------------------------------------------------
    def test_essentials_gradient_flow(self):
        """Backprop through the full diagram with essentials present must
        give non-zero gradients on the finite axis of every essential and
        exactly-zero (and finite) gradients on the infinite axis."""
        dgm_a = self._make_dgm_a().requires_grad_(True)
        dgm_b = self._make_dgm_b()

        cost = oin_diff.wasserstein_cost(
            dgm_a, dgm_b,
            wasserstein_q=2.0, wasserstein_delta=self.DELTA,
            ignore_inf_points=False,
        )
        cost.backward()

        grad = dgm_a.grad
        assert grad is not None
        # No NaN/Inf must propagate from the essentials.
        assert not torch.isnan(grad).any().item(), grad
        assert not torch.isinf(grad).any().item(), grad

        # For each essential family: finite-axis grad must be non-zero,
        # infinite-axis grad must be exactly zero.
        for name, axis in self.A_FINITE_AXIS.items():
            inf_axis = 1 - axis
            for i in self.A_FAM[name]:
                g_finite = grad[i, axis].item()
                g_inf    = grad[i, inf_axis].item()
                assert g_finite != 0.0, (
                    f"{name} a-row {i}: finite-axis grad is zero")
                assert g_inf == 0.0, (
                    f"{name} a-row {i}: inf-axis grad is {g_inf}, expected 0")

        # Finite rows that actually got matched (under L_inf at the chosen
        # input scales they all do) must have at least one non-zero entry.
        for i in self.A_FAM["finite"]:
            row = grad[i]
            assert torch.any(row != 0).item(), (
                f"finite a-row {i} has all-zero gradient: {row}")

        # Compare against the ignore_inf_points=True run: essentials should
        # then contribute zero gradient (only finite-finite matches feed back).
        dgm_a2 = self._make_dgm_a().requires_grad_(True)
        cost2 = oin_diff.wasserstein_cost(
            dgm_a2, dgm_b,
            wasserstein_q=2.0, wasserstein_delta=self.DELTA,
            ignore_inf_points=True,
        )
        cost2.backward()
        grad2 = dgm_a2.grad
        for name in self.A_FINITE_AXIS:
            for i in self.A_FAM[name]:
                assert torch.all(grad2[i] == 0).item(), (
                    f"{name} a-row {i} should have zero grad when essentials"
                    f" are ignored, got {grad2[i]}")

    # ------------------------------------------------------------------
    def test_essentials_gradient_zero_when_ignored(self):
        """Standalone: ignore_inf_points=True backward => all essential
        rows in dgm_a.grad are exactly zero."""
        dgm_a = self._make_dgm_a().requires_grad_(True)
        dgm_b = self._make_dgm_b()

        cost = oin_diff.wasserstein_cost(
            dgm_a, dgm_b,
            wasserstein_q=2.0, wasserstein_delta=self.DELTA,
            ignore_inf_points=True,
        )
        cost.backward()

        grad = dgm_a.grad
        assert grad is not None
        for name in self.A_FINITE_AXIS:
            for i in self.A_FAM[name]:
                assert torch.all(grad[i] == 0).item(), (
                    f"{name} a-row {i}: expected all-zero grad, got {grad[i]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
