"""Tests for wasserstein_matching and DiagramMatching."""

import pytest
import numpy as np
import sys
import os

# Add build directory to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build/bindings/python'))

import oineus


class TestBasicMatching:
    """Test basic finite point matching."""

    def test_identical_diagrams(self):
        """Two identical diagrams should match identity."""
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
        dgm_b = dgm_a.copy()

        matching = oineus.wasserstein_matching(dgm_a, dgm_b)

        assert len(matching.finite_to_finite) == 2
        assert len(matching.a_to_diagonal) == 0
        assert len(matching.b_to_diagonal) == 0
        assert matching.distance < 1e-10

    def test_two_point_matching(self):
        """Simple two-point matching."""
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
        dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)

        # Should have 2 finite-to-finite matches
        assert len(matching.finite_to_finite) == 2
        assert len(matching.a_to_diagonal) == 0
        assert len(matching.b_to_diagonal) == 0

        # Indices should be valid
        for idx_a, idx_b in matching.finite_to_finite:
            assert 0 <= idx_a < len(dgm_a)
            assert 0 <= idx_b < len(dgm_b)

    def test_three_point_matching(self):
        """Three well-separated points."""
        dgm_a = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        dgm_b = np.array([[0.1, 0.9], [1.1, 1.9], [2.1, 2.9]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=1.0)

        assert len(matching.finite_to_finite) == 3
        assert len(matching.a_to_diagonal) == 0
        assert len(matching.b_to_diagonal) == 0


class TestDiagonalMatching:
    """Test matching to diagonal projections."""

    def test_one_extra_point_in_a(self):
        """Diagram A has one extra point."""
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
        dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)

        # Should have 2 finite-to-finite and 1 to diagonal
        assert len(matching.finite_to_finite) == 2
        assert len(matching.a_to_diagonal) == 1
        assert len(matching.b_to_diagonal) == 0

        # Check index validity
        for idx in matching.a_to_diagonal:
            assert 0 <= idx < len(dgm_a)

    def test_one_extra_point_in_b(self):
        """Diagram B has one extra point."""
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
        dgm_b = np.array([[0.1, 0.9], [0.6, 1.8], [1.6, 2.8]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)

        assert len(matching.finite_to_finite) == 2
        assert len(matching.a_to_diagonal) == 0
        assert len(matching.b_to_diagonal) == 1

        for idx in matching.b_to_diagonal:
            assert 0 <= idx < len(dgm_b)

    def test_multiple_diagonal_matches(self):
        """Multiple points matched to diagonal."""
        dgm_a = np.array([[0.0, 1.0]])
        dgm_b = np.array([[0.1, 0.9], [1.1, 1.9], [2.1, 2.9]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=1.0)

        assert len(matching.finite_to_finite) == 1
        assert len(matching.a_to_diagonal) == 0
        assert len(matching.b_to_diagonal) == 2


class TestEssentialPoints:
    """Test matching with essential points."""

    def test_single_essential_category(self):
        """Single category (finite, +inf)."""
        dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
        dgm_b = np.array([[0.1, 0.9], [0.6, np.inf]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b,
                                              ignore_inf_points=False)

        assert len(matching.finite_to_finite) == 1
        assert "(finite, +inf)" in matching.essential_matches
        assert len(matching.essential_matches["(finite, +inf)"]) == 1

        # Check indices refer to original diagrams
        idx_a, idx_b = matching.essential_matches["(finite, +inf)"][0]
        assert dgm_a[idx_a][1] == np.inf
        assert dgm_b[idx_b][1] == np.inf

    def test_multiple_essential_points(self):
        """Multiple essential points in same category."""
        dgm_a = np.array([
            [0.0, 1.0],
            [0.5, np.inf],
            [1.0, np.inf],
            [1.5, np.inf]
        ])
        dgm_b = np.array([
            [0.1, 0.9],
            [0.6, np.inf],
            [1.1, np.inf],
            [1.6, np.inf]
        ])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b,
                                              ignore_inf_points=False)

        assert len(matching.essential_matches["(finite, +inf)"]) == 3

        # All indices should be valid
        for idx_a, idx_b in matching.essential_matches["(finite, +inf)"]:
            assert 0 <= idx_a < len(dgm_a)
            assert 0 <= idx_b < len(dgm_b)

    def test_essential_cardinality_mismatch(self):
        """Should raise error if cardinalities don't match."""
        dgm_a = np.array([[0.5, np.inf], [1.0, np.inf]])
        dgm_b = np.array([[0.6, np.inf]])

        with pytest.raises(ValueError, match="cardinalities must match"):
            oineus.wasserstein_matching(dgm_a, dgm_b, ignore_inf_points=False)

    def test_ignore_essential_points(self):
        """With ignore_inf_points=True, should skip essential."""
        dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
        dgm_b = np.array([[0.1, 0.9], [0.6, np.inf]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b,
                                              ignore_inf_points=True)

        assert len(matching.essential_matches) == 0
        assert len(matching.finite_to_finite) == 1


class TestIndexCorrectness:
    """Test that indices correctly refer to original diagrams."""

    def test_essential_at_different_positions(self):
        """Essential point at position 1 in 4-point diagram."""
        dgm_a = np.array([
            [0.0, 1.0],      # index 0 - finite
            [0.5, np.inf],   # index 1 - essential
            [1.0, 2.0],      # index 2 - finite
            [1.5, 3.0]       # index 3 - finite
        ])
        dgm_b = np.array([
            [0.1, 0.9],      # index 0 - finite
            [0.6, np.inf],   # index 1 - essential
            [1.1, 1.9],      # index 2 - finite
            [1.6, 2.9]       # index 3 - finite
        ])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b,
                                              ignore_inf_points=False)

        # Essential match should involve indices 1 and 1
        assert (1, 1) in matching.essential_matches["(finite, +inf)"]

        # Finite matches should involve indices 0,2,3
        finite_a_indices = set(idx for idx, _ in matching.finite_to_finite)
        finite_b_indices = set(idx for _, idx in matching.finite_to_finite)
        assert finite_a_indices == {0, 2, 3}
        assert finite_b_indices == {0, 2, 3}

    def test_essential_at_end(self):
        """Essential point at the end."""
        dgm_a = np.array([
            [0.0, 1.0],
            [0.5, 2.0],
            [1.5, np.inf]    # index 2 - essential at end
        ])
        dgm_b = np.array([
            [0.1, 0.9],
            [0.6, 1.8],
            [1.6, np.inf]    # index 2 - essential at end
        ])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b,
                                              ignore_inf_points=False)

        assert (2, 2) in matching.essential_matches["(finite, +inf)"]

        # Verify we can index into original diagrams
        idx_a, idx_b = matching.essential_matches["(finite, +inf)"][0]
        assert dgm_a[idx_a][0] == 1.5
        assert dgm_b[idx_b][0] == 1.6


class TestDistanceConsistency:
    """Test matching distance matches wasserstein_distance."""

    def test_distance_matches_wasserstein_distance(self):
        """Distance should match wasserstein_distance function."""
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
        dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])

        q = 2.0
        delta = 0.01

        matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=q, delta=delta)
        distance_direct = oineus.wasserstein_distance(dgm_a, dgm_b, q=q, delta=delta)

        assert abs(matching.distance - distance_direct) < delta * distance_direct

    def test_cost_equals_distance_power_q(self):
        """Cost should equal distance^q."""
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
        dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])

        q = 3.0
        matching = oineus.wasserstein_matching(dgm_a, dgm_b, q=q)

        expected_cost = matching.distance ** q
        assert abs(matching.cost - expected_cost) < 1e-10


class TestPointToDiagonal:
    """Test point_to_diagonal helper function."""

    def test_basic_usage_all_points(self):
        """Project all points at once (default)."""
        dgm = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])

        projs = oineus.point_to_diagonal(dgm)

        assert projs.shape == (3, 2)
        assert np.allclose(projs[0], [0.5, 0.5])
        assert np.allclose(projs[1], [1.25, 1.25])
        assert np.allclose(projs[2], [2.25, 2.25])

    def test_specific_indices(self):
        """Project specific points."""
        dgm = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])

        projs = oineus.point_to_diagonal(dgm, indices=[0, 2])

        assert projs.shape == (2, 2)
        assert np.allclose(projs[0], [0.5, 0.5])
        assert np.allclose(projs[1], [2.25, 2.25])

    def test_with_matching_result(self):
        """Use with matching a_to_diagonal - compute all at once."""
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
        dgm_b = np.array([[0.1, 0.9]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b)

        # Should have 2 points matched to diagonal
        assert len(matching.a_to_diagonal) == 2

        # Get all projections at once (efficient!)
        projs_a = oineus.point_to_diagonal(dgm_a)

        # Check projections for matched points
        for idx in matching.a_to_diagonal:
            assert projs_a[idx, 0] == projs_a[idx, 1]  # On diagonal

    def test_with_list_input(self):
        """Works with list input - returns list."""
        dgm = [[0.0, 1.0], [0.5, 2.0]]
        projs = oineus.point_to_diagonal(dgm)
        assert isinstance(projs, list)
        assert len(projs) == 2
        assert projs[0] == (0.5, 0.5)
        assert projs[1] == (1.25, 1.25)


class TestEmptyDiagrams:
    """Test edge cases with empty diagrams."""

    def test_both_empty(self):
        """Both diagrams empty."""
        dgm_a = np.zeros((0, 2))
        dgm_b = np.zeros((0, 2))

        matching = oineus.wasserstein_matching(dgm_a, dgm_b)

        assert len(matching.finite_to_finite) == 0
        assert len(matching.a_to_diagonal) == 0
        assert len(matching.b_to_diagonal) == 0
        assert matching.distance == 0.0

    def test_one_empty(self):
        """One diagram empty."""
        dgm_a = np.zeros((0, 2))
        dgm_b = np.array([[0.0, 1.0], [0.5, 2.0]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b)

        assert len(matching.finite_to_finite) == 0
        assert len(matching.a_to_diagonal) == 0
        assert len(matching.b_to_diagonal) == 2
        assert matching.distance > 0


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
        dgm_b = np.array([[0.1, 0.9]])

        matching = oineus.wasserstein_matching(dgm_a, dgm_b)

        repr_str = repr(matching)
        assert "DiagramMatching" in repr_str
        assert "finite_to_finite" in repr_str
        assert "distance" in repr_str
