"""Tests for bottleneck_matching and BottleneckMatching."""

import numpy as np
import pytest

import oineus
from oineus import (
    BottleneckMatching,
    EssentialLongestEdge,
    FiniteLongestEdge,
    InfKind,
    bottleneck_matching,
    bottleneck_distance,
)
from oineus._dtype import REAL_DTYPE

# Bottleneck is exact, but Hera computes in REAL_DTYPE so non-machine-representable
# answers (e.g. 0.2 in float32) carry roundoff at the precision floor.
ABS_TOL = 1e-9 if REAL_DTYPE == np.float64 else 1e-5


class TestBasic:
    def test_identical_diagrams(self):
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
        dgm_b = dgm_a.copy()
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert isinstance(m, BottleneckMatching)
        assert m.distance == 0.0
        # Identical-diagrams fast path: matches each to itself, longest is 0-length.
        assert len(m.finite_to_finite) == 2
        # No spurious essential matches
        for edges in m.longest.essential.values():
            assert edges == []

    def test_empty_diagrams(self):
        m = bottleneck_matching(np.empty((0, 2)), np.empty((0, 2)), delta=0.0)
        assert m.distance == 0.0
        assert m.finite_to_finite.shape == (0, 2)
        assert m.longest.finite == []


class TestFiniteMatching:
    def test_two_point_bottleneck_l_inf(self):
        dgm_a = np.array([[0.0, 5.0]])
        dgm_b = np.array([[1.0, 6.0]])
        # bottleneck L-inf: matching A0<->B0 gives distance 1.0 (cheaper than both to diag at 2.5).
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert m.distance == pytest.approx(1.0, abs=ABS_TOL)
        assert len(m.finite_to_finite) == 1
        assert len(m.longest.finite) == 1

        e = m.longest.finite[0]
        assert isinstance(e, FiniteLongestEdge)
        assert e.idx_a == 0
        assert e.idx_b == 0
        assert e.length == pytest.approx(1.0, abs=ABS_TOL)

    def test_match_to_diagonal(self):
        # Single low-persistence point vs empty: must match to diagonal.
        dgm_a = np.array([[0.0, 0.5]])
        dgm_b = np.empty((0, 2))
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert m.distance == pytest.approx(0.25, abs=ABS_TOL)
        assert m.a_to_diagonal.tolist() == [0]
        assert len(m.longest.finite) == 1
        e = m.longest.finite[0]
        assert e.idx_a == 0
        assert e.idx_b is None
        assert e.point_b == pytest.approx((0.25, 0.25))

    def test_integer_grid_ties(self):
        # Each point has persistence 1; optimal is match-to-own-diagonal at cost 0.5.
        # All four of (A0->diag, A1->diag, B0->diag, B1->diag) are tied for the bottleneck.
        dgm_a = np.array([[0.0, 1.0], [10.0, 11.0]])
        dgm_b = np.array([[1.0, 2.0], [11.0, 12.0]])
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert m.distance == pytest.approx(0.5, abs=ABS_TOL)
        # Full matching: every point matched to its own diagonal.
        assert sorted(m.a_to_diagonal.tolist()) == [0, 1]
        assert sorted(m.b_to_diagonal.tolist()) == [0, 1]
        assert m.finite_to_finite.shape == (0, 2)
        # All four edges tied for the bottleneck.
        assert len(m.longest.finite) == 4
        for e in m.longest.finite:
            assert e.length == pytest.approx(0.5, abs=ABS_TOL)
            assert (e.idx_a is None) ^ (e.idx_b is None)  # exactly one side diagonal


class TestEssential:
    def test_essential_matched_by_sort(self):
        # Two (finite, +inf) points in each; match by sorting births.
        dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
        dgm_b = np.array([[0.1, 0.9], [0.7, np.inf]])
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0, ignore_inf_points=False)

        # Mix attr access and enum-indexed access.
        assert [tuple(map(int, p)) for p in m.essential.inf_death] == [(1, 1)]
        assert [tuple(map(int, p)) for p in m.essential[InfKind.INF_DEATH]] == [(1, 1)]
        assert len(m.longest.essential.inf_death) == 1
        e = m.longest.essential[InfKind.INF_DEATH]
        assert isinstance(e[0], EssentialLongestEdge)
        assert e[0].idx_a == 1
        assert e[0].idx_b == 1
        assert e[0].coord_a == pytest.approx(0.5)
        assert e[0].coord_b == pytest.approx(0.7)
        assert e[0].length == pytest.approx(0.2, abs=ABS_TOL)

    def test_essential_cardinality_mismatch_raises(self):
        dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
        dgm_b = np.array([[0.1, 0.9]])  # no +inf point
        with pytest.raises(ValueError, match="cardinalities must match"):
            bottleneck_matching(dgm_a, dgm_b, delta=0.0, ignore_inf_points=False)

    def test_finite_longest_preserved_under_essential_bottleneck(self):
        # Finite part has a small bottleneck; essential part dominates.
        dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
        dgm_b = np.array([[0.1, 0.9], [5.0, np.inf]])  # essential gap = 4.5
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0, ignore_inf_points=False)
        assert m.distance == pytest.approx(4.5, abs=ABS_TOL)
        # Essential longest records the 4.5-long edge.
        ess = m.longest.essential.inf_death
        assert len(ess) == 1
        assert ess[0].length == pytest.approx(4.5, abs=ABS_TOL)
        # Finite longest is still populated with the finite-part's local max.
        assert len(m.longest.finite) >= 1
        for e in m.longest.finite:
            assert e.length < 4.5  # strictly less than the global bottleneck

    def test_ignore_inf_skips_essentials(self):
        dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
        dgm_b = np.array([[0.1, 0.9]])
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0, ignore_inf_points=True)
        # No ValueError raised; essential matches are empty (0, 2) arrays.
        assert all(v.shape == (0, 2) for v in m.essential.values())


class TestCrossCheck:
    def test_distance_matches_bottleneck_distance(self):
        dgm_a = np.array([[0.0, 3.0], [0.5, 4.0]])
        dgm_b = np.array([[0.2, 2.5], [0.7, 3.9]])
        bd = bottleneck_distance(dgm_a, dgm_b, delta=0.0)
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert m.distance == pytest.approx(bd, abs=ABS_TOL)

    def test_approx_distance_within_delta(self):
        rng = np.random.default_rng(42)
        births = rng.uniform(0, 1, size=10)
        deaths = births + rng.uniform(0.1, 0.5, size=10)
        dgm_a = np.column_stack([births, deaths])
        dgm_b = dgm_a + rng.normal(scale=0.05, size=dgm_a.shape)
        dgm_b[:, 1] = np.maximum(dgm_b[:, 0] + 1e-3, dgm_b[:, 1])  # keep above diagonal

        exact = bottleneck_distance(dgm_a, dgm_b, delta=0.0)
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.01)
        # Approx answer is within (1+delta) of exact.
        assert m.distance <= exact * (1 + 0.01 + ABS_TOL)
        assert m.distance >= exact * (1 - 0.01 - ABS_TOL)


class TestTypeAPI:
    def test_bottleneck_matching_is_diagram_matching(self):
        # Subclass relationship as per the plan.
        from oineus import DiagramMatching
        dgm_a = np.array([[0.0, 1.0]])
        dgm_b = np.array([[0.1, 0.9]])
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert isinstance(m, DiagramMatching)
        assert isinstance(m, BottleneckMatching)

    def test_longest_finite_attribute(self):
        dgm_a = np.array([[0.0, 5.0]])
        dgm_b = np.array([[1.0, 6.0]])
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert isinstance(m.longest.finite, list)
        assert len(m.longest.finite) == 1

    def test_essential_longest_attribute_and_enum_indexing(self):
        dgm_a = np.array([[0.0, 1.0]])
        dgm_b = np.array([[0.1, 0.9]])
        m = bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        # Attribute access for every family.
        assert m.longest.essential.inf_death == []
        assert m.longest.essential.neg_inf_death == []
        assert m.longest.essential.inf_birth == []
        assert m.longest.essential.neg_inf_birth == []
        # Enum-indexed access agrees.
        for c in InfKind:
            assert m.longest.essential[c] == []
