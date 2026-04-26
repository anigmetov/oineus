"""Tests for wasserstein_matching and DiagramMatching."""

import pytest
import numpy as np

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
        # Attr access (preferred) and enum-indexed access agree.
        assert len(matching.essential.inf_death) == 1
        assert len(matching.essential[oineus.InfKind.INF_DEATH]) == 1

        # Check indices refer to original diagrams
        idx_a, idx_b = matching.essential.inf_death[0]
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

        assert len(matching.essential.inf_death) == 3

        # All indices should be valid
        for idx_a, idx_b in matching.essential.inf_death:
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

        # All four families are present as empty (0, 2) arrays.
        assert all(v.shape == (0, 2) for v in matching.essential.values())
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
        ess = matching.essential.inf_death
        assert [tuple(map(int, p)) for p in ess] == [(1, 1)]

        # Finite matches should involve indices 0,2,3
        finite_a_indices = set(int(idx) for idx, _ in matching.finite_to_finite)
        finite_b_indices = set(int(idx) for _, idx in matching.finite_to_finite)
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

        # Mix attr access and enum indexing for coverage.
        assert [tuple(map(int, p)) for p in matching.essential.inf_death] == [(2, 2)]
        assert [tuple(map(int, p)) for p in matching.essential[oineus.InfKind.INF_DEATH]] == [(2, 2)]

        # Verify we can index into original diagrams
        idx_a, idx_b = matching.essential.inf_death[0]
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


class TestNearIdenticalDiagrams:
    """Regression: near-identical (ULP-level) inputs used to hang Hera's
    auction because its termination criterion is relative and the true
    cost was below float precision.

    These tests would HANG (not fail) under the bug; if a regression is
    reintroduced and the suite stalls here, that's the symptom.
    """

    def test_translated_point_cloud_no_hang(self):
        """Two diagrams differing only by a tiny non-representable translate
        of the input point cloud (the user-reported reproduction).
        """
        rng = np.random.default_rng(0)
        points_a = rng.random((7, 2))
        points_b = points_a + 0.02  # 0.02 is not exactly representable
        da = oineus.compute_diagrams_vr(points_a, max_dim=1)
        db = oineus.compute_diagrams_vr(points_b, max_dim=1)

        # Used to hang. Should now return promptly with near-zero distance.
        m = oineus.wasserstein_matching(da, db, q=2.0, delta=0.01, dim=0)
        assert m.distance < 1e-6
        assert m.cost < 1e-6

        # Standalone wasserstein_distance used to hang on the same input.
        d = oineus.wasserstein_distance(da, db, q=2.0, delta=0.01, dim=0)
        assert d < 1e-6

    def test_ulp_perturbed_diagrams_no_hang(self):
        """Engineered: diagrams equal in every coordinate except one entry
        that differs by a single ULP. Bypasses Hera's `are_equal` exact
        check; auction would spin forever without our tolerant fast path.
        """
        dgm_a = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
        dgm_b = dgm_a.copy()
        dgm_b[0, 1] = np.nextafter(dgm_b[0, 1], np.inf)  # 1 ULP up

        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0, delta=0.01)
        assert m.distance < 1e-9

        d = oineus.wasserstein_distance(dgm_a, dgm_b, q=2.0, delta=0.01)
        assert d < 1e-9


class TestCostFromMatching:
    """The matching wrapper now derives `result.cost` (and `.distance`)
    from the matching itself rather than calling Hera a second time.
    Verify the derivation agrees with both the matching's own structure
    and `wasserstein_distance` on well-conditioned inputs.
    """

    def test_finite_only_matches_wasserstein_distance(self):
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
        dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])
        for q in [1.0, 2.0, 3.0]:
            m = oineus.wasserstein_matching(dgm_a, dgm_b, q=q, delta=0.01)
            d = oineus.wasserstein_distance(dgm_a, dgm_b, q=q, delta=0.01)
            # Both go through the same auction with the same delta.
            assert abs(m.distance - d) < 0.02 * max(d, 1e-9)
            assert abs(m.cost - m.distance ** q) < 1e-10

    def test_essentials_contribute_to_distance(self):
        """When ignore_inf_points=False, matched essential pairs must
        contribute their |coord_a - coord_b|^q cost to result.cost.
        """
        dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
        dgm_b = np.array([[0.0, 1.0], [0.7, np.inf]])
        q = 2.0
        m = oineus.wasserstein_matching(
            dgm_a, dgm_b, q=q, delta=0.01, ignore_inf_points=False)
        # Finite parts are identical; only contribution is essential.
        # Essential cost = |0.5 - 0.7|^2 = 0.04 → distance = 0.2.
        assert abs(m.cost - 0.04) < 1e-9
        assert abs(m.distance - 0.2) < 1e-9

    def test_internal_p_l1(self):
        """L_1 ground metric: cost contribution from a matched pair is
        the L_1 distance to the q-th power.
        """
        dgm_a = np.array([[0.0, 1.0]])
        dgm_b = np.array([[0.1, 1.2]])
        q = 2.0
        m = oineus.wasserstein_matching(
            dgm_a, dgm_b, q=q, delta=0.01, internal_p=1.0)
        # L_1 distance = 0.1 + 0.2 = 0.3 → cost = 0.09.
        assert abs(m.cost - 0.09) < 1e-9
        assert abs(m.distance - 0.3) < 1e-9


class TestPerfectMatchingCoverage:
    """Hand-crafted diagrams with mixed essentials of every family, in
    deliberately scattered positions. Verify that:

    - Every off-diagonal index in dgm_a is matched exactly once across
      finite_to_finite[:, 0], a_to_diagonal, and the four essential family
      first columns. Same for dgm_b.
    - Each essential pair connects two points of the same family, and the
      A-side / B-side indices in each family match the expected sets.
    - Within an essential family, pairing is by rank-of-finite-coordinate
      (smallest A coord ↔ smallest B coord).
    - finite_to_finite indices on each side are themselves finite points.
    """

    # Hand-crafted layout. Indices into each diagram are committed below.
    DGM_A = np.array([
        [0.0,        5.0],     # 0  finite
        [-np.inf,  100.0],     # 1  neg_inf_birth (death=100)
        [1.0,    np.inf],      # 2  inf_death     (birth=1)
        [10.0,    15.0],       # 3  finite
        [-np.inf,  200.0],     # 4  neg_inf_birth (death=200)
        [5.0,    -np.inf],     # 5  neg_inf_death (birth=5)
        [20.0,    25.0],       # 6  finite
        [np.inf,   50.0],      # 7  inf_birth     (death=50)
    ])
    DGM_B = np.array([
        [-np.inf,  110.0],     # 0  neg_inf_birth (death=110)
        [0.1,       5.1],      # 1  finite
        [1.1,    np.inf],      # 2  inf_death     (birth=1.1)
        [50.0,    51.0],       # 3  finite (far from any A finite)
        [10.1,    15.1],       # 4  finite
        [5.1,    -np.inf],     # 5  neg_inf_death (birth=5.1)
        [20.1,    25.1],       # 6  finite
        [np.inf,   51.0],      # 7  inf_birth     (death=51)
        [-np.inf,  210.0],     # 8  neg_inf_birth (death=210)
        [60.0,    61.0],       # 9  finite (far from any A finite)
    ])

    EXPECTED_A = {
        "finite":         {0, 3, 6},
        "inf_death":      [2],
        "neg_inf_death":  [5],
        "inf_birth":      [7],
        "neg_inf_birth":  [1, 4],
    }
    EXPECTED_B = {
        "finite":         {1, 3, 4, 6, 9},
        "inf_death":      [2],
        "neg_inf_death":  [5],
        "inf_birth":      [7],
        "neg_inf_birth":  [0, 8],
    }

    def _check_essential_family(self, m, name, side_a_finite_axis):
        """Assert that essential.<name> contains exactly the indices we
        expect on each side, that those indices really refer to points of
        that family, and that pairs respect rank-of-finite-coord."""
        pairs = m.essential[name]
        expected_a = self.EXPECTED_A[name]
        expected_b = self.EXPECTED_B[name]

        assert pairs.shape == (len(expected_a), 2), \
            f"{name}: expected {len(expected_a)} pairs, got shape {pairs.shape}"
        assert set(pairs[:, 0].tolist()) == set(expected_a), \
            f"{name}: A-side indices wrong: {pairs[:, 0].tolist()} vs {expected_a}"
        assert set(pairs[:, 1].tolist()) == set(expected_b), \
            f"{name}: B-side indices wrong: {pairs[:, 1].tolist()} vs {expected_b}"

        # Each indexed point really belongs to this family.
        axis = side_a_finite_axis  # 0 = birth-finite (death is ±inf),
                                   # 1 = death-finite (birth is ±inf)
        infinite_axis = 1 - axis
        for ia, ib in pairs:
            pt_a = self.DGM_A[ia]
            pt_b = self.DGM_B[ib]
            assert np.isfinite(pt_a[axis]) and np.isfinite(pt_b[axis]), \
                f"{name}: matched index has non-finite finite-axis coord"
            assert not np.isfinite(pt_a[infinite_axis]), \
                f"{name}: A[{ia}] should have ±inf at axis {infinite_axis}"
            assert not np.isfinite(pt_b[infinite_axis]), \
                f"{name}: B[{ib}] should have ±inf at axis {infinite_axis}"
            assert np.signbit(pt_a[infinite_axis]) == np.signbit(pt_b[infinite_axis]), \
                f"{name}: sign of inf coord differs between matched pair"

        # Pair-by-rank: smallest A finite coord matches smallest B finite coord.
        order_a = np.argsort([self.DGM_A[ia, axis] for ia in pairs[:, 0]])
        order_b = np.argsort([self.DGM_B[ib, axis] for ib in pairs[:, 1]])
        # The pairs themselves should already encode this — verify by
        # constructing the expected pairing from the sorted orders.
        a_sorted = pairs[:, 0][order_a]
        b_sorted = pairs[:, 1][order_b]
        # For every k, (a_sorted[k], b_sorted[k]) should appear as a pair in m.
        as_set = {(int(a), int(b)) for a, b in pairs}
        for a, b in zip(a_sorted, b_sorted):
            assert (int(a), int(b)) in as_set, \
                f"{name}: expected pair ({a}, {b}) by sorted-rank, not in {as_set}"

    def test_full_coverage_with_essentials(self):
        """ignore_inf_points=False: every off-diagonal index in either
        diagram must be matched exactly once."""
        m = oineus.wasserstein_matching(
            self.DGM_A, self.DGM_B, q=2.0, delta=0.01,
            ignore_inf_points=False,
        )

        # 1) Essential families: classification + correctness.
        self._check_essential_family(m, "inf_death",     side_a_finite_axis=0)
        self._check_essential_family(m, "neg_inf_death", side_a_finite_axis=0)
        self._check_essential_family(m, "inf_birth",     side_a_finite_axis=1)
        self._check_essential_family(m, "neg_inf_birth", side_a_finite_axis=1)

        # 2) finite_to_finite indices really refer to finite points on each side.
        for ia in m.finite_to_finite[:, 0]:
            pt = self.DGM_A[ia]
            assert np.isfinite(pt).all(), f"finite_to_finite A index {ia} is not finite"
            assert int(ia) in self.EXPECTED_A["finite"]
        for ib in m.finite_to_finite[:, 1]:
            pt = self.DGM_B[ib]
            assert np.isfinite(pt).all(), f"finite_to_finite B index {ib} is not finite"
            assert int(ib) in self.EXPECTED_B["finite"]

        # 3) Perfect-matching coverage on the A side.
        a_indices_used = set()
        for col in (m.finite_to_finite[:, 0], m.a_to_diagonal,
                    m.essential.inf_death[:, 0], m.essential.neg_inf_death[:, 0],
                    m.essential.inf_birth[:, 0], m.essential.neg_inf_birth[:, 0]):
            for i in col:
                i = int(i)
                assert i not in a_indices_used, f"A index {i} appears twice in matching"
                a_indices_used.add(i)
        # No diagonal points in DGM_A → every input row should be matched.
        assert a_indices_used == set(range(self.DGM_A.shape[0]))

        # 4) Perfect-matching coverage on the B side.
        b_indices_used = set()
        for col in (m.finite_to_finite[:, 1], m.b_to_diagonal,
                    m.essential.inf_death[:, 1], m.essential.neg_inf_death[:, 1],
                    m.essential.inf_birth[:, 1], m.essential.neg_inf_birth[:, 1]):
            for i in col:
                i = int(i)
                assert i not in b_indices_used, f"B index {i} appears twice in matching"
                b_indices_used.add(i)
        assert b_indices_used == set(range(self.DGM_B.shape[0]))

    def test_full_coverage_finite_only(self):
        """ignore_inf_points=True: essentials are dropped from the matching;
        finite parts must still be perfectly covered."""
        m = oineus.wasserstein_matching(
            self.DGM_A, self.DGM_B, q=2.0, delta=0.01,
            ignore_inf_points=True,
        )

        # No essentials anywhere.
        for name in ("inf_death", "neg_inf_death", "inf_birth", "neg_inf_birth"):
            assert m.essential[name].shape == (0, 2), \
                f"{name} should be empty when ignore_inf_points=True"

        # Every finite A index appears exactly once across
        # finite_to_finite[:, 0] and a_to_diagonal.
        a_finite_used = set(int(i) for i in m.finite_to_finite[:, 0]) \
            | set(int(i) for i in m.a_to_diagonal)
        assert a_finite_used == self.EXPECTED_A["finite"]

        # Same on B side.
        b_finite_used = set(int(i) for i in m.finite_to_finite[:, 1]) \
            | set(int(i) for i in m.b_to_diagonal)
        assert b_finite_used == self.EXPECTED_B["finite"]

    def test_full_coverage_bottleneck(self):
        """Same hand-crafted layout under bottleneck. Family classification
        and perfect coverage should hold; longest-edge data should be
        consistent with m.distance."""
        m = oineus.bottleneck_matching(
            self.DGM_A, self.DGM_B, delta=0.0,
            ignore_inf_points=False,
        )

        self._check_essential_family(m, "inf_death",     side_a_finite_axis=0)
        self._check_essential_family(m, "neg_inf_death", side_a_finite_axis=0)
        self._check_essential_family(m, "inf_birth",     side_a_finite_axis=1)
        self._check_essential_family(m, "neg_inf_birth", side_a_finite_axis=1)

        # Perfect coverage on both sides.
        a_used = set()
        for col in (m.finite_to_finite[:, 0], m.a_to_diagonal,
                    m.essential.inf_death[:, 0], m.essential.neg_inf_death[:, 0],
                    m.essential.inf_birth[:, 0], m.essential.neg_inf_birth[:, 0]):
            for i in col:
                a_used.add(int(i))
        assert a_used == set(range(self.DGM_A.shape[0]))

        b_used = set()
        for col in (m.finite_to_finite[:, 1], m.b_to_diagonal,
                    m.essential.inf_death[:, 1], m.essential.neg_inf_death[:, 1],
                    m.essential.inf_birth[:, 1], m.essential.neg_inf_birth[:, 1]):
            for i in col:
                b_used.add(int(i))
        assert b_used == set(range(self.DGM_B.shape[0]))

        # Every recorded longest edge length should be ≤ m.distance, and at
        # least one (across finite + essential) should equal it.
        all_lengths = [e.length for e in m.longest.finite]
        for name in ("inf_death", "neg_inf_death", "inf_birth", "neg_inf_birth"):
            for e in m.longest.essential[name]:
                all_lengths.append(e.length)
        assert all_lengths, "expected at least one longest edge"
        assert max(all_lengths) == pytest.approx(m.distance, abs=1e-9)
        for L in all_lengths:
            assert L <= m.distance + 1e-9


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
