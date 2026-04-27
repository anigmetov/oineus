"""Verify nanobind return-value-policy choices for the matching bindings.

These tests do not check matching correctness; they target the lifetime,
ownership, and aliasing semantics of the C++/Python boundary used by
``DiagramMatching`` (Wasserstein) and ``BottleneckMatching``.

Background
----------
The bindings (see ``bindings/python/oineus_functions.cpp``) use:

* ``nb::rv_policy::move`` for ndarray-returning properties such as
  ``finite_to_finite``, ``a_to_diagonal``, ``b_to_diagonal`` and the four
  per-family essential arrays. Each call goes through ``pairs_to_numpy`` /
  ``ints_to_numpy``, which freshly ``new[]``-allocates an int64 buffer and
  hands ownership to a ``nb::capsule`` deleter. A correct binding must
  therefore yield an *independent* ndarray for every property access.
* ``nb::keep_alive<0, 1>()`` on the ``essential`` property of
  ``DiagramMatching`` (so the ``EssentialMatchesView`` it returns keeps the
  matching alive) and on ``longest`` of ``BottleneckMatching``.
* ``nb::rv_policy::reference_internal`` on
  ``LongestEdges.essential`` (the inner ``EssentialLongestEdges`` lives
  inside the ``LongestEdges`` view).

If any of those policies are wrong, the tests below should crash, return
stale data, or expose silent aliasing.
"""

import gc

import numpy as np
import pytest

import oineus


# ---------------------------------------------------------------------------
# Diagram fixtures
# ---------------------------------------------------------------------------
#
# Each fixture builds *fresh* arrays so no cross-test sharing can mask an
# aliasing bug. The diagrams contain at least one essential point of each of
# the four families, so all four ``essential.*`` arrays are non-empty when
# ``ignore_inf_points=False``.

def _make_dgm_a():
    return np.array([
        [0.0,       1.0],         # 0: finite
        [0.5,       np.inf],      # 1: inf_death
        [0.6,      -np.inf],      # 2: neg_inf_death
        [np.inf,    0.7],         # 3: inf_birth
        [-np.inf,   0.8],         # 4: neg_inf_birth
    ])


def _make_dgm_b():
    return np.array([
        [0.1,       0.9],
        [0.55,      np.inf],
        [0.65,     -np.inf],
        [np.inf,    0.75],
        [-np.inf,   0.85],
    ])


def _wm():
    """Build a fresh Wasserstein matching with all four essential families."""
    return oineus.wasserstein_matching(
        _make_dgm_a(), _make_dgm_b(), q=2.0, ignore_inf_points=False
    )


def _bm():
    """Build a fresh exact bottleneck matching with all four essential families."""
    return oineus.bottleneck_matching(
        _make_dgm_a(), _make_dgm_b(), delta=0.0, ignore_inf_points=False
    )


def _mutate_int_array(arr):
    """Overwrite an integer ndarray to a sentinel value in place."""
    arr[:] = -123456


# ---------------------------------------------------------------------------
# A. Independent buffers
# ---------------------------------------------------------------------------

class TestIndependentBuffers:
    """Each property access must allocate a fresh, owning ndarray."""

    def test_finite_to_finite_returns_fresh_buffer(self):
        m = _wm()
        arr1 = m.finite_to_finite
        arr2 = m.finite_to_finite

        assert arr1 is not arr2, "consecutive accesses returned the SAME object"
        assert arr1.dtype == np.int64
        assert arr1.shape == arr2.shape
        original_bytes = arr1.tobytes()
        assert original_bytes == arr2.tobytes()

        _mutate_int_array(arr1)
        assert arr2.tobytes() == original_bytes, (
            "mutating one ndarray was visible in another — buffers are aliased"
        )

        arr3 = m.finite_to_finite
        assert arr3.tobytes() == original_bytes, (
            "mutation propagated back into the C++ matching struct"
        )

    def test_a_to_diagonal_returns_fresh_buffer(self):
        # Construct asymmetric diagrams so a_to_diagonal is non-empty.
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
        dgm_b = np.array([[0.05, 0.95], [0.55, 1.95]])
        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)
        assert len(m.a_to_diagonal) >= 1

        arr1 = m.a_to_diagonal
        arr2 = m.a_to_diagonal
        assert arr1 is not arr2
        original_bytes = arr1.tobytes()
        assert arr2.tobytes() == original_bytes

        _mutate_int_array(arr1)
        assert arr2.tobytes() == original_bytes
        arr3 = m.a_to_diagonal
        assert arr3.tobytes() == original_bytes

    def test_b_to_diagonal_returns_fresh_buffer(self):
        dgm_a = np.array([[0.05, 0.95], [0.55, 1.95]])
        dgm_b = np.array([[0.0, 1.0], [0.5, 2.0], [1.5, 3.0]])
        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)
        assert len(m.b_to_diagonal) >= 1

        arr1 = m.b_to_diagonal
        arr2 = m.b_to_diagonal
        assert arr1 is not arr2
        original_bytes = arr1.tobytes()
        assert arr2.tobytes() == original_bytes

        _mutate_int_array(arr1)
        assert arr2.tobytes() == original_bytes
        arr3 = m.b_to_diagonal
        assert arr3.tobytes() == original_bytes

    @pytest.mark.parametrize("attr", ["inf_death", "neg_inf_death",
                                       "inf_birth", "neg_inf_birth"])
    def test_essential_attribute_returns_fresh_buffer(self, attr):
        m = _wm()
        ess = m.essential
        arr1 = getattr(ess, attr)
        arr2 = getattr(ess, attr)

        assert arr1.shape == (1, 2), (
            f"expected one essential of family {attr!r} in the test diagram"
        )
        assert arr1 is not arr2, (
            f"essential.{attr} returned the SAME object on two accesses"
        )
        original_bytes = arr1.tobytes()
        assert arr2.tobytes() == original_bytes

        _mutate_int_array(arr1)
        assert arr2.tobytes() == original_bytes, (
            f"essential.{attr} buffers are aliased"
        )
        arr3 = getattr(ess, attr)
        assert arr3.tobytes() == original_bytes, (
            f"mutation of essential.{attr} reached into the C++ struct"
        )

    def test_bottleneck_longest_finite_list_independence(self):
        bm = _bm()

        list1 = bm.longest.finite
        list2 = bm.longest.finite

        assert isinstance(list1, list)
        assert isinstance(list2, list)
        assert len(list1) == len(list2)
        assert len(list1) >= 1

        # Compare element-wise on field values (objects themselves are likely
        # distinct nanobind wrappers).
        for e1, e2 in zip(list1, list2):
            assert e1.length == e2.length
            assert e1.idx_a == e2.idx_a
            assert e1.idx_b == e2.idx_b
            assert e1.point_a == e2.point_a
            assert e1.point_b == e2.point_b

        # Mutating Python's list1 must not propagate to list2.
        sentinel = object()
        list1.append(sentinel)
        assert sentinel not in list2
        assert len(list2) + 1 == len(list1)


# ---------------------------------------------------------------------------
# B. View lifetimes via temporaries
# ---------------------------------------------------------------------------
#
# These exercise nb::keep_alive<0, 1>() (and reference_internal) by binding
# only the *view*, letting the parent matching go out of scope, and then
# touching data that lives in the parent. If keep_alive is wrong, the parent
# struct is freed and reading the view returns garbage / segfaults.

class TestViewLifetimes:

    def test_essential_view_survives_parent_dropped(self):
        view = oineus.wasserstein_matching(
            _make_dgm_a(), _make_dgm_b(), q=2.0, ignore_inf_points=False
        ).essential
        # No name binding for the matching; view must keep it alive.
        gc.collect()

        for attr, expected in [
            ("inf_death",     [(1, 1)]),
            ("neg_inf_death", [(2, 2)]),
            ("inf_birth",     [(3, 3)]),
            ("neg_inf_birth", [(4, 4)]),
        ]:
            arr = getattr(view, attr)
            assert arr.dtype == np.int64
            assert arr.shape == (1, 2)
            got = [tuple(map(int, p)) for p in arr]
            assert got == expected, (
                f"view.{attr} returned {got}; parent was likely freed."
            )

    def test_longest_view_survives_parent_dropped(self):
        longest = oineus.bottleneck_matching(
            _make_dgm_a(), _make_dgm_b(),
            delta=0.0, ignore_inf_points=False,
        ).longest
        gc.collect()

        finite = longest.finite
        assert isinstance(finite, list)
        assert len(finite) >= 1
        # Touch numeric fields to force a read of the underlying memory.
        for e in finite:
            assert isinstance(e.length, float)
            assert isinstance(e.point_a, tuple)
            assert isinstance(e.point_b, tuple)

        ess_inf_death = longest.essential.inf_death
        assert isinstance(ess_inf_death, list)
        assert len(ess_inf_death) >= 1
        for e in ess_inf_death:
            # length, idx_a/idx_b, coord_a/coord_b should be sane numbers.
            assert isinstance(e.length, float)
            assert e.length >= 0.0
            assert int(e.idx_a) == 1
            assert int(e.idx_b) == 1

    def test_two_hop_longest_essential_lifetime(self):
        """Both the keep_alive on .longest AND reference_internal on
        .longest.essential must work together."""
        ess_view = oineus.bottleneck_matching(
            _make_dgm_a(), _make_dgm_b(),
            delta=0.0, ignore_inf_points=False,
        ).longest.essential
        gc.collect()

        for attr in ("inf_death", "neg_inf_death", "inf_birth", "neg_inf_birth"):
            edges = getattr(ess_view, attr)
            assert isinstance(edges, list)
            assert len(edges) >= 1, (
                f"two-hop access to longest.essential.{attr} returned empty list"
            )
            for e in edges:
                assert isinstance(e.length, float)
                assert e.length >= 0.0


# ---------------------------------------------------------------------------
# C. Subscript stability
# ---------------------------------------------------------------------------

class TestSubscriptStability:

    def test_essential_dict_indexing_returns_fresh(self):
        m = _wm()
        ess = m.essential

        arr1 = ess[oineus.InfKind.INF_DEATH]
        arr2 = ess["inf_death"]

        assert arr1.dtype == np.int64
        assert arr2.dtype == np.int64
        assert arr1.shape == arr2.shape == (1, 2)
        assert arr1.tobytes() == arr2.tobytes()
        assert arr1 is not arr2, (
            "enum and string subscripts returned the SAME object"
        )

        original_bytes = arr1.tobytes()
        _mutate_int_array(arr1)
        assert arr2.tobytes() == original_bytes, (
            "subscript-returned arrays are aliased"
        )
        # And another fresh access still matches the original.
        arr3 = ess[oineus.InfKind.INF_DEATH]
        assert arr3.tobytes() == original_bytes


# ---------------------------------------------------------------------------
# D. items / values / keys
# ---------------------------------------------------------------------------

class TestEssentialIteration:

    EXPECTED_KEYS = ["inf_death", "neg_inf_death", "inf_birth", "neg_inf_birth"]

    def test_keys(self):
        m = _wm()
        keys = m.essential.keys()
        assert list(keys) == self.EXPECTED_KEYS

    def test_values(self):
        m = _wm()
        ess = m.essential
        values = ess.values()
        assert isinstance(values, list)
        assert len(values) == 4
        for name, arr in zip(self.EXPECTED_KEYS, values):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.int64
            assert arr.tobytes() == getattr(ess, name).tobytes()

    def test_items(self):
        m = _wm()
        ess = m.essential
        items = list(ess.items())
        assert len(items) == 4

        names = [name for (name, _arr) in items]
        assert names == self.EXPECTED_KEYS

        for name, arr in items:
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.int64
            ref = getattr(ess, name)
            assert arr.shape == ref.shape
            assert arr.tobytes() == ref.tobytes()


# ---------------------------------------------------------------------------
# E. Inheritance is honoured
# ---------------------------------------------------------------------------

class TestInheritance:

    def test_bottleneck_isinstance_diagram_matching(self):
        bm = _bm()
        assert isinstance(bm, oineus.BottleneckMatching)
        assert isinstance(bm, oineus.DiagramMatching), (
            "BottleneckMatching should inherit from DiagramMatching in Python"
        )

    def test_bottleneck_inherits_diagram_matching_properties(self):
        bm = _bm()
        # Inherited properties should all be reachable on the subclass.
        assert isinstance(bm.finite_to_finite, np.ndarray)
        assert isinstance(bm.a_to_diagonal, np.ndarray)
        assert isinstance(bm.b_to_diagonal, np.ndarray)
        # essential view inherited from DiagramMatching
        ess = bm.essential
        assert isinstance(ess.inf_death, np.ndarray)


# ---------------------------------------------------------------------------
# F. Gc-stress
# ---------------------------------------------------------------------------

class TestGcStress:

    def test_repeated_temporary_matchings(self):
        """Tight loop creating temporary matchings, reading essential data,
        forcing gc. Should run without leaking or crashing."""
        for _ in range(100):
            arr = oineus.wasserstein_matching(
                _make_dgm_a(), _make_dgm_b(),
                q=2.0, ignore_inf_points=False,
            ).essential.inf_death
            # Touch the data to force a read.
            assert arr.shape == (1, 2)
            assert int(arr[0, 0]) == 1
            assert int(arr[0, 1]) == 1
            del arr
            gc.collect()

    def test_repeated_temporary_bottleneck_longest(self):
        """Same idea, but for the two-hop bottleneck.longest.essential."""
        for _ in range(100):
            edges = oineus.bottleneck_matching(
                _make_dgm_a(), _make_dgm_b(),
                delta=0.0, ignore_inf_points=False,
            ).longest.essential.inf_death
            assert isinstance(edges, list)
            assert len(edges) >= 1
            assert int(edges[0].idx_a) == 1
            del edges
            gc.collect()
