"""Tests for the non-differentiable full Cech filtration (oineus.cech_filtration).

The combinatorics/value oracle is an independent brute-force implementation:
simplices are enumerated with itertools.combinations and MEB radii come
from the shared subset-circumball oracle in meb_oracle.py.
"""

from itertools import combinations
from math import comb

import numpy as np
import pytest

import oineus as oin

from meb_oracle import meb_radius_sq_oracle


def bruteforce_cech(points, max_dim, max_radius):
    """dict: sorted vertex tuple -> squared MEB radius, all dims <= max_dim."""
    n = len(points)
    result = {}
    for q in range(max_dim + 1):
        for simplex in combinations(range(n), q + 1):
            if q == 0:
                val = 0.0
            else:
                val = meb_radius_sq_oracle(points[list(simplex)])
            if val <= max_radius ** 2:
                result[simplex] = val
    return result


def cells_of(fil):
    """dict: sorted vertex tuple -> value, from a built filtration."""
    return {tuple(sorted(s.vertices)): s.value for s in fil.cells()}


@pytest.mark.parametrize("d,n,seed", [(2, 8, 0), (3, 7, 1), (3, 8, 2)])
def test_matches_bruteforce_enumeration(d, n, seed):
    rng = np.random.default_rng(seed)
    points = rng.random((n, d))
    for max_radius in (0.35, 0.6, None):
        r = oin.max_distance(points) if max_radius is None else max_radius
        fil = oin.cech_filtration(points, max_radius=r)
        expected = bruteforce_cech(points, d, r)
        actual = cells_of(fil)
        assert set(actual) == set(expected)
        for simplex, val in expected.items():
            assert actual[simplex] == pytest.approx(val, abs=1e-10), simplex


@pytest.mark.parametrize("d,n,seed", [(2, 12, 3), (3, 9, 4)])
def test_downward_closure_and_monotone(d, n, seed):
    rng = np.random.default_rng(seed)
    points = rng.random((n, d))
    fil = oin.cech_filtration(points, max_radius=0.5)
    values = cells_of(fil)
    for simplex, val in values.items():
        if len(simplex) == 1:
            continue
        for facet in combinations(simplex, len(simplex) - 1):
            assert facet in values
            assert values[facet] <= val * (1 + 1e-12)


@pytest.mark.parametrize("d,n,seed", [(2, 10, 5), (3, 9, 6)])
def test_default_max_radius_complete_skeleton(d, n, seed):
    rng = np.random.default_rng(seed)
    points = rng.random((n, d))
    fil = oin.cech_filtration(points)
    for q in range(d + 1):
        assert fil.size_in_dimension(q) == comb(n, q + 1)
    assert fil.kind == oin.FiltrationKind.Cech


@pytest.mark.parametrize("d,n,seed", [(2, 10, 7), (3, 8, 8)])
def test_truncation_is_value_cut(d, n, seed):
    rng = np.random.default_rng(seed)
    points = rng.random((n, d))
    full = cells_of(oin.cech_filtration(points))
    r = 0.4
    truncated = cells_of(oin.cech_filtration(points, max_radius=r))
    expected = {s: v for s, v in full.items() if v <= r * r}
    assert truncated == expected


@pytest.mark.parametrize("d,n,seed", [(2, 12, 9), (3, 10, 10)])
def test_subset_is_subcomplex(d, n, seed):
    """Cech on a subset (with vertex_ids) is a genuine subcomplex of Cech on
    the full cloud: same uids (vertex sets) and bitwise-equal values. This
    is the contract the KICR pipeline relies on."""
    rng = np.random.default_rng(seed)
    points = rng.random((n, d))
    idx = np.sort(rng.choice(n, size=n // 2, replace=False))
    r = 0.6
    full = cells_of(oin.cech_filtration(points, max_radius=r))
    sub = cells_of(oin.cech_filtration(points[idx], max_radius=r, vertex_ids=idx))
    for simplex, val in sub.items():
        assert simplex in full
        assert full[simplex] == val  # same arithmetic on same coordinates


def test_input_validation():
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError):
        oin.cech_filtration(rng.random((5, 4)))
    with pytest.raises(ValueError):
        oin.cech_filtration(rng.random((5, 2)), max_dim=3)
    with pytest.raises(ValueError):
        oin.cech_filtration(rng.random(5))
    with pytest.raises(ValueError):
        oin.cech_filtration(rng.random((6, 2)), vertex_ids=np.arange(5))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
