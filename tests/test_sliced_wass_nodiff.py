"""Tests for the non-differentiable (numpy) sliced Wasserstein distance.

These are numpy-only and do not require torch. A separate, torch-gated test
cross-checks agreement with oineus.diff.sliced_wasserstein_distance.
"""

import numpy as np
import pytest

import oineus as oin

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _unit_dirs(n, seed=0):
    rng = np.random.default_rng(seed)
    angles = rng.random(n) * np.pi
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _ref_standard(fin1, fin2, U):
    """Independent per-direction reference for the standard variant."""
    costs = []
    for u in U:
        d1 = (fin1[:, 0] + fin1[:, 1]) / 2.0
        d2 = (fin2[:, 0] + fin2[:, 1]) / 2.0
        diag1 = np.stack([d1, d1], axis=1)
        diag2 = np.stack([d2, d2], axis=1)
        L1 = np.sort(np.concatenate([fin1 @ u, diag2 @ u]))
        L2 = np.sort(np.concatenate([fin2 @ u, diag1 @ u]))
        costs.append(np.sum(np.abs(L1 - L2)))
    return float(np.mean(costs))


def _ref_corrected(fin1, fin2, U):
    """Independent per-direction reference for the diagonal-corrected variant."""
    n1, n2 = len(fin1), len(fin2)
    costs = []
    for u in U:
        d1 = (fin1[:, 0] + fin1[:, 1]) / 2.0
        d2 = (fin2[:, 0] + fin2[:, 1]) / 2.0
        proj1 = fin1 @ u
        proj2 = fin2 @ u
        proj1_self = (np.stack([d1, d1], axis=1)) @ u
        proj2_self = (np.stack([d2, d2], axis=1)) @ u
        L1 = np.concatenate([proj1, (np.stack([d2, d2], axis=1)) @ u])
        L2 = np.concatenate([proj2, (np.stack([d1, d1], axis=1)) @ u])
        i1 = np.argsort(L1)
        i2 = np.argsort(L2)
        L1s, L2s = L1[i1], L2[i2]
        c = 0.0
        for k in range(len(L1s)):
            a, b = i1[k], i2[k]
            is_diag1, is_diag2 = a >= n1, b >= n2
            if is_diag1 and is_diag2:
                continue
            if (not is_diag1) and is_diag2:
                c += abs(proj1[a] - proj1_self[a])
            elif is_diag1 and (not is_diag2):
                c += abs(proj2[b] - proj2_self[b])
            else:
                c += abs(L1s[k] - L2s[k])
        costs.append(c)
    return float(np.mean(costs))


def test_exports_exist():
    assert hasattr(oin, "sliced_wasserstein_distance")
    assert hasattr(oin, "sliced_wasserstein_distance_diag_corrected")


def test_identical_is_zero():
    dgm = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
    assert oin.sliced_wasserstein_distance(dgm, dgm.copy(), n_directions=64, seed=1) == pytest.approx(0.0, abs=1e-9)
    assert oin.sliced_wasserstein_distance_diag_corrected(dgm, dgm.copy(), n_directions=64, seed=1) == pytest.approx(0.0, abs=1e-9)


def test_nonnegative_and_symmetric():
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1], [0.1, 0.9]])
    U = _unit_dirs(80, seed=3)
    d12 = oin.sliced_wasserstein_distance(dgm1, dgm2, directions=U)
    d21 = oin.sliced_wasserstein_distance(dgm2, dgm1, directions=U)
    assert d12 >= 0.0
    assert d12 == pytest.approx(d21, abs=1e-9)


def test_deterministic_with_seed():
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])
    a = oin.sliced_wasserstein_distance(dgm1, dgm2, n_directions=100, seed=42)
    b = oin.sliced_wasserstein_distance(dgm1, dgm2, n_directions=100, seed=42)
    assert a == pytest.approx(b, abs=0.0)


def test_matches_reference_standard():
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0], [0.3, 0.7]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])
    U = _unit_dirs(50, seed=7)
    got = oin.sliced_wasserstein_distance(dgm1, dgm2, directions=U)
    exp = _ref_standard(dgm1, dgm2, U)
    assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)


def test_matches_reference_corrected():
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0], [0.3, 0.7]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])
    U = _unit_dirs(50, seed=11)
    got = oin.sliced_wasserstein_distance_diag_corrected(dgm1, dgm2, directions=U)
    exp = _ref_corrected(dgm1, dgm2, U)
    assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)


def test_essential_points():
    # Both diagrams have one (finite, +inf) essential point -> matched in 1D.
    dgm1 = np.array([[0.0, 1.0], [0.2, np.inf]])
    dgm2 = np.array([[0.1, 1.1], [0.5, np.inf]])
    U = _unit_dirs(40, seed=5)
    finite_only = oin.sliced_wasserstein_distance(
        dgm1[:1], dgm2[:1], directions=U)
    with_ess = oin.sliced_wasserstein_distance(dgm1, dgm2, directions=U)
    # essential contribution is |0.2 - 0.5| = 0.3
    assert with_ess == pytest.approx(finite_only + 0.3, abs=1e-9)


def test_essential_cardinality_mismatch_raises():
    dgm1 = np.array([[0.0, 1.0], [0.2, np.inf]])
    dgm2 = np.array([[0.1, 1.1]])
    with pytest.raises(ValueError):
        oin.sliced_wasserstein_distance(dgm1, dgm2, n_directions=10, seed=0)


def test_ignore_inf_points():
    dgm1 = np.array([[0.0, 1.0], [0.2, np.inf]])
    dgm2 = np.array([[0.1, 1.1]])
    U = _unit_dirs(40, seed=2)
    d = oin.sliced_wasserstein_distance(dgm1, dgm2, directions=U, ignore_inf_points=True)
    d_ref = oin.sliced_wasserstein_distance(dgm1[:1], dgm2, directions=U)
    assert d == pytest.approx(d_ref, abs=1e-9)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_agrees_with_diff_version():
    import torch
    import oineus.diff as oin_diff

    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0], [0.3, 0.7]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])

    nd = 4000
    np_dist = oin.sliced_wasserstein_distance(dgm1, dgm2, n_directions=nd, seed=123)

    t1 = torch.tensor(dgm1, dtype=torch.float64)
    t2 = torch.tensor(dgm2, dtype=torch.float64)
    diff_dist = float(oin_diff.sliced_wasserstein_distance(t1, t2, n_directions=nd).item())

    # Two independent Monte-Carlo estimates of the same quantity; generous tol.
    assert np_dist == pytest.approx(diff_dist, rel=0.1, abs=0.05)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_corrected_agrees_with_diff_version():
    # Confirms the non-diff diagonal-corrected variant implements the same
    # own-projection / zero-diagonal-diagonal semantics as the diff version.
    # Unequal cardinalities (3 vs 2) force diagonal matches, so the correction
    # actually fires.
    import torch
    import oineus.diff as oin_diff

    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0], [0.3, 0.7]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])

    nd = 4000
    np_dist = oin.sliced_wasserstein_distance_diag_corrected(dgm1, dgm2, n_directions=nd, seed=321)

    t1 = torch.tensor(dgm1, dtype=torch.float64)
    t2 = torch.tensor(dgm2, dtype=torch.float64)
    diff_dist = float(oin_diff.sliced_wasserstein_distance_diag_corrected(t1, t2, n_directions=nd).item())

    assert np_dist == pytest.approx(diff_dist, rel=0.1, abs=0.05)
