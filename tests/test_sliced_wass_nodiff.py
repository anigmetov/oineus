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


def _ref_standard(fin1, fin2, U, q=1.0):
    """Independent per-direction reference for the standard variant."""
    costs = []
    for u in U:
        d1 = (fin1[:, 0] + fin1[:, 1]) / 2.0
        d2 = (fin2[:, 0] + fin2[:, 1]) / 2.0
        diag1 = np.stack([d1, d1], axis=1)
        diag2 = np.stack([d2, d2], axis=1)
        L1 = np.sort(np.concatenate([fin1 @ u, diag2 @ u]))
        L2 = np.sort(np.concatenate([fin2 @ u, diag1 @ u]))
        costs.append(np.sum(np.abs(L1 - L2) ** q))
    return float(np.mean(costs))


def _ref_corrected(fin1, fin2, U, q=1.0):
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
                c += abs(proj1[a] - proj1_self[a]) ** q
            elif is_diag1 and (not is_diag2):
                c += abs(proj2[b] - proj2_self[b]) ** q
            else:
                c += abs(L1s[k] - L2s[k]) ** q
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


def test_rejects_zero_directions():
    dgm1 = np.array([[0.0, 1.0]])
    dgm2 = np.array([[0.0, 2.0]])

    with pytest.raises(ValueError, match="n_directions"):
        oin.sliced_wasserstein_distance(dgm1, dgm2, n_directions=0)

    with pytest.raises(ValueError, match="n_directions"):
        oin.sliced_wasserstein_distance_diag_corrected(dgm1, dgm2, n_directions=0)

    with pytest.raises(ValueError, match="directions"):
        oin.sliced_wasserstein_distance(dgm1, dgm2, directions=np.empty((0, 2)))


def test_matches_reference_standard():
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0], [0.3, 0.7]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])
    U = _unit_dirs(50, seed=7)
    got = oin.sliced_wasserstein_distance(dgm1, dgm2, directions=U)
    exp = _ref_standard(dgm1, dgm2, U)
    assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)


def test_w2_squared_matches_reference_standard():
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0], [0.3, 0.7]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])
    U = _unit_dirs(50, seed=7)
    got = oin.sliced_wasserstein_distance(dgm1, dgm2, directions=U, q=2.0)
    exp = _ref_standard(dgm1, dgm2, U, q=2.0)
    assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)


def test_matches_reference_corrected():
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0], [0.3, 0.7]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])
    U = _unit_dirs(50, seed=11)
    got = oin.sliced_wasserstein_distance_diag_corrected(dgm1, dgm2, directions=U)
    exp = _ref_corrected(dgm1, dgm2, U)
    assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)


def test_w2_squared_matches_reference_corrected():
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0], [0.3, 0.7]])
    dgm2 = np.array([[0.2, 1.2], [0.6, 2.1]])
    U = _unit_dirs(50, seed=11)
    got = oin.sliced_wasserstein_distance_diag_corrected(
        dgm1, dgm2, directions=U, q=2.0
    )
    exp = _ref_corrected(dgm1, dgm2, U, q=2.0)
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


@pytest.mark.parametrize("api_name", [
    "sliced_wasserstein_distance",
    "sliced_wasserstein_distance_diag_corrected",
])
@pytest.mark.parametrize("q", [1.0, 2.0])
def test_essential_families_use_q_power(api_name, q):
    dgm1 = np.array([
        [0.2, np.inf],
        [0.8, np.inf],
        [1.3, -np.inf],
        [np.inf, 2.2],
        [-np.inf, 3.4],
    ])
    dgm2 = np.array([
        [0.5, np.inf],
        [1.1, np.inf],
        [1.7, -np.inf],
        [np.inf, 2.5],
        [-np.inf, 3.0],
    ])
    expected = 2 * 0.3 ** q + 0.4 ** q + 0.3 ** q + 0.4 ** q

    actual = getattr(oin, api_name)(dgm1, dgm2, q=q)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


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


API_NAMES = (
    "sliced_wasserstein_distance",
    "sliced_wasserstein_distance_diag_corrected",
)


def _diagrams_with_all_essential_families():
    dgm1 = np.array([
        [0.0, 1.0],
        [0.5, 2.0],
        [0.3, 0.7],
        [0.2, np.inf],
        [0.8, np.inf],
        [1.3, -np.inf],
        [np.inf, 2.2],
        [-np.inf, 3.4],
    ])
    dgm2 = np.array([
        [0.2, 1.2],
        [0.6, 2.1],
        [0.5, np.inf],
        [1.1, np.inf],
        [1.7, -np.inf],
        [np.inf, 2.5],
        [-np.inf, 3.0],
    ])
    return dgm1, dgm2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
@pytest.mark.parametrize("api_name", API_NAMES)
@pytest.mark.parametrize("q", [1.0, 2.0])
def test_numpy_torch_agree_with_explicit_directions(api_name, q):
    import torch
    import oineus.diff as oin_diff

    dgm1, dgm2 = _diagrams_with_all_essential_families()
    directions = np.vstack([
        [1.0, 0.0],
        [0.0, 1.0],
        [np.sqrt(0.5), np.sqrt(0.5)],
        _unit_dirs(73, seed=812),
    ])
    numpy_api = getattr(oin, api_name)
    torch_api = getattr(oin_diff, api_name)
    np_dist = numpy_api(
        dgm1,
        dgm2,
        n_directions=0,
        seed=-1,
        directions=directions,
        q=q,
    )

    t1 = torch.tensor(dgm1, dtype=torch.float64)
    t2 = torch.tensor(dgm2, dtype=torch.float64)
    diff_dist = float(
        torch_api(
            t1,
            t2,
            n_directions=0,
            seed=-1,
            directions=torch.tensor(directions),
            q=q,
        ).item()
    )

    assert np_dist == pytest.approx(diff_dist, rel=1e-12, abs=1e-12)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
@pytest.mark.parametrize("api_name", API_NAMES)
@pytest.mark.parametrize("q", [1.0, 2.0])
def test_numpy_torch_agree_with_seed(api_name, q):
    import torch
    import oineus.diff as oin_diff

    dgm1, dgm2 = _diagrams_with_all_essential_families()
    numpy_api = getattr(oin, api_name)
    torch_api = getattr(oin_diff, api_name)
    np_dist = numpy_api(dgm1, dgm2, n_directions=79, seed=123, q=q)
    diff_dist = float(
        torch_api(
            torch.tensor(dgm1, dtype=torch.float64),
            torch.tensor(dgm2, dtype=torch.float64),
            n_directions=79,
            seed=123,
            q=q,
        ).item()
    )

    assert np_dist == pytest.approx(diff_dist, rel=1e-12, abs=1e-12)


INVALID_ARGUMENTS = (
    pytest.param(
        {"q": 0.0}, ValueError, "q must be a positive finite number", id="zero-q"
    ),
    pytest.param(
        {"q": np.inf},
        ValueError,
        "q must be a positive finite number",
        id="infinite-q",
    ),
    pytest.param(
        {"q": np.nan},
        ValueError,
        "q must be a positive finite number",
        id="nan-q",
    ),
    pytest.param(
        {"q": -1.0},
        ValueError,
        "q must be a positive finite number",
        id="negative-q",
    ),
    pytest.param(
        {"q": "bad"},
        TypeError,
        "q must be a positive finite number",
        id="nonnumeric-q",
    ),
    pytest.param(
        {"n_directions": 0},
        ValueError,
        "n_directions must be positive",
        id="zero-count",
    ),
    pytest.param(
        {"n_directions": 1.5},
        TypeError,
        "n_directions must be an integer",
        id="noninteger-count",
    ),
    pytest.param(
        {"seed": -1},
        ValueError,
        "seed must be a non-negative integer or None",
        id="negative-seed",
    ),
    pytest.param(
        {"seed": 1.5},
        TypeError,
        "seed must be a non-negative integer or None",
        id="noninteger-seed",
    ),
    pytest.param(
        {"directions": np.empty((0, 2))},
        ValueError,
        "directions must contain at least one direction",
        id="empty-directions",
    ),
    pytest.param(
        {"directions": np.ones((2, 3))},
        ValueError,
        "directions must have shape (n_directions, 2)",
        id="direction-shape",
    ),
    pytest.param(
        {"directions": np.array([[np.nan, 1.0]])},
        ValueError,
        "directions must contain only finite values",
        id="nonfinite-direction",
    ),
    pytest.param(
        {"directions": np.zeros((1, 2))},
        ValueError,
        "directions must be nonzero",
        id="zero-direction",
    ),
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
@pytest.mark.parametrize("api_name", API_NAMES)
@pytest.mark.parametrize("kwargs,error_type,message", INVALID_ARGUMENTS)
def test_numpy_torch_reject_invalid_arguments_identically(
    api_name, kwargs, error_type, message
):
    import torch
    import oineus.diff as oin_diff

    dgm1 = np.array([[0.0, 1.0]])
    dgm2 = np.array([[0.0, 2.0]])
    numpy_api = getattr(oin, api_name)
    torch_api = getattr(oin_diff, api_name)

    with pytest.raises(error_type) as np_error:
        numpy_api(dgm1, dgm2, **kwargs)
    with pytest.raises(error_type) as torch_error:
        torch_api(torch.tensor(dgm1), torch.tensor(dgm2), **kwargs)

    assert str(np_error.value) == message
    assert str(torch_error.value) == message


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
@pytest.mark.parametrize("api_name", API_NAMES)
@pytest.mark.parametrize("q", [1.0, 2.0])
def test_numpy_torch_reject_essential_mismatch_identically(api_name, q):
    import torch
    import oineus.diff as oin_diff

    dgm1 = np.array([[0.0, 1.0], [0.2, np.inf]])
    dgm2 = np.array([[0.1, 1.1]])
    numpy_api = getattr(oin, api_name)
    torch_api = getattr(oin_diff, api_name)

    with pytest.raises(ValueError) as np_error:
        numpy_api(dgm1, dgm2, q=q, seed=17)
    with pytest.raises(ValueError) as torch_error:
        torch_api(torch.tensor(dgm1), torch.tensor(dgm2), q=q, seed=17)

    assert str(np_error.value) == str(torch_error.value)
