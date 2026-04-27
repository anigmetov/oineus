import numpy as np
import pytest

import oineus as oin
from oineus._dtype import REAL_DTYPE

# Tight tolerance for "two equivalent paths return the same value" checks.
ABS_TOL = 1e-12 if REAL_DTYPE == np.float64 else 1e-5


def _random_off_diagonal_diagram(rng: np.random.Generator, n_points: int) -> np.ndarray:
    births = rng.uniform(-2.0, 2.0, size=n_points)
    persistence = rng.uniform(0.05, 2.5, size=n_points)
    deaths = births + persistence
    return np.column_stack((births, deaths)).astype(REAL_DTYPE)


def test_distance_values_for_simple_case():
    dgm_1 = np.array([[0.0, 3.0]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[0.0, 3.2]], dtype=REAL_DTYPE)

    bt = oin.bottleneck_distance(dgm_1, dgm_2, delta=0.0)
    # Wasserstein has no exact mode; delta=0 is rejected by Hera. Use a tight
    # positive tolerance and check the result against the analytical value
    # within that relative window.
    delta = 1e-6
    ws_l1 = oin.wasserstein_distance(dgm_1, dgm_2, q=2.0, delta=delta, internal_p=1.0)

    assert bt == pytest.approx(0.2, abs=ABS_TOL)
    assert ws_l1 == pytest.approx(0.2, rel=delta * 10)


def test_wasserstein_zero_fast_path_skips_cpp(monkeypatch):
    dgm_1 = np.array([[0.0, 3.0], [1.0, np.inf]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[1.0, np.inf], [0.0, 3.0]], dtype=REAL_DTYPE)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("Python zero check should return before the C++ call")

    monkeypatch.setattr(oin, "_wasserstein_distance_cpp", _should_not_run)

    assert oin.wasserstein_distance(dgm_1, dgm_2) == 0.0


def test_wasserstein_zero_fast_path_handles_empty_list_and_array(monkeypatch):
    dgm_1 = []
    dgm_2 = np.empty((0, 2), dtype=REAL_DTYPE)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("Python zero check should return before the C++ call")

    monkeypatch.setattr(oin, "_wasserstein_distance_cpp", _should_not_run)

    assert oin.wasserstein_distance(dgm_1, dgm_2) == 0.0


def test_wasserstein_zero_fast_path_respects_infinity_sign(monkeypatch):
    dgm_1 = np.array([[0.0, np.inf]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[0.0, -np.inf]], dtype=REAL_DTYPE)

    monkeypatch.setattr(oin, "_wasserstein_distance_cpp", lambda *args, **kwargs: 7.0)

    assert oin.wasserstein_distance(dgm_1, dgm_2) == 7.0


def test_wasserstein_zero_fast_path_can_be_disabled(monkeypatch):
    dgm = np.array([[0.0, 3.0]], dtype=REAL_DTYPE)

    monkeypatch.setattr(oin, "_wasserstein_distance_cpp", lambda *args, **kwargs: 5.0)

    assert oin.wasserstein_distance(dgm, dgm, check_for_zero=False) == 5.0


def test_distances_match_dionysus_small():
    dion = pytest.importorskip("dionysus")
    if not hasattr(dion, "Diagram") or not hasattr(dion, "bottleneck_distance") or not hasattr(dion, "wasserstein_distance"):
        pytest.skip("installed dionysus does not expose Diagram-distance API")

    dgm_1 = np.array([[0.0, 3.0], [1.0, 2.5]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[1.0, 5.0], [1.5, 3.0]], dtype=REAL_DTYPE)
    delta = 1e-4

    dgm_1_dion = dion.Diagram(dgm_1)
    dgm_2_dion = dion.Diagram(dgm_2)

    ws_dion = dion.wasserstein_distance(dgm_1_dion, dgm_2_dion, q=2, delta=delta, internal_p=-1.0)
    ws_oin = oin.wasserstein_distance(dgm_1, dgm_2, q=2.0, delta=delta, internal_p=np.inf)
    bt_dion = dion.bottleneck_distance(dgm_1_dion, dgm_2_dion, delta=delta)
    bt_oin = oin.bottleneck_distance(dgm_1, dgm_2, delta=delta)

    assert ws_oin == pytest.approx(ws_dion, rel=5e-4, abs=1e-7)
    assert bt_oin == pytest.approx(bt_dion, rel=5e-4, abs=1e-7)


def test_distances_match_dionysus_random_large():
    dion = pytest.importorskip("dionysus")
    if not hasattr(dion, "Diagram") or not hasattr(dion, "bottleneck_distance") or not hasattr(dion, "wasserstein_distance"):
        pytest.skip("installed dionysus does not expose Diagram-distance API")

    rng = np.random.default_rng(7)
    dgm_1 = _random_off_diagonal_diagram(rng, 17)
    dgm_2 = _random_off_diagonal_diagram(rng, 23)
    delta = 1e-2

    dgm_1_dion = dion.Diagram(dgm_1)
    dgm_2_dion = dion.Diagram(dgm_2)

    ws_dion = dion.wasserstein_distance(dgm_1_dion, dgm_2_dion, q=2, delta=delta, internal_p=-1.0)
    ws_oin_np = oin.wasserstein_distance(dgm_1, dgm_2, q=2.0, delta=delta, internal_p=np.inf)

    dgm_1_list = [oin.DiagramPoint(float(b), float(d)) for b, d in dgm_1]
    dgm_2_list = [oin.DiagramPoint(float(b), float(d)) for b, d in dgm_2]
    ws_oin_list = oin.wasserstein_distance(dgm_1_list, dgm_2_list, q=2.0, delta=delta, internal_p=np.inf)

    bt_dion = dion.bottleneck_distance(dgm_1_dion, dgm_2_dion, delta=delta)
    bt_oin = oin.bottleneck_distance(dgm_1, dgm_2, delta=delta)

    assert ws_oin_np == pytest.approx(ws_dion, rel=5e-2, abs=1e-4)
    assert ws_oin_list == pytest.approx(ws_oin_np, rel=ABS_TOL, abs=ABS_TOL)
    assert bt_oin == pytest.approx(bt_dion, rel=5e-2, abs=1e-4)
