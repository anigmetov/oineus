import numpy as np
import pytest

import oineus as oin


def _random_off_diagonal_diagram(rng: np.random.Generator, n_points: int) -> np.ndarray:
    births = rng.uniform(-2.0, 2.0, size=n_points)
    persistence = rng.uniform(0.05, 2.5, size=n_points)
    deaths = births + persistence
    return np.column_stack((births, deaths)).astype(np.float64)


def test_distance_values_for_simple_case():
    dgm_1 = np.array([[0.0, 3.0]], dtype=np.float64)
    dgm_2 = np.array([[0.0, 3.2]], dtype=np.float64)

    bt = oin.bottleneck_distance(dgm_1, dgm_2, delta=0.0)
    ws_l1 = oin.wasserstein_distance(dgm_1, dgm_2, q=2.0, delta=0.0, internal_p=1.0)

    assert bt == pytest.approx(0.2, abs=1e-12)
    assert ws_l1 == pytest.approx(0.2, abs=1e-12)


def test_distances_match_dionysus_small():
    dion = pytest.importorskip("dionysus")
    if not hasattr(dion, "Diagram") or not hasattr(dion, "bottleneck_distance") or not hasattr(dion, "wasserstein_distance"):
        pytest.skip("installed dionysus does not expose Diagram-distance API")

    dgm_1 = np.array([[0.0, 3.0], [1.0, 2.5]], dtype=np.float64)
    dgm_2 = np.array([[1.0, 5.0], [1.5, 3.0]], dtype=np.float64)
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
    assert ws_oin_list == pytest.approx(ws_oin_np, rel=1e-12, abs=1e-12)
    assert bt_oin == pytest.approx(bt_dion, rel=5e-2, abs=1e-4)
