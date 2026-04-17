import numpy as np
import pytest

import oineus as oin

TEST_WASSERSTEIN_DELTA = 1e-3


def _sort_diagram_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape((0, 2))
    idx = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[idx]


def test_frechet_mean_grid_init_single_point_midpoint():
    dgm_1 = np.array([[0.0, 2.0]], dtype=np.float64)
    dgm_2 = np.array([[2.0, 4.0]], dtype=np.float64)

    result = oin.frechet_mean(
        [dgm_1, dgm_2],
        init_strategy=oin.FrechetMeanInit.Grid,
        grid_n_x_bins=1,
        grid_n_y_bins=1,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=10,
    )

    assert result.shape == (1, 2)
    assert result[0, 0] == pytest.approx(1.0, abs=1e-8)
    assert result[0, 1] == pytest.approx(3.0, abs=1e-8)


def test_frechet_mean_identical_diagrams_with_medoid_init():
    dgm = np.array([[0.0, 1.0], [2.0, 5.0]], dtype=np.float64)

    result = oin.frechet_mean(
        [dgm, dgm.copy(), dgm.copy()],
        init_strategy=oin.FrechetMeanInit.MedoidDiagram,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=5,
    )

    assert _sort_diagram_rows(result) == pytest.approx(_sort_diagram_rows(dgm), abs=1e-10)


def test_frechet_mean_respects_weights_for_finite_points():
    dgm_1 = np.array([[0.0, 100.0]], dtype=np.float64)
    dgm_2 = np.array([[10.0, 110.0]], dtype=np.float64)

    result = oin.frechet_mean(
        [dgm_1, dgm_2],
        weights=np.array([0.75, 0.25], dtype=np.float64),
        init_strategy=oin.FrechetMeanInit.Grid,
        grid_n_x_bins=1,
        grid_n_y_bins=1,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=10,
    )

    assert result.shape == (1, 2)
    assert result[0, 0] == pytest.approx(2.5, abs=1e-8)
    assert result[0, 1] == pytest.approx(102.5, abs=1e-8)


def test_frechet_mean_respects_weights_for_infinite_points():
    dgm_1 = np.array([[1.0, np.inf]], dtype=np.float64)
    dgm_2 = np.array([[5.0, np.inf]], dtype=np.float64)

    result = oin.frechet_mean(
        [dgm_1, dgm_2],
        weights=np.array([0.75, 0.25], dtype=np.float64),
        init_strategy=oin.FrechetMeanInit.FirstDiagram,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=5,
    )

    assert result.shape == (1, 2)
    assert np.isinf(result[0, 1])
    assert result[0, 0] == pytest.approx(2.0, abs=1e-8)


def test_frechet_mean_ignore_infinite_points():
    dgm_1 = np.array([[0.0, 2.0], [1.0, np.inf]], dtype=np.float64)
    dgm_2 = np.array([[2.0, 4.0], [3.0, np.inf]], dtype=np.float64)

    result_ignore = oin.frechet_mean(
        [dgm_1, dgm_2],
        init_strategy=oin.FrechetMeanInit.Grid,
        grid_n_x_bins=1,
        grid_n_y_bins=1,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=10,
        ignore_infinite_points=True,
    )

    assert result_ignore.shape == (1, 2)
    assert np.isfinite(result_ignore).all()
    assert result_ignore[0, 0] == pytest.approx(1.0, abs=1e-8)
    assert result_ignore[0, 1] == pytest.approx(3.0, abs=1e-8)

    result_keep = oin.frechet_mean(
        [dgm_1, dgm_2],
        init_strategy=oin.FrechetMeanInit.FirstDiagram,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=10,
        ignore_infinite_points=False,
    )

    assert result_keep.shape == (2, 2)
    inf_mask = np.isinf(result_keep[:, 1])
    assert inf_mask.sum() == 1
    assert result_keep[inf_mask, 0][0] == pytest.approx(2.0, abs=1e-8)


def test_frechet_mean_raises_on_incompatible_infinite_cardinalities():
    dgm_1 = np.array([[0.0, 2.0], [1.0, np.inf]], dtype=np.float64)
    dgm_2 = np.array([[2.0, 4.0], [3.0, np.inf], [5.0, np.inf]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="essential-point cardinalities"):
        oin.frechet_mean(
            [dgm_1, dgm_2],
            init_strategy=oin.FrechetMeanInit.FirstDiagram,
            wasserstein_delta=TEST_WASSERSTEIN_DELTA,
            max_iter=10,
            ignore_infinite_points=False,
        )


def test_frechet_mean_raises_on_nonpositive_wasserstein_delta():
    dgm_1 = np.array([[0.0, 2.0]], dtype=np.float64)
    dgm_2 = np.array([[2.0, 4.0]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="wasserstein_delta must be positive"):
        oin.frechet_mean(
            [dgm_1, dgm_2],
            init_strategy=oin.FrechetMeanInit.Grid,
            wasserstein_delta=0.0,
            max_iter=10,
        )
