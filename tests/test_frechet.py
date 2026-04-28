import numpy as np
import pytest

import oineus as oin
from oineus._dtype import REAL_DTYPE

TEST_WASSERSTEIN_DELTA = 1e-3

# Tight: exact equality up to roundoff (e.g. schedule end at 0, multistart objective tie).
# Loose: numerical equality up to barycenter-iteration noise.
ABS_TIGHT = 1e-12 if REAL_DTYPE == np.float64 else 1e-5
ABS_LOOSE = 1e-8 if REAL_DTYPE == np.float64 else 1e-4


def _sort_diagram_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=REAL_DTYPE)
    if arr.size == 0:
        return arr.reshape((0, 2))
    idx = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[idx]


def test_frechet_mean_grid_init_single_point_midpoint():
    dgm_1 = np.array([[0.0, 2.0]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 4.0]], dtype=REAL_DTYPE)

    result = oin.frechet_mean(
        [dgm_1, dgm_2],
        init_strategy=oin.FrechetMeanInit.Grid,
        grid_n_x_bins=1,
        grid_n_y_bins=1,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=10,
    )

    assert result.shape == (1, 2)
    assert result[0, 0] == pytest.approx(1.0, abs=ABS_LOOSE)
    assert result[0, 1] == pytest.approx(3.0, abs=ABS_LOOSE)


def test_frechet_mean_identical_diagrams_with_medoid_init():
    dgm = np.array([[0.0, 1.0], [2.0, 5.0]], dtype=REAL_DTYPE)

    result = oin.frechet_mean(
        [dgm, dgm.copy(), dgm.copy()],
        init_strategy=oin.FrechetMeanInit.MedoidDiagram,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=5,
    )

    assert _sort_diagram_rows(result) == pytest.approx(_sort_diagram_rows(dgm), abs=ABS_TIGHT)


def test_frechet_mean_respects_weights_for_finite_points():
    dgm_1 = np.array([[0.0, 100.0]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[10.0, 110.0]], dtype=REAL_DTYPE)

    result = oin.frechet_mean(
        [dgm_1, dgm_2],
        weights=np.array([0.75, 0.25], dtype=REAL_DTYPE),
        init_strategy=oin.FrechetMeanInit.Grid,
        grid_n_x_bins=1,
        grid_n_y_bins=1,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=10,
    )

    assert result.shape == (1, 2)
    assert result[0, 0] == pytest.approx(2.5, abs=ABS_LOOSE)
    assert result[0, 1] == pytest.approx(102.5, abs=ABS_LOOSE)


def test_frechet_mean_respects_weights_for_infinite_points():
    dgm_1 = np.array([[1.0, np.inf]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[5.0, np.inf]], dtype=REAL_DTYPE)

    result = oin.frechet_mean(
        [dgm_1, dgm_2],
        weights=np.array([0.75, 0.25], dtype=REAL_DTYPE),
        init_strategy=oin.FrechetMeanInit.FirstDiagram,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=5,
    )

    assert result.shape == (1, 2)
    assert np.isinf(result[0, 1])
    assert result[0, 0] == pytest.approx(2.0, abs=ABS_LOOSE)


def test_frechet_mean_ignore_infinite_points():
    dgm_1 = np.array([[0.0, 2.0], [1.0, np.inf]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 4.0], [3.0, np.inf]], dtype=REAL_DTYPE)

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
    assert result_ignore[0, 0] == pytest.approx(1.0, abs=ABS_LOOSE)
    assert result_ignore[0, 1] == pytest.approx(3.0, abs=ABS_LOOSE)

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
    assert result_keep[inf_mask, 0][0] == pytest.approx(2.0, abs=ABS_LOOSE)


def test_frechet_mean_raises_on_incompatible_infinite_cardinalities():
    dgm_1 = np.array([[0.0, 2.0], [1.0, np.inf]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 4.0], [3.0, np.inf], [5.0, np.inf]], dtype=REAL_DTYPE)

    with pytest.raises(RuntimeError, match="essential-point cardinalities"):
        oin.frechet_mean(
            [dgm_1, dgm_2],
            init_strategy=oin.FrechetMeanInit.FirstDiagram,
            wasserstein_delta=TEST_WASSERSTEIN_DELTA,
            max_iter=10,
            ignore_infinite_points=False,
        )


def test_frechet_mean_raises_on_nonpositive_wasserstein_delta():
    dgm_1 = np.array([[0.0, 2.0]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 4.0]], dtype=REAL_DTYPE)

    with pytest.raises(RuntimeError, match="wasserstein_delta must be positive"):
        oin.frechet_mean(
            [dgm_1, dgm_2],
            init_strategy=oin.FrechetMeanInit.Grid,
            wasserstein_delta=0.0,
            max_iter=10,
        )


def test_frechet_mean_init_helpers():
    dgm_1 = np.array([[0.0, 2.0]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 4.0]], dtype=REAL_DTYPE)

    first = oin.init_frechet_mean_first_diagram([dgm_1, dgm_2])
    medoid = oin.init_frechet_mean_medoid_diagram([dgm_1, dgm_2])
    grid = oin.init_frechet_mean_diagonal_grid([dgm_1, dgm_2], grid_n_x_bins=1, grid_n_y_bins=1)

    assert _sort_diagram_rows(first) == pytest.approx(_sort_diagram_rows(dgm_1), abs=ABS_TIGHT)
    assert medoid.shape == (1, 2)
    assert grid.shape == (1, 2)
    assert grid[0, 0] == pytest.approx(1.0, abs=ABS_LOOSE)
    assert grid[0, 1] == pytest.approx(3.0, abs=ABS_LOOSE)


def test_frechet_mean_objective_matches_expected_midpoint_value():
    dgm_1 = np.array([[0.0, 2.0]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 4.0]], dtype=REAL_DTYPE)
    barycenter = np.array([[1.0, 3.0]], dtype=REAL_DTYPE)

    objective = oin.frechet_mean_objective(
        [dgm_1, dgm_2],
        barycenter,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
    )

    assert objective == pytest.approx(1.0, abs=ABS_LOOSE)


def test_make_frechet_mean_persistence_schedule_is_monotone():
    diagrams = [
        np.array([[0.0, 5.0], [0.0, 2.0], [0.0, 0.5]], dtype=REAL_DTYPE),
        np.array([[1.0, 4.5], [1.0, 1.5]], dtype=REAL_DTYPE),
    ]

    schedule = oin.make_frechet_mean_persistence_schedule(
        diagrams,
        initial_threshold_fraction=0.5,
        max_active_growth=0.5,
        min_persistence=0.0,
    )

    assert schedule[0] >= schedule[-1]
    assert schedule[-1] == pytest.approx(0.0, abs=ABS_TIGHT)
    assert all(schedule[i] > schedule[i + 1] for i in range(len(schedule) - 1))


def test_frechet_mean_multistart_picks_best_requested_start():
    dgm_1 = np.array([[0.0, 2.0]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 4.0]], dtype=REAL_DTYPE)

    barycenter, details = oin.frechet_mean_multistart(
        [dgm_1, dgm_2],
        starts=("first", "grid"),
        grid_n_x_bins=1,
        grid_n_y_bins=1,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=20,
        return_details=True,
    )

    objectives = [run["objective"] for run in details["runs"]]
    assert details["objective"] == pytest.approx(min(objectives), abs=ABS_TIGHT)
    assert barycenter.shape[1] == 2


def test_frechet_mean_multistart_default_starts_do_not_include_grid():
    dgm_1 = np.array([[0.0, 2.0]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 4.0]], dtype=REAL_DTYPE)

    _, details = oin.frechet_mean_multistart(
        [dgm_1, dgm_2],
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=20,
        return_details=True,
    )

    start_names = [run["start"] for run in details["runs"]]
    assert "grid" not in start_names


def test_progressive_frechet_mean_runs_and_records_history():
    dgm_1 = np.array([[0.0, 5.0], [1.0, 1.4]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 7.0], [1.2, 1.5]], dtype=REAL_DTYPE)

    barycenter, details = oin.progressive_frechet_mean(
        [dgm_1, dgm_2],
        thresholds=[3.0, 0.0],
        initial_seed="medoid",
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=20,
        return_details=True,
    )

    assert barycenter.shape[1] == 2
    assert [entry["threshold"] for entry in details["history"]] == pytest.approx([3.0, 0.0])
    assert len(details["history"]) == 2


def test_progressive_frechet_mean_support_hooks_are_used():
    dgm_1 = np.array([[0.0, 5.0], [1.0, 1.4]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 7.0], [1.2, 1.5]], dtype=REAL_DTYPE)
    called = []

    def predicate(**kwargs):
        called.append(("predicate", kwargs["stage_index"]))
        return True

    def add_points(**kwargs):
        called.append(("add", kwargs["stage_index"]))
        return oin.frechet_mean_newborn_points_from_newly_active(
            kwargs["newly_active_diagrams"],
            weights=kwargs["weights"],
        )

    _ = oin.progressive_frechet_mean(
        [dgm_1, dgm_2],
        thresholds=[3.0, 0.0],
        initial_seed="medoid",
        support_update_predicate=predicate,
        support_update_fn=add_points,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=20,
    )

    assert called == [("predicate", 1), ("add", 1)]


def test_progressive_frechet_mean_multistart_picks_best_requested_start():
    dgm_1 = np.array([[0.0, 5.0], [1.0, 1.4]], dtype=REAL_DTYPE)
    dgm_2 = np.array([[2.0, 7.0], [1.2, 1.5]], dtype=REAL_DTYPE)

    barycenter, details = oin.progressive_frechet_mean_multistart(
        [dgm_1, dgm_2],
        starts=("medoid", "grid"),
        thresholds=[3.0, 0.0],
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=20,
        return_details=True,
    )

    objectives = [run["objective"] for run in details["runs"]]
    assert details["objective"] == pytest.approx(min(objectives), abs=ABS_TIGHT)
    assert barycenter.shape[1] == 2
    assert details["thresholds"] == pytest.approx([3.0, 0.0])


def _random_diagram_set(n_diagrams=8, points_per_diagram=20, seed=12345):
    rng = np.random.default_rng(seed)
    diagrams = []
    for _ in range(n_diagrams):
        births = rng.uniform(0.0, 1.0, points_per_diagram)
        deaths = births + rng.uniform(0.1, 1.0, points_per_diagram)
        diagrams.append(np.column_stack((births, deaths)).astype(REAL_DTYPE))
    return diagrams


def test_init_frechet_mean_medoid_diagram_parallel_matches_serial():
    diagrams = _random_diagram_set()
    serial = oin.init_frechet_mean_medoid_diagram(diagrams, n_threads=1)
    parallel = oin.init_frechet_mean_medoid_diagram(diagrams, n_threads=4)
    # Medoid is a discrete argmin: must be bit-identical regardless of n_threads.
    assert np.array_equal(serial, parallel)


def test_frechet_mean_parallel_matches_serial():
    diagrams = _random_diagram_set()
    common = dict(
        init_strategy=oin.FrechetMeanInit.MedoidDiagram,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=20,
    )
    serial = oin.frechet_mean(diagrams, n_threads=1, **common)
    parallel = oin.frechet_mean(diagrams, n_threads=4, **common)
    assert _sort_diagram_rows(serial) == pytest.approx(_sort_diagram_rows(parallel), abs=ABS_LOOSE)


def test_frechet_mean_objective_parallel_matches_serial():
    diagrams = _random_diagram_set()
    barycenter = oin.frechet_mean(
        diagrams,
        init_strategy=oin.FrechetMeanInit.MedoidDiagram,
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=20,
    )
    serial = oin.frechet_mean_objective(
        diagrams, barycenter, wasserstein_delta=TEST_WASSERSTEIN_DELTA, n_threads=1)
    parallel = oin.frechet_mean_objective(
        diagrams, barycenter, wasserstein_delta=TEST_WASSERSTEIN_DELTA, n_threads=4)
    assert serial == pytest.approx(parallel, abs=ABS_TIGHT)


def test_frechet_mean_multistart_accepts_n_threads():
    diagrams = _random_diagram_set(n_diagrams=5)
    serial, _ = oin.frechet_mean_multistart(
        diagrams,
        starts=("medoid", "first"),
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=15,
        return_details=True,
        n_threads=1,
    )
    parallel, _ = oin.frechet_mean_multistart(
        diagrams,
        starts=("medoid", "first"),
        wasserstein_delta=TEST_WASSERSTEIN_DELTA,
        max_iter=15,
        return_details=True,
        n_threads=4,
    )
    assert _sort_diagram_rows(serial) == pytest.approx(_sort_diagram_rows(parallel), abs=ABS_LOOSE)
