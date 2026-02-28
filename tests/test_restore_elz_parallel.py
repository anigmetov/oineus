import random

import numpy as np
import pytest

import oineus as oin
from data_utils import get_pointcloud_data


def _build_filtration():
    random.seed(42)
    np.random.seed(42)
    points = get_pointcloud_data("two_circles", dim=2, n_units=2, n_points_per_unit=8)
    return oin.vr_filtration(points, max_dim=2, n_threads=1)


def _matrix_as_python_lists(matrix):
    return [list(col) for col in matrix]


def _reduce_and_capture(filtration, dualize, params, decomp_threads):
    dcmp = oin.Decomposition(filtration, dualize=dualize, n_threads=decomp_threads)
    dcmp.reduce(params)
    return dcmp, _matrix_as_python_lists(dcmp.r_data), _matrix_as_python_lists(dcmp.v_data)


@pytest.mark.parametrize("dualize", [False, True])
def test_restore_elz_serial_matches_serial_no_clearing(dualize):
    filtration = _build_filtration()

    params_parallel = oin.ReductionParams()
    params_parallel.n_threads = 4
    params_parallel.clearing_opt = True
    params_parallel.compute_v = True
    params_parallel.restore_elz = False

    dcmp_parallel, _, _ = _reduce_and_capture(filtration, dualize, params_parallel, decomp_threads=4)
    dcmp_parallel.restore_elz(v_only=False, n_threads=1)

    params_serial = oin.ReductionParams()
    params_serial.n_threads = 1
    params_serial.clearing_opt = False
    params_serial.compute_v = True
    params_serial.restore_elz = False

    _, r_serial, v_serial = _reduce_and_capture(filtration, dualize, params_serial, decomp_threads=1)

    assert _matrix_as_python_lists(dcmp_parallel.r_data) == r_serial
    assert _matrix_as_python_lists(dcmp_parallel.v_data) == v_serial


@pytest.mark.parametrize("dualize", [False, True])
def test_restore_elz_parallel_matches_serial_restore_and_serial_no_clearing(dualize):
    filtration = _build_filtration()

    params_parallel_restore = oin.ReductionParams()
    params_parallel_restore.n_threads = 4
    params_parallel_restore.clearing_opt = True
    params_parallel_restore.compute_v = True
    params_parallel_restore.restore_elz = True

    _, r_parallel_restore, v_parallel_restore = _reduce_and_capture(
            filtration, dualize, params_parallel_restore, decomp_threads=4)

    params_parallel_then_serial_restore = oin.ReductionParams()
    params_parallel_then_serial_restore.n_threads = 4
    params_parallel_then_serial_restore.clearing_opt = True
    params_parallel_then_serial_restore.compute_v = True
    params_parallel_then_serial_restore.restore_elz = False

    dcmp_serial_restore, _, _ = _reduce_and_capture(
            filtration, dualize, params_parallel_then_serial_restore, decomp_threads=4)
    dcmp_serial_restore.restore_elz(v_only=False, n_threads=1)
    r_parallel_then_serial = _matrix_as_python_lists(dcmp_serial_restore.r_data)
    v_parallel_then_serial = _matrix_as_python_lists(dcmp_serial_restore.v_data)

    params_serial = oin.ReductionParams()
    params_serial.n_threads = 1
    params_serial.clearing_opt = False
    params_serial.compute_v = True
    params_serial.restore_elz = False

    _, r_serial, v_serial = _reduce_and_capture(filtration, dualize, params_serial, decomp_threads=1)

    assert r_parallel_restore == r_parallel_then_serial == r_serial
    assert v_parallel_restore == v_parallel_then_serial == v_serial


def test_restore_elz_requires_compute_v():
    filtration = _build_filtration()
    dcmp = oin.Decomposition(filtration, dualize=False, n_threads=2)

    params = oin.ReductionParams()
    params.n_threads = 2
    params.compute_v = False
    params.restore_elz = True

    with pytest.raises(RuntimeError, match="without V matrix"):
        dcmp.reduce(params)
