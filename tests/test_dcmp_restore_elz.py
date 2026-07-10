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
    params_parallel.use_clearing = True
    params_parallel.compute_v = True
    params_parallel.advanced.dims_to_restore_elz = []

    dcmp_parallel, _, _ = _reduce_and_capture(filtration, dualize, params_parallel, decomp_threads=4)
    dcmp_parallel.restore_elz(v_only=False, n_threads=1)

    params_serial = oin.ReductionParams()
    params_serial.n_threads = 1
    params_serial.use_clearing = False
    params_serial.compute_v = True
    params_serial.advanced.dims_to_restore_elz = []

    _, r_serial, v_serial = _reduce_and_capture(filtration, dualize, params_serial, decomp_threads=1)

    assert _matrix_as_python_lists(dcmp_parallel.r_data) == r_serial
    assert _matrix_as_python_lists(dcmp_parallel.v_data) == v_serial


@pytest.mark.parametrize("dualize", [False, True])
def test_restore_elz_parallel_matches_serial_restore_and_serial_no_clearing(dualize):
    filtration = _build_filtration()

    params_parallel_restore = oin.ReductionParams()
    params_parallel_restore.n_threads = 4
    params_parallel_restore.use_clearing = True
    params_parallel_restore.compute_v = True
    params_parallel_restore.advanced.dims_to_restore_elz = [0, 1, 2]

    _, r_parallel_restore, v_parallel_restore = _reduce_and_capture(
            filtration, dualize, params_parallel_restore, decomp_threads=4)

    params_parallel_then_serial_restore = oin.ReductionParams()
    params_parallel_then_serial_restore.n_threads = 4
    params_parallel_then_serial_restore.use_clearing = True
    params_parallel_then_serial_restore.compute_v = True
    params_parallel_then_serial_restore.advanced.dims_to_restore_elz = []

    dcmp_serial_restore, _, _ = _reduce_and_capture(
            filtration, dualize, params_parallel_then_serial_restore, decomp_threads=4)
    dcmp_serial_restore.restore_elz(v_only=False, n_threads=1)
    r_parallel_then_serial = _matrix_as_python_lists(dcmp_serial_restore.r_data)
    v_parallel_then_serial = _matrix_as_python_lists(dcmp_serial_restore.v_data)

    params_serial = oin.ReductionParams()
    params_serial.n_threads = 1
    params_serial.use_clearing = False
    params_serial.compute_v = True
    params_serial.advanced.dims_to_restore_elz = []

    _, r_serial, v_serial = _reduce_and_capture(filtration, dualize, params_serial, decomp_threads=1)

    assert r_parallel_restore == r_parallel_then_serial == r_serial
    assert v_parallel_restore == v_parallel_then_serial == v_serial


def test_restore_elz_fused_keep_working():
    # oin.reduce (fused RV) keeps the working columns; restore_elz used to pass
    # its has_matrix_v gate and index the empty at-rest r_data/v_data (segfault).
    # It must materialize first, restore, and enable the row-form U solve.
    np.random.seed(1)
    fil = oin.freudenthal_filtration(np.random.rand(12, 12))
    dcmp = oin.reduce(fil, oin.ReductionParams(compute_v=True, n_threads=4), False)

    dcmp.restore_elz()

    assert dcmp.n_elz_violators(n_threads=1) == 0
    dcmp.compute_partial_u_rows(fil, rows=[0], bounds=[1e9], dim=0, cmp="above")

    # the solved U row satisfies U[0] * V == e_0 over Z/2
    u = dcmp.u_as_csr()
    v = dcmp.v_as_csc()
    row = np.asarray((u[[0], :] @ v).todense()).ravel() % 2
    expected = np.zeros(fil.size())
    expected[0] = 1
    assert np.array_equal(row, expected)


def test_restore_elz_default_dim_records_flags_under_dualize():
    # The all-dims default restore_elz() used to push the k_all_dims sentinel
    # through the dualize dim remap and record is_elz_in_dim_ key n_dims(),
    # which no checker reads -- so the row-form U solve rejected the V it had
    # just restored. The sentinel must set the flags every checker reads.
    np.random.seed(1)
    fil = oin.freudenthal_filtration(np.random.rand(12, 12))
    params = oin.ReductionParams(compute_v=True, n_threads=4)

    dcmp = oin.Decomposition(fil, dualize=True)
    dcmp.reduce(params)
    dcmp.restore_elz()

    assert all(dcmp.n_elz_violators_in_dim(d, n_threads=1) == 0 for d in range(3))

    # rejected with 'V is not known to be in ELZ form' before the fix
    dcmp.compute_partial_u_rows(fil, rows=[0], bounds=[1e9], dim=0, cmp="below")

    # a genuine dim-0 block row (last matrix index is a vertex under dualize):
    # the solved row must satisfy U[r] * V == e_r over Z/2
    r = fil.size() - 1
    dcmp.compute_partial_u_rows(fil, rows=[r], bounds=[1e9], dim=0, cmp="below")
    u = dcmp.u_as_csr()
    v = dcmp.v_as_csc()
    row = np.asarray((u[[r], :] @ v).todense()).ravel() % 2
    expected = np.zeros(fil.size())
    expected[r] = 1
    assert np.array_equal(row, expected)

    dcmp.compute_partial_u_rows(fil, rows=[200], bounds=[1e9], dim=1, cmp="below")


def test_restore_elz_explicit_dim_records_only_that_dim_under_dualize():
    np.random.seed(1)
    fil = oin.freudenthal_filtration(np.random.rand(12, 12))
    params = oin.ReductionParams(compute_v=True, n_threads=4)

    dcmp = oin.Decomposition(fil, dualize=True)
    dcmp.reduce(params)
    dcmp.restore_elz(1)

    dcmp.compute_partial_u_rows(fil, rows=[200], bounds=[1e9], dim=1, cmp="below")
    with pytest.raises(RuntimeError, match="ELZ"):
        dcmp.compute_partial_u_rows(fil, rows=[0], bounds=[1e9], dim=0, cmp="below")


def test_restore_elz_requires_compute_v():
    filtration = _build_filtration()
    dcmp = oin.Decomposition(filtration, dualize=False, n_threads=2)

    params = oin.ReductionParams()
    params.n_threads = 2
    params.compute_v = False
    params.advanced.dims_to_restore_elz = [0, 1, 2]

    with pytest.raises(RuntimeError, match="without V matrix"):
        dcmp.reduce(params)


def test_parallel_reduction_timing_fields():
    filtration = _build_filtration()

    params_no_restore = oin.ReductionParams()
    params_no_restore.n_threads = 4
    params_no_restore.use_clearing = True
    params_no_restore.compute_v = True
    params_no_restore.advanced.dims_to_restore_elz = []

    dcmp = oin.Decomposition(filtration, dualize=False, n_threads=4)
    dcmp.reduce(params_no_restore)

    assert dcmp.timings.total >= 0.0
    assert dcmp.timings.restore_elz == 0.0
    assert dcmp.timings.copy_back >= 0.0
    assert dcmp.timings.copy_pivots >= 0.0

    params_with_restore = oin.ReductionParams()
    params_with_restore.n_threads = 4
    params_with_restore.use_clearing = True
    params_with_restore.compute_v = True
    params_with_restore.advanced.dims_to_restore_elz = [0, 1, 2]

    dcmp2 = oin.Decomposition(filtration, dualize=False, n_threads=4)
    dcmp2.reduce(params_with_restore)

    assert dcmp2.timings.total >= 0.0
    assert dcmp2.timings.restore_elz >= 0.0
    assert dcmp2.timings.copy_back >= 0.0
    assert dcmp2.timings.copy_pivots >= 0.0

    params_r_only = oin.ReductionParams()
    params_r_only.n_threads = 4
    params_r_only.use_clearing = True
    params_r_only.compute_v = False
    params_r_only.advanced.dims_to_restore_elz = []

    dcmp3 = oin.Decomposition(filtration, dualize=False, n_threads=4)
    dcmp3.reduce(params_r_only)

    assert dcmp3.timings.total >= 0.0
    assert dcmp3.timings.restore_elz == 0.0
    assert dcmp3.timings.copy_back >= 0.0
    assert dcmp3.timings.copy_pivots >= 0.0


def test_serial_without_clearing_ignores_restore_elz():
    filtration = _build_filtration()
    dcmp = oin.Decomposition(filtration, dualize=False, n_threads=1)

    params = oin.ReductionParams()
    params.n_threads = 1
    params.use_clearing = False
    params.compute_v = False
    params.advanced.dims_to_restore_elz = [0, 1, 2]

    dcmp.reduce(params)
    assert dcmp.timings.restore_elz == 0.0
