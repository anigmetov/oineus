import numpy as np
import oineus as oin


def _assert_diagram_rows_equal(got, expected_rows):
    got_arr = np.asarray(got, dtype=float)
    exp_arr = np.asarray(expected_rows, dtype=float)

    if got_arr.size == 0:
        got_rows = []
    else:
        got_rows = got_arr.reshape((-1, 2)).tolist()

    if exp_arr.size == 0:
        exp_rows = []
    else:
        exp_rows = exp_arr.reshape((-1, 2)).tolist()

    assert sorted(map(tuple, got_rows)) == sorted(map(tuple, exp_rows))


def _run_kicr(K, L):
    # KICR feature toggles belong to KICRParams.
    kicr_params = oin.KICRParams(kernel=True, image=True, cokernel=True)

    # Exercise wrapper forwarding of shared reduction params to params_f/g/ker/im/cok.
    reduction_params = oin.ReductionParams(n_threads=4)
    return oin.compute_kernel_image_cokernel_reduction(
        K, L, params=kicr_params, reduction_params=reduction_params
    )


def test_kernel_1():
    K = [
        [0, [0], 10], [1, [1], 50], [2, [2], 20], [3, [3], 50],
        [4, [4], 15], [5, [5], 12], [6, [0, 1], 50], [7, [1, 2], 60],
        [8, [2, 3], 70], [9, [3, 4], 80], [10, [0, 5], 30], [11, [4, 5], 20],
    ]
    L = [
        [0, [0], 10], [1, [1], 50], [2, [2], 20], [3, [3], 50],
        [4, [4], 15], [5, [0, 1], 50], [6, [1, 2], 60],
        [7, [2, 3], 70], [8, [3, 4], 80],
    ]

    kicr = _run_kicr(K, L)

    _assert_diagram_rows_equal(
        kicr.kernel_diagrams().in_dimension(0),
        [[30.0, 80.0]],
    )
    _assert_diagram_rows_equal(kicr.kernel_diagrams().in_dimension(1), [])
    _assert_diagram_rows_equal(kicr.cokernel_diagrams().in_dimension(0), [[12.0, 20.0]])
    _assert_diagram_rows_equal(kicr.cokernel_diagrams().in_dimension(1), [[80.0, np.inf]])
    _assert_diagram_rows_equal(
        kicr.image_diagrams().in_dimension(0),
        [[15.0, 30.0], [20.0, 60.0], [50.0, 70.0], [10.0, np.inf]],
    )
    _assert_diagram_rows_equal(kicr.image_diagrams().in_dimension(1), [])


def test_kernel_2():
    K = [
        [0, [0], 10], [1, [1], 30], [2, [2], 10], [3, [3], 0],
        [4, [0, 1], 30], [5, [1, 2], 30], [6, [0, 3], 10], [7, [2, 3], 10],
    ]
    L = [
        [0, [0], 10], [1, [1], 30], [2, [2], 10], [3, [0, 1], 30], [4, [1, 2], 30],
    ]

    kicr = _run_kicr(K, L)

    _assert_diagram_rows_equal(kicr.kernel_diagrams().in_dimension(0), [[10.0, 30.0]])
    _assert_diagram_rows_equal(kicr.kernel_diagrams().in_dimension(1), [])
    _assert_diagram_rows_equal(kicr.cokernel_diagrams().in_dimension(0), [[0.0, 10.0]])
    _assert_diagram_rows_equal(kicr.cokernel_diagrams().in_dimension(1), [[30.0, np.inf]])
    _assert_diagram_rows_equal(kicr.image_diagrams().in_dimension(0), [[10.0, np.inf]])
    _assert_diagram_rows_equal(kicr.image_diagrams().in_dimension(1), [])
