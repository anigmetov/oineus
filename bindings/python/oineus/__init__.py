from __future__ import absolute_import

from ._oineus import *

def compute_diagrams_ls_double(data, negate, wrap, top_d, n_threads):
    print("data_dim = ", data.ndim)
    if data.ndim == 1:
        return compute_diagrams_ls_double_1(data, negate, wrap, top_d, n_threads)
    elif data.ndim == 2:
        return compute_diagrams_ls_double_2(data, negate, wrap, top_d, n_threads)
    elif data.ndim == 3:
        return compute_diagrams_ls_double_3(data, negate, wrap, top_d, n_threads)
    else:
        raise RuntimeError("Dimension not supported: " + str(data.ndim))


def compute_diagrams_ls_float(data, negate, wrap, top_d, n_threads):
    print("data_dim = ", data.ndim)
    if data.ndim == 1:
        return compute_diagrams_ls_float_1(data, negate, wrap, top_d, n_threads)
    elif data.ndim == 2:
        return compute_diagrams_ls_float_2(data, negate, wrap, top_d, n_threads)
    elif data.ndim == 3:
        return compute_diagrams_ls_float_3(data, negate, wrap, top_d, n_threads)
    else:
        raise RuntimeError("Dimension not supported: " + str(data.ndim))
