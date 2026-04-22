"""Real-dtype helper for the oineus Python package.

The compile-time C++ ``Real`` type is exposed as ``_oineus.real_dtype``
(``"float32"`` or ``"float64"``). Any numpy array that crosses into a C++
binding declaring ``nb::ndarray<oin_real, ...>`` must match this dtype.
"""

import numpy as np

from . import _oineus

REAL_DTYPE = np.dtype(_oineus.real_dtype)


def as_real_numpy(arr):
    """If ``arr`` is an ``np.ndarray``, return a C-contiguous ``REAL_DTYPE``
    view (casting/copying only when needed). Non-ndarray inputs pass through.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype != REAL_DTYPE or not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=REAL_DTYPE)
    return arr
