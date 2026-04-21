import numpy as np

from .. import _oineus
from .diff_filtration import DiffFiltration
from ._tensor_utils import to_float64_numpy, gather_values


def freudenthal_filtration(data, negate, wrap, max_dim, n_threads):
    tensor, np_data = to_float64_numpy(data)
    fil, cv = _oineus.get_freudenthal_filtration_and_crit_vertices(
        np_data, negate, wrap, max_dim, n_threads
    )
    values = gather_values(tensor, cv)
    return DiffFiltration(fil, values)
