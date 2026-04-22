import numpy as np
import eagerpy as epy

from .. import _oineus
from .diff_filtration import DiffFiltration
from ._tensor_utils import tensor_to_real_numpy, gather_values


def freudenthal_filtration(data, negate, wrap, max_dim, n_threads):
    tensor = epy.astensor(data)
    np_data = tensor_to_real_numpy(tensor)
    fil, cv = _oineus.get_freudenthal_filtration_and_crit_vertices(
        np_data, negate, wrap, max_dim, n_threads
    )
    values = gather_values(tensor, cv)
    return DiffFiltration(fil, values)
