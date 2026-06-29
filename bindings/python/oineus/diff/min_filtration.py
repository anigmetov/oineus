import numpy as np
import eagerpy as epy

from .. import _oineus
from .._dtype import module_of_oineus_obj
from .diff_filtration import DiffFiltration


def min_filtration(fil_1: DiffFiltration, fil_2: DiffFiltration) -> DiffFiltration:
    fil_1_under = fil_1.under_fil
    fil_2_under = fil_2.under_fil

    if fil_1_under.negate != fil_2_under.negate:
        raise ValueError("min_filtration: fil_1 and fil_2 must agree on negate")

    # route to the float32 / float64 backend matching the (already-built) filtrations
    sub = module_of_oineus_obj(fil_1_under)
    # _min_filtration_with_indices returns, for each cell of the result in its FINAL sorted_id
    # order, the source sorted_id in fil_1 / fil_2 -- so inds_k[i] already aligns vals_k with
    # min_fil_under at sorted_id i (no per-cell uid re-derivation needed).
    min_fil_under, inds_1, inds_2 = sub._min_filtration_with_indices(fil_1_under, fil_2_under)
    inds_1 = np.asarray(inds_1, dtype=np.int64)
    inds_2 = np.asarray(inds_2, dtype=np.int64)

    vals_1 = epy.astensor(fil_1.values)[inds_1]
    vals_2 = epy.astensor(fil_2.values)[inds_2]

    # C++ uses filtration order: with negate=True, "earlier" means larger value,
    # so the filtration min is max(v1, v2) in the original scale.
    if fil_1_under.negate:
        min_fil_values = epy.max(epy.stack((vals_1, vals_2)), axis=0).raw
    else:
        min_fil_values = epy.min(epy.stack((vals_1, vals_2)), axis=0).raw

    return DiffFiltration(min_fil_under, min_fil_values)
