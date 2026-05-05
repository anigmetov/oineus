import numpy as np
import eagerpy as epy

from .. import _oineus
from .diff_filtration import DiffFiltration


def min_filtration(fil_1: DiffFiltration, fil_2: DiffFiltration) -> DiffFiltration:
    fil_1_under = fil_1.under_fil
    fil_2_under = fil_2.under_fil

    if fil_1_under.negate != fil_2_under.negate:
        raise ValueError("min_filtration: fil_1 and fil_2 must agree on negate")

    min_fil_under, _, _ = _oineus._min_filtration_with_indices(fil_1_under, fil_2_under)

    # The indices returned by _min_filtration_with_indices reflect the
    # pre-sort order used inside the C++ helper, not the final sort order
    # imposed by the Filtration constructor on the result. Re-derive the
    # source positions via uid so that vals_*[i] aligns with min_fil_under
    # at sorted_id i.
    n = min_fil_under.size()
    inds_1 = np.empty(n, dtype=np.int64)
    inds_2 = np.empty(n, dtype=np.int64)
    for i in range(n):
        uid = min_fil_under.cell(i).uid
        inds_1[i] = fil_1_under.sorted_id_by_uid(uid)
        inds_2[i] = fil_2_under.sorted_id_by_uid(uid)

    vals_1 = epy.astensor(fil_1.values)[inds_1]
    vals_2 = epy.astensor(fil_2.values)[inds_2]

    # C++ uses filtration order: with negate=True, "earlier" means larger value,
    # so the filtration min is max(v1, v2) in the original scale.
    if fil_1_under.negate:
        min_fil_values = epy.max(epy.stack((vals_1, vals_2)), axis=0).raw
    else:
        min_fil_values = epy.min(epy.stack((vals_1, vals_2)), axis=0).raw

    return DiffFiltration(min_fil_under, min_fil_values)
