import numpy as np
import eagerpy as epy

from .. import _oineus
from .diff_filtration import DiffFiltration


def mapping_cylinder_filtration(fil_domain: DiffFiltration, fil_codomain: DiffFiltration,
                                v_domain, v_codomain) -> DiffFiltration:
    assert type(fil_domain) is DiffFiltration
    assert type(fil_codomain) is DiffFiltration

    if isinstance(v_domain, _oineus.Simplex):
        v_domain = v_domain.combinatorial_cell

    if isinstance(v_codomain, _oineus.Simplex):
        v_codomain = v_codomain.combinatorial_cell

    under_fil_dom = fil_domain.under_fil
    under_fil_cod = fil_codomain.under_fil

    under_cyl_fil, cyl_val_inds = _oineus._mapping_cylinder_with_indices(
        under_fil_dom, under_fil_cod, v_domain, v_codomain
    )

    cyl_val_inds = epy.astensor(np.array(cyl_val_inds, dtype=np.int64))
    concat_vals = epy.concatenate((epy.astensor(fil_domain.values), epy.astensor(fil_codomain.values)))

    assert concat_vals.ndim == 1 and concat_vals.shape[0] == fil_domain.size() + fil_codomain.size()

    cyl_values = concat_vals[cyl_val_inds].raw

    return DiffFiltration(under_cyl_fil, cyl_values)
