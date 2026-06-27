import numpy as np
import eagerpy as epy

from .. import _oineus, _SLIM_SIMPLEX_FIL_TYPES
from .diff_filtration import DiffFiltration


def _fatten_under_with_values(under_fil, values):
    """Materialize a slim/packed under-filtration to fat, keeping ``values`` aligned.

    mapping_cylinder's C++ builder is fat-Simplex-only and returns value indices
    aligned to the filtration order it is handed, so a slim/packed under-filtration
    must be fattened first. The fat _Filtration constructor re-sorts the cells by the
    same (value, dim, id) keys that ``under_fil.cells()`` already emits in sorted order
    (id is a total tie-break), so the rebuild is ORDER-PRESERVING: fat cell j is the
    materialization of under_fil cell j, and ``values`` (aligned to the under_fil sorted
    order) is therefore already aligned to the fat order -- no permutation is needed.

    Do NOT reintroduce a per-cell Python permutation here: building one via
    under_fil.sorted_id_by_uid(fat.cell(j).uid) is an O(n) loop with ~3 binding
    round-trips per cell over a (million-cell) filtration, which dwarfs the actual
    reduction (CLAUDE.md: do not wrap a mildly-superlinear C++ core in a linear Python
    loop). The identity is asserted by tests/test_diff_vr_packed.py
    (test_diff_mapping_cylinder_packed_matches_fat, cell-for-cell vs the fat path). A
    hypothetical future encoding whose fat rebuild reorders would need a C++-side
    permutation (a vectorized sorted_id_by_uid binding), never a Python loop.
    """
    if not isinstance(under_fil, _SLIM_SIMPLEX_FIL_TYPES):
        return under_fil, values
    return _oineus._Filtration(under_fil.cells(), under_fil.negate), values


def mapping_cylinder_filtration(fil_domain: DiffFiltration, fil_codomain: DiffFiltration,
                                v_domain, v_codomain,
                                v_domain_value=None, v_codomain_value=None) -> DiffFiltration:
    assert type(fil_domain) is DiffFiltration
    assert type(fil_codomain) is DiffFiltration

    if isinstance(v_domain, _oineus.Simplex):
        v_domain = v_domain.combinatorial_cell

    if isinstance(v_codomain, _oineus.Simplex):
        v_codomain = v_codomain.combinatorial_cell

    # The cylinder is a fat ProductCell<Simplex, Simplex> filtration, so materialize slim/packed
    # under-filtrations to fat and reorder the diff value tensors into the fat sorted order.
    under_fil_dom, vals_dom = _fatten_under_with_values(
        fil_domain.under_fil, epy.astensor(fil_domain.values))
    under_fil_cod, vals_cod = _fatten_under_with_values(
        fil_codomain.under_fil, epy.astensor(fil_codomain.values))

    if v_domain_value is None:
        v_domain_value = under_fil_dom.neg_infinity()
    if v_codomain_value is None:
        v_codomain_value = under_fil_cod.neg_infinity()

    under_cyl_fil, cyl_val_inds = _oineus._mapping_cylinder_with_indices(
        under_fil_dom, under_fil_cod, v_domain, v_codomain, v_domain_value, v_codomain_value
    )

    cyl_val_inds = epy.astensor(np.array(cyl_val_inds, dtype=np.int64))
    concat_vals = epy.concatenate((vals_dom, vals_cod))

    assert concat_vals.ndim == 1 and concat_vals.shape[0] == fil_domain.size() + fil_codomain.size()

    cyl_values = concat_vals[cyl_val_inds].raw

    under_cyl_fil.kind = _oineus.FiltrationKind.MappingCylinder
    return DiffFiltration(under_cyl_fil, cyl_values)
