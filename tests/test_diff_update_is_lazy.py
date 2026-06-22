"""TopologyOptimizer.update(new_values) is lazy: it invalidates both
decompositions instead of eagerly reconstructing them.

After update, the next ensure_hom_reduced / ensure_coh_reduced must
rebuild correctly. The chosen side is inspected via its decomposition's
`is_reduced` / `has_matrix_v` / `has_matrix_u`; the unbuilt side is
inspected via the optimizer's is_hom_built / is_coh_built predicates
(fetching its decmp ref would throw since lazy construction has not
happened)."""

import numpy as np
import torch

import oineus as oin
import oineus.diff as oin_diff


def _values_from_fil(fil):
    return [fil.simplex_value_by_sorted_id(i) for i in range(len(fil))]


def test_update_invalidates_both_sides():
    rng = np.random.default_rng(0)
    pts_np = rng.uniform(-1, 1, size=(12, 2)).astype(np.float64)
    pts = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
    fil = oin_diff.vr_filtration(pts, max_dim=1)

    top_opt = oin_diff.TopologyOptimizer(
        fil, with_crit_sets=True,
        dims_to_restore_elz=[0, 1],
        n_threads=1,
    )
    top_opt.ensure_hom_reduced()
    assert top_opt.homology_decomposition_ref().is_reduced
    assert not top_opt.is_coh_built

    top_opt.update(_values_from_fil(fil), n_threads=1)

    assert not top_opt.is_hom_built
    assert not top_opt.is_coh_built


def test_update_followed_by_ensure_reduced_works():
    """After update, the next ensure_*_reduced rebuilds the side."""
    rng = np.random.default_rng(1)
    pts_np = rng.uniform(-1, 1, size=(10, 2)).astype(np.float64)
    pts = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
    fil = oin_diff.vr_filtration(pts, max_dim=1)

    # dgm-loss recipe: R only.
    top_opt = oin_diff.TopologyOptimizer(
        fil, with_crit_sets=False, n_threads=1,
    )
    top_opt.ensure_hom_reduced()
    assert top_opt.homology_decomposition_ref().is_reduced

    top_opt.update(_values_from_fil(fil), n_threads=1)
    top_opt.ensure_hom_reduced()

    decmp = top_opt.homology_decomposition_ref()
    assert decmp.is_reduced
    # dgm-loss recipe: no V, no U.
    assert not decmp.has_matrix_v()
    assert not decmp.has_matrix_u()


def test_update_preserves_crit_sets_recipe():
    """update() must preserve the per-side reduction recipe across a value
    change. Regression: it reset params to a default Params(), dropping
    compute_v for a with_crit_sets=True optimizer, so the next
    ensure_hom_reduced rebuilt a V-less decomposition and the crit-set
    walkers (which read V) crashed."""
    rng = np.random.default_rng(2)
    pts_np = rng.uniform(-1, 1, size=(15, 2)).astype(np.float64)
    pts = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
    fil = oin_diff.vr_filtration(pts, max_dim=2)

    top_opt = oin_diff.TopologyOptimizer(
        fil, with_crit_sets=True,
        dims_to_restore_elz=[0, 1, 2],
        n_threads=1,
    )

    # Crit-sets recipe builds V on the first reduction.
    top_opt.ensure_hom_reduced()
    assert top_opt.homology_decomposition_ref().has_matrix_v()

    # Re-evaluate at the same values; the recipe must survive update().
    top_opt.update(_values_from_fil(fil), n_threads=1)
    top_opt.ensure_hom_reduced()
    decmp = top_opt.homology_decomposition_ref()
    assert decmp.is_reduced
    # Would be False if update() had reset params to a default Params().
    assert decmp.has_matrix_v()
