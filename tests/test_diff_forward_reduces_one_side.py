"""Forward-reduction policy of PersistenceDiagrams.

dgm-loss: the chosen side is reduced with the cheapest recipe (R only,
no V, no U). The other side is left untouched -- specifically, the
underlying Decomposition is not even constructed (boundary matrix is
cached on the optimizer, decmps are materialized lazily).

crit-sets: the chosen side is reduced with parallel + clearing + V +
restore_ELZ. The other side is left unbuilt until backward proves it
needs anything.

We inspect the chosen side via has_matrix_v / has_matrix_u / is_reduced
and the other side via the optimizer's is_hom_built / is_coh_built
predicates (fetching the other side's decmp ref would throw because
lazy construction has not happened).
"""

import numpy as np
import pytest
import torch

import oineus.diff as oin_diff


def _circle_pts(n=24, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * rng.random(n)
    x = np.cos(angles)
    y = np.sin(angles)
    z = rng.normal(0, noise, n)
    return torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float64,
                        requires_grad=True)


def _chosen_ref(top_opt, dualize):
    if dualize:
        return top_opt.cohomology_decomposition_ref()
    return top_opt.homology_decomposition_ref()


def _other_built(top_opt, dualize):
    return top_opt.is_hom_built if dualize else top_opt.is_coh_built


@pytest.mark.parametrize("dualize", [False, True])
def test_dgm_loss_forward_reduces_only_one_side_r_only(dualize):
    pts = _circle_pts()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss",
                                        dualize=dualize)
    chosen = _chosen_ref(dgms._top_opt, dualize)
    assert chosen.is_reduced
    assert not chosen.has_matrix_v(), "dgm-loss does not need V"
    assert not chosen.has_matrix_u(), "dgm-loss does not need U"
    assert not _other_built(dgms._top_opt, dualize), \
        "other side must not even be built eagerly"


@pytest.mark.parametrize("dualize", [False, True])
def test_crit_sets_forward_reduces_only_one_side_with_v(dualize):
    pts = _circle_pts()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets",
                                        dualize=dualize)
    chosen = _chosen_ref(dgms._top_opt, dualize)
    assert chosen.is_reduced
    assert chosen.has_matrix_v(), "crit-sets forward must compute V"
    assert not chosen.has_matrix_u(), "crit-sets forward does not compute U"
    assert not _other_built(dgms._top_opt, dualize), \
        "other side must not even be built eagerly"
