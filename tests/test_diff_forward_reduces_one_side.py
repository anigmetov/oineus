"""Forward-reduction policy of PersistenceDiagrams.

dgm-loss: the chosen side is reduced with the cheapest recipe (R only,
no V, no U). The other side is left untouched.

crit-sets: the chosen side is reduced with parallel + clearing + V +
restore_ELZ. The other side is left untouched until backward proves
it needs anything.

We inspect the decompositions directly via has_matrix_v / has_matrix_u
/ is_reduced. matrix_summary() was removed in the phase-3 refactor.
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


def _sides(top_opt, dualize):
    hom = top_opt.homology_decomposition_ref()
    coh = top_opt.cohomology_decomposition_ref()
    return (coh, hom) if dualize else (hom, coh)


@pytest.mark.parametrize("dualize", [False, True])
def test_dgm_loss_forward_reduces_only_one_side_r_only(dualize):
    pts = _circle_pts()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss",
                                        dualize=dualize)
    chosen, other = _sides(dgms._top_opt, dualize)
    assert chosen.is_reduced
    assert not chosen.has_matrix_v(), "dgm-loss does not need V"
    assert not chosen.has_matrix_u(), "dgm-loss does not need U"
    assert not other.is_reduced, "other side must not be reduced eagerly"


@pytest.mark.parametrize("dualize", [False, True])
def test_crit_sets_forward_reduces_only_one_side_with_v(dualize):
    pts = _circle_pts()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets",
                                        dualize=dualize)
    chosen, other = _sides(dgms._top_opt, dualize)
    assert chosen.is_reduced
    assert chosen.has_matrix_v(), "crit-sets forward must compute V"
    assert not chosen.has_matrix_u(), "crit-sets forward does not compute U"
    assert not other.is_reduced, "other side must not be reduced eagerly"
