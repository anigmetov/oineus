"""
Correctness tests for the crit-sets backward in oineus.diff.

Phase 1: per-direction smoke + comparison against dgm-loss + Python
combine vs C++ combine equivalence. Phase 2 adds laziness assertions
on the new TopologyOptimizer.matrix_summary, fallback-re-reduce
equivalence, the smarter dualize default keyed on FiltrationKind, and
end-to-end FCA support.

Small inputs only -- intentionally far from any size that could blow
up the VR complex.
"""

import numpy as np
import pytest
import torch

import oineus
import oineus.diff as oin_diff
from oineus.diff import _combine


def _seeded_circle(n=20, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles) + rng.normal(0, noise, n)
    y = np.sin(angles) + rng.normal(0, noise, n)
    z = rng.normal(0, noise, n)
    pts = np.stack([x, y, z], axis=1)
    return torch.tensor(pts, dtype=torch.float64, requires_grad=True)


def _h1_diagram(pts, gradient_method, conflict_strategy="avg", step_size=1.0):
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(
        fil,
        gradient_method=gradient_method,
        step_size=step_size,
        conflict_strategy=conflict_strategy,
    )
    return dgms.in_dimension(1)


def _most_persistent_index(dgm):
    persistence = dgm[:, 1] - dgm[:, 0]
    return int(persistence.argmax().item())


@pytest.mark.parametrize("direction", ["death-up", "death-down", "birth-up", "birth-down"])
def test_crit_sets_backward_runs_all_directions(direction):
    pts = _seeded_circle()
    dgm1 = _h1_diagram(pts, gradient_method="crit-sets")
    if dgm1.shape[0] == 0:
        pytest.skip("no H1 points")

    i = _most_persistent_index(dgm1)
    b_v = dgm1[i, 0].item()
    d_v = dgm1[i, 1].item()
    persistence = d_v - b_v

    if direction == "death-up":
        target = d_v + 0.5 * abs(persistence + 1.0)
        loss = (dgm1[i, 1] - target) ** 2
    elif direction == "death-down":
        target = b_v + 0.5 * (d_v - b_v)
        loss = (dgm1[i, 1] - target) ** 2
    elif direction == "birth-up":
        target = b_v + 0.25 * (d_v - b_v)
        loss = (dgm1[i, 0] - target) ** 2
    else:  # birth-down
        target = b_v - 0.25 * abs(persistence + 1.0)
        loss = (dgm1[i, 0] - target) ** 2

    loss.backward()
    assert pts.grad is not None
    assert torch.isfinite(pts.grad).all()
    assert pts.grad.norm().item() > 0.0


def test_crit_sets_affects_at_least_as_many_simplices_as_dgm_loss():
    pts_a = _seeded_circle()
    pts_b = _seeded_circle()
    assert torch.allclose(pts_a, pts_b)

    # dgm-loss
    dgm_a = _h1_diagram(pts_a, gradient_method="dgm-loss")
    if dgm_a.shape[0] == 0:
        pytest.skip("no H1 points")
    i = _most_persistent_index(dgm_a)
    target_a = dgm_a[i, 1].item() + 0.5
    ((dgm_a[i, 1] - target_a) ** 2).backward()
    n_affected_dgm = (pts_a.grad.abs() > 1e-12).any(dim=1).sum().item()

    # crit-sets
    dgm_b = _h1_diagram(pts_b, gradient_method="crit-sets")
    j = _most_persistent_index(dgm_b)
    target_b = dgm_b[j, 1].item() + 0.5
    ((dgm_b[j, 1] - target_b) ** 2).backward()
    n_affected_crit = (pts_b.grad.abs() > 1e-12).any(dim=1).sum().item()

    assert n_affected_crit >= n_affected_dgm


@pytest.mark.parametrize("strategy", ["avg", "max", "sum"])
def test_all_conflict_strategies_run(strategy):
    pts = _seeded_circle()
    dgm1 = _h1_diagram(pts, gradient_method="crit-sets", conflict_strategy=strategy)
    if dgm1.shape[0] == 0:
        pytest.skip("no H1 points")

    # Push every point's death up by 0.5 -- creates many simultaneous
    # singleton losses, exercising the conflict path.
    targets = dgm1[:, 1].detach() + 0.5
    loss = ((dgm1[:, 1] - targets) ** 2).sum()
    loss.backward()
    assert pts.grad is not None
    assert torch.isfinite(pts.grad).all()


def test_fix_crit_avg_runs_end_to_end():
    # Phase 2 supports FCA via crit_sets_apply (the per-pair input doubles
    # as the critical-prescribed map). Phase 1 raised RuntimeError here.
    pts = _seeded_circle()
    dgm1 = _h1_diagram(pts, gradient_method="crit-sets", conflict_strategy="fca")
    if dgm1.shape[0] == 0:
        pytest.skip("no H1 points")
    target = dgm1[0, 1].item() + 0.5
    loss = (dgm1[0, 1] - target) ** 2
    loss.backward()
    assert pts.grad is not None
    assert torch.isfinite(pts.grad).all()
    assert pts.grad.norm().item() > 0.0


def test_python_combine_matches_cpp_combine_avg():
    pts = _seeded_circle()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    top_opt = oineus._oineus.TopologyOptimizer(fil.under_fil)
    top_opt.reduce_all()
    dgms = top_opt.compute_diagram(include_inf_points=False)

    fil_values = fil.values.detach()
    idx_dgm_h1 = dgms.index_diagram_in_dimension(1, as_numpy=True).astype(np.int64)
    if idx_dgm_h1.size == 0:
        pytest.skip("no H1 points")

    # Build a small batch of singleton targets (all four directions where
    # possible).
    targets_birth = fil_values.numpy()[idx_dgm_h1[:, 0]] - 0.05
    targets_death = fil_values.numpy()[idx_dgm_h1[:, 1]] + 0.05
    indices = np.concatenate([idx_dgm_h1[:, 0], idx_dgm_h1[:, 1]])
    values = np.concatenate([targets_birth, targets_death])

    crit_sets = top_opt.singletons(indices.tolist(), values.tolist())

    # C++ result.
    iv = top_opt.combine_loss(crit_sets, oineus._oineus.ConflictStrategy.Avg)
    cpp_indices = np.asarray(list(iv[0]), dtype=np.int64)
    cpp_targets = np.asarray(list(iv[1]), dtype=np.float64)

    # Python result.
    flat_idx, flat_tgt = _combine.critical_sets_to_flat(
        crit_sets, dtype=fil_values.dtype, device=fil_values.device,
    )
    py_idx, py_tgt = _combine.combine(flat_idx, flat_tgt, "avg",
                                      current_values=fil_values)

    # Compare as sorted (id, value) pairs.
    cpp_sorted = sorted(zip(cpp_indices.tolist(), cpp_targets.tolist()))
    py_sorted = sorted(zip(py_idx.tolist(), py_tgt.tolist()))
    assert len(cpp_sorted) == len(py_sorted)
    for (ci, cv), (pi, pv) in zip(cpp_sorted, py_sorted):
        assert ci == pi
        assert abs(cv - pv) < 1e-9


def test_python_combine_max_picks_largest_displacement():
    current = torch.tensor([0.0, 5.0, 0.0], dtype=torch.float64)
    flat_idx = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    # current[0]=0 -> targets {0.1, -0.4} -> abs disp {0.1, 0.4} -> pick -0.4
    # current[1]=5 -> targets {6.0, 4.0} -> abs disp {1.0, 1.0} -> tie, pick first (6.0)
    # current[2]=0 -> single target 0.2 -> pick 0.2
    flat_tgt = torch.tensor([0.1, -0.4, 6.0, 4.0, 0.2], dtype=torch.float64)
    idx, tgt = _combine.combine(flat_idx, flat_tgt, "max", current_values=current)
    out = dict(zip(idx.tolist(), tgt.tolist()))
    assert out[0] == pytest.approx(-0.4)
    assert out[1] == pytest.approx(6.0)
    assert out[2] == pytest.approx(0.2)


def test_python_combine_sum_preserves_duplicates():
    current = torch.zeros(3, dtype=torch.float64)
    flat_idx = torch.tensor([0, 0, 1], dtype=torch.long)
    flat_tgt = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    idx, tgt = _combine.combine(flat_idx, flat_tgt, "sum", current_values=current)
    assert idx.tolist() == flat_idx.tolist()
    assert tgt.tolist() == flat_tgt.tolist()


def test_python_combine_fca_overrides_critical_simplices():
    current = torch.zeros(3, dtype=torch.float64)
    flat_idx = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    flat_tgt = torch.tensor([1.0, 3.0, 4.0, 6.0, 9.0], dtype=torch.float64)
    # avg per group is {0: 2.0, 1: 5.0, 2: 9.0}
    # FCA overrides id 1 to 7.0; id 0 and 2 stay at avg.
    target_map = {1: 7.0}
    idx, tgt = _combine.combine(flat_idx, flat_tgt, "fca",
                                current_values=current, target_map=target_map)
    out = dict(zip(idx.tolist(), tgt.tolist()))
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(7.0)
    assert out[2] == pytest.approx(9.0)


# ---------------------------------------------------------------------
# Phase 2: per-side lazy reduction + crit_sets_apply
# ---------------------------------------------------------------------


def _matrix_summary(top_opt):
    """Returns (hom_status, coh_status) tuple."""
    return top_opt.matrix_summary()


def test_phase2_death_only_loop_skips_cohomology():
    pts = _seeded_circle()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets")
    dgm1 = dgms.in_dimension(1)
    if dgm1.shape[0] == 0:
        pytest.skip("no H1 points")
    i = _most_persistent_index(dgm1)
    target = dgm1[i, 1].item() + 0.5
    ((dgm1[i, 1] - target) ** 2).backward()

    top_opt = dgms._top_opt
    hom, coh = _matrix_summary(top_opt)
    assert hom.is_reduced
    assert hom.has_v
    assert hom.has_u, "death move requires U_hom"
    assert not coh.is_reduced, (
        "death-only loop must not reduce the cohomology side; "
        f"got coh={coh}"
    )


def test_phase2_birth_only_loop_skips_homology_u():
    pts = _seeded_circle()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets")
    dgm1 = dgms.in_dimension(1)
    if dgm1.shape[0] == 0:
        pytest.skip("no H1 points")

    non_zero_births = dgm1[:, 0] > 1e-6
    if not non_zero_births.any():
        pytest.skip("all births are zero")
    j = int(non_zero_births.nonzero()[0].item())
    # increase_birth: target_birth > current_birth -> needs V_coh, no U
    target = dgm1[j, 0].item() + 0.2
    ((dgm1[j, 0] - target) ** 2).backward()

    top_opt = dgms._top_opt
    hom, coh = _matrix_summary(top_opt)
    # forward already reduced hom (R+V, clearing on, no U)
    assert hom.is_reduced
    assert not hom.has_u, "birth-only loop should leave U_hom unbuilt"
    # coh side reduced for the birth move; increase_birth uses V_coh, no U
    assert coh.is_reduced
    assert coh.has_v
    assert not coh.has_u
    assert coh.clearing_opt_used, "no U on coh -> clearing should be on"


def test_phase2_increase_death_re_reduces_hom_without_clearing():
    pts = _seeded_circle()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="crit-sets")
    dgm1 = dgms.in_dimension(1)
    if dgm1.shape[0] == 0:
        pytest.skip("no H1 points")

    # Confirm the forward reduced hom with clearing on.
    hom_before, _ = _matrix_summary(dgms._top_opt)
    assert hom_before.is_reduced
    assert hom_before.clearing_opt_used
    assert not hom_before.has_u

    # Now drive an increase_death: needs U_hom, must re-reduce hom
    # with clearing off.
    i = _most_persistent_index(dgm1)
    target = dgm1[i, 1].item() + 0.5
    ((dgm1[i, 1] - target) ** 2).backward()

    hom_after, _ = _matrix_summary(dgms._top_opt)
    assert hom_after.has_u
    assert not hom_after.clearing_opt_used


def test_phase2_matches_phase1_numerically():
    """Phase 2 should produce the same crit-sets gradient as Phase 1
    (per-pair via TopologyOptimizer.singletons + combine_loss with
    eager reduce_all)."""
    pts_a = _seeded_circle()
    pts_b = _seeded_circle()
    assert torch.allclose(pts_a, pts_b)

    # Phase-2 path: the production crit-sets backward.
    fil_a = oin_diff.vr_filtration(pts_a, max_dim=2)
    dgms_a = oin_diff.persistence_diagram(fil_a, gradient_method="crit-sets")
    dgm_a = dgms_a.in_dimension(1)
    if dgm_a.shape[0] == 0:
        pytest.skip("no H1 points")
    i = _most_persistent_index(dgm_a)
    target = dgm_a[i, 1].item() + 0.5
    ((dgm_a[i, 1] - target) ** 2).backward()
    grad_phase2 = pts_a.grad.detach().clone()

    # Phase-1-style reference: drive singletons + combine_loss directly.
    fil_b = oin_diff.vr_filtration(pts_b, max_dim=2)
    fil_values_b = fil_b.values.detach()
    top_b = oineus._oineus.TopologyOptimizer(fil_b.under_fil)
    top_b.reduce_all()
    nondiff_b = top_b.compute_diagram(include_inf_points=False)

    arr = nondiff_b.index_diagram_in_dimension(1, as_numpy=True).astype(np.int64)
    # Pick the most-persistent pair on the same diagram.
    fil_values_np = fil_values_b.numpy()
    persist = fil_values_np[arr[:, 1]] - fil_values_np[arr[:, 0]]
    j = int(persist.argmax())
    b_idx, d_idx = int(arr[j, 0]), int(arr[j, 1])
    d_cur = float(fil_values_np[d_idx])
    b_grad_d = 2.0 * (d_cur - target)  # dL/d_d = 2(d - target)
    d_tgt = d_cur - b_grad_d  # step_size = 1.0

    flat_idx = [d_idx]
    flat_tgt = [d_tgt]
    crit_sets = top_b.singletons(flat_idx, flat_tgt)
    indvals = top_b.combine_loss(crit_sets, oineus._oineus.ConflictStrategy.Avg)
    indices = list(indvals[0])
    targets_ref = list(indvals[1])

    # Translate per-simplex grad into pts grad via autograd: build a
    # synthetic loss whose gradient is sum_sigma (f(sigma) - f'(sigma))
    # at the requested simplices.
    indices_t = torch.tensor(indices, dtype=torch.long)
    targets_t = torch.tensor(targets_ref, dtype=torch.float64)
    surrogate = 0.5 * ((fil_b.values[indices_t] - targets_t) ** 2).sum()
    surrogate.backward()
    grad_phase1 = pts_b.grad.detach().clone()

    diff = (grad_phase1 - grad_phase2).norm().item()
    ref = max(grad_phase1.norm().item(), grad_phase2.norm().item(), 1e-12)
    assert diff / ref < 1e-9, (
        f"Phase-1 vs Phase-2 gradients diverge: rel diff {diff / ref:.3e}, "
        f"abs diff {diff:.3e}"
    )


def test_phase2_dualize_default_picks_cohomology_for_vr():
    pts = _seeded_circle()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    # When dualize is left unset on a VR filtration, dgm-loss should
    # default to cohomology.
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss")
    assert dgms._dualize is True


def test_phase2_dualize_default_picks_homology_for_alpha_like():
    """For non-VR filtrations the default stays at homology."""
    pts = _seeded_circle()[:, :2]  # weak_alpha needs 2D or 3D
    fil = oin_diff.weak_alpha_filtration(pts.detach().requires_grad_(True))
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss")
    assert dgms._dualize is False


def test_phase2_explicit_dualize_overrides_kind_default():
    pts = _seeded_circle()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, gradient_method="dgm-loss",
                                        dualize=False)
    assert dgms._dualize is False
