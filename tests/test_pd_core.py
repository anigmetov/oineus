"""Unit tests for the framework-neutral core of oineus.diff (pd_core).

Deliberately imports neither torch nor jax: the core is numpy in / numpy
out, and this file pins down the bookkeeping that used to be trapped
inside the torch autograd backward (matrix-need flags, U-move selection,
the dgm-loss scatter, the crit-sets flow) plus the numpy combine port
against the C++ combine_loss oracle.
"""

import numpy as np
import pytest

import oineus as oin
import oineus.diff.pd_core as pd_core


def _circle_pts(n=20, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles) + rng.normal(0, noise, n)
    y = np.sin(angles) + rng.normal(0, noise, n)
    z = rng.normal(0, noise, n)
    return np.stack([x, y, z], axis=1)


def _vr_setup(max_dim=2):
    # packed=False: the raw C++ TopologyOptimizer used by the combine oracle
    # test is fat-only; the diff wrapper in pd_forward handles both
    fil = oin.vr_filtration(_circle_pts(), max_dim=max_dim, n_threads=1,
                            packed=False)
    values = np.array([fil.cell_value_by_sorted_id(i) for i in range(fil.size())])
    return fil, values


# ---------------------------------------------------------------------------
# determine_needed_matrices / select_u_moves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("negate", [False, True])
def test_determine_needed_matrices_empty(negate):
    assert pd_core.determine_needed_matrices(np.zeros((0, 2)), negate) == \
        (False, False, False, False)


def test_determine_needed_matrices_directions():
    # non-negate mapping: birth grad > 0 (decrease birth) -> U_coh;
    # death grad > 0 (decrease death) -> V_hom; birth grad < 0 -> V_coh;
    # death grad < 0 (increase death) -> U_hom. Order: (v_hom, u_hom, v_coh, u_coh).
    g = lambda b, d: np.array([[b, d]], dtype=np.float64)
    assert pd_core.determine_needed_matrices(g(1.0, 0.0), False) == (False, False, False, True)
    assert pd_core.determine_needed_matrices(g(0.0, 1.0), False) == (True, False, False, False)
    assert pd_core.determine_needed_matrices(g(-1.0, 0.0), False) == (False, False, True, False)
    assert pd_core.determine_needed_matrices(g(0.0, -1.0), False) == (False, True, False, False)
    # negate swaps the assignment
    assert pd_core.determine_needed_matrices(g(1.0, 0.0), True) == (False, False, True, False)
    assert pd_core.determine_needed_matrices(g(0.0, 1.0), True) == (False, True, False, False)
    assert pd_core.determine_needed_matrices(g(-1.0, 0.0), True) == (False, False, False, True)
    assert pd_core.determine_needed_matrices(g(0.0, -1.0), True) == (True, False, False, False)


def test_select_u_moves_filters_directions():
    idx = np.array([10, 11, 12], dtype=np.int64)
    cur = np.array([1.0, 1.0, 1.0])
    tgt = np.array([2.0, 0.5, 1.0])   # up, down, no-op
    move = tgt != cur
    # hom side, non-negate: only increases (tgt > cur) need U
    rows, bounds = pd_core.select_u_moves(idx, cur, tgt, move, side="hom", negate=False)
    assert rows == [10] and bounds == [2.0]
    # coh side, non-negate: only decreases
    rows, bounds = pd_core.select_u_moves(idx, cur, tgt, move, side="coh", negate=False)
    assert rows == [11] and bounds == [0.5]
    # negate flips both
    rows, _ = pd_core.select_u_moves(idx, cur, tgt, move, side="hom", negate=True)
    assert rows == [11]
    rows, _ = pd_core.select_u_moves(idx, cur, tgt, move, side="coh", negate=True)
    assert rows == [10]


# ---------------------------------------------------------------------------
# pd_forward / pd_backward
# ---------------------------------------------------------------------------

def _forward(fil, values, method, **kw):
    args = dict(dualize=None, method=method, dims_to_backprop=None,
                n_threads=1, u_strategy=None, conflict_strategy="avg",
                step_size=1.0, max_dim=2)
    args.update(kw)
    return pd_core.pd_forward(fil, values, **args)


def test_pd_forward_index_diagram_matches_values():
    fil, values = _vr_setup()
    fwd = _forward(fil, values, "dgm-loss")
    assert fwd.dualize is True  # VR defaults to cohomology
    for dim in (0, 1):
        idx = fwd.index_dgm[dim]
        assert idx.dtype == np.int64 and idx.ndim == 2 and idx.shape[1] == 2
        # gathering values at the index diagram reproduces the value diagram
        dgm = fwd.top_opt.compute_diagram(include_inf_points=False)
        np.testing.assert_allclose(values[idx],
                                   dgm.in_dimension(dim, as_numpy=True))


def test_pd_backward_dgm_loss_is_scatter():
    fil, values = _vr_setup()
    fwd = _forward(fil, values, "dgm-loss")
    idx = fwd.index_dgm[1]
    if idx.size == 0:
        pytest.skip("no H1 points")
    rng = np.random.default_rng(3)
    grad = rng.normal(size=idx.shape)
    g = pd_core.pd_backward(fwd, 1, grad)
    expected = np.zeros_like(values)
    np.add.at(expected, idx[:, 0], grad[:, 0])
    np.add.at(expected, idx[:, 1], grad[:, 1])
    np.testing.assert_array_equal(g, expected)


def test_pd_backward_crit_sets_runs_and_moves_the_pair():
    fil, values = _vr_setup()
    fwd = _forward(fil, values, "crit-sets")
    idx = fwd.index_dgm[1]
    if idx.size == 0:
        pytest.skip("no H1 points")
    # push the most persistent pair's death up: grad < 0 on the death axis
    pers = values[idx[:, 1]] - values[idx[:, 0]]
    i = int(pers.argmax())
    grad = np.zeros_like(idx, dtype=np.float64)
    grad[i, 1] = -1.0
    g = pd_core.pd_backward(fwd, 1, grad)
    assert np.isfinite(g).all()
    assert (g ** 2).sum() > 0
    # the driven death simplex itself must be part of the critical set
    d_idx = idx[i, 1]
    assert g[d_idx] != 0.0


def test_pd_backward_unknown_method_raises():
    fil, values = _vr_setup()
    fwd = _forward(fil, values, "dgm-loss")
    fwd.method = "nonsense"
    with pytest.raises(RuntimeError, match="Unknown gradient method"):
        pd_core.pd_backward(fwd, 0, np.zeros((1, 2)))


# ---------------------------------------------------------------------------
# numpy combine vs the C++ combine_loss oracle
# ---------------------------------------------------------------------------

def test_numpy_combine_matches_cpp_combine_avg():
    fil, values = _vr_setup()
    top_opt = oin._oineus.TopologyOptimizer(fil, with_crit_sets=True)
    top_opt.reduce_all()
    dgms = top_opt.compute_diagram(include_inf_points=False)

    idx_dgm_h1 = dgms.index_diagram_in_dimension(1, as_numpy=True).astype(np.int64)
    if idx_dgm_h1.size == 0:
        pytest.skip("no H1 points")

    targets_birth = values[idx_dgm_h1[:, 0]] - 0.05
    targets_death = values[idx_dgm_h1[:, 1]] + 0.05
    indices = np.concatenate([idx_dgm_h1[:, 0], idx_dgm_h1[:, 1]])
    targets = np.concatenate([targets_birth, targets_death])

    crit_sets = top_opt.singletons(indices.tolist(), targets.tolist())

    iv = top_opt.combine_loss(crit_sets, oin.ConflictStrategy.Avg)
    cpp = sorted(zip(list(iv[0]), list(iv[1])))

    flat_idx, flat_tgt = pd_core.critical_sets_to_flat_np(crit_sets)
    np_idx, np_tgt = pd_core.combine(flat_idx, flat_tgt, "avg",
                                     current_values=values)
    ours = sorted(zip(np_idx.tolist(), np_tgt.tolist()))

    assert len(cpp) == len(ours)
    for (ci, cv), (pi, pv) in zip(cpp, ours):
        assert ci == pi
        assert abs(cv - pv) < 1e-9


def test_numpy_combine_max_picks_largest_displacement():
    current = np.array([0.0, 5.0, 0.0])
    flat_idx = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    # current[0]=0 -> targets {0.1, -0.4} -> abs disp {0.1, 0.4} -> pick -0.4
    # current[1]=5 -> targets {6.0, 4.0} -> tie, pick first (6.0)
    # current[2]=0 -> single target 0.2
    flat_tgt = np.array([0.1, -0.4, 6.0, 4.0, 0.2])
    idx, tgt = pd_core.combine(flat_idx, flat_tgt, "max", current_values=current)
    out = dict(zip(idx.tolist(), tgt.tolist()))
    assert out[0] == pytest.approx(-0.4)
    assert out[1] == pytest.approx(6.0)
    assert out[2] == pytest.approx(0.2)


def test_numpy_combine_sum_preserves_duplicates():
    flat_idx = np.array([0, 0, 1], dtype=np.int64)
    flat_tgt = np.array([1.0, 2.0, 3.0])
    idx, tgt = pd_core.combine(flat_idx, flat_tgt, "sum",
                               current_values=np.zeros(3))
    assert idx.tolist() == flat_idx.tolist()
    assert tgt.tolist() == flat_tgt.tolist()


def test_numpy_combine_fca_overrides_critical_simplices():
    flat_idx = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    flat_tgt = np.array([1.0, 3.0, 4.0, 6.0, 9.0])
    # avg per group is {0: 2.0, 1: 5.0, 2: 9.0}; FCA overrides id 1 to 7.0
    idx, tgt = pd_core.combine(flat_idx, flat_tgt, "fca",
                               current_values=np.zeros(3),
                               target_map={1: 7.0})
    out = dict(zip(idx.tolist(), tgt.tolist()))
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(7.0)
    assert out[2] == pytest.approx(9.0)


def test_resolve_strategy_and_u_strategy():
    assert pd_core.resolve_strategy("avg") == oin.ConflictStrategy.Avg
    assert pd_core.resolve_strategy(oin.ConflictStrategy.Max) == oin.ConflictStrategy.Max
    with pytest.raises(ValueError):
        pd_core.resolve_strategy("bogus")
    assert pd_core.resolve_u_strategy(None) == oin._oineus.UStrategy.Auto
    assert pd_core.resolve_u_strategy("row_partial") == oin._oineus.UStrategy.RowPartial
    with pytest.raises(ValueError):
        pd_core.resolve_u_strategy("bogus")
