"""Differentiable slim Freudenthal filtration.

oineus.diff.freudenthal_filtration(..., slim=True) builds the slim (anchor,type)
filtration (FreudenthalFiltration_ND). These tests check that (1) the critical-vertex array
round-trips -- the gathered values equal the filtration's per-cell values and match
the fat path's array -- and (2) gradients through the slim path match finite
differences, and (3) the diff TopologyOptimizer facade dispatches to the slim
per-dim optimizer and runs a crit-set step.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import oineus as oin
import oineus.diff as od
import eagerpy as epy
from oineus._dtype import REAL_DTYPE
from oineus.diff._tensor_utils import gather_values

if REAL_DTYPE == np.float32:
    TORCH_DTYPE = torch.float32
    EPS, ATOL, RTOL, GRAD_NONZERO_SQ = 1e-3, 1e-2, 1e-2, 1e-6
else:
    TORCH_DTYPE = torch.float64
    EPS, ATOL, RTOL, GRAD_NONZERO_SQ = 1e-6, 1e-5, 1e-5, 1e-10


def _fd_grad(f, x_np, eps=EPS):
    g = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        base = x_np.copy(); base[idx] += eps; hi = float(f(base))
        base[idx] -= 2 * eps; lo = float(f(base))
        g[idx] = (hi - lo) / (2 * eps)
    return g


@pytest.mark.parametrize("dim,shape", [(2, (5, 6)), (3, (4, 4, 5))])
def test_crit_vertex_array_round_trips(dim, shape):
    rng = np.random.default_rng(0)
    data_np = rng.uniform(-1.0, 1.0, size=shape).astype(REAL_DTYPE)
    data = torch.tensor(data_np, dtype=TORCH_DTYPE)

    df = od.freudenthal_filtration(data, negate=False, max_dim=dim, slim=True, n_threads=1)
    assert type(df.under_fil).__name__ == f"_FreudenthalFiltration_{dim}D"

    # the gathered values equal the filtration's per-cell values (= data at the
    # lower-star max-value vertex of each simplex)
    fil_values = np.array([c.value for c in df.under_fil], dtype=REAL_DTYPE)
    gathered = np.asarray(df.values, dtype=REAL_DTYPE)
    assert gathered.shape == fil_values.shape
    np.testing.assert_allclose(gathered, fil_values, atol=ATOL, rtol=RTOL)

    # the slim critical-vertex array equals the fat path's array (same emission order)
    grid_cls = {1: oin.Grid_1D, 2: oin.Grid_2D, 3: oin.Grid_3D}[dim]
    grid = grid_cls(data_np, wrap=False, values_on="vertices")
    _, cv_slim = grid.freudenthal_filtration_and_critical_vertices_slim(max_dim=dim, negate=False, n_threads=1)
    _, cv_fat = oin._oineus.get_freudenthal_filtration_and_crit_vertices(data=data_np, negate=False, max_dim=dim, n_threads=1)
    assert np.array_equal(np.asarray(cv_slim), np.asarray(cv_fat))


@pytest.mark.parametrize("dim,shape", [(2, (3, 3)), (3, (3, 3, 3))])
def test_freudenthal_slim_gradient_matches_finite_difference(dim, shape):
    rng = np.random.default_rng(0)
    data_np = rng.uniform(-1.0, 1.0, size=shape).astype(REAL_DTYPE)
    data = torch.tensor(data_np, dtype=TORCH_DTYPE, requires_grad=True)

    df = od.freudenthal_filtration(data, negate=False, max_dim=dim, slim=True, n_threads=1)
    ((df.values ** 2).sum()).backward()
    grad_auto = data.grad.detach().numpy()
    assert float(np.sum(grad_auto ** 2)) > GRAD_NONZERO_SQ, "gradient is (numerically) zero"

    def f(x):
        t = torch.tensor(x, dtype=TORCH_DTYPE)
        d = od.freudenthal_filtration(t, negate=False, max_dim=dim, slim=True, n_threads=1)
        return float((d.values ** 2).sum())

    np.testing.assert_allclose(grad_auto, _fd_grad(f, data_np), atol=ATOL, rtol=RTOL)


def test_diff_optimizer_dispatches_to_slim_freudenthal():
    rng = np.random.default_rng(3)
    data_np = rng.uniform(-1.0, 1.0, size=(8, 8)).astype(REAL_DTYPE)
    data = torch.tensor(data_np, dtype=TORCH_DTYPE, requires_grad=True)

    df = od.freudenthal_filtration(data, negate=False, max_dim=2, slim=True, n_threads=1)
    top = od.TopologyOptimizer(df)
    assert type(top.under_opt).__name__ == "TopologyOptimizerFreudenthal_2D"

    top.reduce_all()
    # the slim optimizer computes a real H1 diagram (not vacuous): an 8x8 random
    # lower-star filtration has H1 features
    dgm = top.compute_diagram(include_inf_points=False)
    h1 = np.asarray(dgm.in_dimension(1))
    assert h1.shape[0] >= 1

    eps = top.get_nth_persistence(1, 1)
    indices, values = top.simplify(eps, oin.DenoiseStrategy.BirthBirth, 1)
    crit_sets = top.singletons(indices, values)
    crit_indices, crit_values = top.combine_loss(crit_sets, oin.ConflictStrategy.Max)
    # combine_loss appends indices and values in lockstep, so equal length is by
    # construction; the meaningful checks are the dispatch (above) and that the
    # whole crit-set pipeline runs on the slim cell without error
    assert len(np.asarray(crit_indices)) == len(np.asarray(crit_values))


def test_min_filtration_slim_matches_fat():
    # oineus.diff.min_filtration over two slim Freudenthal diff-fils must agree with
    # the fat path. This exercises the slim min_filtration_with_indices overload AND
    # the E.1 combinatorial-uid round-trip in diff/min_filtration.py (it keys the
    # result back into the source fils by the materialized fat cell's uid).
    rng = np.random.default_rng(11)
    shape = (5, 6)
    data1 = torch.tensor(rng.uniform(-1, 1, size=shape).astype(REAL_DTYPE), dtype=TORCH_DTYPE)
    data2 = torch.tensor(rng.uniform(-1, 1, size=shape).astype(REAL_DTYPE), dtype=TORCH_DTYPE)

    df_min_s = od.min_filtration(
        od.freudenthal_filtration(data1, max_dim=2, slim=True),
        od.freudenthal_filtration(data2, max_dim=2, slim=True))
    assert type(df_min_s.under_fil).__name__ == "_FreudenthalFiltration_2D"

    df_min_f = od.min_filtration(
        od.freudenthal_filtration(data1, max_dim=2, slim=False),
        od.freudenthal_filtration(data2, max_dim=2, slim=False))

    vs = np.array([c.value for c in df_min_s.under_fil], dtype=REAL_DTYPE)
    vf = np.array([c.value for c in df_min_f.under_fil], dtype=REAL_DTYPE)
    # same complex, same min values (as a multiset, robust to any sort tie-breaking)
    np.testing.assert_allclose(np.sort(vs), np.sort(vf), atol=ATOL, rtol=RTOL)
    # the differentiable values track this filtration's own under_fil per sorted_id
    # (this is the uid round-trip that would silently misalign without E.1)
    np.testing.assert_allclose(np.asarray(df_min_s.values, dtype=REAL_DTYPE), vs, atol=ATOL, rtol=RTOL)
    # and they track the fat path's diff values, matched cell-for-cell by uid (symmetric
    # with the packed test; for Freudenthal the diff value equals the static cell value
    # so this is an exact slim-vs-fat per-cell check of the uid round-trip)
    slim_vals = np.asarray(df_min_s.values, dtype=REAL_DTYPE)
    fat_vals = np.asarray(df_min_f.values, dtype=REAL_DTYPE)
    fat_by_uid = {c.uid: fat_vals[i] for i, c in enumerate(df_min_f.under_fil)}
    for i, c in enumerate(df_min_s.under_fil):
        assert slim_vals[i] == pytest.approx(fat_by_uid[c.uid], abs=ATOL, rel=RTOL)
