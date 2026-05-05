"""Tests covering the bugfix landing for the oineus.diff review.

Each test corresponds to one fix in
~/.claude/plans/here-are-some-comments-optimized-pixel.md so a regression
points back to the originating issue.
"""
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import oineus as oin
import oineus.diff as od
from oineus._dtype import REAL_DTYPE


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
TORCH_DTYPE = torch.float32 if REAL_DTYPE == np.float32 else torch.float64


# ---------------------------------------------------------------------------
# Fix #1 -- min_filtration respects the negate flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("negate", [False, True])
def test_min_filtration_values_match_under_fil(negate):
    """DiffFiltration.values must equal under_fil values for every cell."""
    rng = np.random.default_rng(123)
    data1 = torch.tensor(rng.uniform(-1, 1, size=(4, 5)).astype(REAL_DTYPE),
                         dtype=TORCH_DTYPE, requires_grad=True)
    data2 = torch.tensor(rng.uniform(-1, 1, size=(4, 5)).astype(REAL_DTYPE),
                         dtype=TORCH_DTYPE, requires_grad=True)

    df1 = od.freudenthal_filtration(data1, negate=negate, max_dim=2)
    df2 = od.freudenthal_filtration(data2, negate=negate, max_dim=2)
    df_min = od.min_filtration(df1, df2)

    fil = df_min.under_fil
    assert fil.negate == negate
    diff_values = df_min.values.detach().numpy()
    for i in range(fil.size()):
        assert diff_values[i] == pytest.approx(
            fil.simplex_value_by_sorted_id(i)
        ), f"value mismatch at sorted id {i} (negate={negate})"


def test_min_filtration_rejects_mismatched_negate():
    rng = np.random.default_rng(0)
    data = torch.tensor(rng.uniform(-1, 1, size=(3, 3)).astype(REAL_DTYPE), dtype=TORCH_DTYPE)
    df1 = od.freudenthal_filtration(data, negate=False, max_dim=2)
    df2 = od.freudenthal_filtration(data, negate=True, max_dim=2)
    with pytest.raises(ValueError):
        od.min_filtration(df1, df2)


# ---------------------------------------------------------------------------
# Fix #2 -- crit-sets gradient method works for cube filtrations
# ---------------------------------------------------------------------------

def test_persistence_crit_sets_works_for_cube_filtration():
    rng = np.random.default_rng(7)
    data = torch.tensor(rng.uniform(-1, 1, size=(4, 5)).astype(REAL_DTYPE),
                        dtype=TORCH_DTYPE, requires_grad=True)
    df = od.cube_filtration(data, max_dim=2, values_on="vertices")
    dgms = od.persistence_diagram(df, gradient_method="crit-sets")
    dgm0 = dgms.in_dimension(0)
    assert dgm0.shape[0] > 0
    # Compare to the non-diff cube diagram
    fil_nd = oin.cube_filtration(data.detach().numpy(), max_dim=2, values_on="vertices")
    dcmp = oin.Decomposition(fil_nd, dualize=False)
    dcmp.reduce(oin.ReductionParams())
    nd_dgms = dcmp.diagram(fil_nd, include_inf_points=False)
    nd_dgm0 = np.asarray(nd_dgms.in_dimension(0))
    diff_finite = dgm0[(dgm0[:, 1] != float("inf"))].detach().numpy()
    a = np.asarray(sorted(map(tuple, diff_finite)))
    b = np.asarray(sorted(map(tuple, nd_dgm0)))
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# Fix #3 -- DiffFiltration delegates via __getattr__
# ---------------------------------------------------------------------------

def test_diff_filtration_delegates_methods_correctly():
    rng = np.random.default_rng(0)
    data = torch.tensor(rng.uniform(0, 1, size=(3, 3)).astype(REAL_DTYPE), dtype=TORCH_DTYPE)
    df = od.freudenthal_filtration(data, max_dim=2)
    fil = df.under_fil

    # Bridge methods that the old explicit list got wrong:
    assert df.size_in_dimension(0) == fil.size_in_dimension(0)
    assert df.size_in_dimension(1) == fil.size_in_dimension(1)
    assert df.boundary_matrix() == fil.boundary_matrix()  # default n_threads=1

    # Plus generic delegation -- methods we never explicitly proxied.
    assert df.max_dim() == fil.max_dim()
    assert df.size() == fil.size()
    assert df.cells() == fil.cells()

    # Per-cell uid-based lookups (formerly broken).
    cell0 = fil.cell(0)
    uid = cell0.uid
    assert df.value_by_uid(uid) == fil.value_by_uid(uid)
    assert df.cell_by_uid(uid) == fil.cell_by_uid(uid)

    # under_fil and values still resolve normally despite __getattr__.
    assert df.under_fil is fil
    assert df.values is not None


# ---------------------------------------------------------------------------
# Fix #4 -- freudenthal_filtration has working defaults
# ---------------------------------------------------------------------------

def test_freudenthal_filtration_works_with_defaults():
    data = torch.zeros((4, 4), dtype=TORCH_DTYPE)
    df = od.freudenthal_filtration(data)  # no extra args
    assert df.size() > 0


# ---------------------------------------------------------------------------
# Fix #5 -- dgm-loss backward handles include_inf_points correctly
# ---------------------------------------------------------------------------

def test_dgm_loss_backward_with_inf_points():
    """Backward must not index out-of-range sentinels and must give birth
    gradient for inf-pairs."""
    rng = np.random.default_rng(0)
    pts_np = rng.uniform(-1, 1, size=(8, 2)).astype(REAL_DTYPE)
    pts = torch.tensor(pts_np, dtype=TORCH_DTYPE, requires_grad=True)
    df = od.vr_filtration(pts, max_dim=1)
    dgms = od.persistence_diagram(df, gradient_method="dgm-loss",
                                   include_inf_points=True)
    dgm0 = dgms.in_dimension(0)
    assert dgm0.shape[0] > 0
    # There should be exactly one inf-death point in H0 for a connected VR
    is_inf = ~torch.isfinite(dgm0[:, 1])
    assert int(is_inf.sum().item()) >= 1

    # Loss that touches both finite and inf rows
    finite_part = dgm0[~is_inf]
    inf_part = dgm0[is_inf]
    loss = (finite_part[:, 1] - finite_part[:, 0]).sum() + inf_part[:, 0].sum()
    loss.backward()
    g = pts.grad.detach().numpy()
    assert np.isfinite(g).all()
    assert np.any(g != 0.0)


# ---------------------------------------------------------------------------
# Fix #6 -- TopologyOptimizer.match forwards wasserstein_delta
# ---------------------------------------------------------------------------

def test_topology_optimizer_match_accepts_wasserstein_delta():
    rng = np.random.default_rng(0)
    pts = torch.tensor(rng.uniform(-1, 1, size=(8, 2)).astype(REAL_DTYPE),
                       dtype=TORCH_DTYPE, requires_grad=True)
    df = od.vr_filtration(pts, max_dim=1)
    opt = od.TopologyOptimizer(df)
    template = [oin.DiagramPoint(0.0, 0.5)]
    iv = opt.match(template_dgm=template, dim=0, wasserstein_q=1.0,
                   wasserstein_delta=0.05)
    assert iv is not None


# ---------------------------------------------------------------------------
# Fix #7 -- compute_ker_cok_reduction_cyl works after binding-name repair
# ---------------------------------------------------------------------------

def test_compute_ker_cok_reduction_cyl_smoke():
    fil_2 = oin.list_to_filtration([
        (0, [0], 0.0),
        (1, [1], 0.0),
        (2, [0, 1], 1.0),
    ])
    fil_3 = oin.list_to_filtration([
        (0, [0], 0.0),
        (1, [1], 0.5),
        (2, [0, 1], 2.0),
    ])
    kicr = oin.compute_ker_cok_reduction_cyl(fil_2, fil_3)
    assert kicr is not None


# ---------------------------------------------------------------------------
# Fix #8 -- __all__ exposes core constructors and types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "vr_filtration", "freudenthal_filtration", "cube_filtration",
    "min_filtration", "mapping_cylinder", "multiply_filtration",
    "compute_ker_cok_reduction_cyl",
    "Filtration", "ProdFiltration",
    "CubeFiltration_1D", "CubeFiltration_2D", "CubeFiltration_3D",
    "Simplex", "FiltrationKind", "KerImCokReduced", "KerImCokReducedProd",
    "TopologyOptimizer", "TopologyOptimizerProd",
    "TopologyOptimizerCube_1D", "TopologyOptimizerCube_2D", "TopologyOptimizerCube_3D",
])
def test_public_api_in_all(name):
    assert name in oin.__all__, f"{name} should appear in oineus.__all__"
    assert getattr(oin, name) is not None


# ---------------------------------------------------------------------------
# Fix #9 -- public constructors raise ValueError on bad input under -O
# ---------------------------------------------------------------------------

def test_alpha_rejects_wrong_dim():
    pts = np.zeros((5, 4), dtype=REAL_DTYPE)
    with pytest.raises(ValueError):
        oin.compute_diagrams_alpha(pts)


def test_alpha_rejects_1d_input():
    pts = np.zeros(10, dtype=REAL_DTYPE)
    with pytest.raises(ValueError):
        oin.compute_diagrams_alpha(pts)


def test_max_distance_rejects_too_few_rows():
    with pytest.raises(ValueError):
        oin.max_distance(np.zeros((1, 3), dtype=REAL_DTYPE))


def test_compute_diagrams_vr_rejects_non_2d():
    with pytest.raises(ValueError):
        oin.compute_diagrams_vr(np.zeros(5, dtype=REAL_DTYPE))
