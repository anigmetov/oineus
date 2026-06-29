"""Tests for the differentiable alpha filtration (oineus.diff.alpha).

Combinatorics come from diode (CGAL); critical values are recomputed in
PyTorch as squared circumradii of the attaching simplex tau (a Gabriel
coface), so they should match diode's reported alpha values bitwise (up
to FP noise from the eps-stabilized formulas).
"""
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import diode
    HAS_DIODE = True
except ImportError:
    HAS_DIODE = False

import oineus as oin
import oineus.diff as od


pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_DIODE),
    reason="requires torch and diode",
)


UNIT_TET = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


def _diode_alpha_in_oineus_order(points_np, exact=False):
    """Return the alpha values diode reports, sorted into Oineus filtration
    order (sort by (dim, value, id)).
    """
    triples = diode.fill_alpha_shapes(points_np, exact=exact, with_attachment=True)
    pairs = [(s, a) for s, a, _ in triples]
    fil = oin._oineus.Filtration(pairs, duplicates_possible=False, n_threads=1)
    return np.array([fil.cell(i).value for i in range(fil.size())])


def test_unit_tet_values_match_diode():
    points_np = UNIT_TET.copy()
    points = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)
    df = od.alpha_filtration(points)

    # Expected alphas in Oineus order (dim 0 < dim 1 < dim 2 < dim 3,
    # within each dim sorted by value):
    # 4 vertices @ 0, 3 short edges @ 0.25, 3 long edges @ 0.5,
    # 3 Gabriel triangles @ 0.5, 1 non-Gabriel triangle @ 0.75, 1 cell @ 0.75.
    expected = np.array([0.0]*4 + [0.25]*3 + [0.5]*3 + [0.5]*3 + [0.75, 0.75])

    np.testing.assert_allclose(df.values.detach().numpy(), expected, atol=1e-9)


def _per_dim_sorted_values_from_fil(fil):
    """Read filtration cell values, sort within each dim, concatenate."""
    dim_first = fil.dim_first
    dim_last = fil.dim_last
    blocks = []
    for d in range(fil.max_dim + 1):
        block = [fil.cell(i).value for i in range(dim_first[d], dim_last[d] + 1)]
        blocks.append(np.sort(np.array(block)))
    return np.concatenate(blocks)


def test_match_diode_random_3d():
    rng = np.random.default_rng(0)
    points_np = rng.random((50, 3))
    points = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)

    df = od.alpha_filtration(points)
    expected_sorted = _per_dim_sorted_values_from_fil(df.under_fil)

    actual = df.values.detach().numpy()
    np.testing.assert_allclose(actual, expected_sorted, atol=1e-9)


def test_match_diode_random_2d():
    rng = np.random.default_rng(1)
    points_np = rng.random((40, 2))
    points = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)

    df = od.alpha_filtration(points)
    expected_sorted = _per_dim_sorted_values_from_fil(df.under_fil)

    actual = df.values.detach().numpy()
    np.testing.assert_allclose(actual, expected_sorted, atol=1e-9)


def _strip_zero_bars(arr, tol=1e-12):
    """Drop intervals with birth == death within tolerance."""
    if arr.size == 0:
        return arr
    return arr[np.abs(arr[:, 0] - arr[:, 1]) > tol]


def test_diagram_equality_with_compute_diagrams_alpha():
    """Diagrams from the diff alpha filtration should match those from
    oin.compute_diagrams_alpha after dropping degenerate (birth == death)
    bars, which the two pipelines may emit differently.
    """
    rng = np.random.default_rng(2)
    points_np = rng.random((30, 3))

    diag_ref = oin.compute_diagrams_alpha(points_np)

    points = torch.tensor(points_np, dtype=torch.float64, requires_grad=False)
    df = od.alpha_filtration(points)
    dcmp = oin.Decomposition(df.under_fil, dualize=False)
    dcmp.reduce(oin.ReductionParams())
    diag_diff = dcmp.diagram(fil=df.under_fil, include_inf_points=True)

    for d in range(3):
        ref = _strip_zero_bars(np.array(sorted(map(tuple, diag_ref[d]))))
        new = _strip_zero_bars(np.array(sorted(map(tuple, diag_diff[d]))))
        if ref.size == 0 and new.size == 0:
            continue
        assert ref.shape == new.shape, f"dim {d}: shape mismatch {ref.shape} vs {new.shape}"
        np.testing.assert_allclose(new, ref, atol=1e-9)


def test_grad_nonzero():
    rng = np.random.default_rng(3)
    points_np = rng.random((20, 3))
    points = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)

    df = od.alpha_filtration(points)
    loss = df.values.sum()
    loss.backward()

    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert points.grad.abs().sum().item() > 0.0


def test_guard_raises_when_diode_lacks_with_attachment(monkeypatch):
    import oineus.diff.alpha as alpha_mod

    monkeypatch.setattr(alpha_mod, "_GUARD_RESULT", False)
    pts = torch.tensor(UNIT_TET, dtype=torch.float64, requires_grad=False)
    with pytest.raises(RuntimeError, match="with_attachment"):
        od.alpha_filtration(pts)


@pytest.mark.skipif(np.dtype("float32") not in oin._dtype.REAL_MODULES,
                    reason="extension built without the float32 backend")
def test_diff_alpha_float32_routes_to_f32():
    # A float32 point cloud must build a genuine float32 under-filtration (not float32 values on
    # a float64 filtration), mirroring diff.vr / diff.freudenthal and the non-diff alpha facade.
    rng = np.random.default_rng(2)
    pts32 = torch.tensor(rng.random((40, 3)), dtype=torch.float32, requires_grad=True)
    df = od.alpha_filtration(pts32)
    assert type(df.under_fil).__module__.endswith("._f32")
    assert df.values.dtype == torch.float32
    # recomputed values still match diode's per-dim alpha order to float32 precision
    expected_sorted = _per_dim_sorted_values_from_fil(df.under_fil)
    np.testing.assert_allclose(df.values.detach().numpy(), expected_sorted, atol=1e-5)

    # float64 still routes to the top module
    pts64 = torch.tensor(rng.random((40, 3)), dtype=torch.float64, requires_grad=True)
    assert type(od.alpha_filtration(pts64).under_fil).__module__.endswith("._oineus")
