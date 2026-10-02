"""Tests for the differentiable alpha filtration (oineus.diff.alpha).

Forward values and simplex order come from diode (CGAL). Derivatives come
from the squared circumradius of the attaching simplex tau in either
PyTorch or JAX, without perturbing the native values.
"""
from pathlib import Path

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
    not HAS_DIODE,
    reason="requires diode",
)
torch_required = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")


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


@torch_required
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


@torch_required
def test_match_diode_random_3d():
    rng = np.random.default_rng(0)
    points_np = rng.random((50, 3))
    points = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)

    df = od.alpha_filtration(points)
    expected_sorted = _per_dim_sorted_values_from_fil(df.under_fil)

    actual = df.values.detach().numpy()
    np.testing.assert_array_equal(actual, expected_sorted)


@torch_required
def test_match_diode_random_2d():
    rng = np.random.default_rng(1)
    points_np = rng.random((40, 2))
    points = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)

    df = od.alpha_filtration(points)
    expected_sorted = _per_dim_sorted_values_from_fil(df.under_fil)

    actual = df.values.detach().numpy()
    np.testing.assert_array_equal(actual, expected_sorted)


@torch_required
def test_diagram_equality_with_compute_diagrams_alpha():
    """The two alpha paths must agree without a persistence cutoff."""
    rng = np.random.default_rng(2)
    points_np = rng.random((30, 3))

    diag_ref = oin.compute_diagrams_alpha(points_np, exact=True)

    points = torch.tensor(points_np, dtype=torch.float64, requires_grad=False)
    df = od.alpha_filtration(points, exact=True)
    dcmp = oin.Decomposition(df.under_fil, dualize=False)
    dcmp.reduce(oin.ReductionParams())
    diag_diff = dcmp.diagram(fil=df.under_fil, include_inf_points=True)

    for d in range(3):
        ref = np.array(sorted(map(tuple, diag_ref[d])))
        new = np.array(sorted(map(tuple, diag_diff[d])))
        if ref.size == 0 and new.size == 0:
            continue
        assert ref.shape == new.shape, f"dim {d}: shape mismatch {ref.shape} vs {new.shape}"
        np.testing.assert_array_equal(new, ref)


@torch_required
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


@torch_required
def test_guard_raises_when_diode_lacks_with_attachment(monkeypatch):
    import oineus.diff.alpha as alpha_mod

    monkeypatch.setattr(alpha_mod, "_GUARD_RESULT", False)
    pts = torch.tensor(UNIT_TET, dtype=torch.float64, requires_grad=False)
    with pytest.raises(RuntimeError, match="with_attachment"):
        od.alpha_filtration(pts)


@pytest.mark.skipif(np.dtype("float32") not in oin._dtype.REAL_MODULES,
                    reason="extension built without the float32 backend")
@torch_required
def test_diff_alpha_float32_routes_to_f32():
    # A float32 point cloud must build a genuine float32 under-filtration (not float32 values on
    # a float64 filtration), mirroring diff.vr / diff.freudenthal and the non-diff alpha facade.
    rng = np.random.default_rng(2)
    pts32 = torch.tensor(rng.random((40, 3)), dtype=torch.float32, requires_grad=True)
    df = od.alpha_filtration(pts32)
    assert type(df.under_fil).__module__.endswith("._f32")
    assert df.values.dtype == torch.float32
    # Differentiable values preserve the native float32 values exactly
    expected_sorted = _per_dim_sorted_values_from_fil(df.under_fil)
    np.testing.assert_array_equal(df.values.detach().numpy(), expected_sorted)

    # float64 still routes to the top module
    pts64 = torch.tensor(rng.random((40, 3)), dtype=torch.float64, requires_grad=True)
    assert type(od.alpha_filtration(pts64).under_fil).__module__.endswith("._oineus")


@pytest.fixture(params=["torch", "jax"])
def alpha_backend(request):
    backend = pytest.importorskip(request.param)
    if request.param == "jax":
        backend.config.update("jax_enable_x64", True)
    return backend


def backend_array(backend, data):
    if backend.__name__ == "torch":
        return backend.tensor(data, requires_grad=True)
    return backend.numpy.asarray(data)


def backend_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().numpy()
    return np.asarray(value)


def backend_value_and_grad(backend, function, data):
    points = backend_array(backend, data)
    if backend.__name__ == "torch":
        value = function(points)
        gradient, = backend.autograd.grad(value, points)
        return backend_numpy(value), backend_numpy(gradient)
    value, gradient = backend.value_and_grad(function)(points)
    return np.asarray(value), np.asarray(gradient)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_issue_5_preserves_native_diagrams(alpha_backend, dtype, monkeypatch):
    if np.dtype(dtype) not in oin._dtype.REAL_MODULES:
        pytest.skip(f"extension built without the {np.dtype(dtype)} backend")
    # Original 1000-point fixture from https://github.com/anigmetov/oineus/issues/5
    # SHA256: 72c0d58b03375076b9528baa29748a9984f4d3c57e5ffdd3f267429d36355fb2
    points_np = np.load(Path(__file__).with_name("alpha_roundoff_points.npy")).astype(dtype)
    native = oin.alpha_filtration(points_np, exact=True, packed=False)
    expected_values = np.array([cell.value for cell in native], dtype=dtype)
    decomposition = oin.Decomposition(native, dualize=False)
    params = oin.ReductionParams()
    params.n_threads = 1
    decomposition.reduce(params)
    expected_diagrams = decomposition.diagram(native, include_inf_points=False)

    fill_alpha_shapes = diode.fill_alpha_shapes
    recorded_triples = []

    def record_fill(*args, **kwargs):
        result = fill_alpha_shapes(*args, **kwargs)
        if kwargs.get("with_attachment"):
            recorded_triples.append(result)
        return result

    monkeypatch.setattr(diode, "fill_alpha_shapes", record_fill)
    points = backend_array(alpha_backend, points_np)
    fil = od.alpha_filtration(points, eps=0, exact=True)
    np.testing.assert_array_equal(backend_numpy(fil.values), expected_values)
    np.testing.assert_array_equal(
        backend_numpy(fil.values), np.array([cell.value for cell in fil], dtype=dtype))
    # Independent CGAL calls can order equal-valued simplices differently
    original_order = oin._dtype.REAL_MODULES[np.dtype(dtype)]._Filtration(
        [(simplex, value) for simplex, value, _ in recorded_triples[-1]],
        duplicates_possible=False, n_threads=1)
    for getter in ["get_edges", "get_triangles", "get_tetrahedra"]:
        np.testing.assert_array_equal(getattr(fil, getter)(), getattr(original_order, getter)())
    diagrams = od.persistence_diagram(fil, dualize=False, n_threads=1)
    for dimension, diagram in diagrams.items():
        actual = backend_numpy(diagram)
        expected = expected_diagrams.in_dimension(dimension)
        actual_order = np.lexsort((actual[:, 1], actual[:, 0]))
        expected_order = np.lexsort((expected[:, 1], expected[:, 0]))
        np.testing.assert_array_equal(actual[actual_order], expected[expected_order])
        assert np.all(actual[:, 1] > actual[:, 0]), f"invalid H{dimension} lifetime"

    def loss(x):
        current = od.alpha_filtration(x, eps=0, exact=True)
        current_diagrams = od.persistence_diagram(current, dualize=False, n_threads=1)
        return current.values.sum() + sum(
            ((diagram[:, 1] - diagram[:, 0]) ** 2).sum()
            for diagram in current_diagrams.values())

    value, gradient = backend_value_and_grad(alpha_backend, loss, points_np)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    assert np.any(gradient != 0)


@pytest.mark.parametrize("loss_kind", ["values", "diagram"])
def test_alpha_gradient_matches_directional_difference(alpha_backend, loss_kind):
    rng = np.random.default_rng(5)
    points_np = rng.uniform(-1, 1, size=(8, 3))
    direction = rng.normal(size=points_np.shape)
    direction /= np.linalg.norm(direction)

    def loss(points):
        fil = od.alpha_filtration(points, exact=True)
        if loss_kind == "values":
            weights = backend_array(alpha_backend, np.linspace(1, 2, len(fil)))
            return (weights * fil.values ** 2).sum()
        diagrams = od.persistence_diagram(fil, dualize=False, n_threads=1)
        return sum(((diagram[:, 1] - diagram[:, 0]) ** 2).sum()
                   for diagram in diagrams.values())

    _, gradient = backend_value_and_grad(alpha_backend, loss, points_np)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1e-6
    step = 1e-6
    high = backend_numpy(loss(backend_array(alpha_backend, points_np + step * direction)))
    low = backend_numpy(loss(backend_array(alpha_backend, points_np - step * direction)))
    np.testing.assert_allclose(
        np.sum(gradient * direction), (high - low) / (2 * step), rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_circumradius_helpers_preserve_scale_and_orientation(alpha_backend, dtype):
    from oineus.diff.alpha_utils import (
        triangle_circumradius_sq,
        tetrahedron_circumradius_sq,
    )

    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], dtype=dtype)
    tolerance = 5e-6 if dtype == np.float32 else 1e-12
    for scale in [1, 1e-2, 1e-3, 1e-12, 1e12]:
        points = backend_array(alpha_backend, triangle * scale)
        radius_sq = triangle_circumradius_sq(*(points[i:i + 1] for i in range(3)))
        np.testing.assert_allclose(backend_numpy(radius_sq), scale ** 2 / 3,
                                   rtol=tolerance, atol=0)
    for scale in [1, 1e-4, 1e-12, 1e12]:
        points = backend_array(alpha_backend, UNIT_TET.astype(dtype) * scale)
        for order in [(0, 1, 2, 3), (0, 2, 1, 3)]:
            radius_sq = tetrahedron_circumradius_sq(*(points[i:i + 1] for i in order))
            np.testing.assert_allclose(backend_numpy(radius_sq), 0.75 * scale ** 2,
                                       rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_alpha_values_and_gradients_scale_covariantly(alpha_backend, dtype):
    if np.dtype(dtype) not in oin._dtype.REAL_MODULES:
        pytest.skip(f"extension built without the {np.dtype(dtype)} backend")
    clouds = [
        np.array([[0, 0], [1, 0], [0.3, 0.8]], dtype=dtype),
        np.array([[0, 0, 0], [1.1, 0.1, 0], [0.2, 0.9, 0.1], [0.3, 0.25, 1.2]],
                 dtype=dtype),
    ]
    tolerance = 1e-5 if dtype == np.float32 else 1e-12

    def loss(points):
        return od.alpha_filtration(points, exact=True).values.sum()

    for unit_cloud in clouds:
        unit_value, unit_gradient = backend_value_and_grad(alpha_backend, loss, unit_cloud)
        assert np.isfinite(unit_gradient).all()
        assert np.linalg.norm(unit_gradient) > 1e-6
        for scale in [1e-12, 1e12]:
            cloud = unit_cloud * scale
            fil = od.alpha_filtration(backend_array(alpha_backend, cloud), exact=True)
            np.testing.assert_array_equal(
                backend_numpy(fil.values), np.array([cell.value for cell in fil], dtype=dtype))
            value, gradient = backend_value_and_grad(alpha_backend, loss, cloud)
            assert np.isfinite(value)
            assert np.isfinite(gradient).all()
            np.testing.assert_allclose(value / scale ** 2, unit_value,
                                       rtol=tolerance, atol=0)
            np.testing.assert_allclose(gradient / scale, unit_gradient,
                                       rtol=tolerance, atol=tolerance)


def test_alpha_rejects_nonzero_epsilon(alpha_backend):
    points = backend_array(alpha_backend, UNIT_TET)
    with pytest.raises(ValueError, match="eps"):
        od.alpha_filtration(points, eps=1e-12)
