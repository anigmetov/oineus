import itertools

import numpy as np
import pytest
import torch

import diode
import oineus as oin
import oineus.diff as od


pytestmark = pytest.mark.skipif(
    not hasattr(diode, "fill_periodic_delaunay_lifts_arrays"),
    reason="requires diode.fill_periodic_delaunay_lifts_arrays",
)


def periodic_points_2d(dtype=torch.float64):
    axis = (torch.arange(6, dtype=dtype) + 0.5) / 6
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    background = torch.stack((x.ravel(), y.ravel()), dim=1)
    index = torch.arange(len(background), dtype=dtype)
    background += 0.005 * torch.stack((torch.sin(index), torch.cos(2 * index)), dim=1)
    pair = torch.tensor([[0.01, 0.5], [0.99, 0.5]], dtype=dtype)
    return torch.cat((pair, background))


def periodic_points_3d(dtype=torch.float64):
    return torch.tensor(np.random.default_rng(91).random((80, 3)), dtype=dtype)


def wrap(points, bbox_min, bbox_max):
    return torch.remainder(points - bbox_min, bbox_max - bbox_min) + bbox_min


def minimum_enclosing_radius_sq(points):
    best = np.inf
    for count in range(1, min(len(points), points.shape[1] + 1) + 1):
        for indices in itertools.combinations(range(len(points)), count):
            support = points[list(indices)]
            if count == 1:
                center = support[0]
            else:
                differences = support[1:] - support[0]
                gram = differences @ differences.T
                if np.linalg.matrix_rank(gram) < count - 1:
                    continue
                coefficients = np.linalg.solve(
                    2 * gram, np.sum(differences ** 2, axis=1)
                )
                center = support[0] + differences.T @ coefficients
            distances = np.sum((points - center) ** 2, axis=1)
            radius_sq = np.max(np.sum((support - center) ** 2, axis=1))
            if np.all(distances <= radius_sq + 1e-10):
                best = min(best, radius_sq)
    return best


def shortest_periodic_edge_signature(points):
    points_np = points.detach().numpy()
    vertices, offsets = diode.fill_periodic_delaunay_lifts_arrays(
        points_np, bbox_min=[0, 0], bbox_max=[1, 1]
    )
    edge_vertices = vertices[1]
    edge_offsets = offsets[1]
    lifted = points_np[edge_vertices] + edge_offsets
    values = 0.25 * np.sum((lifted[:, 0] - lifted[:, 1]) ** 2, axis=1)
    edge_idx = np.argmin(values)
    return tuple(edge_vertices[edge_idx]), tuple(edge_offsets[edge_idx].ravel())


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_cech_boundary_value_gradient_and_encoding(exact, packed, dtype):
    points = periodic_points_2d(dtype).requires_grad_()
    filtration = od.cech_delaunay_filtration(
        points,
        periodic=True,
        bbox_min=[0, 0],
        bbox_max=[1, 1],
        exact=exact,
        packed=packed,
    )
    dgm0 = od.persistence_diagram(filtration)[0]
    loss = dgm0[:, 1].min()
    loss.backward()

    assert filtration.values.dtype == dtype
    assert type(filtration.under_fil).__name__ == (
        "_PackedSimplexFiltration_64" if packed else "_Filtration"
    )
    assert loss.item() == pytest.approx(1e-4, rel=2e-5)
    expected = torch.tensor([[0.01, 0.0], [-0.01, 0.0]], dtype=dtype)
    torch.testing.assert_close(points.grad[:2], expected, rtol=2e-5, atol=2e-7)
    assert torch.count_nonzero(points.grad[2:]) == 0


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_cech_3d_backend_and_tetra_gradient(exact, packed, dtype):
    points = periodic_points_3d(dtype).requires_grad_()
    filtration = od.cech_delaunay_filtration(
        points,
        periodic=True,
        bbox_min=[0, 0, 0],
        bbox_max=[1, 1, 1],
        exact=exact,
        packed=packed,
    )
    n_tetrahedra = filtration.size_in_dimension(3)
    filtration.values[-n_tetrahedra:].sum().backward()

    assert filtration.values.dtype == dtype
    assert type(filtration.under_fil).__name__ == (
        "_PackedSimplexFiltration_64" if packed else "_Filtration"
    )
    assert type(filtration.under_fil).__module__.endswith("._f32") == (
        dtype == torch.float32
    )
    assert bool(torch.isfinite(filtration.values).all())
    assert bool(torch.isfinite(points.grad).all())
    assert torch.count_nonzero(points.grad) > 0


@pytest.mark.parametrize("dim", [2, 3])
def test_periodic_cech_values_match_independent_meb_oracle(dim):
    if dim == 2:
        points = periodic_points_2d().numpy()
    else:
        points = periodic_points_3d().numpy()
    vertices, offsets = diode.fill_periodic_delaunay_lifts_arrays(
        points, bbox_min=np.zeros(dim), bbox_max=np.ones(dim)
    )
    filtration = od.cech_delaunay_filtration(
        torch.tensor(points, dtype=torch.float64),
        periodic=True,
        bbox_min=np.zeros(dim),
        bbox_max=np.ones(dim),
    )

    start = 0
    for vertex_rows, offset_rows in zip(vertices, offsets):
        expected = []
        for vertex_row, offset_row in zip(vertex_rows, offset_rows):
            lifted = points[vertex_row] + offset_row
            expected.append(minimum_enclosing_radius_sq(lifted))
        stop = start + len(expected)
        actual = filtration.values[start:stop].detach().numpy()
        np.testing.assert_allclose(actual, np.sort(expected), rtol=1e-9, atol=1e-11)
        start = stop


@pytest.mark.parametrize("dim,expected", [(2, [1, 2, 1]), (3, [1, 3, 3, 1])])
def test_periodic_delaunay_final_complex_has_torus_betti_numbers(dim, expected):
    points = periodic_points_2d() if dim == 2 else periodic_points_3d()
    filtration = od.cech_delaunay_filtration(
        points,
        periodic=True,
        bbox_min=[0] * dim,
        bbox_max=[1] * dim,
    )
    decomposition = oin.Decomposition(filtration.under_fil, False)
    decomposition.reduce(oin.ReductionParams())
    diagram = decomposition.diagram(filtration.under_fil, include_inf_points=True)
    betti = [
        int(np.isinf(np.asarray(diagram.in_dimension(d))[:, 1]).sum())
        for d in range(dim + 1)
    ]
    assert betti == expected


def test_periodic_translation_and_rewrap_preserve_values():
    points = periodic_points_2d()
    translated = wrap(
        points + torch.tensor([0.271, 0.319], dtype=points.dtype),
        torch.zeros(2, dtype=points.dtype),
        torch.ones(2, dtype=points.dtype),
    )
    values = []
    diagrams = []
    for cloud in (points, translated):
        filtration = od.cech_delaunay_filtration(
            cloud, periodic=True, bbox_min=[0, 0], bbox_max=[1, 1], exact=True
        )
        values.append(filtration.values.detach())
        diagrams.append(od.persistence_diagram(filtration))
    torch.testing.assert_close(values[0], values[1], rtol=1e-9, atol=1e-11)
    for dim in diagrams[0]:
        first = diagrams[0][dim].detach().numpy()
        second = diagrams[1][dim].detach().numpy()
        first = first[np.lexsort((first[:, 1], first[:, 0]))]
        second = second[np.lexsort((second[:, 1], second[:, 0]))]
        np.testing.assert_allclose(first, second, rtol=1e-9, atol=1e-11)


def test_periodic_cech_scales_lifts_by_box_width():
    bbox_min = torch.tensor([2.0, -3.0], dtype=torch.float64)
    bbox_max = torch.tensor([5.0, 1.0], dtype=torch.float64)
    points = bbox_min + periodic_points_2d() * (bbox_max - bbox_min)
    filtration = od.cech_delaunay_filtration(
        points, periodic=True, bbox_min=bbox_min, bbox_max=bbox_max
    )
    loss = od.persistence_diagram(filtration)[0][:, 1].min()
    assert loss.item() == pytest.approx(0.0009, rel=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_periodic_lifts_appear_in_three_copy_euclidean_replication(dim):
    points = (
        periodic_points_2d().numpy()
        if dim == 2
        else periodic_points_3d().numpy()
    )
    periodic_vertices, periodic_offsets = diode.fill_periodic_delaunay_lifts_arrays(
        points, bbox_min=[0] * dim, bbox_max=[1] * dim
    )

    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=dim)), dtype=np.int64)
    replicated = np.concatenate([points + shift for shift in shifts])
    replicated_vertices = diode.fill_delaunay_arrays(replicated)
    count = len(points)

    for simplex_dim, periodic_rows in enumerate(periodic_vertices):
        replicated_keys = set()
        for row in replicated_vertices[simplex_dim]:
            original_ids = row % count
            if len(np.unique(original_ids)) != len(row):
                continue
            row_shifts = shifts[row // count]
            order = np.argsort(original_ids)
            sorted_ids = original_ids[order]
            normalized_shifts = row_shifts[order] - row_shifts[order][0]
            replicated_keys.add((tuple(sorted_ids), tuple(normalized_shifts.ravel())))

        for vertex_row, offset_row in zip(periodic_rows, periodic_offsets[simplex_dim]):
            key = (tuple(vertex_row), tuple(offset_row.ravel()))
            assert key in replicated_keys


def test_periodic_h0_gradient_matches_finite_difference():
    initial = periodic_points_2d()

    def loss_at(points):
        filtration = od.cech_delaunay_filtration(
            points, periodic=True, bbox_min=[0, 0], bbox_max=[1, 1]
        )
        return od.persistence_diagram(filtration)[0][:, 1].min()

    points = initial.clone().requires_grad_()
    loss_at(points).backward()
    analytical = points.grad[0, 0].item()
    delta = 1e-6
    plus = initial.clone()
    minus = initial.clone()
    plus[0, 0] += delta
    minus[0, 0] -= delta
    signature = shortest_periodic_edge_signature(initial)
    assert shortest_periodic_edge_signature(plus) == signature
    assert shortest_periodic_edge_signature(minus) == signature
    finite_difference = (loss_at(plus) - loss_at(minus)).item() / (2 * delta)
    assert analytical == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)


def test_periodic_optimization_crosses_then_wraps_and_decreases_loss():
    bbox_min = torch.zeros(2, dtype=torch.float64)
    bbox_max = torch.ones(2, dtype=torch.float64)
    points = periodic_points_2d().requires_grad_()
    optimizer = torch.optim.SGD([points], lr=1.5)

    before_distance = 1 - torch.abs(points[0, 0] - points[1, 0]).item()
    optimizer.zero_grad()
    filtration = od.cech_delaunay_filtration(
        points, periodic=True, bbox_min=bbox_min, bbox_max=bbox_max
    )
    before_loss = od.persistence_diagram(filtration)[0][:, 1].min()
    before_loss.backward()
    optimizer.step()

    assert points[0, 0] < 0
    assert points[1, 0] > 1
    with torch.no_grad():
        points.copy_(wrap(points, bbox_min, bbox_max))
    assert optimizer.param_groups[0]["params"][0] is points
    assert bool(((points >= bbox_min) & (points < bbox_max)).all())

    after_distance = 1 - torch.abs(points[0, 0] - points[1, 0]).item()
    after_filtration = od.cech_delaunay_filtration(
        points, periodic=True, bbox_min=bbox_min, bbox_max=bbox_max
    )
    after_loss = od.persistence_diagram(after_filtration)[0][:, 1].min()
    assert after_distance < before_distance
    assert after_loss < before_loss


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"bbox_min": None, "bbox_max": [1, 1]}, "required"),
        ({"bbox_min": [0], "bbox_max": [1]}, "shape"),
        ({"bbox_min": [1, 0], "bbox_max": [0, 1]}, "greater"),
    ],
)
def test_periodic_cech_validates_box(kwargs, match):
    with pytest.raises(ValueError, match=match):
        od.cech_delaunay_filtration(periodic_points_2d(), periodic=True, **kwargs)


def test_periodic_cech_rejects_box_width_overflow():
    with pytest.raises(ValueError, match="width must be finite"):
        od.cech_delaunay_filtration(
            periodic_points_2d(torch.float32),
            periodic=True,
            bbox_min=[-3e38, -3e38],
            bbox_max=[3e38, 3e38],
        )


def test_periodic_cech_requires_wrapped_points():
    points = periodic_points_2d()
    points[0, 0] = 1.0
    with pytest.raises(ValueError, match="half-open"):
        od.cech_delaunay_filtration(
            points, periodic=True, bbox_min=[0, 0], bbox_max=[1, 1]
        )


def test_periodic_cech_reports_missing_diode_capability(monkeypatch):
    monkeypatch.setattr(oin, "_HAS_DIODE_PERIODIC_LIFTS", False)
    with pytest.raises(RuntimeError, match="fill_periodic_delaunay_lifts_arrays"):
        od.cech_delaunay_filtration(
            periodic_points_2d(),
            periodic=True,
            bbox_min=[0, 0],
            bbox_max=[1, 1],
        )
