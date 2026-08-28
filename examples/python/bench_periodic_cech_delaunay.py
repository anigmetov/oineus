import statistics
import time

import diode
import numpy as np
import torch

import oineus.diff as od


repeats = 5


def median_seconds(function):
    samples = []
    for repeat_idx in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def benchmark(dim, n_points):
    points_np = np.random.default_rng(1000 * dim + n_points).random((n_points, dim))
    bbox_min = [0.0] * dim
    bbox_max = [1.0] * dim

    vertices, _ = diode.fill_periodic_delaunay_lifts_arrays(
        points_np, bbox_min=bbox_min, bbox_max=bbox_max
    )
    n_simplices = sum(len(rows) for rows in vertices)
    diode_seconds = median_seconds(
        lambda: diode.fill_periodic_delaunay_lifts_arrays(
            points_np, bbox_min=bbox_min, bbox_max=bbox_max
        )
    )

    def build_filtration():
        points = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)
        return od.cech_delaunay_filtration(
            points, periodic=True, bbox_min=bbox_min, bbox_max=bbox_max
        )

    forward_seconds = median_seconds(build_filtration)
    backward_samples = []
    for repeat_idx in range(repeats):
        filtration = build_filtration()
        start = time.perf_counter()
        filtration.values.sum().backward()
        backward_samples.append(time.perf_counter() - start)
    backward_seconds = statistics.median(backward_samples)
    return n_simplices, diode_seconds, forward_seconds, backward_seconds


for warmup_dim in (2, 3):
    benchmark(warmup_dim, 256)

print(
    "dim n_points n_simplices diode_s forward_s backward_s "
    "diode_us_per_simplex forward_us_per_simplex backward_us_per_simplex"
)
for dim, n_points in ((2, 512), (2, 2048), (3, 512), (3, 1024), (3, 2048)):
    n_simplices, diode_seconds, forward_seconds, backward_seconds = benchmark(
        dim, n_points
    )
    scale = 1e6 / n_simplices
    print(
        dim,
        n_points,
        n_simplices,
        f"{diode_seconds:.6f}",
        f"{forward_seconds:.6f}",
        f"{backward_seconds:.6f}",
        f"{diode_seconds * scale:.3f}",
        f"{forward_seconds * scale:.3f}",
        f"{backward_seconds * scale:.3f}",
    )
