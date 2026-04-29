#!/usr/bin/env python3
"""Compare serial vs parallel batched topology loss.

Mirrors the per-sample loss used by amorphous training:

  pred_pts -> {cech_delaunay,weak_alpha}_filtration (differentiable)
           -> persistence_diagram[hom_dim]
           -> wasserstein_cost(pred_dgm, target_dgm)

For a batch of point clouds we evaluate the per-sample loss two ways:
  1. serial: a plain Python loop;
  2. parallel: concurrent.futures.ThreadPoolExecutor.

We then check that the per-sample losses and the resulting gradients on the
input tensors agree, and we print wall-clock timings for forward + backward.

Threading is the right primitive: per CLAUDE.md, Oineus C++ work (persistence
reduction, Hera Wasserstein) releases the GIL, and torch ops do too. Diode's
alpha-shape build is the one piece that holds the GIL; we still call it under
the executor to get a realistic apples-to-apples picture.

With --breakdown, the script runs single-threaded and reports per-stage
cumulative time (alpha/diode build, differentiable values, filtration assembly,
persistence reduction, diagram extraction, Wasserstein cost, backward).

Run:
  python bench_diff_topology_parallel.py --n-points 3000
  python bench_diff_topology_parallel.py --n-points 10000 --batch-size 4
  python bench_diff_topology_parallel.py --filtration weak-alpha --breakdown
  python bench_diff_topology_parallel.py --device cuda     # GPU forward, threads still run on host
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

# Pin internal threading. Per-sample work is already small at n<=10000 (alpha
# build, boundary matrix, reduction), and the parallel ThreadPoolExecutor path
# wants each worker single-threaded so torch's intra-op pool and Oineus's
# boundary-matrix construction don't over-subscribe the cores.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already initialized

import oineus as oin
import oineus.diff as oin_diff
from oineus import _alpha_shapes_filtration
from oineus.diff.diff_filtration import DiffFiltration


def make_point_cloud(n_points: int, dim: int, seed: int) -> np.ndarray:
    """A noisy circle in 'dim' dimensions: produces interesting H1."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    radius = 1.0
    coords = np.zeros((n_points, dim))
    coords[:, 0] = radius * np.cos(angles)
    coords[:, 1] = radius * np.sin(angles)
    coords += rng.normal(0.0, 0.05, size=coords.shape)
    return coords


def make_batch(batch_size: int, n_points: int, dim: int, device: torch.device,
               dtype: torch.dtype, base_seed: int = 0) -> list[torch.Tensor]:
    """One requires_grad tensor per sample. Cloned per run so .grad starts fresh."""
    return [
        torch.tensor(
            make_point_cloud(n_points, dim, seed=base_seed + i),
            dtype=dtype, device=device, requires_grad=True,
        )
        for i in range(batch_size)
    ]


def make_target_diagram(points: torch.Tensor, hom_dim: int) -> torch.Tensor:
    pts_np = points.detach().cpu().numpy()
    dgms = oin.compute_diagrams_alpha(pts_np, include_inf_points=False)
    if hom_dim >= len(dgms):
        return torch.zeros((0, 2), dtype=points.dtype, device=points.device)
    arr = np.asarray(dgms.in_dimension(hom_dim, as_numpy=True), dtype=np.float64)
    if arr.size == 0:
        return torch.zeros((0, 2), dtype=points.dtype, device=points.device)
    return torch.as_tensor(arr.reshape((-1, 2)), dtype=points.dtype, device=points.device)


def filtration_builder(kind: str):
    if kind == "cech-delaunay":
        return oin_diff.cech_delaunay_filtration
    if kind == "weak-alpha":
        return oin_diff.weak_alpha_filtration
    raise ValueError(f"unknown filtration kind: {kind}")


def per_sample_loss(points: torch.Tensor, target_dgm: torch.Tensor,
                    hom_dim: int, wasserstein_q: float, kind: str) -> torch.Tensor:
    build = filtration_builder(kind)
    fil = build(points)
    dgms = oin_diff.persistence_diagram(fil, include_inf_points=False, n_threads=1)
    if hom_dim not in dgms.keys():
        return torch.zeros((), dtype=points.dtype, device=points.device)
    pred_dgm = dgms[hom_dim]
    if len(pred_dgm) == 0 or len(target_dgm) == 0:
        return torch.zeros((), dtype=points.dtype, device=points.device)
    return oin_diff.wasserstein_cost(
        pred_dgm,
        target_dgm,
        wasserstein_q=wasserstein_q,
        ignore_inf_points=True,
    )


def run_serial(batch, targets, hom_dim, wasserstein_q, kind):
    losses = [per_sample_loss(p, t, hom_dim, wasserstein_q, kind) for p, t in zip(batch, targets)]
    total = torch.stack(losses).sum()
    total.backward()
    return [float(l.detach()) for l in losses], [p.grad.detach().clone() for p in batch]


def run_parallel(batch, targets, hom_dim, wasserstein_q, kind, n_workers):
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(per_sample_loss, p, t, hom_dim, wasserstein_q, kind)
            for p, t in zip(batch, targets)
        ]
        losses = [f.result() for f in futures]
    total = torch.stack(losses).sum()
    total.backward()
    return [float(l.detach()) for l in losses], [p.grad.detach().clone() for p in batch]


def _values_cech_delaunay(points, alpha_fil):
    n0 = alpha_fil.size_in_dimension(0)
    values = [torch.zeros(n0, requires_grad=True, device=points.device, dtype=points.dtype)]
    for dim in range(1, alpha_fil.max_dim() + 1):
        if dim == 1:
            edges = torch.LongTensor(alpha_fil.get_edges().astype(np.uint64))
            sqd = torch.sum((points[edges[:, 0]] - points[edges[:, 1]]) ** 2, dim=1)
            values.append(0.25 * sqd)
        elif dim == 2:
            tri = torch.LongTensor(alpha_fil.get_triangles().astype(np.uint64))
            _, r2 = oin_diff.triangle_meb(points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]], 0.0)
            values.append(r2)
        elif dim == 3:
            tet = torch.LongTensor(alpha_fil.get_tetrahedra().astype(np.uint64))
            _, r2 = oin_diff.tetrahedron_meb(points[tet[:, 0]], points[tet[:, 1]],
                                             points[tet[:, 2]], points[tet[:, 3]], 0.0)
            values.append(r2)
        else:
            raise RuntimeError(f"unsupported dim {dim}")
    return values


def _values_weak_alpha(points, alpha_fil):
    n0 = alpha_fil.size_in_dimension(0)
    values = [torch.zeros(n0, requires_grad=True, device=points.device, dtype=points.dtype)]
    for dim in range(1, alpha_fil.max_dim() + 1):
        if dim == 1:
            edges = torch.LongTensor(alpha_fil.get_edges().astype(np.uint64))
            values.append(torch.sum((points[edges[:, 0]] - points[edges[:, 1]]) ** 2, dim=1))
        elif dim == 2:
            tri = torch.LongTensor(alpha_fil.get_triangles().astype(np.uint64))
            p0, p1, p2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
            d01 = torch.sum((p0 - p1) ** 2, dim=1)
            d02 = torch.sum((p0 - p2) ** 2, dim=1)
            d12 = torch.sum((p1 - p2) ** 2, dim=1)
            values.append(torch.amax(torch.stack([d01, d02, d12], dim=0), dim=0))
        elif dim == 3:
            tet = torch.LongTensor(alpha_fil.get_tetrahedra().astype(np.uint64))
            p0, p1, p2, p3 = points[tet[:, 0]], points[tet[:, 1]], points[tet[:, 2]], points[tet[:, 3]]
            ds = [
                torch.sum((p0 - p1) ** 2, dim=1),
                torch.sum((p0 - p2) ** 2, dim=1),
                torch.sum((p0 - p3) ** 2, dim=1),
                torch.sum((p1 - p2) ** 2, dim=1),
                torch.sum((p1 - p3) ** 2, dim=1),
                torch.sum((p2 - p3) ** 2, dim=1),
            ]
            values.append(torch.amax(torch.stack(ds, dim=0), dim=0))
        else:
            raise RuntimeError(f"unsupported dim {dim}")
    return values


def run_breakdown(batch, targets, hom_dim, wasserstein_q, kind, sync):
    """Single-threaded forward+backward with per-stage timings (batch-summed)."""
    stages = {k: 0.0 for k in
              ("alpha", "values", "assembly", "reduction", "extract", "wasserstein", "backward")}

    losses = []
    for points, target_dgm in zip(batch, targets):
        sync()
        t0 = time.perf_counter()
        points_np = points.detach().cpu().numpy()
        alpha_fil = _alpha_shapes_filtration(points_np)
        sync()
        stages["alpha"] += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        if kind == "cech-delaunay":
            values_in_dim = _values_cech_delaunay(points, alpha_fil)
        else:
            values_in_dim = _values_weak_alpha(points, alpha_fil)
        sync()
        stages["values"] += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        cd_vals = torch.cat(values_in_dim)
        cd_vals_list = [float(x) for x in cd_vals.clone().detach().cpu()]
        alpha_fil.set_values(cd_vals_list)
        sorted_vals = torch.cat([torch.sort(v)[0] for v in values_in_dim])
        fil = DiffFiltration(alpha_fil, sorted_vals)
        sync()
        stages["assembly"] += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        dgms = oin_diff.persistence_diagram(fil, include_inf_points=False, n_threads=1)
        sync()
        stages["reduction"] += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        if hom_dim in dgms.keys():
            pred_dgm = dgms[hom_dim]
        else:
            pred_dgm = torch.zeros((0, 2), dtype=points.dtype, device=points.device)
        sync()
        stages["extract"] += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        if len(pred_dgm) > 0 and len(target_dgm) > 0:
            loss = oin_diff.wasserstein_cost(
                pred_dgm, target_dgm,
                wasserstein_q=wasserstein_q,
                ignore_inf_points=True,
            )
        else:
            loss = torch.zeros((), dtype=points.dtype, device=points.device)
        sync()
        stages["wasserstein"] += time.perf_counter() - t0
        losses.append(loss)

    total = torch.stack(losses).sum()
    sync()
    t0 = time.perf_counter()
    total.backward()
    sync()
    stages["backward"] += time.perf_counter() - t0

    return stages, [float(l.detach()) for l in losses]


def time_it(fn, *args, sync_device=None, **kwargs):
    if sync_device is not None and sync_device.type == "cuda":
        torch.cuda.synchronize(sync_device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if sync_device is not None and sync_device.type == "cuda":
        torch.cuda.synchronize(sync_device)
    return time.perf_counter() - t0, out


def compare_losses(name_a, losses_a, name_b, losses_b, atol=1e-9, rtol=1e-9):
    arr_a = np.asarray(losses_a)
    arr_b = np.asarray(losses_b)
    diff = np.abs(arr_a - arr_b)
    max_abs = float(diff.max()) if diff.size else 0.0
    print(f"  per-sample loss max |{name_a} - {name_b}|: {max_abs:.3e}")
    return np.allclose(arr_a, arr_b, atol=atol, rtol=rtol)


def compare_grads(grads_a, grads_b, atol=1e-9, rtol=1e-9):
    max_abs = 0.0
    for g_a, g_b in zip(grads_a, grads_b):
        d = (g_a - g_b).abs().max().item() if g_a.numel() else 0.0
        if d > max_abs:
            max_abs = d
    print(f"  gradient max |serial - parallel|: {max_abs:.3e}")
    return all(
        torch.allclose(g_a, g_b, atol=atol, rtol=rtol)
        for g_a, g_b in zip(grads_a, grads_b)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=3000,
                    help="Points per cloud (default 3000; try 10000).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Number of point clouds in the batch.")
    ap.add_argument("--dim", type=int, default=3, choices=(2, 3),
                    help="Ambient dimension (alpha-shapes only support 2 and 3).")
    ap.add_argument("--hom-dim", type=int, default=1,
                    help="Homology dimension for the loss.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Threads for the parallel run. Default: batch-size.")
    ap.add_argument("--device", default="cpu",
                    help="Torch device for input tensors. e.g. 'cpu' or 'cuda'.")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--wasserstein-q", type=float, default=1.0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat each timing run this many times and report best.")
    ap.add_argument("--check-only", action="store_true",
                    help="Skip timing, only verify correctness.")
    ap.add_argument("--filtration", default="cech-delaunay",
                    choices=("cech-delaunay", "weak-alpha"),
                    help="Differentiable filtration to build per sample.")
    ap.add_argument("--breakdown", action="store_true",
                    help="Single-threaded run with per-stage timings (batch-summed).")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    workers = args.workers if args.workers is not None else args.batch_size

    print(f"Config: filtration={args.filtration} n_points={args.n_points} batch={args.batch_size} "
          f"dim={args.dim} hom_dim={args.hom_dim} device={device} dtype={dtype} workers={workers}")

    # Reference targets are computed once on a separate (non-differentiable) batch.
    target_batch = make_batch(args.batch_size, args.n_points, args.dim, device, dtype, base_seed=1000)
    targets = [make_target_diagram(p, args.hom_dim) for p in target_batch]
    print(f"Target diagrams sizes: {[len(t) for t in targets]}")

    # Helper for sync inside breakdown stages.
    def sync_breakdown():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if args.breakdown:
        print("\nPer-stage breakdown (single thread, batch-summed)")
        b = make_batch(args.batch_size, args.n_points, args.dim, device, dtype)
        stages, _ = run_breakdown(b, targets, args.hom_dim, args.wasserstein_q,
                                  args.filtration, sync_breakdown)
        total = sum(stages.values())
        for name, secs in stages.items():
            pct = 100.0 * secs / total if total > 0 else 0.0
            print(f"  {name:11s}: {secs:7.3f} s  ({pct:5.1f}%)")
        print(f"  {'total':11s}: {total:7.3f} s")
        return

    # Correctness check: same inputs, both paths.
    batch_serial = make_batch(args.batch_size, args.n_points, args.dim, device, dtype)
    batch_parallel = make_batch(args.batch_size, args.n_points, args.dim, device, dtype)

    print("\nCorrectness check")
    losses_s, grads_s = run_serial(batch_serial, targets, args.hom_dim, args.wasserstein_q,
                                   args.filtration)
    losses_p, grads_p = run_parallel(batch_parallel, targets, args.hom_dim, args.wasserstein_q,
                                     args.filtration, workers)
    losses_match = compare_losses("serial", losses_s, "parallel", losses_p)
    grads_match = compare_grads(grads_s, grads_p)
    print(f"  losses match: {losses_match}; grads match: {grads_match}")

    if args.check_only:
        return

    # Timing.
    print("\nTiming (forward + backward, best of {})".format(args.repeat))
    sync = device if device.type == "cuda" else None

    serial_best = float("inf")
    for _ in range(args.repeat):
        b = make_batch(args.batch_size, args.n_points, args.dim, device, dtype)
        dt, _ = time_it(run_serial, b, targets, args.hom_dim, args.wasserstein_q,
                        args.filtration, sync_device=sync)
        serial_best = min(serial_best, dt)

    parallel_best = float("inf")
    for _ in range(args.repeat):
        b = make_batch(args.batch_size, args.n_points, args.dim, device, dtype)
        dt, _ = time_it(run_parallel, b, targets, args.hom_dim, args.wasserstein_q,
                        args.filtration, workers, sync_device=sync)
        parallel_best = min(parallel_best, dt)

    print(f"  serial:   {serial_best:.3f} s")
    print(f"  parallel: {parallel_best:.3f} s  (workers={workers})")
    if parallel_best > 0:
        print(f"  speedup:  {serial_best / parallel_best:.2f}x")


if __name__ == "__main__":
    main()
