#!/usr/bin/env python3
"""Benchmark: differentiable alpha filtration vs Cech-Delaunay filtration
on 3D random point clouds (5000 and 10000 points).

Times the full call: combinatorics from CGAL/diode, value tensor build,
and DiffFiltration assembly. Backward pass is benchmarked separately.
"""
import os
import time
import numpy as np
import torch

import oineus.diff as oin_diff


def bench(name, fn, n_warmup=1, n_repeat=3):
    """Time fn() across runs; return (times, total)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def make_points_3d(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)).astype(np.float64)


def run_size(n):
    print(f"\n=== n = {n} points (3D) ===")
    points_np = make_points_3d(n)
    pts_alpha = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)
    pts_cd = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)

    fwd_a_times = bench("alpha forward",
                        lambda: oin_diff.alpha_filtration(pts_alpha))
    fwd_c_times = bench("cech_delaunay forward",
                        lambda: oin_diff.cech_delaunay_filtration(pts_cd))

    print(f"  alpha forward:           min={min(fwd_a_times):.3f}s  "
          f"mean={np.mean(fwd_a_times):.3f}s  (n_repeat={len(fwd_a_times)})")
    print(f"  cech_delaunay forward:   min={min(fwd_c_times):.3f}s  "
          f"mean={np.mean(fwd_c_times):.3f}s  (n_repeat={len(fwd_c_times)})")
    speedup = np.mean(fwd_c_times) / np.mean(fwd_a_times)
    if speedup > 1:
        print(f"  -> alpha is {speedup:.2f}x faster (forward)")
    else:
        print(f"  -> cech_delaunay is {1/speedup:.2f}x faster (forward)")

    # backward pass: build once, then time loss.backward() in isolation.
    def time_backward(filtration_fn, points):
        fil = filtration_fn(points)
        loss = fil.values.pow(2).sum()
        if points.grad is not None:
            points.grad.zero_()
        t0 = time.perf_counter()
        loss.backward()
        return time.perf_counter() - t0

    # Single shot for backward (constructing fresh autograd graph each time
    # would dominate the timing); average over a few independent builds.
    bwd_a = [time_backward(oin_diff.alpha_filtration,
                           torch.tensor(points_np, dtype=torch.float64,
                                        requires_grad=True))
             for _ in range(3)]
    bwd_c = [time_backward(oin_diff.cech_delaunay_filtration,
                           torch.tensor(points_np, dtype=torch.float64,
                                        requires_grad=True))
             for _ in range(3)]
    print(f"  alpha backward (sum^2):   min={min(bwd_a):.3f}s  mean={np.mean(bwd_a):.3f}s")
    print(f"  cd backward    (sum^2):   min={min(bwd_c):.3f}s  mean={np.mean(bwd_c):.3f}s")


def main():
    for n in (5000, 10000):
        run_size(n)


if __name__ == "__main__":
    main()
