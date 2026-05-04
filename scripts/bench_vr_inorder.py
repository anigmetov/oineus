#!/usr/bin/env python3
"""
Benchmark the in-order (VRE) Vietoris-Rips construction in oineus, with the
brute-force C++ reference (``_oineus._get_vr_filtration_naive``, max_dim <= 3)
as a correctness yardstick on small inputs and as a slow-but-honest baseline
on larger ones.

Each (n, point_dim, max_dim, threshold) configuration is pre-flighted with a
combinatorial upper bound to avoid blowing memory: a full VR worst-case is
sum_{j=0..max_dim} C(n, j+1) simplices, each ~120 B in oineus. We skip any
config whose upper bound exceeds MEMORY_CAP_MB (default 1 GB).
"""

import math
import time

import numpy as np

import oineus as oin
from oineus import _oineus

# Approximate bytes per CellWithValue<Simplex<Int>, Real> for d <= 4.
BYTES_PER_SIMPLEX = 120
MEMORY_CAP_MB = 1024  # hard skip if upper bound exceeds this


def upper_bound_simplices(n: int, max_dim: int) -> int:
    """Worst-case simplex count for full VR (no diameter cutoff)."""
    return sum(math.comb(n, j + 1) for j in range(max_dim + 1))


def time_vre(pts, max_dim, max_diameter):
    t0 = time.perf_counter()
    fil = oin.vr_filtration(pts, max_dim=max_dim, max_diameter=max_diameter)
    return time.perf_counter() - t0, fil.size()


def time_naive(pts, max_dim, max_diameter):
    t0 = time.perf_counter()
    fil = _oineus._get_vr_filtration_naive(
        pts, max_dim=max_dim, max_diameter=max_diameter, n_threads=1)
    return time.perf_counter() - t0, fil.size()


def main():
    rng = np.random.default_rng(2024)

    # (n_points, point_dim, max_dim, threshold)
    # Threshold tuned so the curated worst-case stays within MEMORY_CAP_MB.
    # max_dim is capped at 3 so the brute-force naive baseline runs.
    configs = [
        (50,  2, 2, 10.0),
        (50,  3, 3, 10.0),
        (80,  2, 2, 10.0),
        (100, 2, 2, 0.4),
        (100, 3, 3, 0.3),
        (200, 2, 2, 0.2),
    ]

    print(f"{'n':>5} {'pdim':>5} {'maxd':>5} {'thr':>7} "
          f"{'algo':>10} {'size':>9} {'time_s':>9}")
    print("-" * 60)

    rows = []  # (n, pdim, max_dim, thr, algo, size, time)
    for (n, pdim, max_dim, thr) in configs:
        ub = upper_bound_simplices(n, max_dim)
        ub_mb = ub * BYTES_PER_SIMPLEX / 1e6
        if ub_mb > MEMORY_CAP_MB:
            print(f"# skip n={n}, max_dim={max_dim}: upper bound "
                  f"{ub_mb:.0f} MB > cap {MEMORY_CAP_MB} MB")
            continue
        pts = rng.random((n, pdim)).astype(np.float64)
        t_v, sz_v = time_vre(pts, max_dim, thr)
        t_n, sz_n = time_naive(pts, max_dim, thr)
        print(f"{n:>5} {pdim:>5} {max_dim:>5} {thr:>7.2f} "
              f"{'inorder':>10} {sz_v:>9} {t_v:>9.3f}")
        print(f"{n:>5} {pdim:>5} {max_dim:>5} {thr:>7.2f} "
              f"{'naive':>10} {sz_n:>9} {t_n:>9.3f}")
        rows.append((n, pdim, max_dim, thr, sz_v, sz_n, t_v, t_n))
        if sz_v != sz_n:
            print(f"  !! mismatch: VRE={sz_v} naive={sz_n}")

    print()
    print("Speedup summary (time_naive / time_VRE; >1 means VRE faster):")
    print(f"{'n':>5} {'pdim':>5} {'maxd':>5} {'thr':>7} "
          f"{'size':>9} {'naive_s':>9} {'VRE_s':>9} {'ratio':>7}")
    print("-" * 65)
    for (n, pdim, max_dim, thr, sz_v, sz_n, t_v, t_n) in rows:
        ratio = t_n / t_v if t_v > 0 else float("inf")
        print(f"{n:>5} {pdim:>5} {max_dim:>5} {thr:>7.2f} "
              f"{sz_v:>9} {t_n:>9.3f} {t_v:>9.3f} {ratio:>7.2f}")


if __name__ == "__main__":
    main()
