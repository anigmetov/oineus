#!/usr/bin/env python3
"""
Benchmark VRE (in-order) Vietoris-Rips construction against the existing
Bron-Kerbosch implementation.

Each (n, point_dim, max_dim, threshold) configuration is pre-flighted with a
combinatorial upper bound to avoid blowing memory: a full VR worst-case is
sum_{j=0..max_dim} C(n, j+1) simplices, each ~120 B in oineus. We skip any
config whose upper bound exceeds MEMORY_CAP_MB (default 1 GB).

The first pass is a correctness check (simplex counts must match between
algorithms). Wall-clock timings are then printed in a table.

The threshold values are chosen so that BK (which has a pre-existing bug
in the distance-matrix path -- see plan) is not exercised; we always go
through the points path here.
"""

import math
import time

import numpy as np

import oineus as oin

# Approximate bytes per CellWithValue<Simplex<Int>, Real> for d <= 4.
BYTES_PER_SIMPLEX = 120
MEMORY_CAP_MB = 1024  # hard skip if upper bound exceeds this


def upper_bound_simplices(n: int, max_dim: int) -> int:
    """Worst-case simplex count for full VR (no diameter cutoff)."""
    return sum(math.comb(n, j + 1) for j in range(max_dim + 1))


def time_one(pts, max_dim, max_diameter, algorithm):
    t0 = time.perf_counter()
    fil = oin.vr_filtration(pts, max_dim=max_dim, max_diameter=max_diameter,
                            algorithm=algorithm)
    return time.perf_counter() - t0, fil.size()


def main():
    rng = np.random.default_rng(2024)

    # (n_points, point_dim, max_dim, threshold)
    # Threshold tuned so the curated worst-case stays within MEMORY_CAP_MB.
    # threshold = +inf means "no cutoff" -- supplied via large explicit value
    # so we don't depend on the python-side default.
    configs = [
        (50,  2, 2, 10.0),
        (50,  3, 3, 10.0),
        (80,  2, 2, 10.0),
        (100, 2, 2, 0.4),
        (100, 3, 3, 0.3),
        (200, 2, 2, 0.2),
    ]

    print(f"{'n':>5} {'pdim':>5} {'maxd':>5} {'thr':>7} "
          f"{'algo':>14} {'size':>9} {'time_s':>8}")
    print("-" * 65)

    rows = []  # (n, pdim, max_dim, thr, algo, size, time)
    for (n, pdim, max_dim, thr) in configs:
        ub = upper_bound_simplices(n, max_dim)
        ub_mb = ub * BYTES_PER_SIMPLEX / 1e6
        if ub_mb > MEMORY_CAP_MB:
            print(f"# skip n={n}, max_dim={max_dim}: upper bound "
                  f"{ub_mb:.0f} MB > cap {MEMORY_CAP_MB} MB")
            continue
        pts = rng.random((n, pdim)).astype(np.float64)
        sizes = {}
        times = {}
        for algo in ("bron-kerbosch", "inorder"):
            t, sz = time_one(pts, max_dim, thr, algo)
            sizes[algo] = sz
            times[algo] = t
            print(f"{n:>5} {pdim:>5} {max_dim:>5} {thr:>7.2f} "
                  f"{algo:>14} {sz:>9} {t:>8.3f}")
            rows.append((n, pdim, max_dim, thr, algo, sz, t))
        # correctness sanity check
        if sizes["bron-kerbosch"] != sizes["inorder"]:
            print(f"  !! mismatch: BK={sizes['bron-kerbosch']} "
                  f"VRE={sizes['inorder']} -- check thresholds and "
                  f"distance-matrix bug, this case used the points path so "
                  f"both should agree")

    # Summary table: time_BK / time_VRE (>1 means VRE faster).
    print()
    print("Speedup summary (time_BK / time_VRE; >1 means VRE faster):")
    print(f"{'n':>5} {'pdim':>5} {'maxd':>5} {'thr':>7} "
          f"{'size':>9} {'BK_s':>8} {'VRE_s':>8} {'ratio':>7}")
    print("-" * 65)
    by_key = {}
    for r in rows:
        n, pdim, max_dim, thr, algo, sz, t = r
        by_key.setdefault((n, pdim, max_dim, thr), {})[algo] = (sz, t)
    for (n, pdim, max_dim, thr), d in sorted(by_key.items()):
        if "bron-kerbosch" in d and "inorder" in d:
            sz_bk, t_bk = d["bron-kerbosch"]
            sz_vre, t_vre = d["inorder"]
            ratio = t_bk / t_vre if t_vre > 0 else float("inf")
            print(f"{n:>5} {pdim:>5} {max_dim:>5} {thr:>7.2f} "
                  f"{sz_bk:>9} {t_bk:>8.3f} {t_vre:>8.3f} {ratio:>7.2f}")


if __name__ == "__main__":
    main()
