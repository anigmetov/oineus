"""End-to-end (Python) timing of parallel diagram extraction.

Complements the C++ benchmark (bench_diagram.cpp): this measures the full
Python path -- GIL release, taskflow workers, and the numpy export -- whereas
the C++ binary isolates the extraction itself and also A/Bs taskflow against
raw std::thread. Reduction is done once and excluded from the timing.

Run (from the repo root, against a cmake build):
    PYTHONPATH=build/bindings/python python benchmarks/bench_diagram.py
    PYTHONPATH=build/bindings/python python benchmarks/bench_diagram.py --mode vr --n-points 400
"""
import argparse
import time

import numpy as np
import oineus as oin


def build_grid(side):
    rng = np.random.default_rng(42)
    data = np.ascontiguousarray(rng.random((side, side, side)))
    return oin.freudenthal_filtration(data=data, negate=False, max_dim=3)


def build_vr(n_points, max_dim):
    rng = np.random.default_rng(42)
    pts = np.ascontiguousarray(rng.random((n_points, 3)))
    return oin.vr_filtration(pts, max_dim=max_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grid", "vr"], default="grid")
    ap.add_argument("--grid-side", type=int, default=75)
    ap.add_argument("--n-points", type=int, default=400)
    ap.add_argument("--vr-max-dim", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--reduce-threads", type=int, default=8)
    ap.add_argument("--dualize", action="store_true")
    ap.add_argument("--threads", type=str, default="1,2,4,8")
    args = ap.parse_args()

    threads = [int(t) for t in args.threads.split(",") if t]

    if args.mode == "grid":
        fil = build_grid(args.grid_side)
    else:
        fil = build_vr(args.n_points, args.vr_max_dim)
    print(f"mode={args.mode} dualize={args.dualize} n_cols={fil.size()}")

    params = oin.ReductionParams()
    params.n_threads = args.reduce_threads
    t0 = time.perf_counter()
    dcmp = oin.reduce(fil, params, args.dualize)
    print(f"reduce (fused, {args.reduce_threads} threads): "
          f"{1000 * (time.perf_counter() - t0):.1f} ms [excluded]")

    def timed(nt):
        best = float("inf")
        for _ in range(args.reps):
            t = time.perf_counter()
            dcmp.diagram(fil, include_inf_points=True, n_threads=nt)
            best = min(best, time.perf_counter() - t)
        return best * 1000

    dcmp.diagram(fil, include_inf_points=True, n_threads=1)  # warm up
    base = timed(1)
    print(f"\nserial (n_threads=1): {base:.3f} ms (best of {args.reps})\n")
    print(f"{'threads':<8} {'ms':>12} {'speedup':>10}")
    for nt in threads:
        ms = timed(nt)
        print(f"{nt:<8} {ms:>12.3f} {base / ms if ms else 0:>10.2f}")


if __name__ == "__main__":
    main()
