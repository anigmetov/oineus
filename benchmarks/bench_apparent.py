"""Benchmark the apparent-pairs (decorated-matrix) optimization on cubical
filtrations: build time, reduction time, and peak RSS, apparent_opt ON vs OFF,
for cohomology (dualize=True) and homology (dualize=False).

The win lives in the BUILD phase (params.timings.prepare): with apparent_opt the
cohomology builder emits each non-apparent column directly from cube.coboundary()
and never forms the full antitransposed matrix, so build time and peak working
memory drop with the apparent fraction (typically ~90%+ on smooth volumes). The
reduction itself is already near-free for apparent pairs (clearing), so its time
is ~unchanged.

Usage:
    python bench_apparent.py [--n N] [--reps R]          # parent: tabulate all configs
    python bench_apparent.py --worker DUALIZE APPARENT N REPS   # one config (subprocess)

Each config runs in its own subprocess so peak RSS (ru_maxrss) is attributable.
"""
import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np
import oineus as oin


def smooth_field(n, seed=0):
    # smooth scalar field on an n x n x n grid: sum of a few low-frequency
    # sinusoids (high apparent fraction) plus tiny noise to break exact ties
    lin = np.linspace(0.0, 2.0 * np.pi, n)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    f = (np.sin(x) + np.sin(1.3 * y) + np.sin(0.7 * z)
         + 0.5 * np.cos(0.5 * x + 0.9 * y)
         + 0.3 * np.sin(0.4 * z - 0.6 * x))
    rng = np.random.default_rng(seed)
    f = f + 1e-6 * rng.standard_normal(f.shape)
    return np.ascontiguousarray(f, dtype=np.float64)


def maxrss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def run_one(dualize, apparent, n, reps, n_threads=8):
    a = smooth_field(n)
    fil = oin.cube_filtration(a, n_threads=n_threads, values_on="vertices")
    N = fil.size()

    prepare_best = float("inf")
    reduce_best = float("inf")
    wall_best = float("inf")
    for _ in range(reps + 1):  # one warmup
        p = oin.ReductionParams()
        p.n_threads = n_threads
        p.compute_v = True
        p.apparent_opt = apparent
        t0 = time.perf_counter()
        dcmp = oin.reduce(fil, p, dualize)
        wall = time.perf_counter() - t0
        prepare_best = min(prepare_best, p.timings.prepare)
        reduce_best = min(reduce_best, p.timings.reduce)
        wall_best = min(wall_best, wall)
        del dcmp
    return dict(N=N, prepare=prepare_best, reduce=reduce_best, wall=wall_best,
                maxrss=maxrss_mb())


def worker(dualize, apparent, n, reps):
    res = run_one(dualize, apparent, n, reps)
    print(json.dumps(res))


def parent(n, reps):
    print(f"apparent-pairs benchmark: smooth {n}x{n}x{n} volume, "
          f"compute_v=True, n_threads=8, best of {reps}\n")
    header = f"{'mode':<12}{'apparent':<10}{'cells':>12}{'prepare(s)':>13}{'reduce(s)':>12}{'wall(s)':>10}{'maxRSS(MB)':>13}"
    print(header)
    print("-" * len(header))
    summary = {}
    for dualize in (True, False):
        mode = "cohomology" if dualize else "homology"
        row = {}
        for apparent in (False, True):
            cmd = [sys.executable, __file__, "--worker",
                   str(int(dualize)), str(int(apparent)), str(n), str(reps)]
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print(f"  worker failed ({mode}, apparent={apparent}):\n{out.stderr}")
                continue
            r = json.loads(out.stdout.strip().splitlines()[-1])
            row[apparent] = r
            print(f"{mode:<12}{str(apparent):<10}{r['N']:>12,}{r['prepare']:>13.3f}"
                  f"{r['reduce']:>12.3f}{r['wall']:>10.3f}{r['maxrss']:>13.1f}")
        summary[mode] = row
        if False in row and True in row:
            off, on = row[False], row[True]
            sp = off["prepare"] / on["prepare"] if on["prepare"] > 0 else float("nan")
            dm = off["maxrss"] - on["maxrss"]
            print(f"  -> {mode}: prepare speedup {sp:.2f}x, "
                  f"peak RSS {off['maxrss']:.0f} -> {on['maxrss']:.0f} MB ({dm:+.0f} MB)\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--worker", nargs=4, metavar=("DUALIZE", "APPARENT", "N", "REPS"))
    args = ap.parse_args()
    if args.worker:
        d, a, n, reps = (int(x) for x in args.worker)
        worker(bool(d), bool(a), n, reps)
    else:
        parent(args.n, args.reps)


if __name__ == "__main__":
    main()
