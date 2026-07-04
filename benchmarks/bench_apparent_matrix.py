"""Measurement matrix for the apparent-pairs (decorated-matrix, null-marked
materialization) optimization, extending bench_apparent.py to the full grid:

    kind      x  cubical / Freudenthal (both slim by default)
    field     x  smooth volume (high apparent fraction) / uniform random (control)
    size      x  96^3, 128^3 (smooth); 96^3 (random)
    side      x  homology (dualize=False) / cohomology (dualize=True)
    threads   x  1, 8
    flag      x  on (apparent, compute_v=True) / off (compute_v=True)
                 / off_nov (compute_v=False -- cheapest R-only reference)
                 / on_nov (apparent, compute_v=False -- R-only apparent)

Every (config, rep) runs in a FRESH SUBPROCESS so peak RSS (ru_maxrss; BYTES on
darwin, KB on Linux) is attributable per run; the parent aggregates medians.
`wall` times the oin.reduce call only; `rss_fil` is the child's high-water mark
right after the filtration build, so rss_peak - rss_fil bounds the reduction's
own contribution to the peak. All runs are cold (no in-process warmup).

Note: use_apparent_pairs only activates on the fused parallel path
(n_threads > 1, compute_v either way); at n_threads=1 the flag is inert
(classic serial fallback, n_apparent_pairs() == 0). The 1-thread rows measure
exactly that.

Usage:
    python bench_apparent_matrix.py [--reps 3] [--csv PATH] [--quick]
        [--flags on_nov,...] [--threads 8,...] [--fields smooth,...]
    python bench_apparent_matrix.py --worker KIND FIELD N DUALIZE THREADS FLAG
"""
import argparse
import csv
import json
import os
import resource
import statistics
import subprocess
import sys
import time

import numpy as np

FLAGS = ("on", "off", "off_nov", "on_nov")


def smooth_field(n, seed=0):
    # same generator as bench_apparent.py: sum of low-frequency sinusoids
    # (high apparent fraction) plus tiny noise to break exact ties
    lin = np.linspace(0.0, 2.0 * np.pi, n)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    f = (np.sin(x) + np.sin(1.3 * y) + np.sin(0.7 * z)
         + 0.5 * np.cos(0.5 * x + 0.9 * y)
         + 0.3 * np.sin(0.4 * z - 0.6 * x))
    rng = np.random.default_rng(seed)
    f = f + 1e-6 * rng.standard_normal(f.shape)
    return np.ascontiguousarray(f, dtype=np.float64)


def random_field(n, seed=1):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((n, n, n)), dtype=np.float64)


def maxrss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def worker(kind, field, n, dualize, n_threads, flag):
    # run from a directory containing the oineus package (e.g. build/tests);
    # script-file invocation does not put the cwd on sys.path
    sys.path.insert(0, os.getcwd())
    import oineus as oin

    a = smooth_field(n) if field == "smooth" else random_field(n)
    t0 = time.perf_counter()
    if kind == "cube":
        fil = oin.cube_filtration(a, n_threads=8, values_on="vertices")
    else:
        fil = oin.freudenthal_filtration(a, n_threads=8)
    fil_build = time.perf_counter() - t0
    rss_fil = maxrss_mb()

    p = oin.ReductionParams()
    p.n_threads = n_threads
    p.compute_v = flag in ("on", "off")
    p.use_apparent_pairs = flag in ("on", "on_nov")
    t0 = time.perf_counter()
    dcmp = oin.reduce(fil, p, dualize)
    wall = time.perf_counter() - t0

    t = dcmp.timings
    print(json.dumps(dict(
        n_cells=fil.size(), n_apparent=dcmp.n_apparent_pairs(),
        fil_build=fil_build, wall=wall,
        prepare=t.prepare, reduce=t.reduce, bauer=t.bauer,
        copy_back=t.copy_back, copy_pivots=t.copy_pivots,
        timings_total=t.reduction_total,
        rss_fil_mb=rss_fil, rss_peak_mb=maxrss_mb())))


def run_config(kind, field, n, dualize, n_threads, flag, reps):
    runs = []
    for _ in range(reps):
        cmd = [sys.executable, __file__, "--worker",
               kind, field, str(n), str(int(dualize)), str(n_threads), flag]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"worker failed: {' '.join(cmd[3:])}\n{out.stderr}", file=sys.stderr)
            return None
        runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
    med = lambda k: statistics.median(r[k] for r in runs)
    r0 = runs[0]
    return dict(
        kind=kind, field=field, n=n,
        side="coh" if dualize else "hom", threads=n_threads, flag=flag,
        n_cells=r0["n_cells"], n_apparent=r0["n_apparent"],
        app_fraction=r0["n_apparent"] / r0["n_cells"],
        wall_med=med("wall"), wall_min=min(r["wall"] for r in runs),
        prepare_med=med("prepare"), reduce_med=med("reduce"),
        bauer_med=med("bauer"), copy_back_med=med("copy_back"),
        fil_build_med=med("fil_build"),
        rss_fil_med_mb=med("rss_fil_mb"), rss_peak_med_mb=med("rss_peak_mb"),
        reps=reps)


CSV_FIELDS = ["kind", "field", "n", "side", "threads", "flag",
              "n_cells", "n_apparent", "app_fraction",
              "wall_med", "wall_min", "prepare_med", "reduce_med",
              "bauer_med", "copy_back_med", "fil_build_med",
              "rss_fil_med_mb", "rss_peak_med_mb", "reps"]


def print_block(rows):
    r0 = rows[0]
    print(f"\n{r0['kind']} {r0['field']} {r0['n']}^3: "
          f"{r0['n_cells']:,} cells")
    header = (f"{'side':<5}{'thr':>4} {'flag':<8}{'wall(s)':>9}{'min':>8}"
              f"{'prepare':>9}{'reduce':>9}{'bauer':>8}{'copyback':>10}"
              f"{'peakRSS(MB)':>13}{'n_app':>12}{'frac':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['side']:<5}{r['threads']:>4} {r['flag']:<8}"
              f"{r['wall_med']:>9.3f}{r['wall_min']:>8.3f}"
              f"{r['prepare_med']:>9.3f}{r['reduce_med']:>9.3f}"
              f"{r['bauer_med']:>8.3f}{r['copy_back_med']:>10.3f}"
              f"{r['rss_peak_med_mb']:>13.0f}"
              f"{r['n_apparent']:>12,}{r['app_fraction']:>7.3f}")


def parent(reps, csv_path, quick, flags=None, threads=None, fields=None):
    # flags / threads / fields: optional axis subsets for partial re-runs
    flags_axis = tuple(flags) if flags else FLAGS
    if quick:
        combos = [("cube", "smooth", 48), ("freud", "smooth", 48)]
        threads_axis = (8,)
        reps = 1
    else:
        combos = [(kind, field, n)
                  for kind in ("cube", "freud")
                  for field, n in (("smooth", 96), ("smooth", 128), ("random", 96))
                  if fields is None or field in fields]
        threads_axis = tuple(threads) if threads else (1, 8)

    print(f"apparent-pairs matrix: medians of {reps} fresh-subprocess reps, "
          f"sequential; wall = oin.reduce only")
    all_rows = []
    for kind, field, n in combos:
        block = []
        for dualize in (False, True):
            for n_threads in threads_axis:
                for flag in flags_axis:
                    row = run_config(kind, field, n, dualize, n_threads, flag, reps)
                    if row is None:
                        continue
                    block.append(row)
                    print(f"  done {kind} {field} {n} "
                          f"{row['side']} t{n_threads} {flag}: "
                          f"wall {row['wall_med']:.3f}s rss {row['rss_peak_med_mb']:.0f}MB",
                          file=sys.stderr)
        all_rows.extend(block)
        if block:
            print_block(block)

    if csv_path and all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {len(all_rows)} rows to {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--csv", default="bench_apparent_matrix.csv")
    ap.add_argument("--quick", action="store_true",
                    help="48^3 smooth only, 8 threads, 1 rep (smoke test)")
    ap.add_argument("--flags", default=None,
                    help="comma-separated flag subset (default: all)")
    ap.add_argument("--threads", default=None,
                    help="comma-separated thread axis (default: 1,8)")
    ap.add_argument("--fields", default=None,
                    help="comma-separated field subset (default: smooth,random)")
    ap.add_argument("--worker", nargs=6,
                    metavar=("KIND", "FIELD", "N", "DUALIZE", "THREADS", "FLAG"))
    args = ap.parse_args()
    if args.worker:
        kind, field, n, dualize, n_threads, flag = args.worker
        worker(kind, field, int(n), bool(int(dualize)), int(n_threads), flag)
    else:
        parent(args.reps, args.csv, args.quick,
               flags=args.flags.split(",") if args.flags else None,
               threads=[int(t) for t in args.threads.split(",")] if args.threads else None,
               fields=args.fields.split(",") if args.fields else None)


if __name__ == "__main__":
    main()
