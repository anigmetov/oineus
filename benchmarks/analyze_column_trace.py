"""Offline replay of a fused R-only column trace (OINEUS_COLUMN_TRACE).

Reads the binary dump written by ColumnTraceCtx::dump (see
include/oineus/column_trace.h) and computes the high-water mark of live
working-column bytes under four policies:

  eager       -- every column alive [0, end], including apparent-null slots
                 at their would-be (recorded) size: approximates the plain
                 off_nov working matrix.
  null_marked -- apparent + cleared columns cost 0, everything else alive
                 [0, end]: the spec model of the current on_nov design.
  as_impl     -- apparent columns cost 0, everything else alive from the
                 start; frees (clearing / reduced-to-zero) only shrink the
                 curve, so the high-water mark is the full non-apparent build:
                 what the current implementation actually holds at its peak.
  ideal       -- true column oracle: apparent columns cost 0, every other
                 column alive [t_first_touch, t_last_touch] only; columns
                 never touched cost 0.

The answer to the Direction A question is the ratio ideal / null_marked.

Byte model (per column): obj_bytes + (nnz > inline_cap ? nnz * int_bytes : 0)
with defaults obj_bytes=56 (sizeof OinColumn<long>), inline_cap=4
(OINEUS_COL_INLINE_CAP), int_bytes=8. The model prices columns at their BUILD
size throughout: fill-in growth from left-additions during reduction is NOT
tracked, and the atomic pointer-slot array itself (n_cols * 8 bytes, identical
under every policy) is excluded. Event order is approximate under concurrency
(relaxed global tick counter) -- fine for a high-water replay.

Usage:
    python analyze_column_trace.py [--markdown] LABEL=TRACE_FILE [...]
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = b"OINCTRC1"
REC_DTYPE = np.dtype([
    ("nnz", "<u4"), ("flags", "<u4"), ("n_touches", "<u4"),
    ("t_build", "<u4"), ("t_first", "<u4"), ("t_last", "<u4"), ("t_free", "<u4"),
])
FLAG_APPARENT = 1
FLAG_CLEARED = 2
FLAG_ZEROED = 4


def read_trace(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        n_cols, final_clock = struct.unpack("<QQ", f.read(16))
        recs = np.fromfile(f, dtype=REC_DTYPE, count=n_cols)
    if len(recs) != n_cols:
        raise ValueError(f"{path}: truncated ({len(recs)} of {n_cols} records)")
    return recs, final_clock


def interval_hwm(starts, ends, weights):
    """High-water mark of sum of weights over intervals [start, end] (inclusive).

    Ties can only be a column's own start==end (ticks are otherwise unique);
    stable sort with the +delta block first counts such columns as alive for
    their single tick.
    """
    times = np.concatenate([starts, ends])
    deltas = np.concatenate([weights, -weights])
    order = np.argsort(times, kind="stable")
    live = np.cumsum(deltas[order])
    peak_pos = int(np.argmax(live))
    return float(live[peak_pos]), float(times[order][peak_pos])


def analyze(path, obj_bytes, inline_cap, int_bytes):
    recs, final_clock = read_trace(path)
    nnz = recs["nnz"].astype(np.int64)
    flags = recs["flags"]
    apparent = (flags & FLAG_APPARENT) != 0
    cleared = (flags & FLAG_CLEARED) != 0
    zeroed = (flags & FLAG_ZEROED) != 0
    touched = recs["t_first"] > 0

    col_bytes = (obj_bytes + np.where(nnz > inline_cap, nnz * int_bytes, 0)).astype(np.float64)

    res = {
        "path": path,
        "n_cols": len(recs),
        "final_clock": final_clock,
        "n_apparent": int(apparent.sum()),
        "n_cleared": int(cleared.sum()),
        "n_zeroed": int(zeroed.sum()),
        "n_untouched_nonapparent": int((~apparent & ~touched).sum()),
        "n_touches_total": int(recs["n_touches"].sum()),
        "max_touches_one_col": int(recs["n_touches"].max()),
    }

    res["eager"] = float(col_bytes.sum())
    res["null_marked"] = float(col_bytes[~apparent & ~cleared].sum())
    res["as_impl"] = float(col_bytes[~apparent].sum())

    live = ~apparent & touched
    starts = recs["t_first"][live].astype(np.int64)
    ends = recs["t_last"][live].astype(np.int64)
    if len(starts):
        res["ideal"], peak_t = interval_hwm(starts, ends, col_bytes[live])
    else:
        res["ideal"], peak_t = 0.0, 0.0
    res["ideal_peak_frac"] = peak_t / final_clock if final_clock else 0.0

    # bytes-weighted mean relative alive-interval length under the oracle
    span = (ends - starts).astype(np.float64)
    w = col_bytes[live]
    res["mean_rel_interval"] = float((span * w).sum() / (w.sum() * final_clock)) if final_clock else 0.0

    res["ratio_ideal_null"] = res["ideal"] / res["null_marked"] if res["null_marked"] else 0.0
    res["ratio_null_eager"] = res["null_marked"] / res["eager"] if res["eager"] else 0.0
    res["ratio_ideal_eager"] = res["ideal"] / res["eager"] if res["eager"] else 0.0
    return res


def mb(x):
    return x / (1024.0 * 1024.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="LABEL=PATH or PATH")
    ap.add_argument("--obj-bytes", type=int, default=56, help="per-column object overhead (sizeof OinColumn)")
    ap.add_argument("--inline-cap", type=int, default=4, help="SBO inline capacity (OINEUS_COL_INLINE_CAP)")
    ap.add_argument("--int-bytes", type=int, default=8, help="sizeof(Int) of a column entry")
    ap.add_argument("--markdown", action="store_true", help="also print a markdown summary table")
    args = ap.parse_args()

    rows = []
    for spec in args.traces:
        label, _, path = spec.rpartition("=")
        if not label:
            label = path
        r = analyze(path, args.obj_bytes, args.inline_cap, args.int_bytes)
        r["label"] = label
        rows.append(r)

        print(f"== {label} ({path})")
        print(f"   n_cols={r['n_cols']:,}  events={r['final_clock']:,}  "
              f"apparent={r['n_apparent']:,}  cleared={r['n_cleared']:,}  "
              f"zeroed={r['n_zeroed']:,}  untouched(non-app)={r['n_untouched_nonapparent']:,}")
        print(f"   touches total={r['n_touches_total']:,}  max/col={r['max_touches_one_col']:,}  "
              f"mean rel interval={r['mean_rel_interval']:.4f}  ideal peak at {r['ideal_peak_frac']:.3f} of run")
        print(f"   eager       = {mb(r['eager']):9.1f} MB")
        print(f"   null_marked = {mb(r['null_marked']):9.1f} MB   ({r['ratio_null_eager']:.3f} of eager)")
        print(f"   as_impl     = {mb(r['as_impl']):9.1f} MB")
        print(f"   ideal       = {mb(r['ideal']):9.1f} MB   ideal/null_marked = {r['ratio_ideal_null']:.3f}  "
              f"ideal/eager = {r['ratio_ideal_eager']:.3f}")
        print()

    if args.markdown:
        print("| config | eager (MB) | null-marked (MB) | as-impl (MB) | ideal oracle (MB) | ideal/null | ideal/eager |")
        print("|--------|-----------:|-----------------:|-------------:|------------------:|-----------:|------------:|")
        for r in rows:
            print(f"| {r['label']} | {mb(r['eager']):.0f} | {mb(r['null_marked']):.0f} | "
                  f"{mb(r['as_impl']):.0f} | {mb(r['ideal']):.0f} | "
                  f"{r['ratio_ideal_null']:.3f} | {r['ratio_ideal_eager']:.3f} |")


if __name__ == "__main__":
    sys.exit(main())
