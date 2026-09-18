"""Benchmark selective restoration and separate R/V access in fresh processes.

Run with PYTHONPATH pointing to the newly built bindings directory.
Peak RSS includes setup; rss_growth_mib is growth over the setup high-water mark.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


def peak_rss_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def run_case(case):
    import numpy as np
    import oineus as oin

    fil = oin.freudenthal_filtration(
        np.random.default_rng(1701).random((case["side"], case["side"])), n_threads=1)
    params = oin.ReductionParams(n_threads=4, compute_v=True, use_clearing=True)
    if case["storage"] == "working":
        dcmp = oin.reduce(fil, params, case["dualize"])
    else:
        dcmp = oin.Decomposition(fil, case["dualize"])
        dcmp.reduce(params)
    n = fil.size()
    k = min(32, n) if case["request"] == "few" else n // 2
    indices = np.linspace(0, n - 1, k, dtype=int).tolist()
    # Access is measured after restoration, using an otherwise fresh decomposition
    if case["operation"] != "restore":
        dcmp.restore_elz_columns(indices, n_threads=4)
    gc.collect()
    rss_before = peak_rss_mib()
    start = time.perf_counter()
    operation = case["operation"]
    if operation == "restore":
        result = dcmp.restore_elz_columns(indices, n_threads=case["threads"])
    elif operation == "getters":
        result = [(dcmp.r_col(j), dcmp.v_col(j)) for j in indices]
    elif operation == "cached_lists":
        r, v = dcmp.r_data, dcmp.v_data
        result = [(r[j], v[j]) for j in indices]
    else:
        r, v = dcmp.r_as_csc(), dcmp.v_as_csc()
        result = [(r.indices[r.indptr[j]:r.indptr[j + 1]],
                   v.indices[v.indptr[j]:v.indptr[j + 1]]) for j in indices]
    elapsed = time.perf_counter() - start
    rss_after = peak_rss_mib()
    return dict(case, n_columns=n, n_requested=k, seconds=elapsed,
                peak_rss_mib=rss_after, rss_growth_mib=max(0, rss_after - rss_before),
                returned_none=result is None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(run_case(json.loads(args.case))))
        return
    import oineus as oin

    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    module_path = oin._oineus.__file__
    cases = []
    for dualize in (False, True):
        for storage in ("working", "materialized"):
            for request in ("few", "many"):
                for operation, threads in (("restore", 1), ("restore", 4),
                                           ("getters", 1), ("cached_lists", 1), ("cached_csc", 1)):
                    cases.append(dict(side=args.side, dualize=dualize, storage=storage,
                                      request=request, operation=operation, threads=threads))
    trials = []
    summaries = []
    for case in cases:
        group = []
        for repeat in range(args.repeats):
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    "--case", json.dumps(case)], check=True, capture_output=True, text=True)
            result = json.loads(child.stdout)
            group.append(result)
            trials.append(dict(result, repeat=repeat))
        summary = dict(case, n_columns=group[0]["n_columns"], n_requested=group[0]["n_requested"],
                       median_seconds=statistics.median(r["seconds"] for r in group),
                       max_peak_rss_mib=max(r["peak_rss_mib"] for r in group),
                       max_rss_growth_mib=max(r["rss_growth_mib"] for r in group))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        if args.output:
            args.output.write_text(json.dumps(dict(
                source_revision=revision, module_path=module_path,
                platform=platform.platform(), python=sys.version, executable=sys.executable,
                cpu_count=os.cpu_count(), repetitions=args.repeats,
                memory_note="Process peak RSS includes filtration/reduction setup; growth is over the pre-phase high-water mark.",
                summaries=summaries, trials=trials), indent=2) + "\n")


if __name__ == "__main__":
    main()
