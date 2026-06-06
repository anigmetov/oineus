"""Benchmark decomposition-manipulation strategies for iterated persistent
homology, comparing the algorithms of

  * Piekenbrock & Perea 2021 (vineyards adjacent transpositions; move schedules)
  * Luo & Nelson 2021 (warm-start updates: permutation, and insertion/deletion)

against a full from-scratch recomputation. The headline metric of both papers
is the number of column operations, so we report that alongside wall-clock.

Workloads:
  (a) 1-parameter families -- a filtration whose values drift over a sequence
      of steps; at each step every strategy updates its decomposition and is
      validated against a full recompute:
        * lower-star on a grid (perturbed pixel values)  -> pure reorder
        * Vietoris-Rips on a point cloud (perturbed positions) -> pure reorder
        * Vietoris-Rips with a shrinking radius -> insertions/deletions (Alg 3)
  (b) optimization step -- a target-driven reorder via targets_to_permutation
      (the "warm starts" use case already scaffolded in oineus).

Usage (from repo root, cmake build on PYTHONPATH):
  PYTHONPATH="$PWD/build/bindings/python" ./.venv/bin/python agents_outputs/bench_dcmp_manips.py
"""
import argparse
import os
import sys
import time

import numpy as np

import oineus as oin

REORDER_METHODS = ["vineyards", "moves", "warm_perm"]   # pure-reorder strategies


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def reduce_params():
    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 1
    p.clearing_opt = False
    return p


def fresh_decomposition(fil):
    d = oin.Decomposition(fil, False)
    d.reduce(reduce_params())
    return d


def diagram_key(dcmp, fil, max_dim):
    dgm = dcmp.diagram(fil, include_inf_points=True)
    out = []
    for d in range(max_dim + 1):
        a = dgm.in_dimension(d)
        out.append(np.round(np.sort(a, axis=0), 6) if len(a) else np.zeros((0, 2)))
    return out


def keys_equal(a, b):
    if len(a) != len(b):
        return False
    return all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


def uid_to_sorted_id(fil):
    # cell.uid is a single integer that canonically identifies the cell
    return {c.uid: c.sorted_id for c in fil.cells()}


def new_to_old_reorder(fil_old, fil_new):
    """Pure reorder (same cell set): new_to_old[k] = old sorted id of the cell
    now at new position k, matched by UID."""
    old = uid_to_sorted_id(fil_old)
    return [old[c.uid] for c in fil_new.cells()]


def new_to_old_and_boundary_edits(fil_old, fil_new):
    """For an edited cell set: new_to_old[k] = old sorted id or -1 (inserted),
    plus the new boundary matrix (sorted order)."""
    old = uid_to_sorted_id(fil_old)
    new_to_old = [old.get(c.uid, -1) for c in fil_new.cells()]
    # boundary of the new filtration in sorted order = d_data of a fresh decomposition
    new_boundary = [list(col) for col in oin.Decomposition(fil_new, False).d_data]
    return new_to_old, new_boundary


def col_ops(stats):
    return stats.n_column_additions()


# --------------------------------------------------------------------------
# Workload builders
# --------------------------------------------------------------------------
def lower_star_family(side, n_steps, sigma, seed):
    rng = np.random.default_rng(seed)
    base = rng.random((side, side))
    for k in range(n_steps + 1):
        data = base + sigma * k * rng.standard_normal(base.shape)
        yield oin.freudenthal_filtration(np.ascontiguousarray(data), max_dim=2)


def vr_reorder_family(n_points, n_steps, sigma, diam, seed, dim=2):
    rng = np.random.default_rng(seed)
    base = rng.random((n_points, dim))
    for k in range(n_steps + 1):
        pts = base + sigma * k * rng.standard_normal(base.shape)
        yield oin.vr_filtration(np.ascontiguousarray(pts), max_dim=2, max_diameter=diam)


def vr_radius_family(n_points, radii, seed, dim=2):
    rng = np.random.default_rng(seed)
    pts = np.ascontiguousarray(rng.random((n_points, dim)))
    for r in radii:
        yield oin.vr_filtration(pts, max_dim=2, max_diameter=float(r))


# --------------------------------------------------------------------------
# One reorder step: update each running decomposition, validate, record.
# --------------------------------------------------------------------------
def run_reorder_family(label, fils, max_dim, rows):
    fils = list(fils)
    fil0 = fils[0]
    # one running decomposition per strategy, all starting from the same state
    running = {m: fresh_decomposition(fil0) for m in REORDER_METHODS}

    for step in range(1, len(fils)):
        fil_prev, fil_new = fils[step - 1], fils[step]
        if fil_prev.size() != fil_new.size():
            print(f"  [{label}] step {step}: cell count changed "
                  f"({fil_prev.size()} -> {fil_new.size()}), skipping reorder step")
            continue
        nto = new_to_old_reorder(fil_prev, fil_new)

        t0 = time.perf_counter()
        full = fresh_decomposition(fil_new)
        t_full = time.perf_counter() - t0
        key_full = diagram_key(full, fil_new, max_dim)

        for m in REORDER_METHODS:
            dcmp = running[m]
            stats = oin.DecompositionManipStats()
            t0 = time.perf_counter()
            if m == "vineyards":
                dcmp.transpose_to(nto, stats)
            elif m == "moves":
                dcmp.apply_move_schedule(nto, stats)
            elif m == "warm_perm":
                dcmp.update_with_permutation(nto, stats)
            t_wall = time.perf_counter() - t0
            ok = keys_equal(diagram_key(dcmp, fil_new, max_dim), key_full)
            rows.append({
                "input": label, "step": step, "method": m, "n_cells": fil_new.size(),
                "col_ops": col_ops(stats), "n_transp": stats.n_transpositions,
                "n_moves": stats.n_moves, "n_scan": stats.n_columns_scanned,
                "nnz_after": stats.nnz_r_after + stats.nnz_v_after,
                "t_method": t_wall, "t_full": t_full, "ok": ok,
            })
            if not ok:
                print(f"  [{label}] step {step} method {m}: DIAGRAM MISMATCH")


def run_edit_family(label, fils, max_dim, rows):
    """Insertion/deletion family driven by Luo-Nelson Alg 3 (update_with_edits)."""
    fils = list(fils)
    running = fresh_decomposition(fils[0])
    for step in range(1, len(fils)):
        fil_prev, fil_new = fils[step - 1], fils[step]
        nto, new_b = new_to_old_and_boundary_edits(fil_prev, fil_new)

        t0 = time.perf_counter()
        full = fresh_decomposition(fil_new)
        t_full = time.perf_counter() - t0
        key_full = diagram_key(full, fil_new, max_dim)

        stats = oin.DecompositionManipStats()
        t0 = time.perf_counter()
        running.update_with_edits(nto, new_b, list(fil_new.dim_first), list(fil_new.dim_last), stats)
        t_wall = time.perf_counter() - t0
        ok = keys_equal(diagram_key(running, fil_new, max_dim), key_full)
        n_ins = sum(1 for x in nto if x < 0)
        n_del = fil_prev.size() - (len(nto) - n_ins)
        rows.append({
            "input": label, "step": step, "method": "warm_edits", "n_cells": fil_new.size(),
            "col_ops": col_ops(stats), "n_transp": stats.n_transpositions, "n_moves": stats.n_moves,
            "n_scan": stats.n_columns_scanned,
            "nnz_after": stats.nnz_r_after + stats.nnz_v_after, "t_method": t_wall,
            "t_full": t_full, "ok": ok, "n_ins": n_ins, "n_del": n_del,
        })
        if not ok:
            print(f"  [{label}] step {step} update_with_edits: DIAGRAM MISMATCH")


# --------------------------------------------------------------------------
# Optimization-step demo: target-driven reorder via targets_to_permutation.
# --------------------------------------------------------------------------
def run_optimization_step(rows, seed=0):
    """One topology-optimization-style step: push the most persistent H1 pair
    towards the diagonal by lowering its death (2-cell) value, then re-sort the
    filtration. This is the kind of reorder that gradient-based optimization of
    a persistence-based loss induces -- the "warm starts" use case."""
    rng = np.random.default_rng(seed)
    pts = np.ascontiguousarray(rng.random((40, 2)))
    fil = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0)
    max_dim = 2

    full0 = fresh_decomposition(fil)
    dgm = full0.diagram(fil, include_inf_points=False).in_dimension(1)
    if len(dgm) == 0:
        print("  [optimization] no H1 features, skipping")
        return

    j = int(np.argmax(dgm[:, 1] - dgm[:, 0]))
    death_val, birth_val = float(dgm[j, 1]), float(dgm[j, 0])
    new_vals = [fil.cell_value_by_sorted_id(s) for s in range(fil.size())]
    for s in range(fil.size()):
        if fil.cell(s).dim == 2 and abs(new_vals[s] - death_val) < 1e-12:
            new_vals[s] = birth_val
    # target order: stable sort by (dim, new value, old sorted id)
    new_to_old = sorted(range(fil.size()),
                        key=lambda s: (fil.cell(s).dim, new_vals[s], s))

    for m in REORDER_METHODS:
        dcmp = fresh_decomposition(fil)
        stats = oin.DecompositionManipStats()
        t0 = time.perf_counter()
        if m == "vineyards":
            dcmp.transpose_to(new_to_old, stats)
        elif m == "moves":
            dcmp.apply_move_schedule(new_to_old, stats)
        else:
            dcmp.update_with_permutation(new_to_old, stats)
        t_wall = time.perf_counter() - t0
        # validate against from-scratch reduction of the reordered boundary
        ok = (pairing_of(dcmp) == pairing_of_boundary(dcmp.d_data))
        rows.append({
            "input": "optimization", "step": 1, "method": m, "n_cells": fil.size(),
            "col_ops": col_ops(stats), "n_transp": stats.n_transpositions, "n_moves": stats.n_moves,
            "n_scan": stats.n_columns_scanned,
            "nnz_after": stats.nnz_r_after + stats.nnz_v_after, "t_method": t_wall,
            "t_full": 0.0, "ok": ok,
        })


def pairing_of(dcmp):
    return frozenset((max(col), c) for c, col in enumerate(dcmp.r_data) if col)


def pairing_of_boundary(boundary):
    d = oin.Decomposition(boundary, len(boundary), False, True)
    d.reduce(reduce_params())
    return frozenset((max(col), c) for c, col in enumerate(d.r_data) if col)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_results(rows, path_csv, path_md):
    cols = ["input", "step", "method", "n_cells", "col_ops", "n_transp", "n_moves",
            "n_scan", "nnz_after", "t_method", "t_full", "ok"]
    with open(path_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # aggregate per (input, method): total col_ops, mean speedup, all-ok
    inputs = []
    for r in rows:
        if r["input"] not in inputs:
            inputs.append(r["input"])
    with open(path_md, "w") as f:
        f.write("# Decomposition-manipulation benchmark\n\n")
        f.write("Strategies: **full** (recompute), **vineyards** "
                "(`transpose_to`), **moves** (`apply_move_schedule`), **warm_perm** "
                "(Luo-Nelson Alg 2 `update_with_permutation`), **warm_edits** "
                "(Luo-Nelson Alg 3 `update_with_edits`).\n\n")
        f.write("Headline column-operation count plus **col_scans** = columns "
                "visited in whole-matrix passes (row relabels, pivot rebuilds, "
                "reductions, materializations) -- the global O(nnz)/O(n) work the "
                "column-op count hides. Wall-clock in seconds.\n\n")
        for inp in inputs:
            f.write(f"## {inp}\n\n")
            f.write("| method | total col_ops | total col_scans | total transp | "
                    "total moves | sum t_method (s) | sum t_full (s) | all ok |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            methods = []
            for r in rows:
                if r["input"] == inp and r["method"] not in methods:
                    methods.append(r["method"])
            for m in methods:
                sub = [r for r in rows if r["input"] == inp and r["method"] == m]
                f.write(f"| {m} | {sum(r['col_ops'] for r in sub)} | "
                        f"{sum(r['n_scan'] for r in sub)} | "
                        f"{sum(r['n_transp'] for r in sub)} | {sum(r['n_moves'] for r in sub)} | "
                        f"{sum(r['t_method'] for r in sub):.4f} | "
                        f"{sum(r['t_full'] for r in sub):.4f} | "
                        f"{all(r['ok'] for r in sub)} |\n")
            f.write("\n")
    print(f"\nWrote {path_csv}\nWrote {path_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller/faster sizes")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rows = []

    if args.quick:
        ls_side, ls_steps = 12, 4
        vr_n, vr_steps = 25, 4
    else:
        ls_side, ls_steps = 24, 8
        vr_n, vr_steps = 40, 8

    print("== lower-star grid family (pure reorder) ==")
    run_reorder_family("lower_star", lower_star_family(ls_side, ls_steps, 0.02, 0),
                       max_dim=2, rows=rows)

    print("== VR point-cloud family (pure reorder) ==")
    run_reorder_family("vr_reorder", vr_reorder_family(vr_n, vr_steps, 0.01, 1.5, 1),
                       max_dim=2, rows=rows)

    print("== VR shrinking-radius family (insert/delete, Alg 3) ==")
    radii = list(np.linspace(1.5, 0.8, 6))
    run_edit_family("vr_edits", vr_radius_family(vr_n, radii, 2), max_dim=2, rows=rows)

    print("== optimization step (targets_to_permutation) ==")
    run_optimization_step(rows)

    n_bad = sum(1 for r in rows if not r["ok"])
    print(f"\n{len(rows)} measurements, {n_bad} diagram mismatches")
    write_results(rows, os.path.join(here, "bench_dcmp_manips_results.csv"),
                  os.path.join(here, "bench_dcmp_manips_results.md"))
    if n_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
