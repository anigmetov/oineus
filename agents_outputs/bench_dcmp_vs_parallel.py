"""Compare warm decomposition updates against a from-scratch PARALLEL
pairing-only reduction (compute_v=False, compute_u=False, clearing on).

This answers: if you only need the persistence pairing, is warm-updating the
decomposition worth it versus just recomputing R from scratch with all cores?

Workload: a lower-star 2D filtration whose pixel values drift by a small
perturbation each step (the iterated-persistence setting). Each step:
  * warm_perm  -- Luo-Nelson Alg 2 update_with_permutation (maintains R and V)
  * moves      -- apply_move_schedule (small sizes only; m conjugations)
  * full@Kt    -- fresh Decomposition + reduce(compute_v=False, n_threads=K)

Times are the C++-side reduction/update time (ReductionParams.elapsed for full,
DecompositionManipStats.elapsed_total for warm), excluding filtration build and
new_to_old computation (both shared / needed by either approach). Every result
is validated to have the same index pairing.

Usage:
  PYTHONPATH="$PWD/build/bindings/python" ./.venv/bin/python agents_outputs/bench_dcmp_vs_parallel.py [--quick]
"""
import argparse
import os
import statistics

import numpy as np

import oineus as oin

THREADS = [1, 2, 4, 8, 16]


def pairing(r_data):
    return frozenset((max(col), c) for c, col in enumerate(r_data) if col)


def lower_star(side, data):
    return oin.freudenthal_filtration(np.ascontiguousarray(data), max_dim=2)


def new_to_old_reorder(fil_old, fil_new):
    old = {c.uid: c.sorted_id for c in fil_old.cells()}
    return [old[c.uid] for c in fil_new.cells()]


def warm_decomp(fil):
    d = oin.Decomposition(fil, False)
    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 1
    p.clearing_opt = False
    d.reduce(p)
    return d


def full_pairing_time(fil, n_threads):
    d = oin.Decomposition(fil, False)
    p = oin.ReductionParams()
    p.compute_v = False
    p.compute_u = False
    p.clearing_opt = True
    p.n_threads = n_threads
    d.reduce(p)
    return pairing(d.r_data), p.elapsed


def make_fils_global(side, n_steps, sigma, rng):
    base = rng.random((side, side))
    return [lower_star(side, base + sigma * k * rng.standard_normal(base.shape))
            for k in range(n_steps + 1)]


def make_fils_local(side, n_steps, k_pixels, rng):
    """Each step perturbs only k_pixels random pixels -> only the cells in their
    lower stars reorder, so the permutation is near-identity."""
    data = rng.random((side, side))
    fils = [lower_star(side, data.copy())]
    for _ in range(n_steps):
        idx = rng.integers(0, side, size=(k_pixels, 2))
        for r, c in idx:
            data[r, c] = rng.random()
        fils.append(lower_star(side, data.copy()))
    return fils


def run(side, n_steps, sigma, seed, do_moves, k_pixels=None):
    rng = np.random.default_rng(seed)
    if k_pixels is None:
        fils = make_fils_global(side, n_steps, sigma, rng)
    else:
        fils = make_fils_local(side, n_steps, k_pixels, rng)
    n_cells = fils[0].size()

    wp = warm_decomp(fils[0])
    mv = warm_decomp(fils[0]) if do_moves else None

    times = {f"full@{k}t": [] for k in THREADS}
    times["warm_perm"] = []
    if do_moves:
        times["moves"] = []
    moved_frac = []

    for step in range(1, len(fils)):
        nto = new_to_old_reorder(fils[step - 1], fils[step])
        moved_frac.append(sum(1 for k, o in enumerate(nto) if k != o) / n_cells)

        key_full, t = full_pairing_time(fils[step], 1)
        times["full@1t"].append(t)
        for k in THREADS[1:]:
            _, tk = full_pairing_time(fils[step], k)
            times[f"full@{k}t"].append(tk)

        st = oin.DecompositionManipStats()
        wp.update_with_permutation(nto, st)
        times["warm_perm"].append(st.elapsed_total)
        assert pairing(wp.r_data) == key_full, f"warm_perm mismatch at step {step}"

        if do_moves:
            st = oin.DecompositionManipStats()
            mv.apply_move_schedule(nto, st)
            times["moves"].append(st.elapsed_total)
            assert pairing(mv.r_data) == key_full, f"moves mismatch at step {step}"

    return n_cells, statistics.mean(moved_frac), times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    sizes = [64, 128] if args.quick else [64, 128, 256]
    print(f"cores={os.cpu_count()}  threads swept={THREADS}")

    hdr = (f"{'size':>6} {'cells':>9} {'moved%':>8}  "
           + "  ".join(f"{m:>10}" for m in ["warm_perm", "moves"] + [f'full@{k}t' for k in THREADS]))

    def fmt(times, name):
        if name not in times or not times[name]:
            return f"{'-':>10}"
        return f"{1000 * statistics.mean(times[name]):>9.2f}m"  # ms

    def row(side, n_cells, mf, times):
        cells = [fmt(times, "warm_perm"), fmt(times, "moves")] + [fmt(times, f"full@{k}t") for k in THREADS]
        print(f"{side:>6} {n_cells:>9} {100*mf:>7.3f}%  " + "  ".join(cells), flush=True)

    # global: ~all cells reorder. moves is prohibitive here (m ~ n, each O(nnz)):
    # measured separately as 7.8s@24k, 192s@97k -- omit from the live sweep.
    print("\n== global perturbation (~all cells reorder; worst case for warm) ==", flush=True)
    print(hdr, flush=True)
    for side in sizes:
        n_cells, mf, times = run(side, n_steps=3, sigma=0.01, seed=side, do_moves=False)
        row(side, n_cells, mf, times)

    # localized: only k pixels' lower stars reorder, so the permutation is
    # near-identity -- the regime warm starts target.
    print("\n== localized perturbation (k pixels changed/step; warm's intended regime) ==", flush=True)
    print(hdr, flush=True)
    big = 128 if args.quick else 256
    for k in [1, 10, 100]:
        n_cells, mf, times = run(big, n_steps=2, sigma=0.0, seed=1000 + k, do_moves=True, k_pixels=k)
        row(f"{big}/{k}", n_cells, mf, times)

    print("\n(times are mean per-step C++ reduction/update in ms; "
          "full = parallel pairing-only reduce, warm = maintains R+V)", flush=True)


if __name__ == "__main__":
    main()
