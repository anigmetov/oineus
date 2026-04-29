#!/usr/bin/env python3
"""
Phase-3 partial-U vs full compute_u_from_v_1 microbenchmark.

Times the U-inversion stage in isolation -- no autograd, no torch. The
question this answers: given a freshly reduced decomposition (V correct,
no U) and a set of (d_idx, target_death) pairs that the future
crit_sets_apply_partial would receive, how does
compute_partial_u_from_v_1(cols, bounds, ...) compare to
compute_u_from_v_1(dim, ...) (the Phase-2 full inversion)?

The cols/bounds derivation here mirrors what crit_sets_apply_partial
will do once it lands (todo step 4 in the broader Phase-3 plan):
  - For each (d_idx, target_death) pair on the homology side,
    the walker reads u_data_t[d_idx] and visits tau in the contiguous
    range [d_idx, last-tau-with-value-le-target_death] within dim.
  - cols = union of these per-pair ranges.
  - per-column bound = min(value(d_p)) over pairs whose range includes
    that column; cmp_op = stop_below (matches increase_death walker).

Sweeps:
  --filtration {vr, weak_alpha, freudenthal}    (cube dispatch tbd)
  --size <list>                                 problem size axis
  --n-pairs <list>                              {1, 4, 16, 64, all}
  --eps <list>                                  fraction of dim-d range:
                                                  small => towards diagonal,
                                                  large => far away
  --n-threads <list>

For each (filtration, size, n_pairs, eps, n_threads) combo we report:
  t_full_ms     compute_u_from_v_1(dim, n_threads) wall time
  t_partial_ms  compute_partial_u_from_v_1(...) wall time
  speedup       t_full / t_partial
  cols_frac     |cols| / |dim|        (>= 1.0 means partial covers everything)

Note: the Phase-3 backward path will use this driver only when at
least one direction needs U (increase_death on hom; decrease_birth on
coh). Pairs going *towards* the diagonal use V columns and never enter
this code -- those are off-axis here. Eps in this benchmark is the
death-up distance (away from diagonal); larger eps means the partial
pass covers more of the dim and is more likely to lose to full.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "tests"))
sys.path.insert(0, TESTS_DIR)

import oineus as oin

try:
    from data_utils import sample_annulus, random_gaussian_2d  # noqa: E402
except ImportError as e:
    raise SystemExit(
        "Run with PYTHONPATH=build/bindings/python so data_utils.py is on sys.path: "
        f"{e}"
    )


# ---------------------------------------------------------------------
# Filtration construction (no torch -- pure numpy + oineus)
# ---------------------------------------------------------------------


def make_pointcloud(n_points, seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        pts = sample_annulus(n_points=n_points, inner_radius=1.0,
                             outer_radius=2.0, sigma=0.05)
    finally:
        np.random.set_state(state)
    return pts.astype(np.float64)


def make_grid(grid_shape, seed):
    nx, ny = grid_shape
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        arr = random_gaussian_2d(nx, ny, complexity=5)
    finally:
        np.random.set_state(state)
    return arr.astype(np.float64)


def build_filtration(name, size, seed):
    if name == "vr":
        pts = make_pointcloud(size, seed)
        return oin.vr_filtration(pts, max_dim=2, max_diameter=1e9, n_threads=1)
    if name == "freudenthal":
        nx = ny = size
        data = make_grid((nx, ny), seed)
        return oin.freudenthal_filtration(
            data=data, negate=False, wrap=False, max_dim=2, n_threads=1)
    raise ValueError(f"unknown filtration {name!r}")


# ---------------------------------------------------------------------
# Reduce + extract diagram
# ---------------------------------------------------------------------


def reduce_v_only(fil, n_threads):
    """Phase-2-style reduction: clearing on, restore on, V correct, no U."""
    decmp = oin.Decomposition(fil, dualize=False, n_threads=n_threads)
    p = oin.ReductionParams()
    p.compute_v = True
    p.compute_u = False
    p.clearing_opt = True
    p.restore_elz = True
    p.n_threads = n_threads
    decmp.reduce(p)
    return decmp


def extract_dgm_for_dim(fil, decmp, dim):
    """Return list of (b_idx_filtration, d_idx_filtration, b_value, d_value)
    for finite pairs in homology dimension `dim`."""
    dgm = decmp.diagram(fil, include_inf_points=False).in_dimension(
        dim, as_numpy=False)
    out = []
    for p in dgm:
        out.append((int(p.birth_index), int(p.death_index),
                    float(p.birth), float(p.death)))
    return out


# ---------------------------------------------------------------------
# Pair selection and cols/bounds derivation
# ---------------------------------------------------------------------


def pick_pairs(dgm_pts, n_pairs, seed):
    """Pick n_pairs from dgm_pts. Bias toward most persistent."""
    n = len(dgm_pts)
    if n == 0:
        return []
    if n_pairs == "all" or int(n_pairs) >= n:
        return list(dgm_pts)
    n_pairs = int(n_pairs)
    rng = np.random.default_rng(seed)
    persistence = np.array([d - b for (_, _, b, d) in dgm_pts])
    order = np.argsort(persistence)[::-1]
    head = order[:max(1, n_pairs // 2)]
    rest_pool = order[max(1, n_pairs // 2):]
    if len(rest_pool):
        tail = rng.choice(rest_pool, size=min(n_pairs - len(head), len(rest_pool)),
                          replace=False)
        chosen = np.concatenate([head, tail])
    else:
        chosen = head
    return [dgm_pts[i] for i in chosen]


def derive_cols_bounds_increase_death(fil, decmp, hom_dim, picks, eps_frac):
    """Mimic crit_sets_apply_partial's derivation for the increase_death
    direction (uses hom-side U). For an H_{hom_dim} pair the death-
    creator lives in dim hom_dim+1, which is where U inversion happens.

    For each pair (d_idx, target_death) where
    target_death = current_death + eps_frac * death_dim_range, build
    the contiguous range [d_idx, last-tau-in-death-dim-with-value-le-
    target_death]. Take the union across pairs; per-col bound =
    min(value(d_p)) over containing pairs.

    Assumes non-dualize hom side; matrix index == filtration index.
    Assumes non-negate filtration (value < target_death is the move).
    """
    if not picks:
        return [], 0
    death_dim = hom_dim + 1
    if death_dim >= len(decmp.dim_first):
        return [], 0
    dim_first = decmp.dim_first[death_dim]
    dim_last = decmp.dim_last[death_dim]
    if dim_first > dim_last:
        return [], 0
    n_dim_cols = dim_last - dim_first + 1

    dim_values = np.array(
        [fil.simplex_value_by_sorted_id(c) for c in range(dim_first, dim_last + 1)])
    dim_min = dim_values.min()
    dim_max = dim_values.max()
    dim_range = dim_max - dim_min if dim_max > dim_min else 1.0

    cols_to_bound = {}
    for (b_idx, d_idx, b_val, d_val) in picks:
        target_death = d_val + eps_frac * dim_range
        # all cols c in [d_idx, dim_last] with value(c) <= target_death
        for c in range(d_idx, dim_last + 1):
            v_c = fil.simplex_value_by_sorted_id(c)
            if v_c > target_death:
                break
            cur = cols_to_bound.get(c)
            if cur is None or cur > d_val:
                cols_to_bound[c] = d_val

    cols = sorted(cols_to_bound.keys())
    bounds = [cols_to_bound[c] for c in cols]
    return cols, bounds, n_dim_cols, death_dim


# ---------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------


def time_call(fn, n_repeat=3, reset=None):
    """Run fn n_repeat times after one warmup; return min wall time (s).
    If reset is provided, call it before every measured run (to undo
    any side effects of the previous fn invocation).
    """
    if reset is not None:
        reset()
    fn()
    times = []
    for _ in range(n_repeat):
        if reset is not None:
            reset()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


@dataclass
class Result:
    fil: str
    size: int
    fil_size: int
    dim: int
    n_pairs: object
    eps: float
    n_threads: int
    n_dim_cols: int
    n_partial_cols: int
    cols_frac: float
    t_full_ms: float
    t_partial_ms: float
    speedup: float

    def header(self):
        return ("fil       size  fil_size  dim  n_pairs  eps    "
                "thr  dim_cols  partial  frac  "
                "t_full(ms)  t_partial(ms)  speedup")

    def row(self):
        return (f"{self.fil:<10}{self.size:>5}  {self.fil_size:>8}  "
                f"{self.dim:>3}  {str(self.n_pairs):>7}  {self.eps:<5.2f}  "
                f"{self.n_threads:>3}  {self.n_dim_cols:>8}  {self.n_partial_cols:>7}  "
                f"{self.cols_frac:>4.2f}  {self.t_full_ms:>10.2f}  "
                f"{self.t_partial_ms:>13.2f}  {self.speedup:>7.2f}")


def run_combo(fil_name, size, n_pairs, eps, n_threads, hom_dim, seed, repeat):
    fil = build_filtration(fil_name, size, seed)
    # Reduce once; both timings re-use it (with u_data_t cleared
    # before each measured call). This isolates the U-inversion cost.
    decmp_full = reduce_v_only(fil, n_threads)
    decmp_partial = reduce_v_only(fil, n_threads)

    dgm_pts = extract_dgm_for_dim(fil, decmp_full, hom_dim)
    picks = pick_pairs(dgm_pts, n_pairs, seed)
    derived = derive_cols_bounds_increase_death(
        fil, decmp_full, hom_dim, picks, eps)
    if not derived[0]:
        return None
    cols, bounds, n_dim_cols, death_dim = derived

    cols_frac = len(cols) / max(1, n_dim_cols)

    def reset_full():
        decmp_full.u_data_t = []
    def call_full():
        decmp_full.compute_u_from_v_1(death_dim, n_threads)

    def reset_partial():
        decmp_partial.u_data_t = []
    def call_partial():
        decmp_partial.compute_partial_u_from_v_1(
            fil, cols, bounds, n_threads=n_threads)

    t_full = time_call(call_full, repeat, reset=reset_full)
    t_partial = time_call(call_partial, repeat, reset=reset_partial)

    return Result(
        fil=fil_name, size=size, fil_size=fil.size(), dim=death_dim,
        n_pairs=n_pairs, eps=eps, n_threads=n_threads,
        n_dim_cols=n_dim_cols, n_partial_cols=len(cols),
        cols_frac=cols_frac,
        t_full_ms=t_full * 1e3,
        t_partial_ms=t_partial * 1e3,
        speedup=t_full / t_partial if t_partial > 0 else float("inf"),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filtration", choices=["vr", "freudenthal"],
                   default="freudenthal")
    p.add_argument("--size", type=int, nargs="+",
                   default=[64, 128, 256])
    p.add_argument("--n-pairs", nargs="+",
                   default=["1", "4", "16", "all"])
    p.add_argument("--eps", type=float, nargs="+",
                   default=[0.02, 0.10, 0.50])
    p.add_argument("--n-threads", type=int, nargs="+",
                   default=[1, 4])
    p.add_argument("--hom-dim", type=int, default=1,
                   help="homology dim (the partial-U pass operates on dim+1)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--time-budget", type=float, default=180.0,
                   help="bail after a single configuration's measurement exceeds (s)")
    args = p.parse_args()

    print(f"# filtration={args.filtration} hom_dim={args.hom_dim} repeat={args.repeat}")
    print(Result(args.filtration, 0, 0, args.hom_dim + 1, "", 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0).header())

    for size in args.size:
        for n_pairs in args.n_pairs:
            for eps in args.eps:
                for n_threads in args.n_threads:
                    t0 = time.perf_counter()
                    try:
                        r = run_combo(args.filtration, size, n_pairs, eps,
                                      n_threads, args.hom_dim, args.seed,
                                      args.repeat)
                    except Exception as e:
                        print(f"  ERROR {args.filtration} size={size} "
                              f"n_pairs={n_pairs} eps={eps}: {e}")
                        continue
                    if r is None:
                        print(f"  SKIP  {args.filtration:<10} size={size} "
                              f"n_pairs={n_pairs} eps={eps}: no pairs")
                        continue
                    elapsed = time.perf_counter() - t0
                    print(r.row())
                    if elapsed > args.time_budget:
                        print(f"# hit time budget {args.time_budget:.0f}s; "
                              f"skipping rest of this size")
                        break


if __name__ == "__main__":
    main()
