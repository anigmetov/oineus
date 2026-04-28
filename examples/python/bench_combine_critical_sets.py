#!/usr/bin/env python3
"""
Phase-1 benchmark: C++ TopologyOptimizer.combine_loss vs the
PyTorch-vectorised oineus.diff._combine.combine.

Inputs come from real persistence pipelines on point clouds drawn from
tests/data_utils.py. We sweep one parameter (n_points) at a time, ramping
up cautiously and printing per-stage timings before each iteration so a
configuration that gets uncomfortable can be Ctrl-C'd before the next one
starts. Weak-alpha filtrations are preferred over Vietoris-Rips for
size scaling.

Run:
    PYTHONPATH=build/bindings/python venv_build/bin/python \
        examples/python/bench_combine_critical_sets.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

import oineus
import oineus.diff as oin_diff
from oineus.diff import _combine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "tests"))
sys.path.insert(0, TESTS_DIR)
from data_utils import sample_annulus  # noqa: E402


def make_filtration(n_points: int, seed: int):
    rng = np.random.default_rng(seed)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        pts = sample_annulus(n_points=n_points, inner_radius=1.0,
                             outer_radius=2.0, sigma=0.05)
    finally:
        np.random.set_state(state)
    pts = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
    fil = oin_diff.weak_alpha_filtration(pts)
    return pts, fil


def build_critical_sets(top_opt, fil_values_np, dim_dgm_pairs, perturb=0.05, seed=0):
    """
    From an index diagram, build a synthetic batch of singleton targets
    that exercises all four directions (increase/decrease birth/death).

    `dim_dgm_pairs` is an (M, 2) ndarray of (birth_idx, death_idx) sorted
    ids. We perturb each entry with mixed signs, drop no-ops, hand to
    top_opt.singletons, and return the critical-sets list.
    """
    rng = np.random.default_rng(seed)
    n_pairs = dim_dgm_pairs.shape[0]
    if n_pairs == 0:
        return []
    b_idx = dim_dgm_pairs[:, 0]
    d_idx = dim_dgm_pairs[:, 1]
    # mixed sign perturbations so all four directions appear
    b_sign = rng.choice([-1.0, 1.0], size=n_pairs)
    d_sign = rng.choice([-1.0, 1.0], size=n_pairs)
    b_tgt = fil_values_np[b_idx] + b_sign * perturb
    d_tgt = fil_values_np[d_idx] + d_sign * perturb

    flat_idx = np.concatenate([b_idx, d_idx]).tolist()
    flat_tgt = np.concatenate([b_tgt, d_tgt]).tolist()
    return top_opt.singletons(flat_idx, flat_tgt)


def time_cpp_combine(top_opt, crit_sets, strategy_enum, n_repeats):
    # Warm-up.
    top_opt.combine_loss(crit_sets, strategy_enum)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        top_opt.combine_loss(crit_sets, strategy_enum)
    return (time.perf_counter() - t0) / n_repeats


def time_python_combine(crit_sets, strategy, current_values, n_repeats):
    flat_idx, flat_tgt = _combine.critical_sets_to_flat(
        crit_sets, dtype=current_values.dtype, device=current_values.device,
    )
    # Warm-up.
    _combine.combine(flat_idx, flat_tgt, strategy, current_values=current_values)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _combine.combine(flat_idx, flat_tgt, strategy, current_values=current_values)
    return (time.perf_counter() - t0) / n_repeats


def time_python_combine_with_flatten(crit_sets, strategy, current_values, n_repeats):
    """Includes the flatten cost (apples-to-apples vs C++ which takes the
    list-of-tuples form directly)."""
    # Warm-up.
    flat_idx, flat_tgt = _combine.critical_sets_to_flat(
        crit_sets, dtype=current_values.dtype, device=current_values.device,
    )
    _combine.combine(flat_idx, flat_tgt, strategy, current_values=current_values)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        flat_idx, flat_tgt = _combine.critical_sets_to_flat(
            crit_sets, dtype=current_values.dtype, device=current_values.device,
        )
        _combine.combine(flat_idx, flat_tgt, strategy, current_values=current_values)
    return (time.perf_counter() - t0) / n_repeats


def benchmark_one(n_points, seed, time_budget):
    print(f"\n=== n_points={n_points} (seed={seed}) ===")
    t0 = time.perf_counter()
    pts, fil = make_filtration(n_points, seed)
    print(f"  build filtration: {time.perf_counter() - t0:.3f}s, "
          f"|fil|={fil.size()}")

    t0 = time.perf_counter()
    top_opt = oineus._oineus.TopologyOptimizer(fil.under_fil)
    top_opt.reduce_all()
    nondiff_dgms = top_opt.compute_diagram(include_inf_points=False)
    print(f"  reduce_all + compute_diagram: {time.perf_counter() - t0:.3f}s")

    if (time.perf_counter() - t0) > time_budget:
        print(f"  reduce_all exceeded budget {time_budget:.0f}s; stopping")
        return False

    fil_values = fil.values.detach()
    fil_values_np = fil_values.numpy()

    # Aggregate H1 (and H0 to thicken the diagram for higher conflict density).
    pair_arrays = []
    for dim in (0, 1):
        arr = nondiff_dgms.index_diagram_in_dimension(dim, as_numpy=True).astype(np.int64)
        if arr.size == 0:
            continue
        # drop infinite points (death idx out of range)
        finite = (arr[:, 0] >= 0) & (arr[:, 0] < fil.size()) \
                 & (arr[:, 1] >= 0) & (arr[:, 1] < fil.size())
        pair_arrays.append(arr[finite])
    if not pair_arrays:
        print("  no finite diagram points; skipping")
        return True
    pairs = np.concatenate(pair_arrays, axis=0)
    print(f"  finite (H0+H1) pairs: {pairs.shape[0]}")

    t0 = time.perf_counter()
    crit_sets = build_critical_sets(top_opt, fil_values_np, pairs, perturb=0.05, seed=seed)
    print(f"  build {len(crit_sets)} critical sets: {time.perf_counter() - t0:.3f}s")

    total_contribs = sum(len(cs[1]) for cs in crit_sets)
    if total_contribs == 0:
        print("  no contributions; skipping")
        return True
    print(f"  total contributions M={total_contribs}, "
          f"avg group size={total_contribs / max(1, len(crit_sets)):.1f}")

    n_repeats = max(3, min(50, int(2_000_000 / max(1, total_contribs))))
    print(f"  n_repeats={n_repeats}")

    print(f"  {'strategy':>8} {'cpp':>10} {'py-pre':>10} {'py-full':>10} "
          f"{'cpp/py-pre':>10} {'cpp/py-full':>10}")

    enum = oineus._oineus.ConflictStrategy
    strategies = [("avg", enum.Avg), ("max", enum.Max), ("sum", enum.Sum)]

    for name, strategy_enum in strategies:
        t_cpp = time_cpp_combine(top_opt, crit_sets, strategy_enum, n_repeats)
        t_py_pre = time_python_combine(crit_sets, name, fil_values, n_repeats)
        t_py_full = time_python_combine_with_flatten(crit_sets, name, fil_values, n_repeats)
        print(f"  {name:>8} {t_cpp*1e3:>9.3f}ms {t_py_pre*1e3:>9.3f}ms "
              f"{t_py_full*1e3:>9.3f}ms {t_cpp/t_py_pre:>10.2f} {t_cpp/t_py_full:>10.2f}")

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[64, 128, 256, 512, 1024, 2048],
                   help="point-cloud sizes to sweep")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget", type=float, default=30.0,
                   help="bail out if a single reduce_all takes longer (s)")
    args = p.parse_args()

    print(f"sizes to sweep: {args.sizes}")
    print(f"per-stage time budget: {args.time_budget}s")

    for n in args.sizes:
        ok = benchmark_one(n, args.seed, args.time_budget)
        if not ok:
            break


if __name__ == "__main__":
    main()
