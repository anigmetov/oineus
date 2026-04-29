#!/usr/bin/env python3
"""
Phase-2 backward benchmark for oineus.diff.persistence_diagram (crit-sets).

Times one full forward + loss + backward iteration on a fixed-shape
synthetic loss, with knobs to sweep:

  --filtration    {vr, weak_alpha, freudenthal, cubical}
  --n-points / --grid-shape
  --n-moves        small, medium, "all" -- exercises both
                   "few moves on heavy U/V rows" and "many moves"
  --directions    {death-up, death-down, birth-up, birth-down, mixed}
  --strategy      {avg, max, sum, fca}

Also reports which matrices were materialised (TopologyOptimizer.matrix_summary)
to confirm the per-side lazy reduction is actually skipping work.

Inputs come from real persistence pipelines via tests/data_utils.py.
Sweep cautiously: print per-stage timings before each iteration; bail
out if a single iteration exceeds --time-budget. Weak-alpha is preferred
over VR for size scaling; VR is in scope for the load-bearing
"clearing wins on cohomology" case but capped at small n_points.

Run:
    PYTHONPATH=build/bindings/python venv_build/bin/python \
        examples/python/bench_crit_sets_backward.py \
        --filtration vr --n-points 20 50 100 --n-moves 1 5 all \
        --directions mixed
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

import oineus
import oineus.diff as oin_diff

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "tests"))
sys.path.insert(0, TESTS_DIR)
from data_utils import (  # noqa: E402
    sample_annulus,
    random_gaussian_2d,
    rosenbrock_2d,
)


# ---------------------------------------------------------------------
# Filtration construction
# ---------------------------------------------------------------------


def make_pointcloud_2d(n_points, seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        pts = sample_annulus(n_points=n_points, inner_radius=1.0,
                             outer_radius=2.0, sigma=0.05)
    finally:
        np.random.set_state(state)
    return torch.tensor(pts, dtype=torch.float64, requires_grad=True)


def make_grid_2d(grid_shape, seed):
    nx, ny = grid_shape
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        arr = random_gaussian_2d(nx, ny, complexity=5)
    finally:
        np.random.set_state(state)
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)


def build_filtration(name, n_points, grid_shape, seed, max_dim=2):
    if name == "vr":
        pts = make_pointcloud_2d(n_points, seed)
        return pts, oin_diff.vr_filtration(pts, max_dim=max_dim)
    if name in ("weak_alpha", "alpha"):
        # Differentiable alpha-complex combinatorics with weak-alpha
        # values. The actual alpha values from CGAL aren't
        # differentiable; this is the closest stand-in.
        pts = make_pointcloud_2d(n_points, seed)
        return pts, oin_diff.weak_alpha_filtration(pts)
    if name == "freudenthal":
        data = make_grid_2d(grid_shape, seed)
        return data, oin_diff.freudenthal_filtration(
            data, max_dim=max_dim, negate=False, wrap=False, n_threads=1)
    if name == "cubical":
        data = make_grid_2d(grid_shape, seed)
        return data, oin_diff.cube_filtration(
            data, max_dim=max_dim, negate=False, wrap=False, n_threads=1)
    raise ValueError(f"unknown filtration {name!r}")


# ---------------------------------------------------------------------
# Loss construction (no Wasserstein)
# ---------------------------------------------------------------------


def pick_pair_indices(dgm, n_moves, seed):
    """Return integer indices of which dgm pairs to perturb."""
    n = dgm.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n_moves == "all":
        return np.arange(n)
    n_moves = int(n_moves)
    if n_moves >= n:
        return np.arange(n)
    # Bias towards the most persistent pairs (likely to have heavier
    # critical sets through long U/V columns).
    persistence = (dgm[:, 1] - dgm[:, 0]).detach().numpy()
    order = np.argsort(persistence)[::-1]
    rng = np.random.default_rng(seed)
    head = order[:max(1, n_moves // 2)]
    tail = rng.choice(order[max(1, n_moves // 2):], size=n_moves - len(head),
                      replace=False) if len(order) > len(head) else np.empty(0, dtype=np.int64)
    return np.concatenate([head, tail]).astype(np.int64)


def build_loss(dgm, picks, directions, perturb=0.05, rng_seed=0):
    """Build a hand-coded matching loss: each picked pair is matched
    to its perturbed target along a deterministic direction pattern.
    Returns a scalar torch.Tensor connected to the input graph.

    No Wasserstein call -- the matching is fixed by `directions`."""
    if len(picks) == 0:
        return None
    rng = np.random.default_rng(rng_seed)
    bs = []
    ds = []
    cur_b = dgm[picks, 0].detach().numpy()
    cur_d = dgm[picks, 1].detach().numpy()
    if directions == "death-up":
        bs[:] = cur_b.tolist()
        ds[:] = (cur_d + perturb).tolist()
    elif directions == "death-down":
        bs[:] = cur_b.tolist()
        ds[:] = np.maximum(cur_b + 1e-3, cur_d - perturb).tolist()
    elif directions == "birth-up":
        bs[:] = (cur_b + perturb).tolist()
        ds[:] = cur_d.tolist()
    elif directions == "birth-down":
        bs[:] = np.maximum(cur_b - perturb, np.full_like(cur_b, -1e9)).tolist()
        ds[:] = cur_d.tolist()
    elif directions == "mixed":
        b_sign = rng.choice([-1.0, 1.0], size=len(picks))
        d_sign = rng.choice([-1.0, 1.0], size=len(picks))
        bs[:] = (cur_b + b_sign * perturb).tolist()
        ds[:] = (cur_d + d_sign * perturb).tolist()
    else:
        raise ValueError(f"unknown directions {directions!r}")
    target_b = torch.tensor(bs, dtype=dgm.dtype, device=dgm.device)
    target_d = torch.tensor(ds, dtype=dgm.dtype, device=dgm.device)
    diff_b = dgm[picks, 0] - target_b
    diff_d = dgm[picks, 1] - target_d
    return (diff_b ** 2 + diff_d ** 2).sum()


# ---------------------------------------------------------------------
# One benchmark iteration
# ---------------------------------------------------------------------


def run_one(filtration, n_points, grid_shape, n_moves, directions,
            strategy, seed, label, gradient_method="crit-sets", n_threads=1):
    pts, fil = build_filtration(filtration, n_points, grid_shape, seed)

    t = time.perf_counter()
    dgms = oin_diff.persistence_diagram(
        fil,
        gradient_method=gradient_method,
        step_size=1.0,
        conflict_strategy=strategy,
        n_threads=n_threads,
    )
    t_forward = time.perf_counter() - t

    # Use H1 if non-empty, else H0 (for filtrations that don't produce H1).
    dgm = None
    for d in (1, 0):
        if d in dgms:
            cand = dgms.in_dimension(d)
            if cand.shape[0]:
                dgm = cand
                break
    if dgm is None or dgm.shape[0] == 0:
        return {"label": label, "skipped": "no diagram points"}

    picks = pick_pair_indices(dgm, n_moves, seed)
    loss = build_loss(dgm, picks, directions, perturb=0.05, rng_seed=seed)
    if loss is None:
        return {"label": label, "skipped": "no picks"}

    # Steady-state timing: re-run a few times after a warm-up to amortise
    # the fallback re-reduce that may happen on the first backward.
    pts_grad_clone = None
    n_warmup = 1
    n_repeat = 3
    times = []
    for i in range(n_warmup + n_repeat):
        if pts.grad is not None:
            pts.grad = None
        t = time.perf_counter()
        loss.backward(retain_graph=(i < n_warmup + n_repeat - 1))
        times.append(time.perf_counter() - t)
        if i == 0:
            pts_grad_clone = pts.grad.detach().clone()
    t_backward_cold = times[0]
    t_backward_warm = sum(times[n_warmup:]) / max(1, n_repeat)

    if hasattr(dgms._top_opt, "matrix_summary"):
        hom_status, coh_status = dgms._top_opt.matrix_summary()
    else:
        # dgm-loss uses a single Decomposition, not a TopologyOptimizer
        class _S: pass
        s = _S()
        s.is_reduced = True
        s.has_v = False
        s.has_u = False
        s.clearing_opt_used = False
        hom_status = coh_status = s

    return {
        "label": label,
        "fil_size": fil.size(),
        "n_pairs_in_dgm": dgm.shape[0],
        "n_moves": int(picks.size),
        "t_forward": t_forward,
        "t_backward_cold": t_backward_cold,
        "t_backward_warm": t_backward_warm,
        "grad_norm": float(pts_grad_clone.norm().item()),
        "hom_is_reduced": hom_status.is_reduced,
        "hom_has_v": hom_status.has_v,
        "hom_has_u": hom_status.has_u,
        "hom_clearing": hom_status.clearing_opt_used,
        "coh_is_reduced": coh_status.is_reduced,
        "coh_has_v": coh_status.has_v,
        "coh_has_u": coh_status.has_u,
        "coh_clearing": coh_status.clearing_opt_used,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def fmt_status(r):
    if "skipped" in r:
        return f"  SKIPPED: {r['skipped']}"
    return (
        f"  fil={r['fil_size']:>6}  pairs={r['n_pairs_in_dgm']:>5}  moves={r['n_moves']:>5}  "
        f"fwd={r['t_forward']*1e3:>8.2f}ms  bwd-cold={r['t_backward_cold']*1e3:>8.2f}ms  "
        f"bwd-warm={r['t_backward_warm']*1e3:>8.2f}ms  ||grad||={r['grad_norm']:.3e}\n"
        f"    hom: reduced={r['hom_is_reduced']}, V={r['hom_has_v']}, U={r['hom_has_u']}, clearing={r['hom_clearing']}\n"
        f"    coh: reduced={r['coh_is_reduced']}, V={r['coh_has_v']}, U={r['coh_has_u']}, clearing={r['coh_clearing']}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filtration",
                   choices=["vr", "weak_alpha", "alpha", "freudenthal", "cubical"],
                   default="weak_alpha")
    p.add_argument("--n-points", type=int, nargs="+",
                   default=[64, 256, 1024, 2048],
                   help="point-cloud sizes (vr, weak_alpha)")
    p.add_argument("--grid-shape", type=int, nargs=2, action="append",
                   help="add a grid (nx, ny); pass once per shape")
    p.add_argument("--n-moves", nargs="+",
                   default=["1", "all"],
                   help="moves per backward; integers or 'all'")
    p.add_argument("--directions",
                   choices=["death-up", "death-down", "birth-up", "birth-down", "mixed"],
                   default="mixed")
    p.add_argument("--strategy", choices=["avg", "max", "sum", "fca"], default="avg")
    p.add_argument("--gradient-method",
                   choices=["crit-sets", "crit-sets-partial",
                            "crit-sets-row-partial", "dgm-loss",
                            "all", "both"],
                   default="all",
                   help="'all' runs every variant (dgm-loss + Phase-2 + "
                        "Phase-3 + Phase-4); 'both' is dgm-loss-free "
                        "Phase-2 vs Phase-3 comparison; individual values "
                        "run just that one")
    p.add_argument("--n-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget", type=float, default=30.0,
                   help="bail out after a single iteration exceeds (s)")
    args = p.parse_args()

    if args.filtration in ("freudenthal", "cubical") and not args.grid_shape:
        args.grid_shape = [(8, 8), (16, 16), (32, 32), (64, 64)]
    if args.filtration in ("vr", "weak_alpha", "alpha") and args.grid_shape:
        # ignore stray --grid-shape for point-cloud filtrations
        args.grid_shape = None

    print(f"filtration: {args.filtration}")
    print(f"directions: {args.directions}, strategy: {args.strategy}")
    print(f"n-moves to sweep: {args.n_moves}")
    if args.filtration in ("vr", "weak_alpha"):
        print(f"n-points to sweep: {args.n_points}")
    else:
        print(f"grid shapes to sweep: {args.grid_shape}")
    print(f"per-iter time budget: {args.time_budget}s\n")

    if args.filtration in ("vr", "weak_alpha", "alpha"):
        size_iter = [(n, None) for n in args.n_points]
        size_label = lambda n, g: f"n_points={n}"
    else:
        size_iter = [(None, tuple(g)) for g in args.grid_shape]
        size_label = lambda n, g: f"grid={g}"

    if args.gradient_method == "all":
        methods = ["dgm-loss", "crit-sets", "crit-sets-partial",
                   "crit-sets-row-partial"]
    elif args.gradient_method == "both":
        methods = ["crit-sets", "crit-sets-partial"]
    else:
        methods = [args.gradient_method]

    # Collect rows for the summary table
    summary_rows = []

    for n, g in size_iter:
        for nm in args.n_moves:
            results = {}
            for method in methods:
                label = f"{size_label(n, g)} moves={nm} method={method}"
                print(f"=== {label} ===")
                t0 = time.perf_counter()
                r = run_one(args.filtration, n, g, nm, args.directions,
                            args.strategy, args.seed, label,
                            gradient_method=method, n_threads=args.n_threads)
                elapsed = time.perf_counter() - t0
                print(fmt_status(r))
                print(f"  iteration wall-clock: {elapsed:.2f}s")
                results[method] = r
                if elapsed > args.time_budget:
                    print(f"  hit time budget {args.time_budget:.0f}s; stopping")
                    break
            else:
                # Both / all comparison line
                if "crit-sets" in results and "crit-sets-partial" in results \
                        and "skipped" not in results["crit-sets"] \
                        and "skipped" not in results["crit-sets-partial"]:
                    t2 = results["crit-sets"]["t_backward_cold"]
                    t3 = results["crit-sets-partial"]["t_backward_cold"]
                    if t3 > 0:
                        print(f"  >>> COLD backward speedup (phase2/phase3): "
                              f"{t2/t3:.2f}x  (phase2={t2*1e3:.2f}ms, "
                              f"phase3={t3*1e3:.2f}ms)")
                # Stash for summary
                size_str = size_label(n, g)
                for method, r in results.items():
                    if "skipped" in r:
                        continue
                    summary_rows.append((
                        args.filtration, size_str, r["fil_size"], nm,
                        r["n_moves"], method, r["t_forward"],
                        r["t_backward_cold"], r["t_backward_warm"]))
            print()

    # Final summary table: forward + cold backward + total per variant
    if summary_rows:
        print("=" * 100)
        print("SUMMARY TABLE: forward + cold backward (cold = full reduction "
              "on each step, realistic for optimization loops)")
        print(f"# filtration  size  fil_size  moves  variant            "
              f"t_fwd(ms)  t_bwd_cold(ms)  t_total_cold(ms)  t_bwd_warm(ms)")
        for (fil, sz, fs, mv_req, mv, method, fwd, bcold, bwarm) in summary_rows:
            total = fwd + bcold
            print(f"  {fil:<10} {sz:<10}  {fs:>7}  {str(mv):>5}  "
                  f"{method:<18}  {fwd*1e3:>9.2f}  {bcold*1e3:>14.2f}  "
                  f"{total*1e3:>16.2f}  {bwarm*1e3:>14.2f}")


if __name__ == "__main__":
    main()
