#!/usr/bin/env python3
"""
U-strategy production-default sweep for oineus.diff.persistence_diagram.

Compares 5 strategies for computing U on top of a parallel V-only
reduction (with restore_elz):

  legacy_in_band -- Phase-2 in-band U during reduction (clearing off,
                    compute_u=true). Reference / control.
  col_R          -- Algorithm 3, R U = D, full dim
                    (decmp.compute_u_from_v).
  col_V          -- Algorithm 4, V U = I, full dim, columns + transpose
                    (decmp.compute_u_from_v_1).
  row_full       -- Phase-4 row-form, all rows of dim
                    (decmp.compute_full_u_rows).
  row_partial    -- Phase-4 row-form, only rows the walker reads,
                    with PHASE4_PARTIAL_THRESHOLD-based fallback to
                    full (decmp.compute_partial_u_rows).

For each scenario (filtration kind, size, hom/coh side, sampler,
n_pairs, strategy), we record:
  - t_u_stage_ms: time of just the U-computation step on a freshly
    reduced decomposition (in isolation, no walks/scatter).
  - t_total_ms: end-to-end cold backward (forward + reduction + U +
    walks + scatter).

Output:
  - CSV row per scenario.
  - Final "best strategy per (kind, side, n_pairs band)" matrix that
    suggests defaults for production code.

Pair selection samplers are pluggable; the registry holds a wide menu
(see SAMPLER_REGISTRY at the bottom). The CLI selects which subset to
run via --samplers.

Usage:
    PYTHONPATH=build/bindings/python uv run --no-sync python \
        examples/python/bench_u_strategies.py \
        --filtration freudenthal --grid-shape 64 64 --grid-shape 128 128 \
        --side hom --strategies all --samplers default --n-pairs 1 4 16 all \
        --n-threads 8 --csv out.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "tests"))
sys.path.insert(0, TESTS_DIR)

import oineus
import oineus.diff as oin_diff

try:
    from data_utils import sample_annulus, random_gaussian_2d  # noqa: E402
except ImportError as e:
    raise SystemExit(
        "Run with PYTHONPATH=build/bindings/python so data_utils.py is on "
        f"sys.path: {e}"
    )


# ---------------------------------------------------------------------
# Filtration generators
# ---------------------------------------------------------------------


def _make_pointcloud(n_points, seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        pts = sample_annulus(n_points=n_points, inner_radius=1.0,
                             outer_radius=2.0, sigma=0.05).astype(np.float64)
    finally:
        np.random.set_state(state)
    return pts


def _make_grid(grid_shape, seed):
    nx, ny = grid_shape
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        arr = random_gaussian_2d(nx, ny, complexity=5).astype(np.float64)
    finally:
        np.random.set_state(state)
    return arr


def build_diff_filtration(kind, size, grid_shape, seed):
    """Returns (data_tensor, DiffFiltration). data_tensor has
    requires_grad=True so loss.backward() flows through fil.values."""
    if kind == "vr":
        pts = _make_pointcloud(size, seed)
        data = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
        fil = oin_diff.vr_filtration(data, max_dim=2)
        return data, fil
    if kind == "alpha":
        # Use weak_alpha (alpha-complex combinatorics, weak-alpha values --
        # the only differentiable variant; the actual alpha values from
        # CGAL aren't differentiable).
        pts = _make_pointcloud(size, seed)
        data = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
        fil = oin_diff.weak_alpha_filtration(data)
        return data, fil
    if kind == "freudenthal":
        arr = _make_grid(grid_shape, seed)
        data = torch.tensor(arr, dtype=torch.float64, requires_grad=True)
        fil = oin_diff.freudenthal_filtration(
            data, max_dim=2, negate=False, wrap=False, n_threads=1)
        return data, fil
    raise ValueError(f"unknown filtration {kind!r}")


# ---------------------------------------------------------------------
# Reference U (precomputed once per scenario, used by U-density samplers)
# ---------------------------------------------------------------------


def precompute_reference_u(under_fil, side, dim_for_u):
    """Build a reference Decomposition with full U on the requested side
    at the right dim. dim_for_u is the *matrix-dim* in that side's
    decomposition where U should be computed:
      - hom (non-dualize): matrix_dim == filtration_dim. For an H_d
        pair, the death-creator is in original dim d+1, so dim_for_u
        for the H_d-needing U is hom_dim+1.
      - coh (dualize): matrix_dim is REVERSED. For an H_d pair the
        birth-creator is in original dim d, which maps to coh matrix
        dim = top_dim - d. The caller is responsible for passing the
        right matrix dim."""
    dualize = (side == "coh")
    decmp = oineus.Decomposition(under_fil, dualize=dualize, n_threads=4)
    p = oineus.ReductionParams()
    p.compute_v = True
    p.compute_u = False
    p.clearing_opt = True
    p.restore_elz = True
    p.n_threads = 4
    decmp.reduce(p)
    decmp.compute_u_from_v_1(dim_for_u, n_threads=4)
    return decmp


def coh_dim_for_hom_dim(under_fil, hom_dim):
    """For an H_{hom_dim} pair the birth-creator's original dim is
    hom_dim. In the cohomology decomposition cells are reversed by
    matrix-index, so original dim d maps to coh matrix dim
    top_dim - d. (top_dim = max_dim of the filtration.)"""
    top_dim = under_fil.max_dim()
    return top_dim - hom_dim


# ---------------------------------------------------------------------
# Pair-selection samplers
# ---------------------------------------------------------------------
#
# Common signature:
#   sampler(dgm_pts, ref_decmp, under_fil, n_pairs, rng, side, kwargs)
#       -> list of (pair_idx_in_dgm_pts, target_value)
#
# The dispatcher converts (pair_idx, target_value) into the right
# tensor entries for the autograd loss according to the sweep's
# direction (death-up vs birth-down).
#
# dgm_pts: list of dicts {b_idx, d_idx, b_value, d_value, persistence}.
# side: "hom" or "coh", determines which walker the move targets.
# ref_decmp: a precomputed Decomposition with u_data_t populated for
#   the relevant dim, used by U-density-aware samplers.


def _persistence_array(dgm_pts):
    return np.array([p["persistence"] for p in dgm_pts])


def _emit_picks(chosen_indices, dgm_pts, direction, eps, rng=None):
    """Build (pair_idx, target_value, axis) tuples for a chosen subset
    of pairs and a chosen direction. axis is 'b' or 'd' indicating
    which simplex of the pair the target applies to. Most directions
    return one entry per pair; 'mixed' returns one entry per pair but
    randomly picks death-up vs birth-down per pair."""
    out = []
    for i in chosen_indices:
        i = int(i)
        p = dgm_pts[i]
        if direction == "death-up":
            out.append((i, p["d_value"] + eps, "d"))
        elif direction == "birth-down":
            tgt = max(p["b_value"] - eps,
                      p["b_value"] - p["persistence"] * 0.99)
            out.append((i, tgt, "b"))
        elif direction == "toward-diagonal":
            # Move death down to birth (death-down, V-only).
            out.append((i, p["b_value"], "d"))
        elif direction == "mixed":
            # Per-pair coin flip between death-up and birth-down so
            # each backward exercises BOTH hom-side U and coh-side U
            # in one call. Models a real Wasserstein-style loss where
            # different pairs need to move in different directions.
            r = rng.random() if rng is not None else (i % 2) * 0.5
            if r < 0.5:
                out.append((i, p["d_value"] + eps, "d"))
            else:
                tgt = max(p["b_value"] - eps,
                          p["b_value"] - p["persistence"] * 0.99)
                out.append((i, tgt, "b"))
        else:
            raise ValueError(f"unknown direction {direction!r}")
    return out


def sampler_top_persistence(dgm_pts, ref_decmp, under_fil, n_pairs,
                            rng, direction, kwargs):
    """Top-N most persistent pairs. The direction parameter picks which
    axis of the pair to move:
      - death-up   : target = death + eps, walker = increase_death (hom-U)
      - birth-down : target = birth - eps, walker = decrease_birth (coh-U)
      - mixed      : per-pair coin flip between death-up and birth-down
    """
    eps_frac = kwargs.get("eps_frac", 0.20)
    if not dgm_pts: return []
    pers = _persistence_array(dgm_pts)
    order = np.argsort(pers)[::-1]
    chosen = order[:n_pairs]
    val_range = max(p["d_value"] for p in dgm_pts) - min(p["b_value"] for p in dgm_pts)
    eps = eps_frac * max(val_range, 1e-6)
    return _emit_picks(chosen, dgm_pts, direction, eps, rng=rng)


def sampler_wasserstein_to_empty(dgm_pts, ref_decmp, under_fil, n_pairs,
                                 rng, direction, kwargs):
    """Move pairs toward the diagonal. Standard topological cleanup
    loss; this is intrinsically toward-diagonal -- the direction
    parameter only picks which axis to move (and 'mixed' alternates
    randomly per pair). Both death-down and birth-up are V-only, so
    U-strategy choice does NOT matter for this sampler."""
    if not dgm_pts: return []
    pers = _persistence_array(dgm_pts)
    order = np.argsort(pers)
    chosen = order[:n_pairs]
    out = []
    for i in chosen:
        i = int(i)
        p = dgm_pts[i]
        if direction == "birth-down":
            # Move birth up to meet death (V-only, "cleanup" semantics).
            out.append((i, p["d_value"], "b"))
        elif direction == "mixed":
            r = rng.random() if rng is not None else (i % 2) * 0.5
            if r < 0.5:
                out.append((i, p["b_value"], "d"))   # death-down
            else:
                out.append((i, p["d_value"], "b"))   # birth-up
        else:
            # death-up / toward-diagonal: move death down to meet birth.
            out.append((i, p["b_value"], "d"))
    return out


def sampler_adversarial_spanning(dgm_pts, ref_decmp, under_fil, n_pairs,
                                 rng, direction, kwargs):
    """Pair targets spread across the entire dim's value range. Forces
    partial cols/rows to span almost the whole dim. Direction-aware:
    death-up spreads upward toward val_max; birth-down spreads downward
    toward val_min; mixed alternates per pair."""
    if not dgm_pts: return []
    chosen = rng.choice(len(dgm_pts), size=min(n_pairs, len(dgm_pts)),
                        replace=False)
    val_max = max(p["d_value"] for p in dgm_pts)
    val_min = min(p["b_value"] for p in dgm_pts)
    val_range = val_max - val_min
    out = []
    for rank, i in enumerate(chosen):
        i = int(i)
        p = dgm_pts[i]
        frac = (rank + 1) / max(len(chosen), 1)
        if direction == "mixed":
            r = rng.random() if rng is not None else (rank % 2) * 0.5
            local = "death-up" if r < 0.5 else "birth-down"
        else:
            local = direction
        if local == "death-up":
            cur = p["d_value"]
            tgt = cur + frac * (val_max + 0.1 * val_range - cur)
            out.append((i, tgt, "d"))
        elif local == "birth-down":
            cur = p["b_value"]
            tgt = cur - frac * (cur - (val_min - 0.1 * val_range))
            out.append((i, tgt, "b"))
        else:
            out.append((i, p["b_value"], "d"))   # toward diagonal
    return out


def sampler_top_u_density(dgm_pts, ref_decmp, under_fil, n_pairs,
                          rng, direction, kwargs):
    """Pairs whose U row (u_data_t at the relevant matrix index) is
    densest. Worst case for per-row methods; each row solve is itself
    expensive. The 'relevant matrix index' depends on direction:
      - death-up:   d_p_filtration (= matrix idx on hom)
      - birth-down: fil_size - b_p_filtration - 1 (matrix idx on coh)
      - mixed:      use the larger of the two density estimates per pair
    The caller is responsible for passing a ref_decmp on the matching
    side; for 'mixed' the caller passes the hom ref_decmp by
    convention (density picked is hom-side; the actual move per pair
    still alternates randomly)."""
    if not dgm_pts or ref_decmp is None: return []
    fil_size = under_fil.size()
    densities = []
    for i, p in enumerate(dgm_pts):
        if direction in ("death-up", "mixed"):
            r = p["d_idx"]
        elif direction == "birth-down":
            r = fil_size - p["b_idx"] - 1
        else:
            r = p["d_idx"]
        if 0 <= r < len(ref_decmp.u_data_t):
            densities.append((len(ref_decmp.u_data_t[r]), i))
    if not densities: return []
    densities.sort(reverse=True)
    chosen = [i for _, i in densities[:n_pairs]]
    eps_frac = kwargs.get("eps_frac", 0.20)
    val_range = max(p["d_value"] for p in dgm_pts) - min(p["b_value"] for p in dgm_pts)
    eps = eps_frac * max(val_range, 1e-6)
    return _emit_picks(chosen, dgm_pts, direction, eps, rng=rng)


def sampler_wasserstein_to_template(dgm_pts, ref_decmp, under_fil, n_pairs,
                                    rng, direction, kwargs):
    """Match the diagram to a template (default: shifted-deaths or
    shifted-births template). Pick the n_pairs largest matched
    displacements; target axis depends on direction. For 'mixed',
    each pair gets a random axis assignment with the matching
    template displacement direction."""
    if not dgm_pts: return []
    n = len(dgm_pts)
    if direction == "death-up":
        targets = [p["d_value"] + 0.3 * p["persistence"] for p in dgm_pts]
        axes = ["d"] * n
        disp = [t - p["d_value"] for t, p in zip(targets, dgm_pts)]
    elif direction == "birth-down":
        targets = [p["b_value"] - 0.3 * p["persistence"] for p in dgm_pts]
        axes = ["b"] * n
        disp = [p["b_value"] - t for t, p in zip(targets, dgm_pts)]
    elif direction == "mixed":
        targets, axes = [], []
        for i, p in enumerate(dgm_pts):
            r = rng.random() if rng is not None else (i % 2) * 0.5
            if r < 0.5:
                targets.append(p["d_value"] + 0.3 * p["persistence"])
                axes.append("d")
            else:
                targets.append(p["b_value"] - 0.3 * p["persistence"])
                axes.append("b")
        disp = [(t - p["d_value"] if a == "d" else p["b_value"] - t)
                for t, a, p in zip(targets, axes, dgm_pts)]
    else:
        targets = [p["b_value"] for p in dgm_pts]
        axes = ["d"] * n
        disp = [p["d_value"] - t for t, p in zip(targets, dgm_pts)]
    order = np.argsort(disp)[::-1]
    chosen = order[:n_pairs]
    return [(int(i), float(targets[i]), axes[i]) for i in chosen]


# Additional samplers (kept in registry; not in default sweep)


def sampler_bottom_persistence(dgm_pts, ref_decmp, under_fil, n_pairs,
                               rng, direction, kwargs):
    if not dgm_pts: return []
    pers = _persistence_array(dgm_pts)
    order = np.argsort(pers)
    chosen = order[:n_pairs]
    out = []
    for i in chosen:
        i = int(i)
        p = dgm_pts[i]
        if direction == "birth-down":
            out.append((i, p["d_value"], "b"))
        elif direction == "mixed":
            r = rng.random() if rng is not None else (i % 2) * 0.5
            if r < 0.5: out.append((i, p["b_value"], "d"))
            else: out.append((i, p["d_value"], "b"))
        else:
            out.append((i, p["b_value"], "d"))
    return out


def sampler_uniform_random(dgm_pts, ref_decmp, under_fil, n_pairs,
                           rng, direction, kwargs):
    if not dgm_pts: return []
    chosen = rng.choice(len(dgm_pts), size=min(n_pairs, len(dgm_pts)),
                        replace=False)
    eps_frac = kwargs.get("eps_frac", 0.10)
    val_range = max(p["d_value"] for p in dgm_pts) - min(p["b_value"] for p in dgm_pts)
    eps = eps_frac * max(val_range, 1e-6)
    return _emit_picks(chosen, dgm_pts, direction, eps, rng=rng)


def sampler_stratified_persistence(dgm_pts, ref_decmp, under_fil, n_pairs,
                                   rng, direction, kwargs):
    if not dgm_pts: return []
    pers = _persistence_array(dgm_pts)
    order = np.argsort(pers)
    n = len(dgm_pts)
    per_q = max(1, n_pairs // 4)
    out_idx = []
    for q_start, q_end in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        s = int(q_start * n)
        e = int(q_end * n)
        bin_idx = order[s:e]
        if len(bin_idx) > 0:
            ch = rng.choice(bin_idx, size=min(per_q, len(bin_idx)),
                            replace=False)
            out_idx.extend(int(i) for i in ch)
    out_idx = out_idx[:n_pairs]
    eps_frac = kwargs.get("eps_frac", 0.15)
    val_range = max(p["d_value"] for p in dgm_pts) - min(p["b_value"] for p in dgm_pts)
    eps = eps_frac * max(val_range, 1e-6)
    return _emit_picks(out_idx, dgm_pts, direction, eps, rng=rng)


def sampler_bottom_u_density(dgm_pts, ref_decmp, under_fil, n_pairs,
                             rng, direction, kwargs):
    if not dgm_pts or ref_decmp is None: return []
    fil_size = under_fil.size()
    densities = []
    for i, p in enumerate(dgm_pts):
        if direction in ("death-up", "mixed"):
            r = p["d_idx"]
        elif direction == "birth-down":
            r = fil_size - p["b_idx"] - 1
        else:
            r = p["d_idx"]
        if 0 <= r < len(ref_decmp.u_data_t):
            densities.append((len(ref_decmp.u_data_t[r]), i))
    if not densities: return []
    densities.sort()
    chosen = [i for _, i in densities[:n_pairs]]
    eps_frac = kwargs.get("eps_frac", 0.10)
    val_range = max(p["d_value"] for p in dgm_pts) - min(p["b_value"] for p in dgm_pts)
    eps = eps_frac * max(val_range, 1e-6)
    return _emit_picks(chosen, dgm_pts, direction, eps, rng=rng)


SAMPLER_REGISTRY = {
    # Default sweep set (5 samplers)
    "top_persistence": sampler_top_persistence,
    "wasserstein_to_empty": sampler_wasserstein_to_empty,
    "adversarial_spanning": sampler_adversarial_spanning,
    "top_u_density": sampler_top_u_density,
    "wasserstein_to_template": sampler_wasserstein_to_template,
    # Extras (registered but not in default sweep)
    "bottom_persistence": sampler_bottom_persistence,
    "uniform_random": sampler_uniform_random,
    "stratified_persistence": sampler_stratified_persistence,
    "bottom_u_density": sampler_bottom_u_density,
}

DEFAULT_SAMPLERS = [
    "top_persistence",
    "wasserstein_to_empty",
    "adversarial_spanning",
    "top_u_density",
    "wasserstein_to_template",
]


# ---------------------------------------------------------------------
# Strategy enumeration
# ---------------------------------------------------------------------


STRATEGY_NAMES = ["legacy_in_band", "col_R", "col_V", "row_full",
                  "row_partial"]
EXTRA_STRATEGIES = ["col_partial"]   # Phase-3, kept for double-checks


# ---------------------------------------------------------------------
# Diagram extraction
# ---------------------------------------------------------------------


def extract_dgm_pts(under_fil, hom_dim):
    """One-shot reduction + diagram extraction. Returns a list of
    {b_idx, d_idx, b_value, d_value, persistence} dicts for finite
    pairs in homology dim hom_dim (death lives in dim hom_dim+1)."""
    decmp = oineus.Decomposition(under_fil, dualize=False, n_threads=1)
    p = oineus.ReductionParams()
    p.compute_v = True
    p.compute_u = False
    p.clearing_opt = True
    p.restore_elz = True
    p.n_threads = 1
    decmp.reduce(p)
    dgm = decmp.diagram(under_fil, include_inf_points=False).in_dimension(
        hom_dim, as_numpy=False)
    out = []
    for q in dgm:
        out.append({
            "b_idx": int(q.birth_index),
            "d_idx": int(q.death_index),
            "b_value": float(q.birth),
            "d_value": float(q.death),
            "persistence": float(q.death - q.birth),
        })
    return out


# ---------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------


def time_call(fn, n_repeat=3, reset=None):
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
class ScenarioResult:
    kind: str
    size_str: str
    fil_size: int
    direction: str    # death-up / birth-down / toward-diagonal
    sides_used: str   # which decomposition U was actually computed on
    hom_dim: int
    sampler: str
    n_pairs_req: int
    n_pairs_used: int
    strategy: str
    n_threads: int
    t_u_stage_ms: float
    t_total_ms: float


def time_strategy_total(build_args, hom_dim, sampler_name,
                        n_pairs_req, direction, strategy, n_threads,
                        ref_decmp, rng_seed):
    """End-to-end cold backward (forward + reduction + U + walks +
    scatter) for one strategy on one (sampler, n_pairs, direction)."""
    rng = np.random.default_rng(rng_seed)

    data, fil = build_diff_filtration(*build_args)
    under_fil = fil.under_fil

    dgm_pts = extract_dgm_pts(under_fil, hom_dim)
    if not dgm_pts:
        return None, None, 0

    sampler = SAMPLER_REGISTRY[sampler_name]
    picks = sampler(dgm_pts, ref_decmp, under_fil, n_pairs_req, rng,
                    direction, {})
    if not picks:
        return None, None, 0

    # Each pick is (pair_idx, target_value, axis) where axis is 'b'
    # or 'd' indicating which simplex of the pair the target applies to.
    # The dispatcher uses axis directly; this lets 'mixed' direction
    # produce per-pair routing.
    contributions = []
    for pick in picks:
        if len(pick) == 3:
            pair_idx, tgt_value, axis = pick
        else:
            pair_idx, tgt_value = pick
            axis = "d" if direction in ("death-up", "toward-diagonal") else "b"
        p = dgm_pts[pair_idx]
        contributions.append((axis, p["b_idx"], p["d_idx"], tgt_value))

    # Build loss
    dgms = oin_diff.persistence_diagram(
        fil, gradient_method="crit-sets-strategy",
        u_strategy=strategy, step_size=1.0,
        conflict_strategy="avg", n_threads=n_threads)
    if hom_dim not in dgms:
        return None, None, 0
    dgm_t = dgms[hom_dim]
    if dgm_t.shape[0] == 0:
        return None, None, 0

    # Map our sampled (b_idx, d_idx) to rows in dgm_t. The diagram tensor's
    # row order matches index_diagram_in_dimension's order (finite pairs).
    # Build a lookup from (b_idx, d_idx) -> row in dgm_t.
    index_dgm = dgm_pts  # same source
    n_dgm = len(index_dgm)
    bd_to_row = {(p["b_idx"], p["d_idx"]): r for r, p in enumerate(index_dgm)}

    # Compute loss: sum over picks of (current - target)^2 on the right axis
    loss = torch.zeros((), dtype=dgm_t.dtype, device=dgm_t.device)
    for sym, b_dgm_idx, d_dgm_idx, tgt_value in contributions:
        row = bd_to_row.get((b_dgm_idx, d_dgm_idx))
        if row is None or row >= dgm_t.shape[0]: continue
        if sym == "b":
            cur_t = dgm_t[row, 0]
        else:
            cur_t = dgm_t[row, 1]
        loss = loss + (cur_t - tgt_value) ** 2

    if loss.requires_grad and float(loss.item()) > 0:
        # cold backward
        t0 = time.perf_counter()
        loss.backward()
        t_total = time.perf_counter() - t0
    else:
        t_total = 0.0

    return contributions, t_total, len(picks)


def time_strategy_u_stage(under_fil, hom_dim, direction, strategy,
                          picks_data, ref_decmp, n_threads):
    """Time only the U-computation step on a freshly reduced
    decomposition. The decomposition's `dualize` and the dim of U are
    derived from the direction:
      - death-up: hom side (dualize=False), U at dim hom_dim+1.
      - birth-down: coh side (dualize=True), U at coh_dim_for_hom_dim.
      - toward-diagonal: V-only, no U is computed; returns 0.
      - mixed: per-pair direction; some picks need hom-U, some need
        coh-U. Time BOTH and report the SUM (representing the
        backward's combined U-stage cost).
    """
    if not picks_data: return 0.0
    if direction == "toward-diagonal":
        return 0.0
    if direction == "death-up":
        return _time_u_stage_one_side(
            under_fil, hom_dim, "hom", strategy, picks_data, n_threads)
    if direction == "birth-down":
        return _time_u_stage_one_side(
            under_fil, hom_dim, "coh", strategy, picks_data, n_threads)
    if direction == "mixed":
        # Split picks by axis ('d' -> hom-U, 'b' -> coh-U). Time each
        # side separately on its own freshly-reduced decomposition,
        # sum the wall times -- the cold backward must do both.
        hom_picks = [p for p in picks_data if p[0] == "d"]
        coh_picks = [p for p in picks_data if p[0] == "b"]
        t_hom = _time_u_stage_one_side(
            under_fil, hom_dim, "hom", strategy, hom_picks, n_threads
        ) if hom_picks else 0.0
        t_coh = _time_u_stage_one_side(
            under_fil, hom_dim, "coh", strategy, coh_picks, n_threads
        ) if coh_picks else 0.0
        return t_hom + t_coh
    return 0.0


def _time_u_stage_one_side(under_fil, hom_dim, side, strategy,
                           picks_data, n_threads):
    """Per-side U-stage timing helper. picks_data should already be
    filtered to the picks that hit this side."""
    if not picks_data: return 0.0
    if side == "hom":
        dualize = False
        u_dim = hom_dim + 1
        partial_axis = "d"
        partial_cmp = "above"
    else:
        dualize = True
        u_dim = coh_dim_for_hom_dim(under_fil, hom_dim)
        partial_axis = "b"
        partial_cmp = "below"

    def reduce_v_only():
        decmp = oineus.Decomposition(under_fil, dualize=dualize,
                                     n_threads=n_threads)
        p = oineus.ReductionParams()
        p.compute_v = True
        p.compute_u = False
        p.clearing_opt = True
        p.restore_elz = True
        p.n_threads = n_threads
        decmp.reduce(p)
        return decmp

    fil_size = under_fil.size()

    if strategy == "legacy_in_band":
        # No separate U step; reduction with compute_u=true produces it
        # in-band. The parallel reducer doesn't support compute_u=true,
        # so we MUST use serial (n_threads=1). This is part of why
        # legacy_in_band loses on big inputs: U is built by a serial
        # reduction.
        def call_with_u():
            decmp = oineus.Decomposition(under_fil, dualize=dualize,
                                         n_threads=1)
            p = oineus.ReductionParams()
            p.compute_v = True
            p.compute_u = True
            p.clearing_opt = False
            p.n_threads = 1
            decmp.reduce(p)
        return time_call(call_with_u, n_repeat=2)

    if strategy == "col_R":
        decmp = reduce_v_only()
        return time_call(lambda: decmp.compute_u_from_v(u_dim, n_threads),
                         n_repeat=2,
                         reset=lambda: setattr(decmp, "u_data_t", []))
    if strategy == "col_V":
        decmp = reduce_v_only()
        return time_call(lambda: decmp.compute_u_from_v_1(u_dim, n_threads),
                         n_repeat=2,
                         reset=lambda: setattr(decmp, "u_data_t", []))
    if strategy == "row_full":
        decmp = reduce_v_only()
        return time_call(lambda: decmp.compute_full_u_rows(under_fil, u_dim, n_threads=n_threads),
                         n_repeat=2,
                         reset=lambda: setattr(decmp, "u_data_t", []))
    if strategy == "row_partial":
        decmp = reduce_v_only()
        rows, bounds = [], []
        for sym, b_idx, d_idx, tgt in picks_data:
            if sym != partial_axis:
                continue
            if side == "hom":
                rows.append(d_idx)
            else:
                rows.append(fil_size - b_idx - 1)
            bounds.append(tgt)
        if not rows:
            return 0.0
        from oineus.diff.persistence_diagram import PHASE4_PARTIAL_THRESHOLD
        dim_size = decmp.dim_last[u_dim] - decmp.dim_first[u_dim] + 1
        if len(rows) / dim_size > PHASE4_PARTIAL_THRESHOLD:
            return time_call(lambda: decmp.compute_full_u_rows(under_fil, u_dim, n_threads=n_threads),
                             n_repeat=2,
                             reset=lambda: setattr(decmp, "u_data_t", []))
        return time_call(lambda: decmp.compute_partial_u_rows(under_fil, rows, bounds, u_dim, cmp=partial_cmp, n_threads=n_threads),
                         n_repeat=2,
                         reset=lambda: setattr(decmp, "u_data_t", []))

    raise ValueError(f"unknown strategy {strategy}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_n_pairs(spec, dgm_size):
    if spec == "all":
        return dgm_size
    return min(int(spec), dgm_size)


def run_sweep(args):
    rng_master = np.random.default_rng(args.seed)
    results: List[ScenarioResult] = []

    if args.filtration in ("freudenthal",):
        size_iter = [(None, tuple(g)) for g in (args.grid_shape or [(64, 64)])]
        size_label = lambda n, g: f"grid={g[0]}x{g[1]}"
    else:
        size_iter = [(n, None) for n in args.n_points]
        size_label = lambda n, g: f"n_pts={n}"

    samplers = (DEFAULT_SAMPLERS if args.samplers == ["default"]
                else args.samplers)
    strategies = STRATEGY_NAMES.copy()
    if args.strategies != "default":
        strategies = args.strategies

    directions = args.direction  # list

    for n, g in size_iter:
        data, fil = build_diff_filtration(args.filtration, n, g, args.seed)
        under_fil = fil.under_fil
        fil_size = under_fil.size()

        hom_dim = args.hom_dim
        dgm_pts = extract_dgm_pts(under_fil, hom_dim)
        if not dgm_pts:
            print(f"# {size_label(n, g)}: no H{hom_dim} pairs, skipping")
            continue
        n_dgm = len(dgm_pts)

        n_pairs_list = [parse_n_pairs(s, n_dgm) for s in args.n_pairs]
        n_pairs_list = sorted(set(n_pairs_list))

        size_str = size_label(n, g)
        print(f"# {size_str} fil_size={fil_size} H{hom_dim}={n_dgm}",
              flush=True)

        # Precompute reference U on each side that any selected
        # direction will exercise (used by U-density samplers).
        # Mixed needs both; the U-density samplers default to the hom
        # ref for mixed, since picking by max(hom_density, coh_density)
        # would double-count.
        ref_decmp_by_direction = {}
        need_hom_ref = any(d in ("death-up", "mixed") for d in directions)
        need_coh_ref = "birth-down" in directions
        ref_hom = ref_coh = None
        if need_hom_ref:
            ref_dim = hom_dim + 1
            print(f"  ref_decmp side=hom u_dim={ref_dim} "
                  f"(top_dim={under_fil.max_dim()}, hom_dim={hom_dim})",
                  flush=True)
            ref_hom = precompute_reference_u(under_fil, "hom", ref_dim)
        if need_coh_ref:
            ref_dim = coh_dim_for_hom_dim(under_fil, hom_dim)
            print(f"  ref_decmp side=coh u_dim={ref_dim} "
                  f"(top_dim={under_fil.max_dim()}, hom_dim={hom_dim})",
                  flush=True)
            ref_coh = precompute_reference_u(under_fil, "coh", ref_dim)
        for direction in directions:
            if direction == "death-up": ref_decmp_by_direction[direction] = ref_hom
            elif direction == "birth-down": ref_decmp_by_direction[direction] = ref_coh
            elif direction == "mixed": ref_decmp_by_direction[direction] = ref_hom
            else: ref_decmp_by_direction[direction] = None

        for direction in directions:
            sides_used = ("hom" if direction == "death-up"
                          else "coh" if direction == "birth-down"
                          else "both" if direction == "mixed"
                          else "none")
            ref_decmp = ref_decmp_by_direction[direction]
            for sampler_name in samplers:
                for n_pairs_req in n_pairs_list:
                    for strategy in strategies:
                        seed_for_run = int(rng_master.integers(0, 2**31))
                        try:
                            picks_data, t_total, n_used = time_strategy_total(
                                (args.filtration, n, g, args.seed),
                                hom_dim, sampler_name, n_pairs_req,
                                direction, strategy, args.n_threads,
                                ref_decmp, seed_for_run)
                        except Exception as e:
                            print(f"  ERROR {sampler_name} n={n_pairs_req} "
                                  f"dir={direction} strat={strategy}: {e}",
                                  flush=True)
                            continue
                        if picks_data is None:
                            continue
                        try:
                            t_u = time_strategy_u_stage(
                                under_fil, hom_dim, direction, strategy,
                                picks_data, ref_decmp, args.n_threads)
                        except Exception as e:
                            print(f"  ERROR u-stage {sampler_name} "
                                  f"strat={strategy}: {e}", flush=True)
                            t_u = 0.0

                        r = ScenarioResult(
                            kind=args.filtration, size_str=size_str,
                            fil_size=fil_size, direction=direction,
                            sides_used=sides_used, hom_dim=hom_dim,
                            sampler=sampler_name, n_pairs_req=n_pairs_req,
                            n_pairs_used=n_used,
                            strategy=strategy, n_threads=args.n_threads,
                            t_u_stage_ms=t_u * 1e3,
                            t_total_ms=t_total * 1e3,
                        )
                        results.append(r)
                        print(f"  {sampler_name:<24} dir={direction:<16} "
                              f"n={n_pairs_req:>5} strat={strategy:<14} "
                              f"u={t_u*1e3:>7.2f}ms total={t_total*1e3:>8.2f}ms",
                              flush=True)

    return results


def write_csv(results, path):
    if not results:
        print(f"# no results to write to {path}")
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "size", "fil_size", "direction", "sides_used",
                    "hom_dim", "sampler", "n_pairs_req", "n_pairs_used",
                    "strategy", "n_threads", "t_u_stage_ms", "t_total_ms"])
        for r in results:
            w.writerow([r.kind, r.size_str, r.fil_size, r.direction,
                        r.sides_used, r.hom_dim, r.sampler,
                        r.n_pairs_req, r.n_pairs_used, r.strategy,
                        r.n_threads,
                        f"{r.t_u_stage_ms:.4f}", f"{r.t_total_ms:.4f}"])
    print(f"# wrote {len(results)} rows to {path}")


def best_strategy_matrix(results, metric="t_total_ms"):
    """Group results by (kind, direction, n_pairs_band, sampler) and
    pick the best strategy by metric."""
    bands = [(0, 4), (4, 16), (16, 64), (64, 256), (256, 10**9)]
    def band_label(n):
        for lo, hi in bands:
            if lo <= n < hi:
                if hi >= 10**9: return f">={lo}"
                return f"{lo}-{hi-1}"
        return "?"

    groups = defaultdict(list)
    for r in results:
        if getattr(r, metric) <= 0:
            continue
        key = (r.kind, r.direction, band_label(r.n_pairs_used), r.sampler)
        groups[key].append(r)

    print()
    print("=" * 100)
    print(f"BEST STRATEGY PER GROUP (metric: {metric})", flush=True)
    print(f"  {'kind':<12} {'side':<5} {'n_pairs':<8} {'sampler':<26} "
          f"{'best':<14} {'time(ms)':>10} {'runner-up':<14} {'gap':>6}",
          flush=True)
    for key in sorted(groups):
        rs = groups[key]
        # Pick the best run per strategy first, then compare strategies.
        # Prevents "runner-up" from being the same strategy at a different size.
        per_strat = defaultdict(list)
        for r in rs:
            per_strat[r.strategy].append(r)
        strat_best = sorted(
            ((min(rl, key=lambda r: getattr(r, metric))) for rl in per_strat.values()),
            key=lambda r: getattr(r, metric))
        if not strat_best:
            continue
        best = strat_best[0]
        if len(strat_best) < 2:
            print(f"  {best.kind:<12} {best.side:<5} {key[2]:<8} "
                  f"{best.sampler:<26} {best.strategy:<14} "
                  f"{getattr(best, metric):>10.2f} {'-':<14} {'-':>6}",
                  flush=True)
        else:
            second = strat_best[1]
            gap = (getattr(second, metric) - getattr(best, metric)) / max(getattr(best, metric), 1e-9)
            print(f"  {best.kind:<12} {best.direction:<16} {key[2]:<8} "
                  f"{best.sampler:<26} {best.strategy:<14} "
                  f"{getattr(best, metric):>10.2f} {second.strategy:<14} "
                  f"{gap*100:>5.1f}%", flush=True)
    import sys as _sys
    _sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filtration",
                   choices=["vr", "alpha", "freudenthal"],
                   default="freudenthal")
    p.add_argument("--n-points", type=int, nargs="+",
                   default=[64, 256, 1024])
    p.add_argument("--grid-shape", type=int, nargs=2, action="append",
                   help="add a grid (nx, ny); pass once per shape")
    p.add_argument("--direction", nargs="+",
                   default=["death-up", "birth-down", "mixed"],
                   choices=["death-up", "birth-down", "toward-diagonal",
                            "mixed"],
                   help="Move direction(s) for the sampled pairs. "
                        "death-up triggers hom-side U via increase_death; "
                        "birth-down triggers coh-side U via "
                        "decrease_birth; toward-diagonal is V-only (no U); "
                        "mixed is per-pair coin flip between death-up and "
                        "birth-down -- exercises BOTH hom- and coh-side U "
                        "in one backward, models a Wasserstein-style loss "
                        "where different pairs need different directions.")
    p.add_argument("--hom-dim", type=int, default=1)
    p.add_argument("--strategies", nargs="+", default="default",
                   help="'default' = the 5 production candidates; or "
                        f"a list from {STRATEGY_NAMES + EXTRA_STRATEGIES}")
    p.add_argument("--samplers", nargs="+", default=["default"],
                   help="'default' or a list of names from "
                        f"{list(SAMPLER_REGISTRY.keys())}")
    p.add_argument("--n-pairs", nargs="+", default=["1", "4", "16", "all"])
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default=None,
                   help="write per-row CSV to this path")
    args = p.parse_args()

    results = run_sweep(args)
    if args.csv:
        write_csv(results, args.csv)
    best_strategy_matrix(results, metric="t_total_ms")
    best_strategy_matrix(results, metric="t_u_stage_ms")


if __name__ == "__main__":
    main()
