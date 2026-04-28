"""
Pure-PyTorch reimplementation of TopologyOptimizer.combine_loss for the
crit-sets backward pass.

Given a flat batch of (simplex_id, target_value) contributions it
produces the conflict-resolved (indices, targets) arrays that the
autograd backward scatters into the gradient. This is the PyTorch
counterpart of the C++ unordered_map-based combine_loss in
include/oineus/top_optimizer.h:454.

Convention: contributions come in two parallel 1-D tensors

    flat_indices: long  (M,)   simplex (sorted) ids, possibly with repeats
    flat_targets: real  (M,)   target value contributed for that id

`combine` aggregates per-simplex contributions according to the
strategy. For Sum the duplicates are kept (the backward scatter-adds
them); Max/Avg/FixCritAvg merge to one (id, target) per unique simplex.

Benchmark policy (see examples/python/bench_combine_critical_sets.py).
On annulus weak-alpha pipelines, n_points in [64, 4096], CPU fp64:

  - On already-flat tensors, this PyTorch combine is 1.4-5x faster
    than C++ combine_loss for Avg/Max.
  - However, flattening from a CriticalSets list-of-tuples into flat
    tensors costs ~10x more than the C++ combine itself, because the
    nanobind conversion produces one Python list per pair and the
    flatten loop iterates them all.
  - Net: while the only handle into the per-pair data is the
    CriticalSets list, the C++ combine_loss wins end-to-end. The
    PyTorch path becomes the winner once the per-pair data is already
    in flat-tensor form (a Phase-2 binding change can deliver that).

Therefore PersistenceDiagramHelper.backward currently calls C++
combine_loss; this module is exposed primarily for the benchmark and
for direct use when the caller already holds flat tensors (e.g. unit
tests, future fused entry points).
"""

from typing import Optional

import numpy as np
import torch

from .. import _oineus


def critical_sets_to_flat(critical_sets, dtype, device):
    """Flatten C++-shaped CriticalSets [(target, [ids]), ...] to two tensors.

    Uses numpy concat + a single tensor copy. Building one torch tensor
    per critical set in a Python loop is ~10x slower for typical sizes.
    """
    if len(critical_sets) == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=dtype, device=device))

    sizes = np.fromiter((len(cs[1]) for cs in critical_sets),
                        dtype=np.int64, count=len(critical_sets))
    total = int(sizes.sum())
    if total == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=dtype, device=device))

    target_per_set = np.fromiter((cs[0] for cs in critical_sets),
                                 dtype=np.float64, count=len(critical_sets))
    flat_idx = np.empty(total, dtype=np.int64)
    pos = 0
    for cs, n in zip(critical_sets, sizes):
        if n:
            flat_idx[pos:pos + n] = cs[1]
            pos += n
    flat_tgt = np.repeat(target_per_set, sizes)

    return (torch.from_numpy(flat_idx).to(device=device),
            torch.from_numpy(flat_tgt).to(dtype=dtype, device=device))


def combine(flat_indices: torch.Tensor,
            flat_targets: torch.Tensor,
            strategy,
            current_values: torch.Tensor,
            target_map: Optional[dict] = None):
    """
    Resolve conflicts and return (indices, targets) tensors.

    flat_indices: (M,) long
    flat_targets: (M,) real
    strategy: _oineus.ConflictStrategy or one of "avg"/"max"/"sum"/"fca".
    current_values: (N,) real, indexed by simplex id; required for "max".
    target_map: dict[int, float], required for "fca". Maps a critical
        simplex's sorted id to its prescribed target value.

    For "sum" the returned arrays preserve duplicates -- the caller is
    expected to scatter_add into the per-simplex gradient.
    """
    strategy = _resolve(strategy)
    if flat_indices.numel() == 0:
        return flat_indices, flat_targets

    if strategy == _oineus.ConflictStrategy.Sum:
        return flat_indices, flat_targets

    unique_ids, inverse = torch.unique(flat_indices, return_inverse=True)
    n_groups = unique_ids.shape[0]
    dtype = flat_targets.dtype
    device = flat_targets.device

    if strategy == _oineus.ConflictStrategy.Avg:
        sums = torch.zeros(n_groups, dtype=dtype, device=device)
        sums.scatter_add_(0, inverse, flat_targets)
        counts = torch.zeros(n_groups, dtype=dtype, device=device)
        counts.scatter_add_(0, inverse, torch.ones_like(flat_targets))
        return unique_ids, sums / counts

    if strategy == _oineus.ConflictStrategy.Max:
        cur = current_values[flat_indices]
        disp = (flat_targets - cur).abs()
        max_disp = torch.full((n_groups,), -1.0, dtype=dtype, device=device)
        max_disp.scatter_reduce_(0, inverse, disp, reduce="amax", include_self=False)
        is_max = disp == max_disp[inverse]
        # Resolve ties deterministically: pick the smallest position in the
        # flat batch among entries that achieve the per-group max.
        sentinel = flat_indices.shape[0] + 1
        positions = torch.arange(flat_indices.shape[0], device=device)
        cand = torch.where(is_max, positions, torch.full_like(positions, sentinel))
        picked = torch.full((n_groups,), sentinel, dtype=positions.dtype, device=device)
        picked.scatter_reduce_(0, inverse, cand, reduce="amin", include_self=True)
        return unique_ids, flat_targets[picked]

    if strategy == _oineus.ConflictStrategy.FixCritAvg:
        if target_map is None:
            raise ValueError("FixCritAvg requires target_map")
        sums = torch.zeros(n_groups, dtype=dtype, device=device)
        sums.scatter_add_(0, inverse, flat_targets)
        counts = torch.zeros(n_groups, dtype=dtype, device=device)
        counts.scatter_add_(0, inverse, torch.ones_like(flat_targets))
        avg = sums / counts
        # Override entries that appear in target_map.
        if target_map:
            override_keys = torch.tensor(list(target_map.keys()), dtype=torch.long,
                                         device=device)
            override_vals = torch.tensor(list(target_map.values()), dtype=dtype,
                                         device=device)
            # Map override_keys to positions in unique_ids via searchsorted.
            sorted_unique, sort_idx = torch.sort(unique_ids)
            pos_in_sorted = torch.searchsorted(sorted_unique, override_keys)
            in_range = pos_in_sorted < sorted_unique.shape[0]
            pos_in_sorted_clamped = pos_in_sorted.clamp(max=sorted_unique.shape[0] - 1)
            hit_mask = in_range & (sorted_unique[pos_in_sorted_clamped] == override_keys)
            if hit_mask.any():
                pos_in_unique = sort_idx[pos_in_sorted[hit_mask]]
                avg[pos_in_unique] = override_vals[hit_mask]
        return unique_ids, avg

    raise ValueError(f"unsupported strategy {strategy!r}")


def _resolve(strategy):
    if isinstance(strategy, _oineus.ConflictStrategy):
        return strategy
    table = {
        "avg": _oineus.ConflictStrategy.Avg,
        "max": _oineus.ConflictStrategy.Max,
        "sum": _oineus.ConflictStrategy.Sum,
        "fca": _oineus.ConflictStrategy.FixCritAvg,
    }
    try:
        return table[strategy.lower()]
    except (AttributeError, KeyError):
        raise ValueError(f"unknown strategy {strategy!r}")
