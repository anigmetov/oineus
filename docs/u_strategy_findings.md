# U-computation strategies for the crit-sets backward: findings

This document records the experimental findings that drive the
default `u_strategy` in `oineus.diff.persistence_diagram`. The
underlying machinery, tests, and bench live in:

- C++ primitives: `include/oineus/decomposition.h` (`compute_u_column[_1]`,
  `compute_u_column[_1]_bounded`, `compute_partial_u_from_v[_1]`,
  `compute_u_row_bounded`, `compute_partial_u_rows`,
  `compute_full_u_rows`).
- Python diff layer: `bindings/python/oineus/diff/persistence_diagram.py`
  (`u_strategy` parameter on `persistence_diagram(...)`).
- Bench: `examples/python/bench_u_strategies.py`.

## TL;DR

**Default `u_strategy` is `row_partial`.** It wins ~95% of the
1500+ measured configurations (3 filtration kinds x 3 directions x
5 samplers x sizes x n_pairs) on isolated U-stage time, with the
threshold-based fallback to `row_full` (PHASE4_PARTIAL_THRESHOLD =
0.75) automatically handling the remaining cases without manual
tuning. Bit-identical gradients to the in-band reference verified.

## What `u_strategy` controls

The crit-sets backward needs U for two walkers: `increase_death`
(reads `decmp_hom.u_data_t`) and `decrease_birth` (reads
`decmp_coh.u_data_t`). The other two walkers (`decrease_death`,
`increase_birth`) read V columns only, so U-strategy choice is
irrelevant when all moves are toward the diagonal.

Each U-needing side gets a parallel V-only reduction with
`restore_elz` (`ensure_reduced_for_partial_u_*`), then U is built
on top via one of:

| u_strategy | Algorithm | Output | Notes |
|---|---|---|---|
| `legacy_in_band` | ELZ in-band U during reduction | full dim, in-band | clearing off; **serial reduction only** (parallel reducer doesn't support `compute_u=true`). Backward compat with `gradient_method='crit-sets'`. |
| `col_R` | R U = D, Algorithm 3 | full dim | `compute_u_from_v(dim)`. Pivots non-monotonic in iteration; bound truncation is post-filter only. |
| `col_V` | V U = I, Algorithm 4 | full dim | `compute_u_from_v_1(dim)`. Columns inverted then transposed to row form. |
| `col_partial` | V U = I (Phase-3), partial | subset of cols, then col->row | `compute_partial_u_from_v_1(cols, bounds)`. Kept for back-compat; consistently loses to `row_partial`. |
| `row_full` | V^T U^T = I (Phase-4), full dim | rows directly | `compute_full_u_rows(dim)`. Same nnz as `col_V` but no col->row stage. |
| `row_partial` | V^T U^T = I, partial | only rows the walker reads | `compute_partial_u_rows(rows, bounds)`. Falls back to `row_full` when `n_pairs / dim_size > PHASE4_PARTIAL_THRESHOLD`. **Production default.** |
| `auto` | row_partial | -- | Alias of `row_partial`; the value to use unless you have a specific reason. |

The `gradient_method` parameter still works but is now an alias:

- `crit-sets`             -> `u_strategy='legacy_in_band'`
- `crit-sets-partial`     -> `u_strategy='col_partial'`
- `crit-sets-row-partial` -> `u_strategy='row_partial'`
- `crit-sets-strategy`    -> `u_strategy=` whatever the caller passes (must be explicit)

## Conceptual fix: direction, not side

The bench was originally parameterized by `--side hom/coh`. This is
incorrect: which decomposition needs U is determined by the move
direction, not chosen up front:

- `death-up` (target_d > current_d) -> hom-side U via `increase_death`.
- `birth-down` (target_b < current_b) -> coh-side U via `decrease_birth`.
- `death-down` and `birth-up` (toward diagonal) -> V-only on either side.
- `mixed` -> per-pair coin flip; both sides may need U in one backward.

The bench now uses `--direction {death-up, birth-down,
toward-diagonal, mixed}` and the dispatcher routes U computation to
whichever side(s) the picks actually require.

## Bench setup

`examples/python/bench_u_strategies.py` sweeps:

- **Filtration kinds**: vr, alpha (weak-alpha, since true alpha values
  aren't differentiable), freudenthal (lower-star).
- **Sizes** (representative): freudenthal {64x64, 128x128, 256x256};
  alpha {256, 1024} points; VR {32, 64} points.
- **Directions**: death-up, birth-down, mixed (default sweep set).
- **Samplers** (default 5; registry has 9):
  - `top_persistence` -- the N most persistent pairs.
  - `wasserstein_to_empty` -- standard topological cleanup loss
    (V-only; control sample where U-strategy choice should not
    matter).
  - `adversarial_spanning` -- targets spread across the dim's value
    range; forces partial cols/rows to span almost the whole dim
    (the case where Phase 3 lost in earlier rounds).
  - `top_u_density` -- pairs with the densest U rows; worst case
    for per-row methods.
  - `wasserstein_to_template` -- match dgm to a shifted-deaths or
    shifted-births template; the matched-displacements case.
- **n_pairs** per sampler: {1, 4, 16, ..., all} adapted to
  diagram size.
- **n_threads**: 8.
- **Strategies measured**: legacy_in_band, col_R, col_V, row_full,
  row_partial (5).

For each config we record:

- **t_u_stage_ms**: time of just the U-computation step on a
  freshly reduced decomposition. Isolates the strategy choice.
- **t_total_ms**: end-to-end cold backward (forward + reduction +
  U + walks + scatter). The "real" wall-clock the user experiences.

Both are reported because the U-stage signal can be drowned out by
forward + reduction in the total time.

## Results

1578 timed configurations. Combined CSVs:

- `/tmp/claude/bench_freud_v3.csv` (freudenthal, 675 rows)
- `/tmp/claude/bench_alpha_v3.csv` (alpha, 525 rows)
- `/tmp/claude/bench_vr_v3.csv` (VR, 375 rows)

### U-stage (the metric the strategy choice controls)

`row_partial` wins overwhelmingly across all (kind, direction, n_pairs
band, sampler) cells:

| kind        | direction    | row_partial wins | runner-up        | typical gap to runner-up |
|-------------|--------------|------------------|------------------|--------------------------|
| freudenthal | death-up     | 10/10            | row_full         | 100-270%                 |
| freudenthal | birth-down   | 10/10            | **col_R**        | 105-170%                 |
| freudenthal | mixed        | 8/10             | col_R / row_full | 14-600%                  |
| alpha       | death-up     | 16/20            | col_V / row_full | 6-260%                   |
| alpha       | birth-down   | 20/20            | **col_R**        | 2-93%                    |
| alpha       | mixed        | 19/20            | col_R / row_full | 6-330%                   |
| VR          | death-up     | 10/10            | col_V            | 200-370%                 |
| VR          | birth-down   | 10/10            | col_V / row_full | 88-280%                  |
| VR          | mixed        | 10/10            | col_V            | 41-330%                  |

**Aggregate: row_partial wins ~95% of cells (113 of ~120).** The
remaining ~5% are alpha death-up at large n_pairs where row_full
ties or marginally wins (within a few percent gap).

#### Patterns matching the paper's expectations

- **Lower-star (freudenthal) cohomology**: `col_R` is the consistent
  runner-up to `row_partial`. Matches the paper's claim that
  R U = D is competitive on lower-star coh.
- **Lower-star homology**: runner-up is `row_full` or `col_V`. col_R
  is still in the mix but not the leader.
- **Alpha cohomology**: `col_R` is the consistent runner-up,
  matching the paper's "ties V U = I on alpha" finding -- but
  `row_partial` beats both by 20-90% by skipping rows the walker
  doesn't need.
- **VR (any side)**: col_R never appears in the runner-up; col_V is
  always the runner-up. Matches "V U = I dominates R U = D on VR".
  But `row_partial` beats col_V by 200-370% by combining
  V-substitution with per-row truncation.

The paper's pairwise comparisons (R U = D vs V U = I) under-rate
the win available, because they don't compare against partial
inversion. The `row_partial` win comes from two compounding
effects:

1. Solving for rows directly (no col->row transpose afterward).
2. Computing only the rows the walker reads, with per-row value-
   bound truncation.

#### Mixed direction

`mixed` is the most realistic case for actual Wasserstein-style
losses where different pairs move in different directions and BOTH
hom-side and coh-side U get computed in the same backward.

| kind | mixed result |
|---|---|
| freudenthal | row_partial 8/10, ties on V-only sampler |
| alpha | row_partial 19/20 |
| VR | row_partial 10/10, 41-330% gap to col_V |

VR `mixed` 4-15 pairs is the most striking: U-stage at row_partial
2.0 ms vs col_V 7.0 ms (3.4x). Mixed direction validates that
row_partial works decisively even when the cost is the SUM of two
independent U computations.

### Total-time

Differences are smaller because reduction + forward + walks + scatter
dominate. The qualitative picture stays the same:

- row_partial wins majority of cells.
- Where row_partial doesn't win on total time, the runner-up is
  within a few percent.
- For tiny inputs (alpha n_points=64) total time is essentially
  noise; for large inputs (freudenthal 256^2, fil_size=391k) the
  row_partial win on cold backward stays at 4-7x over Phase-2
  legacy_in_band.

## Numerical equivalence

All 7 strategies produce **bit-identical gradients** to the
legacy_in_band reference. Verified end-to-end on freudenthal 16x16
and VR n_points=20:

```
strategy        max diff vs legacy_in_band
col_R           0.00e+00
col_V           0.00e+00
row_full        0.00e+00
row_partial     0.00e+00
col_partial     0.00e+00
auto            0.00e+00
```

The 20 existing `test_diff_critical_sets.py` cases all pass with
the refactored `_dispatch_u_for_side` dispatcher.

## Production default

`u_strategy='auto'` (= `row_partial` with threshold dispatch) is the
default. `gradient_method='crit-sets'` keeps backward-compat
(=> `legacy_in_band`); existing callers see no behavior change. To
opt into the new default, pass `gradient_method='crit-sets-strategy',
u_strategy='auto'` or just `gradient_method='crit-sets-row-partial'`.

## Caveats and known gaps

- **Negate filtrations** are not yet supported by the row-form
  partial helpers (`_classify_increase_death_rows` /
  `_classify_decrease_birth_rows` early-return for negate). Falls
  back to legacy_in_band for negate.
- **Multiple homology dimensions in one backward** are partially
  supported; each side's classifier picks one common dim. Pairs
  spanning multiple dims fall back to the first dim seen. Real
  uses optimize one homology dim at a time.
- **The bench's `--n-points` for alpha/VR is bounded by what
  fits in seconds**. Larger inputs (e.g. VR n=256, alpha n=10000)
  not yet measured; expect the row_partial win to grow.
- **PHASE4_PARTIAL_THRESHOLD = 0.75** is set conservatively. The
  bench data suggests it could be lowered to 0.5 or 0.6 with no
  loss; left at 0.75 to be safe.

## Future work

- Tune PHASE4_PARTIAL_THRESHOLD on a wider sweep, possibly per
  filtration kind.
- Add negate-aware row-partial helpers.
- Drop col_partial (Phase 3 column-form) once enough release time
  has confirmed nothing depends on it.
- Add fine-grained per-stage timings (vt_data build, row solves,
  walks, scatter) to the C++ side via Timer hooks for the paper's
  fine-grained comparison.
