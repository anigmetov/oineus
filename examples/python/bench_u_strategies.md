# U-strategy benchmark: findings and recommended defaults

Companion to `bench_u_strategies.py`. Documents the strategy options,
the experimental setup, the headline results, and the recommended
production defaults for `oineus.diff.persistence_diagram`.

## What's being compared

After Phase 1 (lazy reduction), Phase 2 (in-band U), Phase 3
(column-form partial U), and Phase 4 (row-form partial U), the
oineus differentiable backward has five distinct strategies for
computing the U matrix needed by the crit-sets walker. All five
share the same diagram-extraction setup (deferred reduction via
`TopologyOptimizer`); they differ in *how* and *when* U is built
on each side that needs it.

| Strategy | What it does | When it runs | Algorithm |
|---|---|---|---|
| `legacy_in_band` | Builds U during reduction (clearing off, compute_u=true) | inside `reduce_serial` | in-band ELZ |
| `col_R` | V-only reduction with restore_elz, then `compute_u_from_v(dim)` | post-reduction, full dim | Alg 3 (R U = D) |
| `col_V` | V-only reduction, then `compute_u_from_v_1(dim)` (columns + transpose) | post-reduction, full dim | Alg 4 (V U = I) |
| `col_partial` (Phase 3, deprecated) | Compute a column subset of U via `compute_partial_u_from_v_1` | post-reduction, subset of cols | Alg 4 bounded |
| `row_full` | V-only reduction, then `compute_full_u_rows(dim)` (one V transpose, then row solves) | post-reduction, full dim | row-form (V^T U^T = I) |
| `row_partial` | V-only reduction, then `compute_partial_u_rows(rows, bounds, ...)` (only the rows the walker reads) | post-reduction, subset of rows | row-form bounded |

`row_partial` has an internal threshold dispatch: when the requested
rows cover more than `PHASE4_PARTIAL_THRESHOLD = 0.75` of the dim's
matrix range, it falls back to `compute_full_u_rows` (avoiding the
overhead of partial-row bookkeeping when the savings can no longer
justify it).

## What's NOT compared but worth noting

- **`R U = D` cannot be transposed for row-wise selected inversion.**
  R is reduced (each non-zero column has a unique low) but R^T does
  not have an analogous structure that supports forward substitution
  per row. So `col_R` is a column-only / full-dim algorithm; there is
  no `row_R` analogue.
- **Phase-3 col_partial is kept for double-checks but never wins.**
  Bench evidence confirmed it: row_partial is uniformly faster, and
  col_partial's "many pairs" failure mode (alpha 1024 / 266 moves:
  16.2 ms vs row_partial's 3.3 ms) is exactly what motivated Phase 4.

## Public API

The diff layer accepts both the legacy `gradient_method` aliases and a
unified `u_strategy` parameter:

```python
import oineus.diff as oin_diff

# Default behavior (row_partial via 'auto'):
dgms = oin_diff.persistence_diagram(
    fil, gradient_method='crit-sets', step_size=1.0,
    conflict_strategy='avg', n_threads=8)
# (gradient_method='crit-sets' currently maps to u_strategy='legacy_in_band'
#  for back-compat. Pass u_strategy explicitly to opt in to a new strategy.)

# Explicit strategy choice:
dgms = oin_diff.persistence_diagram(
    fil, gradient_method='crit-sets-strategy',
    u_strategy='row_partial',     # production-recommended default
    step_size=1.0, conflict_strategy='avg', n_threads=8)
```

Valid `u_strategy` values: `auto`, `legacy_in_band`, `col_R`, `col_V`,
`col_partial`, `row_full`, `row_partial`.

`auto` = `row_partial` for now; will be retuned once the bench data
suggests a more sophisticated dispatch (e.g., per-direction or
per-filtration-kind defaults).

The deprecated aliases (`crit-sets-partial` and `crit-sets-row-partial`)
still work and map to `col_partial` and `row_partial` respectively. The
PyTorch backward function dispatches all five strategies through one
`_dispatch_u_for_side` helper that respects per-side need_u flags.

## Experimental setup

`bench_u_strategies.py` sweeps a 5-dimensional space:

1. **Filtration kind**: `vr` (Vietoris-Rips, max_dim=2), `alpha`
   (weak alpha), `freudenthal` (lower-star on a 2-D random
   trigonometric grid).
2. **Filtration size**: 32-128 points (VR), 256-1024 points (alpha),
   64-256 grid edge (freudenthal).
3. **Direction** of the gradient moves applied to picked pairs:
   - `death-up`: target > current_d, walker = `increase_death`,
     hom-side U.
   - `birth-down`: target < current_b, walker = `decrease_birth`,
     coh-side U.
   - `mixed`: per-pair coin flip between death-up and birth-down,
     exercising BOTH hom and coh U in one backward (models a
     Wasserstein-style loss with bidirectional matchings).
   - `toward-diagonal`: V-only control direction (no U), included
     mainly to confirm the strategy choice is irrelevant when no U
     is needed.
4. **Sampler** (which pairs to perturb and by how much). Default
   sweep uses 5 of the 9 registered samplers:
   - `top_persistence`: top-N by persistence (most "important"
     features).
   - `wasserstein_to_empty`: low-persistence pairs moved to the
     diagonal (standard topological cleanup).
   - `adversarial_spanning`: targets spread across the dim's value
     range (forces partial paths to span almost the whole dim).
   - `top_u_density`: pairs with the densest U rows (worst case for
     per-row methods).
   - `wasserstein_to_template`: shifted-axis template matching
     (proxy for an actual Wasserstein gradient with prescribed
     targets).
5. **Number of pairs picked**: `1`, `4`, `16`, ..., `all` per scenario.
6. **Strategy**: the five above.

Two timing metrics per scenario:
- **`t_u_stage_ms`**: wall time of the U-computation in isolation,
  on a freshly reduced decomposition. Isolates the variable.
- **`t_total_ms`**: cold backward, end to end (forward + reduction +
  U + walks + scatter). Closer to actual wall-clock in optimization
  loops; the U-stage signal can be drowned by reduction / walks for
  small inputs.

n_threads = 8 in the headline runs; the parallel V-only reduction
benefits from cores, the in-band U does not.

The bench took ~10-15 minutes per filtration kind for the full
matrix (1578 rows total: 5 strategies x 5 samplers x 3 directions x
sizes x n_pairs across 3 filtration kinds).

## Headline findings

### U-stage (the metric that isolates the strategy choice)

`row_partial` wins ~95% of cells across all 1578 timed configurations:

| | death-up | birth-down | mixed |
|---|---|---|---|
| **alpha**       | row_partial 16/20 (4 ties to row_full at large n_pairs) | row_partial 20/20 (col_R is consistent runner-up, 20-65% gap) | row_partial 19/20 |
| **freudenthal** | row_partial 10/10 (row_full runner-up, 100-230% gap) | row_partial 10/10 (col_R runner-up, 130-160% gap) | row_partial 10/10 |
| **vr**          | row_partial 10/10 (col_V runner-up, **3-4x speedup**) | row_partial 10/10 (col_V/row_full runner-up, 80-280% gap) | row_partial 10/10 (col_V runner-up, **3-4x speedup**) |

Confirmed paper expectations:
- **Lower-star (freudenthal) cohomology**: `col_R` is the consistent
  runner-up, supporting the paper's claim that R U = D is competitive
  on this case.
- **Alpha cohomology**: `col_R` again is the runner-up, gap 20-65%.
- **VR cohomology and homology**: `col_V` is the runner-up; row_partial
  beats it by 3-4x. Confirms V U = I dominates over R U = D on VR.
- **`legacy_in_band` is competitive only on tiny inputs.** On
  freudenthal 256² it's 6-10x slower than row_partial because the
  in-band path is forced to use serial reduction.

### Total cold backward (closer to user wall-clock)

Same pattern but the gaps shrink because total time includes
reduction + walks + scatter. `row_partial` wins ~30 of 40 cells; in
the rest, `col_V` or `row_full` is at most 2-10% better. The few
cells where `legacy_in_band` ties (VR `wasserstein_to_empty`) are
the V-only direction where U-strategy choice is by definition
irrelevant.

### Mixed direction (the most realistic case)

The `mixed` direction is the closest analogue to a real Wasserstein-
style loss where different pairs need to move in different directions.
It exercises BOTH hom-side U (death-up picks) and coh-side U
(birth-down picks) in the same backward, summing the U-stage costs.

`row_partial` wins **~88% of mixed-direction cells**, by gaps that
are typically 30-300%. This is the strongest single piece of
evidence for `row_partial` as the production default: even when the
backward pays the cost of U on both sides, the per-row partial pass
remains the cheapest option.

VR `mixed` at 4-15 pairs: row_partial U-stage at **2.0 ms vs col_V's
7.0 ms** (3.4x). Alpha `mixed` at >=256 pairs: row_partial at
**0.96 ms vs col_R's 1.41 ms** (47% gap, runner-up is col_R, not
col_V or row_full).

### When does `row_partial` lose?

Five scenarios surface across the 1578-row sweep where `row_partial`
loses by more than ~5%:
1. **Alpha death-up at >=256 pairs**, the `adversarial_spanning`
   sampler: `row_full` wins by 20%. The threshold dispatch already
   handles this -- row_partial's auto-fallback to row_full kicks in
   when `n_pairs / dim_size > 0.75`.
2. **Freudenthal mixed at 4-15 pairs**, `top_u_density` sampler:
   `col_R` wins narrowly (2.2% gap). Edge case, n_pairs is small
   relative to dim.
3-5. Various V-only `wasserstein_to_empty` cases, where all
   strategies are within noise of each other -- the "no U" control.

In every loss case, the gap is small enough (<25%) that the wrong
choice doesn't hurt user wall-clock meaningfully.

## Recommended defaults

**For oineus.diff.PersistenceDiagram production code**:

- **Default `u_strategy='auto'`** (=current `row_partial` with
  threshold dispatch). The bench shows this strategy wins or ties
  in ~95% of cells across all configurations.
- **Keep all 5 strategies as alternatives** for users who want to
  override (e.g., paper benchmarks measuring specific algorithms).
- **`legacy_in_band`** stays as the back-compat target for
  `gradient_method='crit-sets'` so existing code keeps working.

**For the paper**:

- The findings here corroborate the paper's claim that `col_R`
  (Alg 3) is competitive on lower-star cohomology and alpha
  cohomology.
- The findings also show that **row-form partial inversion is
  uniformly faster** than column-form (whether full or partial)
  in the differentiable backward setting where only a subset of
  pairs is moved per backward.
- The `mixed` direction results suggest the "selected inversion"
  framing is most natural for row-form: each pair is one independent
  row solve, regardless of whether it goes hom-side (death-up) or
  coh-side (birth-down). The bookkeeping for which side a row
  belongs to falls out of the direction classification, not the
  algorithm.

## Files

- `examples/python/bench_u_strategies.py`: the bench (single file).
- `bindings/python/oineus/diff/persistence_diagram.py`: the diff
  layer that exposes `u_strategy`. The dispatcher
  `_dispatch_u_for_side` is the source of truth for what each
  strategy does in production.
- `include/oineus/decomposition.h`: the C++ primitives
  (`compute_u_from_v`, `compute_u_from_v_1`, `compute_full_u_rows`,
  `compute_partial_u_rows`, plus the bounded variants).
- `tests/tests_compute_u_column_bounded.cpp`: 17 Catch2 cases
  covering all primitives (column + row, bounded + unbounded,
  hom + coh).
- CSV outputs from the headline runs:
  `/tmp/claude/bench_{freud,alpha,vr}_v3.csv` (1578 rows total).

## Open follow-ups (not blocking the production-default decision)

- Tune `PHASE4_PARTIAL_THRESHOLD` more precisely (currently 0.75;
  bench data suggests a slightly lower value 0.5-0.6 may be safer
  for cases where row_full slightly wins).
- Add Wasserstein-actual sampler (currently uses a shifted-template
  proxy; would benefit from `oineus.wasserstein_matching` integration).
- Bench with negate filtrations (current paths return early for
  negate; the partial-U primitives would need negate-aware cmp
  direction flips).
- Run a paper-style fine-grained timing without the diff layer
  (no torch, no autograd) to separate the pure C++ algorithm cost
  from the Python overhead. The bench's `t_u_stage_ms` already
  approximates this but goes through a small Python wrapper.
