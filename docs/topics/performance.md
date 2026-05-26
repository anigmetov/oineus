# Performance

Oineus is built around shared-memory parallelism. The expensive stages
(reduction, VR enumeration, large-diagram Hera calls) all expose an
explicit `n_threads` parameter. There is no global thread setting and no
implicit "use every core" -- you choose.

## Reduction

```{code-block} python
import oineus as oin

params = oin.ReductionParams(
    n_threads=8,        # default
    chunk_size=256,     # default; rarely worth tuning
    clearing_opt=True,  # default; turn off only to compare with literature
    compute_v=False,    # default; True if you need cycle reps
    compute_u=False,    # default; cannot be combined with n_threads > 1
)
dcmp = oin.Decomposition(fil, dualize=True)
dcmp.reduce(params)
```

What actually matters:

- **`n_threads`.** Persistence reduction parallelizes well up to ~8-16
  threads for most realistic inputs and then plateaus -- the late stages
  serialize. For very small filtrations (under ~10^5 cells) the overhead
  of spinning up workers dominates; drop to 1 thread.
- **`dualize=True`** (cohomology reduction). For VR and other
  high-dimensional simplicial filtrations the cohomology boundary is
  much sparser; reducing it can be 2-10x faster on the same cells.
  {py:func}`oineus.compute_diagrams_vr` defaults to `dualize=True` for
  exactly this reason. For grid filtrations the choice is less clear-cut;
  both run.
- **`clearing_opt`** -- on by default. Skip columns paired in lower
  dimensions. The Morozov-Nigmetov SPAA 2020 paper has the details. Turn
  off only for benchmark comparisons with codes that do not implement it.
- **`compute_v` / `compute_u`.** Each one significantly increases the
  memory footprint (you are storing a full $V$ or $U$ matrix alongside
  $R$). Only enable them when you actually need cycle representatives,
  matrix sanity checks, or critical-set / ELZ workflows.

## Filtration construction

The filtration builders themselves can be the bottleneck on dense inputs:

- **`vr_filtration`** -- cap `max_dim` and `max_diameter` aggressively.
  The cell count grows as $\binom{n}{k+1}$; halving the diameter often
  cuts the cells by an order of magnitude.
- **`freudenthal_filtration` / `cube_filtration`** -- the `n_threads`
  argument parallelizes the sort. For a $256^3$ volume this is a
  meaningful speedup; for small grids leave it at 1.

## Diagram distances

Hera (vendored under `extern/hera/`) backs both
{py:func}`oineus.bottleneck_distance` and
{py:func}`oineus.wasserstein_distance`. Empirical complexity on our
diagrams (this is the empirical scaling, not the worst-case bound):

- **Wasserstein** -- geometric auction, approximately $O(n^{1.6})$.
- **Bottleneck** -- approximately $O(n^{1.2})$.

In practice this means that for diagrams under ~10^6 finite points, both
calls return in seconds. Loosen `delta` (the relative-error tolerance)
before you reach for parallelism.

## Memory

The dominant memory cost is the boundary matrix plus $V$/$U$ if requested.
For a VR filtration on $n$ points with `max_dim=2`, expect
$\approx 8 \cdot \binom{n}{3}$ bytes for $D$ alone. The cubical builder
on a $d$-array of shape $(N_1, \dots, N_d)$ produces on the order of
$N_1 \cdots N_d \cdot 2^d$ cells; for a $256^3$ float64 volume that is
already ~3 GB of cells before $R$ and $V$.

If you are tight on memory:

- Lower `max_dim` / tighten `max_diameter`.
- Set `compute_v = compute_u = False`.
- Pick `dualize` deliberately -- cohomology reduction is sparser in
  practice.
- Consider one-shot helpers ({py:func}`oineus.compute_diagrams_vr`,
  etc.), which free intermediate state earlier than holding onto a
  long-lived `Decomposition` object.

## Diagnostics

Pass `verbose=True` to `ReductionParams` and `KICRParams` to get a
per-stage trace of cell counts and times. This is the fastest way to
identify which stage is the bottleneck on your input.

## Ctrl-C safety

All long-running Python entry points (reduction, filtration construction,
Hera calls, KICR) respond to Ctrl-C within tens of milliseconds via a
cooperative-cancellation hook. Interrupted work raises
`KeyboardInterrupt` rather than wedging.

## See also

- {doc}`decomposition` -- the parameters live on `ReductionParams`.
- {doc}`differentiable` -- the differentiable forward shares the same
  reduction; everything here applies to the diff path too.
- The Morozov-Nigmetov SPAA 2020 paper for the parallel algorithm.
