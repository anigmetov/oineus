# Performance

Oineus is built around shared-memory parallelism. The expensive stages
(reduction, VR enumeration, large-diagram Hera calls) all expose an
explicit `n_threads` parameter. There is no global thread setting and no
implicit "use every core" -- you choose.

## Reduction

```{code-block} python
import oineus as oin

params = oin.ReductionParams(
    n_threads=8,        # default is 1 -- you choose the thread count
    chunk_size=256,     # advanced knob (params.advanced); rarely worth tuning
    use_clearing=True,  # default; turn off only to compare with literature
    compute_v=False,    # default; True if you need cycle reps
    compute_u=False,    # True adds a parallel full-U solve and retains V
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
- **`use_clearing`** -- on by default. Skip columns paired in lower
  dimensions. The Morozov-Nigmetov SPAA 2020 paper has the details. Turn
  off only for benchmark comparisons with codes that do not implement it.
- **`compute_v` / `compute_u`.** Each one significantly increases the
  memory footprint. With parallel reduction, `compute_u=True` implies
  `compute_v=True`: Oineus Bauer-fills cleared V columns, retains the full V,
  and then computes $U=V^{-1}$ by parallel VTUT. Only enable these outputs
  when you actually need cycle representatives, matrix sanity checks, or
  critical-set / ELZ workflows.

### Column representation (advanced -- you almost never need this)

While a column is being reduced it lives in a transient *working* data
structure; the stored columns themselves are always sorted integer
vectors. `ReductionParams.advanced.col_repr` selects that working structure. The
options follow the accelerated column representations of the PHAT library
([Bauer et al., 2017](https://doi.org/10.1016/j.jsc.2016.03.008)):

- **`BitTree`** (default) -- a hierarchical 64-ary dense bit-set (PHAT's
  `bit_tree`). Fastest on essentially every input we have measured and the
  best behaved at high thread counts.
- **`Full`** -- a dense bit-set paired with a max-heap. Within ~15% of
  `BitTree`, occasionally a hair faster on the homology of large, sparse
  grid filtrations. A fine alternative default.
- **`Heap`** -- a lazy max-heap. Lower constant memory than the dense
  options and ~2x faster than `Set`, but slower than `Full` / `BitTree`.
- **`Set`** -- a `std::set`. The simplest representation and the previous
  default; kept for comparison and reproducibility. Typically 2.5-7x
  slower than `BitTree`.

```{code-block} python
# The default (BitTree) is right for almost everyone; this is opt-in tuning.
params = oin.ReductionParams(col_repr=oin.ColumnRepr.Full)
```

This is a knob for experts and benchmarking. Reach for `Full` only to
sanity-check that `BitTree` is not pathological on an unusual input (the
two should agree to within ~15%), or `Set` to reproduce results from
before the knob existed. The dense representations (`Full`, `BitTree`)
allocate roughly one bit per cell per worker thread for the working
column -- negligible next to the boundary matrix itself.

## Fused reduction and the timing breakdown (advanced)

{py:func}`oineus.reduce` is the *fused* one-shot path (see
{doc}`decomposition` for the user-facing view). "Fused" means the filtration
builds the parallel reducer's working-column array **directly**: there is no
intermediate at-rest boundary matrix and no prepare-copy. Relative to the
explicit `Decomposition(fil) + reduce`, it drops two $O(\mathrm{nnz})$ column
copies (boundary $\to D \to R$) and, for the parallel $R{+}V$ path, the
copy-back of $R/V$.

### Reading the per-phase timings

After `reduce`, the decomposition's read-only field `dcmp.timings`
({py:class}`oineus.ReductionTimings`) holds the wall-clock breakdown in
seconds:

| phase | meaning | nonzero when |
|---|---|---|
| `prepare` | build the working atomic-pointer column array | parallel only |
| `reduce` | the lock-free reduction core itself | always |
| `bauer` | dedicated eager fill of cleared V columns from paired R columns | parallel R+V when keeping/restoring working columns |
| `restore_elz` | restore the canonical ELZ form of $V$ | only if `dims_to_restore_elz` is set |
| `copy_back` | move the working columns back into `r_data`/`v_data` | parallel, *materializing* paths |
| `copy_pivots` | copy the pivot array into the at-rest `_pivots` | parallel only |
| `compute_u` | solve $V^T U^T=I$ for every row | parallel `compute_u=True` |

`dcmp.timings.reduction_total` (synonym: `dcmp.timings.total`) is the
path-comparable sum. The serial path reduces in place, so it has no
`prepare` / `copy_back` / `copy_pivots`. Pass `verbose=True` for a printed
trace as well.

What the fuse changes in this breakdown:

- The boundary build is folded into `prepare`; there is no separate "build
  $D$, copy into $R$" cost before the reduction starts.
- For the parallel keep-working $R{+}V$ path -- the default of
  {py:func}`oineus.reduce` with `compute_v=True`, and of the topology
  optimizer -- **`copy_back` is ~0**: the reduced columns are kept in the
  working form and materialized into `r_data`/`v_data` only lazily, on first
  access. That materialization runs *after* `reduce` returns and so is not part
  of these timings.
- Parallel `compute_u=True` is different: VTUT needs V immediately, so R and V
  are materialized before the solve and `compute_u` records the full transpose
  plus row-solve time. The plain copy-back performs any required Bauer fills;
  any requested `restore_elz` phase runs first.

On realistic inputs the boundary build dominates `fil -> reduced`; everything
above it is the reduction-side plumbing the fuse trims (do not assume the
reduction core itself dominates -- it usually does not).

### Measured

2D Freudenthal grid, 8 threads, vs `Decomposition(fil)+reduce`:

- {py:func}`oineus.reduce` diagram-only (R): ~1.5x faster.
- {py:func}`oineus.reduce` with `compute_v=True` (R+V): ~1.5x faster -- both
  the prepare-copy and the copy-back are gone.

The topology optimizer ({doc}`optimization`, {doc}`differentiable`) reduces
through the same fused, keep-working path by default and reads $R$/$V$/$U$
straight out of the working form, so its forward reduction inherits these
savings without ever materializing the full matrices.

### Why the stored working-column type is fixed

`col_repr` (above) selects only the per-thread *scratch* used while reducing a
column; the columns actually *stored* in the working array are always sorted
integer vectors, independent of `col_repr`. That is what lets the fused path
and the keep-working optimizer hand one working array to any `col_repr` core
and keep it around afterwards.

### Apparent pairs on grids (`use_apparent_pairs`)

On complete cubical / Freudenthal grid filtrations, `use_apparent_pairs`
leaves the *apparent* (Bauer) columns -- typically 43-50% of all columns --
out of the parallel working matrix and resolves them on demand. It is
tri-state:

- `None` (default, auto) -- ON only where measured as a pure win: the
  parallel diagram-only (`compute_v=False`) **homology** reduction of a
  **cubical** grid, where it ties or wins on wall time *and* sets the
  peak-RSS floor (~-20% vs plain R-only, ~-40% vs R+V at $128^3$).
- `True` -- force ON wherever supported. Peak RSS drops against the
  same-mode baseline everywhere (-7..-22% for diagram-only runs,
  -15..-31% for R+V on smooth fields), at a wall cost: ~1.8-2.4x on
  cohomology with `compute_v=True`, ~1.4x on Freudenthal homology at
  $128^3$. Worth it for memory-bound runs that need $V$ (the optimizer)
  or Freudenthal grids; avoid it for wall-sensitive cohomology runs.
- `False` -- never.

The flag is inert at `n_threads=1` and on VR / user filtrations, and the
diagrams are identical either way.

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
