# Inside the decomposition

The persistence pipeline is

```
filtration   -->   Decomposition   -->   reduce   -->   diagram
```

The one-shot helpers ({py:func}`oineus.compute_diagrams_ls`,
{py:func}`oineus.compute_diagrams_vr`, {py:func}`oineus.compute_diagrams_alpha`)
collapse the middle two stages, but
you can have more control using a more explicit approach. {py:class}`oineus.Decomposition` is the
reduction engine: it holds the boundary matrix $D$ derived from a
{py:class}`oineus.Filtration`, performs the column reduction
$R = D V$, and exposes the resulting matrices.

## The manual workflow

```{code-block} python
import oineus as oin

# 1. Build a filtration (any builder; here, a hand-written one)
simplices = [
    oin.Simplex([0],       0.2),
    oin.Simplex([1],       0.1),
    oin.Simplex([2],       0.3),
    oin.Simplex([0, 1],    0.9),
    oin.Simplex([0, 2],    0.5),
    oin.Simplex([1, 2],    0.8),
    oin.Simplex([0, 1, 2], 1.0),
]
fil = oin.Filtration(simplices, negate=False, n_threads=1)

# 2. Construct a Decomposition object (no reduction performed yet)
dcmp = oin.Decomposition(fil, dualize=False)

# 3. Configure and reduce
params = oin.ReductionParams()
params.n_threads = 2          # parallel reduction
params.compute_v = True       # retain V (also implied by parallel compute_u)
params.compute_u = True       # recover U by parallel VTUT after reduction
params.use_clearing = True
dcmp.reduce(params)

# 4. Extract the diagram
dgms = dcmp.diagram(fil, include_inf_points=True)
print(dgms.in_dimension(0))   # H0
print(dgms.in_dimension(1))   # H1
```

The same pattern works with any {py:class}`oineus.Filtration` -- swap step 1
for {py:func}`oineus.freudenthal_filtration`, {py:func}`oineus.vr_filtration`,
{py:func}`oineus.cube_filtration`, etc. See {doc}`filtrations`.

## The fast path: `oineus.reduce`

When performance is the priority and you mostly want diagrams (or cycle
representatives), skip the explicit `Decomposition(fil)` constructor and use
the one-shot {py:func}`oineus.reduce`:

```{code-block} python
params = oin.ReductionParams()
params.n_threads = 8
params.compute_v = False          # diagram only; True if you also need V
dcmp = oin.reduce(fil, params, dualize=False)
dgms = dcmp.diagram(fil)
```

`oineus.reduce` builds the reduction matrix **directly from the filtration**
and feeds it straight to the parallel reducer, skipping the intermediate
boundary-matrix copies the explicit `Decomposition(fil) + reduce` path makes.
The diagrams are identical; this is simply the recommended default when speed
matters. (The per-phase timings land on the returned decomposition as
`dcmp.timings` -- see {doc}`performance`.) `dualize=True` selects cohomology,
exactly as for `Decomposition`.

Two post-reduce details, both invisible if you only call `dcmp.diagram(fil)`:

- **`compute_v=False`, `compute_u=False`, parallel.** Once the pairing is known the reduced
  columns are freed ("pivots-only" state). The diagram still works -- it reads
  the pivots -- but `dcmp.r_data` / `dcmp.r_as_csc()` then raise a clear error
  instead of returning an empty matrix. Use `compute_v=True`, or the explicit
  `Decomposition(fil) + reduce`, if you need the reduced $R$ itself.
- **`compute_v=True`, `compute_u=False`, parallel.** $R$ and $V$ are kept in a compact working
  form and **materialized lazily**: the first access to `dcmp.r_data` /
  `dcmp.v_data`, pickling, or `sanity_check` reconstructs the at-rest matrices.
  `dcmp.diagram(fil)` does not trigger that, so diagram-only callers never pay
  for it.
- **`compute_u=True`, parallel.** This implies `compute_v=True`: cleared
  columns are Bauer-filled, $V$ is retained, and the full $U=V^{-1}$ is
  recovered by a parallel solve of $V^T U^T=I$. Because VTUT needs the matrix
  immediately, this path returns materialized $R$, $V$, and $U$ rather than a
  lazy working representation.

The serial path (`n_threads=1`) reduces in place and always leaves `r_data`
populated. A fused decomposition does not hold the original boundary $D$, so
`sanity_check` needs it passed explicitly:
`dcmp.sanity_check(fil.boundary_matrix())`.

## What the matrices are

Every stored $R,V$ factorization satisfies

$$ R \;=\; D V. $$

On the parallel `compute_u=True` path, the retained $V$ and computed $U$
additionally satisfy

$$ U=V^{-1}, \qquad R U \;=\; D, $$

where $U$ and $V$ are unit upper-triangular over $\mathbb{F}_2$.

After {py:meth}`oineus.Decomposition.reduce`:

- `dcmp.r_data` -- columns of the reduced boundary matrix $R$.
- `dcmp.v_data` -- columns of $V$, the column operations applied during
  reduction. Populated when `params.compute_v = True`; parallel
  `params.compute_u = True` also computes and retains it automatically.
- `dcmp.u_data_t` -- stored rows of $U$. Parallel `compute_u=True` fills it
  with the complete inverse of the retained $V$.
- `dcmp.r_as_csc()`, `dcmp.v_as_csc()`, `dcmp.d_as_csc()`,
  `dcmp.u_as_csr()` -- SciPy-compatible sparse views over $\mathbb{F}_2$.

Only `has_full_matrix_u() == True` guarantees that the stored rows form a
complete $U$ for the reduction recipe that produced them. In particular, the
critical-set implementation may use `u_data_t` as scratch storage for bounded
row prefixes when `params.compute_u = False`; those internal rows carry no
public completeness guarantee. `n_computed_u_rows` reports row-solve work,
while `n_valid_u_rows` is either zero or the size of a complete $U$.
Post-reduction $U$ solvers still expose their raw row results through `u_row`,
but do not upgrade that global guarantee. Oineus deliberately does not keep a
per-simplex validity bitmap.

With `n_threads > 1`, `compute_u = True` runs reduction first and full VTUT
second. `advanced.dims_to_restore_elz` keeps its independent meaning. If it is
empty, the returned matrices satisfy $R=DV$ and $U=V^{-1}$, but the parallel
reduction's Bauer-filled $V$ need not be in canonical ELZ form. If dimensions
are requested, their ELZ restoration runs before VTUT, so U is the inverse of
the restored V. VTUT always uses the full V, including unrecovered Bauer
columns in dimensions that were not requested for restoration.

## Reduction parameters

{py:class}`oineus.ReductionParams` controls the algorithm. 

- `n_threads` -- threads for the parallel column reduction. Set to `1`
  for deterministic ordering or to debug.
- `use_clearing` -- skip columns whose row was already paired in a
  lower dimension. Usually a big win; turn it off only to compare with
  literature timings that don't use it.
- `compute_v`, `compute_u` -- see above.
- `advanced.col_repr` -- the working-column data structure used during
  reduction. The default is the fastest choice; see
  {doc}`performance` for when (rarely) to change it.

## Cohomology and the `dualize` switch

`Decomposition(fil, dualize=True)` reduces the coboundary matrix (cohomology) 
instead of the bounday matrix (homology). The diagrams are identical, 
but for **VR the dual is normally much faster**. {py:func}`oineus.compute_diagrams_vr`
sets `dualize=True` by default for exactly this reason. For grid filtrations
the choice is less clear-cut; both run.

## Extracting the diagram

```{code-block} python
dgms = dcmp.diagram(fil, include_inf_points=True)
arr  = dgms.in_dimension(1)         # (n, 2) NumPy array
pts  = dgms.in_dimension(1, as_numpy=False)   # list of DiagramPoint
for p in pts:
    p.birth, p.death, p.birth_index, p.death_index
```

`birth_index` and `death_index` are positions in `fil.simplices()` (i.e.,
`sorted_id` values), so you can map every diagram point back to the pair of
cells that created and killed the homology class.

## Zero-persistence diagrams

The standard diagram filters out pairs with `birth == death` -- these are
"zero-persistence" pairs, generated and immediately killed by simplices with
the same filtration value (very common on grids with plateaus, or so-called
apparent pairs in VR filtrations). When you actually want them, Oineus
exposes two routes:

```{code-block} python
# Route 1: just the zero-persistence pairs
zero_dgms = dcmp.zero_pers_diagram(fil)
print(zero_dgms.in_dimension(0))

# Route 2: include them in the regular diagram
params = oin.ReductionParams()
# The flag actually lives on the per-call diagram args / KICR params depending
# on which path you take; for the most common case:
dgms = dcmp.diagram(fil, include_inf_points=True)
# To also include zero-persistence pairs, request them explicitly:
zero_dgms = dcmp.zero_pers_diagram(fil)
```

Use `zero_pers_diagram` when you need them as a separate set (for example, to
verify that an apparent absence of features is matched by the filtration's
zero-pers structure). The {py:class}`oineus.KICRParams.include_zero_persistence`
flag does the analogous thing for kernel/image/cokernel diagrams; see
{doc}`kicr`.

## See also

- {doc}`filtrations` -- where the input `Filtration` comes from.
- {doc}`diagrams_distances` -- what to do with the output `Diagrams`.
- {doc}`performance` -- the reduction is the heavy stage; this is where
  threading and `dualize` matter.
- `examples/python/example_manual.py` -- the full manual snippet in
  executable form.
