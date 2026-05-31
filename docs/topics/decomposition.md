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
params.compute_v = True       # we want matrix V (cycle representatives)
params.compute_u = False      # cannot directly compute U with parallel reduction
params.clearing_opt = True
dcmp.reduce(params)

# 4. Extract the diagram
dgms = dcmp.diagram(fil, include_inf_points=True)
print(dgms.in_dimension(0))   # H0
print(dgms.in_dimension(1))   # H1
```

The same pattern works with any {py:class}`oineus.Filtration` -- swap step 1
for {py:func}`oineus.freudenthal_filtration`, {py:func}`oineus.vr_filtration`,
{py:func}`oineus.cube_filtration`, etc. See {doc}`filtrations`.

## What the matrices are

The reduction maintains

$$ R \;=\; D V, R U \;=\; D, \qquad D, R, U, V \in \mathrm{GL}(\mathbb{F}_2). $$

After {py:meth}`oineus.Decomposition.reduce`:

- `dcmp.r_data` -- columns of the reduced boundary matrix $R$.
- `dcmp.v_data` -- columns of $V$, the column operations applied during
  reduction. Populated when `params.compute_v = True`.
- `dcmp.u_data_t` -- rows of the inverse $U$ (such that $D U^{-1} = R$, in
  the standard convention). Populated when `params.compute_u = True`.
- `dcmp.r_as_csc()`, `dcmp.v_as_csc()`, `dcmp.d_as_csc()`,
  `dcmp.u_as_csr()` -- SciPy-compatible sparse views over $\mathbb{F}_2$.

`compute_u = True` cannot be combined with multi-threaded reduction; Oineus
will silently use a single thread if you set both.
## Reduction parameters

{py:class}`oineus.ReductionParams` controls the algorithm. 

- `n_threads` -- threads for the parallel column reduction. Set to `1`
  for deterministic ordering or to debug.
- `clearing_opt` -- skip columns whose row was already paired in a
  lower dimension. Usually a big win; turn it off only to compare with
  literature timings that don't use it.
- `compute_v`, `compute_u` -- see above.

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
