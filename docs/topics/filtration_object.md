# Working with a filtration

Every builder in {doc}`filtrations` -- alpha, VR, Freudenthal, cubical, or
custom -- returns the same kind of thing: an opaque *filtration* object
that you hand off to a decomposition, a one-shot helper, or a
differentiable layer. Most of the time you do not need to do anything to
it before passing it along. When you do, this page collects the methods
worth knowing.

For brevity, examples below build a small simplicial filtration with
{py:func}`oineus.list_to_filtration`. The methods shown work identically
on filtrations produced by any other builder; the only differences for
cubical and product filtrations are noted at the end.

## A running example

```{code-block} python
import numpy as np
import oineus as oin

#   2 ---- 0.8 ---- 1
#    \              /
#     0.5        0.9
#       \        /
#        \      /
#         \    /
#          \  /
#           0
data = [
    (0, [0],       0.2),
    (1, [1],       0.1),
    (2, [2],       0.3),
    (3, [0, 1],    0.9),
    (4, [0, 2],    0.5),
    (5, [1, 2],    0.8),
    (6, [0, 1, 2], 1.0),
]
fil = oin.list_to_filtration(data)
```

## Inspecting cells

```{code-block} python
print(fil.size())                 # 7
print(fil.size_in_dimension(1))   # 3 edges
print(fil.max_dim)                # 2 (highest cell dimension)

for sigma in fil.cells():
    print(sigma)                  # human-readable form

first_cell = fil.cell(0)          # index in filtration order
print(first_cell.vertices, first_cell.value)
```

`cells()` returns a copy of every cell in filtration order. For
simplicial filtrations the alias `simplices()` is also available.

Indices into the filtration:

- `fil.cell(i)` -- cell at filtration-order index `i` (this is the
  `sorted_id`).
- `fil.cell_value_by_sorted_id(i)` -- just the filtration value.
- `fil.id_by_sorted_id(i)` -- the `id` you supplied when building.
- `fil.sorted_id_by_id(id)` -- inverse lookup.

Diagram points reference cells by their filtration-order index, so this
is the right side of every "which cell killed this class?" question.

## Filtering and subsetting

A *subfiltration* keeps a subset of cells (which must itself be a valid
filtration -- if you keep a triangle, you keep its edges and vertices).
Pass any Python callable that decides which cells survive:

```{code-block} python
# Keep cells with value <= 0.8.
sub = fil.subfiltration(lambda cell: cell.value <= 0.8)
print(sub.size())                 # 5 (drops the [0,1] edge and the triangle)
```

This is the natural way to slice a filtration at a threshold without
re-building from scratch. The result is itself a filtration with
`.is_subfiltration_` set internally; you can reduce it like any other.

## Bulk value updates

Setting new filtration values on the existing cells (for example, after a
gradient step in a topology-optimization loop):

```{code-block} python
new_values = np.asarray([0.0, 0.0, 0.0, 0.7, 0.4, 0.6, 0.9])
fil.set_values(new_values, n_threads=1)
```

The array must have length `fil.size()` and is indexed by filtration
order (`sorted_id`). After `set_values`, the filtration is re-sorted to
keep the cells in non-decreasing value order, which means the old
filtration-order indices into `new_values` no longer point at the same
cells.

## Boundary and coboundary matrices

```{code-block} python
D    = fil.boundary_matrix(n_threads=1)            # full boundary matrix
D1   = fil.boundary_matrix_in_dimension(1)         # only the d_1 block
Dco  = fil.coboundary_matrix(n_threads=1)          # cohomology
```

Each returns a sparse Z_2 column matrix (a list of columns, each a list
of row indices). For SciPy-friendly views, use
{py:func}`oineus.to_scipy_matrix`. After reducing a {py:class}`oineus.Decomposition`
the matrices you typically want are `R`, `V`, `U` on the
`Decomposition` object itself (see {doc}`decomposition`); the methods here
are for accessing the unreduced boundary of the filtration.

## Filtration-order arithmetic

Building a mapping cylinder, a relative pair, or a product filtration
sometimes needs a "value that is strictly earlier (or later) than every
cell". Hard-coding `-inf` / `+inf` is wrong when the data already
contains `-inf`; use the filtration's sentinels:

```{code-block} python
fil.neg_infinity()      # earlier than every cell
fil.infinity()          # later than every cell
fil.fil_min(a, b)       # the one that enters earlier (respects negate)
fil.fil_max(a, b)       # the one that enters later
```

`fil_min` / `fil_max` know whether the filtration is sublevel or
superlevel, so they always return "the right one" rather than the
numerical min / max. {py:func}`oineus.mapping_cylinder` and
{py:func}`oineus.multiply_filtration` use these internally.

## Sorting permutations

```{code-block} python
perm     = fil.sorting_permutation()       # original_id -> sorted_id
inv_perm = fil.inv_sorting_permutation()   # sorted_id -> original_id
```

Mostly useful when you have parallel data (gradients, weights, ...)
indexed by the `id` you supplied at construction time and want to align
it with the filtration-order indices returned by the decomposition.

## Cubical filtrations

A cubical filtration built with {py:func}`oineus.cube_filtration` has the
same surface: `cells()`, `size()`, `size_in_dimension(d)`, `max_dim`,
`boundary_matrix(...)`, `subfiltration(predicate)`, `set_values(...)`,
`neg_infinity()` / `infinity()` / `fil_min` / `fil_max`. The only
difference is that the objects yielded by `cells()` are cubical cells
(with a vertex set living on the grid) rather than simplices.

```{code-block} python
import numpy as np
import oineus as oin

img = np.array([[1.0, 2.0, 1.5],
                [0.5, 3.0, 2.5],
                [2.0, 1.0, 0.8]])

fil = oin.cube_filtration(img, negate=False, values_on="vertices")
print(fil.size(), fil.max_dim)

low = fil.subfiltration(lambda c: c.value <= 1.5)
```

The decomposition machinery and the one-shot helpers work the same way
for cubical filtrations; nothing on the user side cares about the cell
type.

## Products of simplices

Product cells appear when you build a *mapping cylinder* (an inclusion
$L \hookrightarrow K$ as a simplicial subcomplex of $K \times [0, 1]$)
or when you multiply a filtration by a single auxiliary cell. The
factory functions hide the product-cell bookkeeping:

```{code-block} python
import oineus as oin

# Two simplicial filtrations on overlapping vertex sets.
fil_K = oin.list_to_filtration([
    (0, [0], 0.0), (1, [1], 0.0), (2, [2], 0.0),
    (3, [0, 1], 0.6), (4, [1, 2], 0.7), (5, [0, 2], 0.8),
])
fil_L = oin.list_to_filtration([
    (0, [0], 0.0), (1, [1], 0.0), (2, [2], 0.0),
    (3, [0, 1], 0.5),
])

# Two auxiliary vertices, used as the cone points of the cylinder.
v_top    = oin.Simplex([10], 0.0)
v_bottom = oin.Simplex([11], 0.0)

fil_cyl = oin.mapping_cylinder(fil_L, fil_K, v_top, v_bottom)
print(fil_cyl.size())
print(fil_cyl.cell(0))
```

You can also build a plain product of a filtration with a single
auxiliary simplex:

```{code-block} python
sigma = oin.Simplex([10], 0.0)
fil_prod = oin.multiply_filtration(fil_K, sigma)
```

The same methods used above (`cells()`, `size()`, `subfiltration(...)`,
`boundary_matrix(...)`, `neg_infinity()`, ...) all work here. For the
common kernel/cokernel-of-a-cylinder workflow, the high-level helper
{py:func}`oineus.compute_ker_cok_reduction_cyl` builds the cylinder, the
inclusion, and the KICR object in one call -- see {doc}`mapping_cylinder`
and {doc}`kicr`.

## See also

- {doc}`filtrations` -- which builder produces which filtration.
- {doc}`decomposition` -- once you have a filtration, the reduction
  pipeline.
- {doc}`mapping_cylinder` -- product cells and the cylinder construction
  in more depth.
