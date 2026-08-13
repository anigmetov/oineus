# Kernel, image, cokernel persistence

Given a simplicial map $f \colon L \hookrightarrow K$ (in Oineus, an
inclusion of one filtration into a larger one), the *map-induced
persistence module* decomposes into three new persistence modules: the
kernel, the image, and the cokernel. Their diagrams describe, respectively:
the classes in $L$ that die under the map, the classes in $K$ that are
hit by $L$, and the classes in $K$ that come from outside $L$. See
Cohen-Steiner-Edelsbrunner-Harer-Morozov for the foundations.

Oineus computes all three diagrams in one pass.

## Quick example

```{code-block} python
import oineus as oin

# K: a hollow square (4 vertices + 4 edges). Values are chosen so
# that the kernel and cokernel both have a finite H0 point.
K = [
    [0, [0],    10.0],
    [1, [1],    30.0],
    [2, [2],    10.0],
    [3, [3],     0.0],
    [4, [0, 1], 30.0],
    [5, [1, 2], 30.0],
    [6, [0, 3], 10.0],
    [7, [2, 3], 10.0],
]

# L: a 2-edge subcomplex (vertices + two edges).
L = [
    [0, [0],    10.0],
    [1, [1],    30.0],
    [2, [2],    10.0],
    [3, [0, 1], 30.0],
    [4, [1, 2], 30.0],
]

kicr = oin.compute_kernel_image_cokernel_reduction(K, L)

print(kicr.kernel_diagrams().in_dimension(0))
print(kicr.image_diagrams().in_dimension(0))
print(kicr.cokernel_diagrams().in_dimension(0))
```

Either filtration can be passed as a Python list of
`(id, vertices, value)` triples (auto-converted via
{py:func}`oineus.list_to_filtration`) or as a pre-built
{py:class}`oineus.Filtration`. The vertex labels in $L$ must be a subset
of those in $K$, and every cell of $L$ must also be in $K$ at the same
filtration value -- this is what "inclusion" means.

## Reading the output

{py:func}`oineus.compute_kernel_image_cokernel_reduction` returns a
{py:class}`oineus.KerImCokReduced` object with five diagram accessors:

- `kicr.domain_diagrams()` -- persistence of $L$ alone.
- `kicr.codomain_diagrams()` -- persistence of $K$ alone (available when
  `KICRParams.codomain=True`).
- `kicr.kernel_diagrams()` -- diagram of $\ker f_*$.
- `kicr.image_diagrams()` -- diagram of $\mathrm{im}\, f_*$.
- `kicr.cokernel_diagrams()` -- diagram of $\mathrm{coker}\, f_*$.

Each accessor returns a {py:class}`oineus.Diagrams` object indexed by
homology dimension; use `.in_dimension(d)` to extract a 2D NumPy array.

### Index diagrams and cell lookup

The returned {py:class}`oineus.Diagrams` also contains the persistence
pairing. Continuing the example above:

```{code-block} python
import numpy as np

kernel_dgms = kicr.kernel_diagrams()
index_h0 = kernel_dgms.index_diagram_in_dimension(0)
print(index_h0)  # [[5 7]]

# Filter essential rows before treating both columns as cell indices
sentinel = np.iinfo(index_h0.dtype).max
finite_index_h0 = index_h0[index_h0[:, 1] != sentinel]

# Keep the values and index metadata together as DiagramPoint objects
for point in kernel_dgms.in_dimension(0, as_numpy=False):
    birth_cell = kicr.fil_K.cell(point.birth_index)
    death_cell = None if point.is_inf() else kicr.fil_K.cell(point.death_index)
    print(point.birth_index, birth_cell)
    if death_cell is not None:
        print(point.death_index, death_cell)
```

For the **kernel, image, and cokernel** diagrams, every birth index and
every finite death index is a position in the filtration order of the
ambient, or "big", filtration $K$ -- a `sorted_id` in `kicr.fil_K`. Use
`kicr.fil_K.cell(i)` for the lookup, not `kicr.fil_L.cell(i)`. This is true
even when the endpoint cell belongs to $L$: Oineus has already translated
its $L$ index to the corresponding position in $K$. The pair `[5, 7]`
above, for example, ends at edge `[1, 2]`; that edge has index 7 in $K$ but
index 4 in $L$.

Essential points have no death cell and store an integer sentinel in
`death_index`. In the NumPy index diagram it is the largest value of the
array's unsigned integer dtype; filter those rows, as above, before looking
up both endpoints or casting the array to a signed dtype. When using
`DiagramPoint` objects, test `point.is_inf()` before looking up the death
cell. The index spaces of the two ordinary diagram accessors are different:
`domain_diagrams()` uses filtration order in $L$, while
`codomain_diagrams()` uses filtration order in $K$.

For a mapping-cylinder computation, the ambient $K$ is the product/cylinder
filtration stored in `kicr.fil_K`; map its product cells back to the original
filtrations afterward if needed.

## Configuring what gets computed

By default all three of kernel, image, and cokernel are computed. If you
only need one, pass a {py:class}`oineus.KICRParams`:

```{code-block} python
params = oin.KICRParams(kernel=True, image=False, cokernel=False)
kicr = oin.compute_kernel_image_cokernel_reduction(K, L, params=params)
```

Other useful `KICRParams` fields:

- `include_zero_persistence` -- include zero-persistence pairs in the
  output diagrams (default `False`). The analogue of
  `Decomposition.zero_pers_diagram(fil)` for KICR; see
  {doc}`decomposition`.
- `verbose`, `sanity_check` -- diagnostic flags.
- `n_threads` -- threads used for the reductions; defaults to the value
  used by the individual `ReductionParams` blocks below.
- `params_f`, `params_g`, `params_ker`, `params_im`, `params_cok` --
  per-stage {py:class}`oineus.ReductionParams`. Most users do not need to
  touch these. Pass a single `reduction_params=...` to
  `compute_kernel_image_cokernel_reduction` to populate all five with the
  same settings.

```{code-block} python
rp = oin.ReductionParams(n_threads=8, use_clearing=True)
kicr = oin.compute_kernel_image_cokernel_reduction(K, L, reduction_params=rp)
```

## Product filtrations

For map-induced persistence on a *mapping cylinder* of $f$ (rather than on
the inclusion $L \hookrightarrow K$ directly), Oineus has a separate
binding that operates on product filtrations:
{py:class}`oineus.KerImCokReducedProd`. The high-level helper
{py:func}`oineus.compute_ker_cok_reduction_cyl` wires together the two
inputs, builds the cylinder, and returns a `KerImCokReducedProd` -- see
{doc}`mapping_cylinder`.

`compute_kernel_image_cokernel_reduction` dispatches between the simplex
and product-simplex variants automatically by inspecting the cell type of
the first filtration.

## When to reach for this vs. mapping cylinders

- **Use KICR directly** when the inclusion is a literal subcomplex of the
  same complex with the same filtration values on the shared cells. This
  is the case for thresholded VR / alpha at different radii, sublevel-set
  pairs at different thresholds, etc.
- **Use the mapping cylinder route** when the map is not an inclusion of
  the underlying complex but is some other simplicial map -- merging,
  collapsing, gluing. The cylinder construction turns a general
  simplicial map into an inclusion, after which KICR applies. The
  one-shot helper is {py:func}`oineus.compute_ker_cok_reduction_cyl`.

## See also

- {doc}`mapping_cylinder` -- how to handle non-inclusion simplicial maps.
- {doc}`relative_homology` -- relative diagrams of a pair $(K, L)$;
  related but algebraically different.
- {doc}`decomposition` -- the underlying reduction machinery; `KICRParams`
  carries one `ReductionParams` per sub-decomposition.
- `examples/python/example_kernel.py`, `tests/test_kicr.py`,
  `tests/test_api_kicr.py` -- runnable demos.
