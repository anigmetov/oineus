# Limitations and caveats

Oineus is built for large-scale persistence and differentiable topology, and a
few of its design choices trade generality or safety for speed. This page
collects the sharp edges worth knowing before you hit them. None of these are
bugs -- they are consequences of the library's scope -- but each one has bitten
someone.

## Cell construction

**Simplices are limited to dimension 13.** The combinatorial uid that keys every
simplex packs the vertex count into the top four bits of a 128-bit integer, so a
simplex can have at most 14 vertices (dimension 13). Constructors and `join()`
reject anything larger with a `ValueError` rather than let two simplices silently
share a uid. In practice this is never a constraint -- Vietoris-Rips and alpha
complexes almost never exceed dimension 3 or 4 -- but if you build simplicial
complexes by hand, keep it in mind. (The slim, bit-packed encodings the factory
functions prefer cap out lower still -- how much lower depends on the number of
points -- and fall back to the fat encoding automatically when a cell no longer
fits.)

**Vertices must be non-negative and distinct.** `oineus.Simplex([0, 0, 1])` and
`oineus.Simplex([-1, 0])` raise `ValueError`: a duplicated or negative vertex
would mis-encode the uid and corrupt the boundary. The same applies to both
factors of a product cell, and to the vertex added by `join()`.

**Cubes must fit inside their grid.** A cube's anchor *and* its opposite corner
(the anchor plus one step in each spanned dimension) must lie in the domain,
unless the grid wraps. An edge anchored at the last vertex of a non-wrapping axis
is rejected -- it would have a vertex outside the grid and a malformed boundary.

These checks live only at the Python boundary, where hand-built cells enter. The
bulk factory functions (`vr_filtration`, `freudenthal_filtration`,
`cube_filtration`, alpha) build valid cells in C++ and pay nothing for them.

## `max_distance` is an enclosing radius, not a diameter

{py:func}`oineus.max_distance` returns the **enclosing radius**
$\min_i \max_j \lVert x_i - x_j \rVert$ -- the smallest radius from which some
single point sees every other. This is the standard Vietoris-Rips cutoff: beyond
it the complex is a cone and carries no more topology, exactly the threshold
Ripser uses. It is *not* the diameter $\max_{i,j} \lVert x_i - x_j \rVert$: for
three collinear points at 0, 1, 2 it returns 1, not 2. Feed it straight to
`vr_filtration` as `max_diameter`; do not read it as the largest pairwise
distance.

It rejects non-finite input with `ValueError`, and also raises when the
coordinate spread is so large (beyond roughly `1.3e154` in a single axis) that a
squared distance overflows float64. Rescale such data first. A constant cloud
correctly returns 0.

## Printing large objects

`repr()` on a {py:class}`~oineus.Filtration`, a
{py:class}`~oineus.Decomposition`, or a grid is a **bounded one-line summary** --
sizes, dimensions, which matrices are stored -- never the cells or matrix
entries. A million-cell object would otherwise hang your REPL or notebook. When
you actually want the full dump, ask for it explicitly:

```{code-block} python
print(dcmp)                 # Decomposition(n_rows=..., reduced=true, has_V=..., ...)
print(dcmp.to_str_debug())  # every column of D, R, V, U -- only for small inputs
print(fil.to_str_debug())   # every cell, grouped by dimension
```

## Mutable objects are hashable

`Simplex`, the cube types, and `DiagramPoint` expose writable fields (`value`,
`id`, `birth`, `death`, ...) that also feed equality and `__hash__`. If you put
one in a `set`, or use it as a `dict` key, and then mutate one of those fields,
the object lands in the wrong bucket and membership tests silently fail. Treat an
object as frozen once it is in a hashed container, or copy it before mutating.

## Pickling

Most heavyweight objects pickle, for use with `multiprocessing`, `joblib`, and
result caches: {py:class}`~oineus.Filtration`,
{py:class}`~oineus.Decomposition`, the kernel/image/cokernel result,
`oineus.diff.DiffFiltration`, and the diagram containers all round-trip.

Caveats:

- **Grids do not pickle.** A grid is a thin non-owning view over your NumPy array
  (see below), with no state of its own to serialize.
- **Matching views do not pickle.** `EssentialMatches`, `LongestEdges`, and the
  other longest-edge / essential-match views are lightweight windows into another
  object; snapshot the arrays you need instead.
- **Pickles are not versioned.** They are meant for same-build, same-session
  round-trips. A pickle written by one Oineus version may fail to load in
  another, and there is no migration path -- do not use them for long-term
  storage.
- **PyTorch tensors detach on pickle.** Unpickling a `DiffFiltration` gives you
  its values as plain tensors; the autograd graph does not survive. This is
  ordinary tensor-pickle behavior, but it means a restored object is not
  differentiable.

## Grids borrow their data

`oineus.Grid_2D(array, ...)` (and the other dimensions) stores a **non-owning
pointer** into `array` and pins the array alive for the grid's lifetime -- so you
do not need to keep a separate reference. But because the grid is a view, do not
*mutate* the array underneath a live grid, and build the filtration from the grid
before you are done with the data.

## Differentiable diagrams

`oineus.diff` is an **eager boundary**: diagram sizes are data-dependent, so you
cannot `jax.jit`, `jax.vmap`, or `torch.compile` *through* a diagram call. Build
the filtration and the diagram inside the function you differentiate, and jit or
compile the surrounding network and loss instead. See {doc}`differentiable` for
the pattern.

For the crit-sets gradient method the backward pass re-reduces from the
underlying filtration. Do not change that filtration between the forward diagram
call and the backward pass: a **structural** change (different cells) raises
`RuntimeError`, and while a pure **value** change is tolerated (the forward values
are restored before re-reducing), there is no good reason to do it. Note also
that a JAX crit-sets loss spanning $k$ dimensions performs $k$ reductions in the
backward, so keep the number of differentiated dimensions small.

## Interrupting long computations

Most long-running calls poll for Ctrl-C and raise `KeyboardInterrupt` within tens
of milliseconds. The **apparent-pairs detection** phase (used by the lean,
`use_apparent_pairs` reduction path) is the exception: its parallel loop skips
only a small fraction of the remaining cells on interrupt and cannot preempt a
cell already running, so an interrupt raised during that phase can take nearly as
long as the phase itself to take effect. It always terminates and raises -- it
never hangs -- but it is markedly less responsive than the rest of the library.

## Numeric types

Integers are `long int`. Both `float64` and `float32` reals ship in the standard
build: `float32` lives in an internal `_f32` submodule and is selected
automatically from your input array's dtype -- no special build flag is needed.
The differentiable layer likewise routes `float32` tensors to the `float32`
backend. (`OINEUS_PYTHON_REAL` only changes which real the top-level default and
the standalone array builders use; it does not turn `float32` on.) Mixing a
`float32` filtration with `float64` diagram arithmetic in one pipeline is not
supported -- keep a single real type end to end.
