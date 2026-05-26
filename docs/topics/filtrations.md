# Filtration types

A *filtration* is a sequence of cells sorted by a real-valued function. Oineus
ships builders for the four filtrations you will reach for most often
(Freudenthal, cubical, Vietoris-Rips, alpha) plus a constructor for
user-defined simplicial filtrations. Every builder returns a filtration
object that you can hand to {py:class}`oineus.Decomposition`, the one-shot
helpers ({py:func}`oineus.compute_diagrams_ls`,
{py:func}`oineus.compute_diagrams_vr`,
{py:func}`oineus.compute_diagrams_alpha`), or any of the differentiable
builders in {py:mod}`oineus.diff`.

## Picking a filtration

| Builder | Input | Cells produced | Typical use |
| --- | --- | --- | --- |
| {py:func}`oineus.freudenthal_filtration` | $d$-D NumPy array | triangulated $d$-cube grid, lower/upper star | Image and volume sublevel-set persistence; smaller than cubical. |
| {py:func}`oineus.cube_filtration` | $d$-D NumPy array | genuine cubical complex | When you need pixel/voxel adjacency or want to match GUDHI/cubical-literature conventions. |
| {py:func}`oineus.vr_filtration` | $(n, d)$ points or $(n, n)$ pairwise distances | full $k$-skeleton of VR | Generic point clouds in any dimension. |
| {py:func}`oineus.alpha_filtration` | $(n, 2)$ or $(n, 3)$ points | alpha complex via diode | 2D/3D point clouds; much fewer cells than VR. |
| {py:func}`oineus.list_to_filtration` | list of `(id, vertices, value)` | user-specified simplicial complex | Bring-your-own filtration. |

For low-dimensional point clouds the rule of thumb is alpha first, VR only if
you need cells of dimension $\geq d$ (where $d$ is the ambient dimension) or
your geometry breaks Delaunay assumptions.

## Function data on a grid

### Freudenthal (triangulated grid)

```{code-block} python
import numpy as np
import oineus as oin

img = np.load("scan.npy")               # shape (Z, Y, X), dtype float
fil = oin.freudenthal_filtration(img, negate=False, max_dim=2, n_threads=4)

dcmp = oin.Decomposition(fil, dualize=True)
dcmp.reduce(oin.ReductionParams(n_threads=4))
dgms = dcmp.diagram(fil)

print(dgms.in_dimension(0).shape, dgms.in_dimension(1).shape)
```

The one-shot equivalent is {py:func}`oineus.compute_diagrams_ls`, which builds
the Freudenthal filtration, reduces, and returns
{py:class}`oineus.Diagrams` in one call.

### Cubical

```{code-block} python
cube_fil = oin.cube_filtration(img, negate=False, values_on="vertices")
```

`values_on` is the cubical-only knob worth knowing about:

- `"vertices"` (default): each vertex gets its data value; higher-dimensional
  cubes inherit the max (or min, with `negate=True`) of their vertices. This
  is the lower/upper-star convention.
- `"top_cells"`: each top-dimensional cube gets its data value; lower cubes
  inherit the min (or max with `negate=True`). Useful when the data lives on
  voxels rather than vertices, and matches the cubical convention in GUDHI.

### Sublevel vs. superlevel sets

By default the filtration is built in increasing order of value (sublevel
sets). Pass `negate=True` to flip the sign and obtain superlevel persistence.
The resulting diagrams have the same persistence pairs, just reflected through
the diagonal.

### Periodic boundaries

`freudenthal_filtration(..., wrap=True)` wraps the grid into a $d$-torus.
Lower-star persistence on a constant array then reports $H_1$ of the torus,
$H_2$ of the torus, etc. Currently the cubical builder does not support
`wrap=True`.

## Point clouds

### Vietoris-Rips

```{code-block} python
pts = np.load("points.npy")             # shape (n, d)
fil = oin.vr_filtration(pts, max_dim=2, max_diameter=2.0, n_threads=4)

dcmp = oin.Decomposition(fil, dualize=True)
dcmp.reduce(oin.ReductionParams(n_threads=4))
dgms = dcmp.diagram(fil)
```

Notes:

- `max_dim` caps the simplex dimension. For $H_k$ you need cells up through
  dimension $k+1$.
- `max_diameter` truncates the filtration. With no truncation the VR complex
  on $n$ points has $\binom{n}{k+1}$ $k$-simplices, which is huge; tightening
  `max_diameter` is usually the difference between "runs in a second" and
  "exhausts memory".
- `from_pwdists=True` takes an $(n, n)$ symmetric distance matrix instead of
  raw coordinates. Combine with a custom metric (geodesic, $L^1$, ...) to do
  VR on any metric space.
- The in-order enumeration is the VRE algorithm from Vejdemo-Johansson,
  Matuszewski & Bauer; see the docstring for the citation.

### Alpha

```{code-block} python
dgms = oin.compute_diagrams_alpha(pts)         # one-shot

# or, if you want to keep the filtration around for downstream work:
fil  = oin.alpha_filtration(pts)
```

Alpha needs the optional [`diode`](https://github.com/mrzv/diode) dependency,
which provides the CGAL-backed Delaunay construction. The defaults are
suitable for most point clouds; the knobs you might want:

- `exact=True` -- use CGAL's exact kernel for the Delaunay construction. Slow
  but immune to numerical pathologies.
- `periodic=True` plus a bounding box (`bbox_min`, `bbox_max`, or
  `compute_bounding_box=True`) -- periodic alpha shapes on a $d$-torus.
- `weights` -- 1D array of vertex weights; produces weighted (regular) alpha
  shapes in 3D.

## Custom simplicial filtrations

For one-off filtrations -- toy examples, programmatically constructed
complexes, anything you cannot get out of a builder -- pass a list of
`(id, vertices, value)` triples:

```{code-block} python
data = [
    (0, [0],    0.2),
    (1, [1],    0.1),
    (2, [2],    0.3),
    (3, [0, 1], 0.9),
    (4, [0, 2], 0.5),
    (5, [1, 2], 0.8),
    (6, [0, 1, 2], 1.0),
]
fil = oin.list_to_filtration(data)
```

Or build the simplices explicitly with {py:class}`oineus.Simplex` and pass
them to {py:class}`oineus.Filtration`:

```{code-block} python
v0, v1, v2 = oin.Simplex([0], 0.2), oin.Simplex([1], 0.1), oin.Simplex([2], 0.3)
e1, e2, e3 = oin.Simplex([0, 1], 0.9), oin.Simplex([0, 2], 0.5), oin.Simplex([1, 2], 0.8)
t = oin.Simplex([0, 1, 2], 1.0)

fil = oin.Filtration([v0, v1, v2, e1, e2, e3, t], negate=False, n_threads=1)
```

The `Filtration` constructor sorts the simplices into a valid filtration
order and assigns each one a `sorted_id`. The `id` you supplied is kept as
`simplex.id` so you can recover the original ordering.

Vertices must be tagged with their vertex labels in sorted order
(`Simplex([0, 1, 2], ...)`, not `Simplex([2, 0, 1], ...)`), and any face of a
simplex in the list must also be in the list (you cannot include the triangle
$\{0, 1, 2\}$ without the three edges and three vertices).

## See also

- {doc}`decomposition` -- once you have a `Filtration`, what to do with it.
- {doc}`differentiable` -- the same filtrations as differentiable layers
  over PyTorch tensors.
- {doc}`performance` -- `max_dim`, `max_diameter`, and `n_threads` are where
  most of the runtime knobs live.
