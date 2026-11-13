---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 1. Grid & Image Filtrations

Scalar fields sampled on regular 1D, 2D, 3D, or 4D grids—such as grayscale images, CT/MRI volumes, elevation maps, or fluid simulations—are one of the two primary data sources in topological data analysis. Oineus provides two distinct cell complexes over regular arrays:

1. **Freudenthal (Kuhn) simplicial triangulation** via {py:func}`oineus.freudenthal_filtration` (and the one-call shortcut {py:func}`oineus.compute_diagrams_ls`).
2. **Cubical complexes** via {py:func}`oineus.cube_filtration`.

This tutorial explains how sublevel and superlevel filtrations work, how Freudenthal and cubical complexes differ on the exact same array, how `values_on="vertices"` vs. `values_on="cells"` assigns values, and how periodic boundary conditions (`wrap=True`) turn a grid into a torus.

---

## 1. Sublevel vs. Superlevel Persistence (`negate`)

Given a scalar field $f: \Omega \to \mathbb{R}$:

- **Sublevel filtration (`negate=False`, default)** sweeps $f^{-1}((-\infty, t])$ as $t$ increases:
  - $H_0$ births occur at **local minima** (basins) and merge at **saddles**.
  - $H_1$ loops wrap around **peaks** (regions where $f > t$) and die when the peak is submerged.
- **Superlevel filtration (`negate=True`)** sweeps $f^{-1}([t, +\infty))$ as $t$ *decreases*—retaining the **original function values** in the diagram (so `birth > death`, placing points *below* the diagonal):
  - $H_0$ births occur at **local maxima** (peaks) and merge at **saddles**.

Let's construct a 2D synthetic landscape with two Gaussian peaks and one circular moat, and compare its sublevel and superlevel diagrams side by side:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import oineus as oin

lin = np.linspace(-2.5, 2.5, 80)
X, Y = np.meshgrid(lin, lin)

peak_1 = 1.4 * np.exp(-((X + 0.9) ** 2 + (Y - 0.5) ** 2) / 0.35)
peak_2 = 1.0 * np.exp(-((X - 1.0) ** 2 + (Y + 0.4) ** 2) / 0.30)
moat = -0.8 * np.exp(-((np.hypot(X, Y) - 1.8) ** 2) / 0.12)
field = peak_1 + peak_2 + moat

dgms_sub = oin.compute_diagrams_ls(field, negate=False)
dgms_sup = oin.compute_diagrams_ls(field, negate=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
im = axes[0].imshow(field, origin="lower", extent=[-2.5, 2.5, -2.5, 2.5], cmap="terrain")
fig.colorbar(im, ax=axes[0], fraction=0.046)
axes[0].set_title("Scalar field f(x, y)")

oin.plot_diagram(dgms_sub, dims=(0, 1), ax=axes[1], title="Sublevel (negate=False)")
oin.plot_diagram(dgms_sup, dims=(0, 1), ax=axes[2], title="Superlevel (negate=True)")
fig.tight_layout()
```

Notice how in `dgms_sup` (`negate=True`), every finite point satisfies $\text{birth} > \text{death}$: the secondary peak is born around $f \approx 1.0$ in $H_0$ and merges into the taller peak's component when the threshold drops to the saddle between them!

---

## 2. Freudenthal Triangulation vs. Cubical Complex

On a $d$-dimensional grid:

- **Cubical complex** ({py:func}`oineus.cube_filtration`) keeps axis-aligned vertices, edges, squares, cubes, and hypercubes directly ($4$-connectivity on 2D vertices when `values_on="vertices"`).
- **Freudenthal triangulation** ({py:func}`oineus.freudenthal_filtration`) subdivides each $d$-cube into $d!$ simplices along the main diagonal ($2$ triangles per 2D square, $6$ tetrahedra per 3D voxel, $24$ pentachora per 4D cell), introducing diagonal adjacencies ($6$-connectivity in 2D).

Let's compare cell counts and subtle saddle differences on a checkerboard-like saddle test:

```{code-cell} ipython3
fil_freud = oin.freudenthal_filtration(field, max_dim=2)
fil_cube = oin.cube_filtration(field, values_on="vertices")

print(f"Freudenthal cells (80x80): {fil_freud.size():7d}  (kind={fil_freud.kind.as_str()})")
print(f"Cubical cells     (80x80): {fil_cube.size():7d}  (kind={fil_cube.kind.as_str()})")

# Small 3x3 diagonal saddle example
saddle_patch = np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.5, 1.0],
    [1.0, 1.0, 0.0],
], dtype=np.float64)

h0_freud = oin.reduce(oin.freudenthal_filtration(saddle_patch)).diagram(
    oin.freudenthal_filtration(saddle_patch), include_inf_points=False
).in_dimension(0)
h0_cube = oin.reduce(oin.cube_filtration(saddle_patch, values_on="vertices")).diagram(
    oin.cube_filtration(saddle_patch, values_on="vertices"), include_inf_points=False
).in_dimension(0)

print("Diagonal corners merge at (Freudenthal):", h0_freud)
print("Diagonal corners merge at (Cubical):    ", h0_cube)
```

In 2D, `cube_filtration` uses ~20% fewer cells than `freudenthal_filtration`; in 3D ($d! = 6$) and 4D ($d! = 24$), the cubical complex is dramatically smaller while avoiding directional triangulation bias. Use `freudenthal_filtration` when downstream algorithms require simplicial complexes (such as `mapping_cylinder` or simplicial cochain solvers).

---

## 3. Cubical Data Placement: `values_on="vertices"` vs. `"cells"`

Image-processing libraries fall into two camps regarding where pixel/voxel intensities live:

- **`values_on="vertices"` (default)**: Each array entry is a **0-cube (vertex)** on a grid of shape `(N_0, ..., N_{d-1})`. Higher-dimensional cubes take the **maximum** (or minimum if `negate=True`) of their incident vertices (lower-star rule).
- **`values_on="cells"`**: Each array entry is a **top-dimensional $d$-cube** (pixel/voxel), so the vertex grid has shape `(N_0 + 1, ..., N_{d-1} + 1)`. Lower-dimensional faces take the **minimum** (or maximum if `negate=True`) of their incident top-dimensional cells (T-construction, matching GUDHI's `CubicalComplex(top_dimensional_cells=...)`).

:::{important}
Pass `values_on="cells"` (not `"top_cells"`). Passing any string other than `"vertices"` or `"cells"` raises a `ValueError`.
:::

Let's see how isolated bright/dark single pixels behave under the two constructions:

```{code-cell} ipython3
# A 5x5 dark ring of zeros surrounding a bright center pixel (value 1.0)
ring_img = np.ones((5, 5), dtype=np.float64)
ring_img[1:4, 1:4] = 0.0
ring_img[2, 2] = 1.0

fil_v = oin.cube_filtration(ring_img, values_on="vertices")
fil_c = oin.cube_filtration(ring_img, values_on="cells")

dgm_v = oin.reduce(fil_v).diagram(fil_v, include_inf_points=False).in_dimension(1)
dgm_c = oin.reduce(fil_c).diagram(fil_c, include_inf_points=False).in_dimension(1)

print("Vertices mode size:", fil_v.size(), "| H1 bars:", dgm_v)
print("Cells mode size:   ", fil_c.size(), "| H1 bars:", dgm_c)
```

Both detect the `[0.0, 1.0)` loop around the bright center pixel! Notice `fil_c` has more cells on a $5 \times 5$ array because $5 \times 5$ top squares span a $6 \times 6$ grid of corner vertices.

---

## 4. Periodic Boundary Conditions (`wrap=True`)

When simulating periodic domains (cosmological fields, molecular dynamics boxes, turbulent flows, textures), set `wrap=True` on {py:func}`oineus.freudenthal_filtration` or {py:func}`oineus.compute_diagrams_ls` to identify opposite faces into a flat torus $\mathbb{T}^d$. (Note: `wrap=True` is currently implemented for Freudenthal grid filtrations; `cube_filtration(..., wrap=True)` raises `RuntimeError`.)

On a smooth periodic field over $\mathbb{T}^2$, homology has Betti numbers $\beta_0 = 1$, $\beta_1 = 2$ (two independent non-contractible torus cycles), and $\beta_2 = 1$ (the closed torus surface itself):

```{code-cell} ipython3
torus_field = 0.05 * np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False))[:, None] * np.ones((1, 24))

fil_open = oin.freudenthal_filtration(torus_field, wrap=False, max_dim=2)
fil_torus = oin.freudenthal_filtration(torus_field, wrap=True, max_dim=2)

dgms_open = oin.reduce(fil_open).diagram(fil_open, include_inf_points=True)
dgms_torus = oin.reduce(fil_torus).diagram(fil_torus, include_inf_points=True)

for q in range(3):
    n_inf_open = int(np.isinf(dgms_open[q]).any(axis=1).sum())
    n_inf_torus = int(np.isinf(dgms_torus[q]).any(axis=1).sum())
    print(f"Dimension {q} essential bars -> open grid: {n_inf_open}, periodic torus (wrap=True): {n_inf_torus}")
```

---

## 5. Scaling From 1D Signals to 3D & 4D Volumes

Both `freudenthal_filtration` and `cube_filtration` natively support **1D, 2D, 3D, and 4D** NumPy arrays (`float64` or `float32`). If you only need low-dimensional homology (say $H_0$ and $H_1$ of a 3D volume), pass `max_dim=2` to `cube_filtration` or `freudenthal_filtration` to skip constructing 3-cells and save memory:

```{code-cell} ipython3
rng = np.random.default_rng(42)

# 3D scalar volume containing a hollow spherical shell -> creates an H2 void!
g3 = np.linspace(-1.5, 1.5, 26)
Z, Y3, X3 = np.meshgrid(g3, g3, g3, indexing="ij")
R3 = np.sqrt(X3**2 + Y3**2 + Z**2)
shell_3d = -np.exp(-((R3 - 0.85) ** 2) / 0.06) + 0.03 * rng.standard_normal(R3.shape)

fil_3d = oin.cube_filtration(shell_3d, n_threads=4)
dgms_3d = oin.reduce(fil_3d, n_threads=4).diagram(fil_3d, include_inf_points=False)

h2 = dgms_3d.in_dimension(2)
top_h2 = h2[np.argmax(h2[:, 1] - h2[:, 0])]
print(f"3D cubical cells: {fil_3d.size():,d} | prominent H2 void (birth, death): {np.round(top_h2, 4)}")

# 4D spacetime / hypervolume cubical filtration works identically
vol_4d = rng.random((6, 6, 6, 6), dtype=np.float32)
fil_4d = oin.cube_filtration(vol_4d, max_dim=2, n_threads=2)
print(f"4D float32 cubical 2-skeleton cells: {fil_4d.size():,d} (max_dim={fil_4d.max_dim})")
```

### Accessing Underlying Grid Objects & Critical Indices

When you need the flat array index of the vertex (or top cell) responsible for every simplex's filtration value, pass `with_critical_vertices=True` to `freudenthal_filtration`—or instantiate {py:class}`oineus.Grid_2D` (or `Grid_1D`..`Grid_4D`) directly:

```{code-cell} ipython3
fil_crit, crit_verts = oin.freudenthal_filtration(saddle_patch, with_critical_vertices=True)
print("Filtration size:", fil_crit.size(), "| critical vertex map shape:", crit_verts.shape)

grid2d = oin.Grid_2D(saddle_patch, wrap=False, values_on="vertices")
fil_g, cube_crit = grid2d.cube_filtration_and_critical_indices(max_dim=2)
print("Grid_2D data_location:", grid2d.data_location, "| cube critical indices count:", len(cube_crit))
```
