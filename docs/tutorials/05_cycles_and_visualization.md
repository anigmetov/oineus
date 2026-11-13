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

# 5. Representative Cycles & Visualization (`oineus.vis`)

Persistence diagrams tell you *when* topological features appear and disappear; the reduced matrices $R$ and $V$ tell you **where** those features live in your data! Combined with {py:mod}`oineus.vis`, you can extract representative cycles, bounding chains, and critical sets and render them directly over 2D scalar fields and 2D/3D point clouds.

---

## 1. How Representative Cycles Live in $R$ and $V$

Suppose `dcmp = oin.Decomposition(fil, dualize=False)` has been reduced with `compute_v=True`, and a finite $H_k$ persistence pair has `(birth_index, death_index) = (i, j)` (obtained from `dgms.in_dimension(k, as_numpy=False)`):

- **Death-time cycle representative `dcmp.r_data[j]`**: Since $R_{\cdot, j} = D V_{\cdot, j}$ and $\operatorname{low}(R_{\cdot, j}) = i$, the column `r_data[j]` is a $\mathbb{Z}_2$ $k$-cycle born at $i$ (all entries in `r_data[j]` are $\le i$, so the cycle already exists at the birth time of $i$!) that becomes a boundary at $j$.
- **Bounding $(k+1)$-chain `dcmp.v_data[j]`**: The $(k+1)$-cells in `v_data[j]` are precisely the $(k+1)$-chain whose $\mathbb{Z}_2$ boundary equals `r_data[j]`—filling in the hole at death time!
- **Essential cycle representative `dcmp.v_data[i]`**: If an unpaired positive $k$-cell $i$ never dies (`death == +inf`, so `r_data[i] == []`), then $D V_{\cdot, i} = R_{\cdot, i} = 0$, meaning **`v_data[i]` itself is the $k$-cycle** representing that essential homology class!

Let's extract **both** the $H_1$ loop (`r_data[j]`) and its filling 2-chain of triangles (`v_data[j]`) on a 2D point cloud with interior clutter, and plot them with {py:func}`oineus.vis.plot_chain`:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import oineus as oin

rng = np.random.default_rng(4)
theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
ring_pts = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.05 * rng.standard_normal((32, 2))
clutter = rng.uniform(-1.5, 1.5, size=(35, 2))
# Keep clutter outside the central hole so the main loop stays crisp
clutter = clutter[np.linalg.norm(clutter, axis=1) > 0.75]
pts = np.vstack([ring_pts, clutter])

fil = oin.alpha_filtration(pts)
dcmp = oin.Decomposition(fil, dualize=False)
dcmp.reduce(oin.ReductionParams(n_threads=2, use_clearing=True, compute_v=True))

dgms = dcmp.diagram(fil, include_inf_points=False)
h1_pts = dgms.in_dimension(1, as_numpy=False)
best_bar = max(h1_pts, key=lambda p: p.persistence)

cycle_edges = dcmp.r_data[best_bar.death_index]
filling_tris = dcmp.v_data[best_bar.death_index]
print(f"Longest H1 bar: birth={best_bar.birth:.3f}, death={best_bar.death:.3f}")
print(f"Cycle has {len(cycle_edges)} edges; bounding chain has {len(filling_tris)} triangles.")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
oin.plot_chain(pts, fil, cycle_edges, ax=axes[0], title="H1 cycle representative (r_data[death])")
oin.plot_chain(
    pts, fil, filling_tris, ax=axes[1],
    chain_color=oin.OKABE_ITO_BLUE,
    title="Bounding 2-chain of triangles (v_data[death])",
)
fig.tight_layout()
```

---

## 2. Highlighting Saddles & Basins on 2D Scalar Fields

{py:func}`oineus.vis.plot_chain` automatically inspects `fil.kind`: when given a 2D array `source` and a `Cubical` or `Freudenthal` filtration, it renders the scalar field with `imshow` and overlays 0-cells (dots), 1-cells (grid segments), and 2-cells (shaded squares or triangles)!

Let's visualize:
1. The **saddle edge** that merges two basins in a 2D cubical sublevel filtration.
2. The **basin of attraction** (cohomology $V$ column) of the secondary basin—every vertex that belongs to that basin before the saddle merge!

```{code-cell} ipython3
H, W = 48, 64
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
basin1 = -1.0 * np.exp(-((xx - 0.30 * W) ** 2 + (yy - 0.50 * H) ** 2) / (2 * (0.11 * W) ** 2))
basin2 = -0.82 * np.exp(-((xx - 0.72 * W) ** 2 + (yy - 0.50 * H) ** 2) / (2 * (0.11 * W) ** 2))
landscape = basin1 + basin2 + 0.01 * rng.standard_normal((H, W))

fil_cube = oin.cube_filtration(landscape, values_on="vertices")
opt = oin.TopologyOptimizer(fil_cube)
dgms_field = opt.compute_diagram(include_inf_points=False)
h0_bars = dgms_field.in_dimension(0, as_numpy=False)
merge_bar = max(h0_bars, key=lambda p: p.persistence)

# 1. Saddle edge (death_index) + its two boundary vertices
saddle_chain = [merge_bar.death_index]
# 2. Critical set for raising the secondary minimum's birth up to the saddle -> entire right basin!
opt.ensure_coh_reduced()
basin_vertices = opt.increase_birth(merge_bar.birth_index)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
oin.plot_chain(
    landscape, fil_cube, saddle_chain, ax=axes[0],
    field_cmap="coolwarm", title="Merging saddle edge (H0 death cell)",
)
oin.plot_chain(
    landscape, fil_cube, basin_vertices, ax=axes[1],
    field_cmap="coolwarm", vertex_style=oin.default_chain_vertex_style() | {"s": 14, "alpha": 0.75},
    title=f"Basin critical set ({len(basin_vertices)} vertices)",
)
fig.tight_layout()
```

:::{note}
If you extracted matrix indices from a **cohomology** decomposition (`dualize=True`), pass `dualize=True` (or `dualize=dcmp`) to `oin.plot_chain(source, fil, chain, dualize=True)` so matrix indices are converted back to filtration `sorted_id`s automatically!
:::

---

## 3. Customizing Persistence Diagram Plots & Hybrid Density Mode

When a diagram contains tens of thousands of low-persistence noise points near the diagonal alongside a handful of prominent features, plotting every point as an SVG/PNG marker creates visual clutter and slow renders.

{py:func}`oineus.vis.plot_diagram` solves this with **hybrid scatter-density rendering** (`scatter_only=False`, powered by `mpl-scatter-density`):
- Points inside a strip of width `near_diagonal_fraction` along the diagonal (when total points exceed `density_threshold`) are rasterized into a smooth 2D pixel-density histogram via `mpl-scatter-density`.
- High-persistence signal points and essential `InfKind` points (drawn along dashed $\pm\infty$ margin lines) remain crisp individual scatter markers!

You can also customize any visual layer non-destructively using the `default_*_style()` helpers and Okabe–Ito colorblind-safe palette constants (`OKABE_ITO_BLUE`, `OKABE_ITO_VERMILLION`):

```{code-cell} ipython3
import importlib.util

# Generate a noisy 96x96 field producing thousands of diagram points
xx2, yy2 = np.meshgrid(np.linspace(-2, 2, 96), np.linspace(-2, 2, 96))
noisy_img = -np.exp(-((np.hypot(xx2, yy2) - 1.0) ** 2) / 0.1) + 0.35 * rng.standard_normal((96, 96))
dgms_large = oin.compute_diagrams_ls(noisy_img)
print("Total H0 + H1 points:", len(dgms_large[0]) + len(dgms_large[1]))

has_mpl_density = importlib.util.find_spec("mpl_scatter_density") is not None
custom_pt_style = oin.default_point_style() | {"s": 28, "alpha": 0.85}
custom_inf_style = oin.default_inf_point_style() | {"s": 55}

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
oin.plot_diagram(
    dgms_large, dims=(0, 1), ax=axes[0],
    scatter_only=True, point_style=custom_pt_style, inf_point_style=custom_inf_style,
    title="Multi-dimension scatter (H0 & H1)",
)
oin.plot_diagram(
    dgms_large, dims=(1,), ax=axes[1],
    scatter_only=not has_mpl_density, density_threshold=200, near_diagonal_fraction=0.15,
    color=oin.OKABE_ITO_VERMILLION, point_style=custom_pt_style,
    title="H1 with Okabe-Ito Vermillion styling",
)
fig.tight_layout()
```
