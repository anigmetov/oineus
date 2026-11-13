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

(quickstart)=
# 5-Minute Quickstart

Throughout the documentation we use the standard import alias `import oineus as oin`. Let's compute persistence diagrams on a 2D scalar field and a noisy circle point cloud, compare two diagrams with Wasserstein distance, and plot the results.

## 1. Persistence of a 2D Scalar Field

Given a regular 1D, 2D, 3D, or 4D NumPy array, {py:func}`oineus.compute_diagrams_ls` triangulates the grid (Freudenthal triangulation), sweeps sublevel sets from $-\infty$ to $+\infty$, and returns a {py:class}`oineus.Diagrams` container:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import oineus as oin

rng = np.random.default_rng(7)
grid = np.linspace(-2.2, 2.2, 64)
X, Y = np.meshgrid(grid, grid)

# Ring-shaped valley (-1 near radius 1.1) with mild Gaussian noise
R = np.hypot(X, Y)
field = -np.exp(-((R - 1.1) ** 2) / 0.12) + 0.06 * rng.standard_normal(X.shape)

dgms_grid = oin.compute_diagrams_ls(field)
print("H0 finite + essential pairs:", dgms_grid.in_dimension(0).shape)
print("H1 finite pairs:", dgms_grid.in_dimension(1).shape)
```

Calling `dgms.in_dimension(q)` returns an `(n_q, 2)` NumPy array of `(birth, death)` coordinates (`include_inf_points=True` by default, so essential classes carry `np.inf` death). Let's inspect the scalar field alongside its persistence diagram using {py:func}`oineus.vis.plot_diagram`:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

im = axes[0].imshow(field, origin="lower", extent=[-2.2, 2.2, -2.2, 2.2], cmap="coolwarm")
fig.colorbar(im, ax=axes[0], fraction=0.046)
axes[0].set_title("2D scalar field (circular valley)")

oin.plot_diagram(dgms_grid, dims=(0, 1), ax=axes[1], title="Sublevel persistence diagram")
fig.tight_layout()
```

Notice the prominent $H_1$ point far above the diagonal: the circular valley forms a loop at $f \approx -0.95$ that fills in only near $f \approx 0$.

## 2. Point-Cloud Persistence: Vietoris–Rips vs. Alpha

For a general metric point cloud in $\mathbb{R}^d$, {py:func}`oineus.compute_diagrams_vr` computes Vietoris–Rips persistence up to diagram dimension `max_dim` (automatically building the `(max_dim + 1)`-skeleton needed to bound `max_dim`-cycles):

```{code-cell} ipython3
theta = np.linspace(0.0, 2.0 * np.pi, 70, endpoint=False)
pts_a = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.06 * rng.standard_normal((70, 2))
pts_b = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.12 * rng.standard_normal((70, 2))

dgms_vr_a = oin.compute_diagrams_vr(pts_a, max_dim=1, max_diameter=2.2)
dgms_vr_b = oin.compute_diagrams_vr(pts_b, max_dim=1, max_diameter=2.2)

h1_a = dgms_vr_a.in_dimension(1)
print("Longest VR H1 bar (birth, death):", np.round(h1_a[np.argmax(h1_a[:, 1] - h1_a[:, 0])], 4))
```

When points live in low-dimensional Euclidean space ($\mathbb{R}^2$ or $\mathbb{R}^3$) and `diode` is installed, {py:func}`oineus.compute_diagrams_alpha` computes exact **Alpha** persistence on the Delaunay triangulation—producing far fewer simplices than Vietoris–Rips with **no** `max_diameter` cutoff required (note that Alpha values are **squared circumradii** $r^2$, whereas Vietoris–Rips uses **edge diameters** $2r$):

```{code-cell} ipython3
dgms_alpha = oin.compute_diagrams_alpha(pts_a, include_inf_points=False)
h1_alpha = dgms_alpha.in_dimension(1)
top_alpha = h1_alpha[np.argmax(h1_alpha[:, 1] - h1_alpha[:, 0])]
print("Longest Alpha H1 bar in r^2:", np.round(top_alpha, 4))
print("Converted to diameter 2*sqrt(r^2):", np.round(2.0 * np.sqrt(top_alpha), 4))
```

## 3. Distances & Optimal Matchings

Oineus integrates **Hera** to compute exact or $(1+\delta)$-approximate Bottleneck and $q$-Wasserstein distances—and can return the explicit bijection for inspection or plotting via {py:func}`oineus.wasserstein_matching` and {py:func}`oineus.vis.plot_matching`:

```{code-cell} ipython3
h1_a_fin = h1_a[np.isfinite(h1_a).all(axis=1)]
h1_b = dgms_vr_b.in_dimension(1)
h1_b_fin = h1_b[np.isfinite(h1_b).all(axis=1)]

b_dist = oin.bottleneck_distance(h1_a_fin, h1_b_fin, delta=0.0)
w2_match = oin.wasserstein_matching(h1_a_fin, h1_b_fin, q=2.0)

print(f"H1 bottleneck distance:    {b_dist:.4f}")
print(f"H1 2-Wasserstein distance: {w2_match.distance:.4f} ({len(w2_match.finite_to_finite)} finite pairs matched)")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
axes[0].scatter(pts_a[:, 0], pts_a[:, 1], s=20, label="Cloud A (low noise)")
axes[0].scatter(pts_b[:, 0], pts_b[:, 1], s=20, marker="^", alpha=0.8, label="Cloud B (higher noise)")
axes[0].set_aspect("equal")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("Noisy circle point clouds")

oin.plot_matching(h1_a_fin, h1_b_fin, w2_match, ax=axes[1], title="2-Wasserstein H1 matching")
fig.tight_layout()
```

## 4. Differentiating Through Persistence (`oineus.diff`)

Need gradients with respect to point coordinates or pixel values? Switch to `oineus.diff` (supporting both PyTorch and JAX):

```{code-cell} ipython3
import torch
import oineus.diff as oin_diff

pts_t = torch.tensor(pts_b, dtype=torch.float64, requires_grad=True)
dfil = oin_diff.vr_filtration(pts_t, max_dim=2, max_diameter=2.2)
dgms_t = oin_diff.persistence_diagram(dfil, include_inf_points=False)

# Maximize the persistence (death - birth) of the longest H1 bar
h1 = dgms_t[1]
pers = h1[:, 1] - h1[:, 0]
loss = -pers.max()
loss.backward()

print("Non-zero gradient vertices:", int((pts_t.grad.norm(dim=1) > 0).sum()), "of", len(pts_t))
```

## Next Steps

- Read {doc}`concepts` for the architectural mental model and decision tables across filtrations, distances, and optimization methods.
- Dive into the {doc}`../tutorials/index` to explore any of the 16 executable guides.
