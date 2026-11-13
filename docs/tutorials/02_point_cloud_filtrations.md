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

# 2. Point-Cloud Filtrations

Given a point cloud $P = \{x_0, \dots, x_{N-1}\} \subset \mathbb{R}^d$ (or an $N \times N$ pairwise distance matrix), Oineus offers multiple geometric filtration families with distinct combinatorial sizes and scale parameterizations:

- **Vietoris–Rips** ({py:func}`oineus.vr_filtration`, {py:func}`oineus.compute_diagrams_vr`),
- **Alpha** ({py:func}`oineus.alpha_filtration`, {py:func}`oineus.compute_diagrams_alpha`, via `diode`),
- **Čech** ({py:func}`oineus.cech_filtration`),
- plus differentiable **Weak Alpha** and **Čech–Delaunay** filtrations in `oineus.diff` ({py:func}`oineus.diff.weak_alpha_filtration`, {py:func}`oineus.diff.cech_delaunay_filtration`).

---

## 1. Vietoris–Rips: Filtration `max_dim` vs. Diagram `max_dim`

In the Vietoris–Rips complex $\operatorname{VR}_t(P)$, a $k$-simplex $[v_0, \dots, v_k]$ enters at its maximum pairwise Euclidean distance (diameter):

$$
\operatorname{val}([v_0, \dots, v_k]) = \max_{0 \le i < j \le k} \|x_{v_i} - x_{v_j}\|_2.
$$

:::{important}
**Understand `max_dim` in `vr_filtration` vs. `compute_diagrams_vr`:**
- {py:func}`oineus.vr_filtration` builds simplices **up to cell dimension `max_dim`**. To compute finite death times in homology dimension $k$, you must pass `max_dim = k + 1` so $(k+1)$-simplices are present to bound $k$-cycles!
- {py:func}`oineus.compute_diagrams_vr` interprets `max_dim` as the **maximum diagram dimension**: passing `max_dim=1` automatically builds the 2-skeleton (`vr_filtration(..., max_dim=2)`) and trims the returned {py:class}`oineus.Diagrams` to $H_0$ and $H_1$.
:::

Let's verify both routes on a noisy torus curve in $\mathbb{R}^3$, and also show how `from_pwdists=True` works directly on a custom distance matrix:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import oineus as oin

rng = np.random.default_rng(12)
t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
# Trefoil-like loop in R^3
pts = np.column_stack([
    np.sin(t) + 2.0 * np.sin(2.0 * t),
    np.cos(t) - 2.0 * np.cos(2.0 * t),
    -np.sin(3.0 * t),
]) + 0.08 * rng.standard_normal((90, 3))

# High-level shortcut: max_dim=1 means H0 and H1 diagrams (builds 2-simplices internally)
dgms_shortcut = oin.compute_diagrams_vr(
    pts, max_dim=1, max_diameter=3.0, params=oin.ReductionParams(n_threads=4)
)

# Explicit filtration + reduction: pass max_dim=2 so triangles kill 1-cycles!
fil_vr, crit_edges = oin.vr_filtration(
    pts, max_dim=2, max_diameter=3.0, with_critical_edges=True, n_threads=4
)
dcmp = oin.reduce(fil_vr, dualize=True, n_threads=4)
dgms_explicit = dcmp.diagram(fil_vr)

assert np.allclose(dgms_shortcut.in_dimension(1), dgms_explicit.in_dimension(1))
print(f"VR 2-skeleton size (max_diameter=3.0): {fil_vr.size():,d} simplices")
print(f"Critical edges shape: {crit_edges.shape} (endpoints of longest edge per simplex)")

# Distance-matrix input (e.g., geodesic or correlation distances)
D = cdist(pts, pts, metric="cityblock")
dgms_l1 = oin.compute_diagrams_vr(D, from_pwdists=True, max_dim=1, max_diameter=4.5)
print("L1-metric H1 top bar:", np.round(dgms_l1[1][np.argmax(dgms_l1[1][:, 1] - dgms_l1[1][:, 0])], 4))
```

:::{tip}
Use {py:func}`oineus.max_distance` (`oin.max_distance(pts)` or `oin.max_distance(D, from_pwdists=True)`) if you want to inspect the diameter of your dataset before choosing `max_diameter`.
:::

---

## 2. Alpha Filtrations: Exact Nerve with Far Fewer Simplices

For points in $\mathbb{R}^2$ or $\mathbb{R}^3$, the **Alpha complex** restricts the Čech complex to the Delaunay triangulation (computed via CGAL in `diode`). By the Nerve Theorem, the union of balls $\bigcup_i B_r(x_i)$ is homotopy equivalent to the Alpha sublevel complex at **squared radius** $r^2$, yet the complex has orders of magnitude fewer simplices than Vietoris–Rips!

{py:func}`oineus.alpha_filtration` and {py:func}`oineus.compute_diagrams_alpha` also support:

- **Weighted Alpha shapes** (`weights=...`, power-cell radii $r^2 - w_i$),
- **Periodic boxes** (`periodic=True`, `bbox_min=...`, `bbox_max=...`),
- **Exact CGAL geometric predicates** (`exact=True`).

```{code-cell} ipython3
# Sample 250 points on a hollow 3D sphere + mild radial noise
u = rng.standard_normal((250, 3))
sphere_pts = u / np.linalg.norm(u, axis=1, keepdims=True) + 0.04 * rng.standard_normal((250, 3))

fil_alpha = oin.alpha_filtration(sphere_pts, exact=True)
fil_vr_3d = oin.vr_filtration(sphere_pts, max_dim=3, max_diameter=2.1, n_threads=4)

print(f"250-pt 3D sphere -> Alpha cells: {fil_alpha.size():,d} vs. VR 3-skeleton: {fil_vr_3d.size():,d}")

dgms_alpha = oin.reduce(fil_alpha, n_threads=4).diagram(fil_alpha, include_inf_points=False)
h2 = dgms_alpha.in_dimension(2)
print("Prominent Alpha H2 void (squared radius r^2):", np.round(h2[np.argmax(h2[:, 1] - h2[:, 0])], 4))

# Weighted Alpha: assign larger weights to half the sphere so balls grow earlier there
weights = np.where(sphere_pts[:, 2] > 0, 0.08, 0.0)
dgms_weighted = oin.compute_diagrams_alpha(sphere_pts, weights=weights, include_inf_points=False)
print("Weighted Alpha H2 top bar:", np.round(dgms_weighted[2][np.argmax(dgms_weighted[2][:, 1] - dgms_weighted[2][:, 0])], 4))
```

### Periodic Alpha on a Flat Torus

Passing `periodic=True` computes the periodic Delaunay triangulation on $[x_{\min}, x_{\max}]^d$ (note: CGAL's 3D/2D periodic triangulation requires points strictly inside a cubic/square domain with sufficiently dense cover, or falls back cleanly when needed):

```{code-cell} ipython3
pts_torus = rng.uniform(0.05, 0.95, size=(120, 3))
fil_per = oin.alpha_filtration(
    pts_torus, periodic=True, compute_bounding_box=False,
    bbox_min=[0.0, 0.0, 0.0], bbox_max=[1.0, 1.0, 1.0],
)
dgms_per = oin.reduce(fil_per).diagram(fil_per, include_inf_points=True)
print("3D periodic Alpha essential counts (H0, H1, H2, H3):", [
    int(np.isinf(dgms_per[q]).any(axis=1).sum()) for q in range(4)
])
```

Notice the exact Betti numbers of the 3-torus $\mathbb{T}^3$: $(1, 3, 3, 1)$!

---

## 3. Čech, Weak Alpha & Čech–Delaunay: Scale Conventions Side by Side

Let's compare all point-cloud complexes on a clean 2D ring of radius $R = 1.0$, paying close attention to each filtration's **value parameterization**:

| Function | Complex | Filtration Value | Death of Unit Circle |
| --- | --- | --- | --- |
| {py:func}`oineus.vr_filtration` | All cliques $\le \text{max\_diameter}$ | Edge **diameter** $2r$ | $\sqrt{3} \approx 1.732$ (equilateral triangle side) |
| {py:func}`oineus.cech_filtration` | All subsets $\le \text{max\_radius}$ | MEB **radius** $r$ | $1.0$ (unit circumcircle radius) |
| {py:func}`oineus.alpha_filtration` | Gabriel / Delaunay cells | Circumradius **squared** $r^2$ | $1.0^2 = 1.0$ |
| {py:func}`oineus.diff.cech_delaunay_filtration` | Delaunay simplices with MEB values | MEB **squared radius** $r^2$ | $1.0^2 = 1.0$ |
| {py:func}`oineus.diff.weak_alpha_filtration` | Delaunay simplices with circumsphere values (made monotone) | Circumradius **squared** $r^2$ | $1.0^2 = 1.0$ |

Let's verify this numerically and plot the diagrams converted to a common radius scale $r$:

```{code-cell} ipython3
import torch
import oineus.diff as oin_diff

angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
ring = np.column_stack([np.cos(angles), np.sin(angles)]) + 0.03 * rng.standard_normal((24, 2))

# 1. Vietoris-Rips (values = diameter 2r -> divide by 2 for radius r)
dgm_vr_r = oin.compute_diagrams_vr(ring, max_dim=1, include_inf_points=False)[1] / 2.0

# 2. Full Cech (values = radius r directly!)
fil_cech = oin.cech_filtration(ring, max_dim=2, max_radius=1.2, n_threads=4)
dgm_cech_r = oin.reduce(fil_cech).diagram(fil_cech, include_inf_points=False)[1]

# 3. Alpha (values = squared radius r^2 -> sqrt for radius r)
dgm_alpha_r = np.sqrt(np.maximum(0.0, oin.compute_diagrams_alpha(ring, include_inf_points=False)[1]))

# 4. Cech-Delaunay via oineus.diff (values = squared radius r^2 -> sqrt for radius r)
dfil_cd = oin_diff.cech_delaunay_filtration(torch.tensor(ring, dtype=torch.float64))
dgm_cd_r = np.sqrt(np.maximum(0.0, oin_diff.persistence_diagram(dfil_cd)[1].detach().numpy()))

print("H1 longest bar in radius r:")
print("  VR (diam / 2):      ", np.round(dgm_vr_r[np.argmax(dgm_vr_r[:, 1] - dgm_vr_r[:, 0])], 4))
print("  Full Cech (r):      ", np.round(dgm_cech_r[np.argmax(dgm_cech_r[:, 1] - dgm_cech_r[:, 0])], 4))
print("  Alpha (sqrt(r^2)):  ", np.round(dgm_alpha_r[np.argmax(dgm_alpha_r[:, 1] - dgm_alpha_r[:, 0])], 4))
print("  Cech-Delaunay (r):  ", np.round(dgm_cd_r[np.argmax(dgm_cd_r[:, 1] - dgm_cd_r[:, 0])], 4))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
axes[0].scatter(ring[:, 0], ring[:, 1], s=30, color=oin.OKABE_ITO_BLUE)
axes[0].set_aspect("equal")
axes[0].set_title("24-point noisy unit circle")

axes[1].plot([0, 1.1], [0, 1.1], "--", color="0.6")
axes[1].scatter(dgm_vr_r[:, 0], dgm_vr_r[:, 1], s=45, label="VR (diameter / 2)", marker="o")
axes[1].scatter(dgm_alpha_r[:, 0], dgm_alpha_r[:, 1], s=55, label="Alpha / Cech / Cech-Delaunay (r)", marker="x")
axes[1].set_xlabel("Birth radius r")
axes[1].set_ylabel("Death radius r")
axes[1].legend(fontsize=8)
axes[1].set_title("VR vs. Nerve (Cech / Alpha) in radius units")
fig.tight_layout()
```

Notice that **Full Čech**, **Alpha**, and **Čech–Delaunay** produce the *exact same* $H_1$ diagram in radius units ($d \approx 1.0$), whereas **Vietoris–Rips** fills the circle earlier at $r = \frac{\sqrt{3}}{2} \approx 0.866$ because an inscribed equilateral triangle has edge length $\sqrt{3} R$ even before the three balls share a triple intersection!
