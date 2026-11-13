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

# 11. Topological Optimization Loops

Once persistence diagrams and distances are differentiable, you can compose them freely with standard geometric regularizers and PyTorch/JAX optimizers (`Adam`, `SGD`). This tutorial walks through three practical topological optimization patterns—plus how Oineus keeps `"crit-sets"` reduction fast via lazy partial-$U$ recovery.

By the end of this tutorial you will know how to:
- Sculpt a noisy 2D point cloud into a clean topological loop with {py:func}`oineus.diff.cech_delaunay_filtration` and {py:func}`oineus.diff.wasserstein_cost`.
- Optimize point clouds on a **flat periodic torus** (`periodic=True` + `torch.remainder`).
- Denoise a 2D scalar field by driving spurious $H_0$ minima to the diagonal with `"crit-sets"` and {py:class}`oineus.diff.TopologyOptimizer`.
- Configure {py:class}`oineus.UStrategy` (`Auto` / `RowPartial` vs `LegacyInBand`) for large complexes.

---

## 1. Sculpting a Point Cloud with `cech_delaunay_filtration`

For point clouds in $\mathbb{R}^2$ or $\mathbb{R}^3$, {py:func}`oineus.diff.cech_delaunay_filtration` is ideal for optimization:
- Unlike Vietoris–Rips ($O(n^3)$ triangles), the Delaunay triangulation has $O(n)$ simplices in 2D.
- Unlike raw circumradius (`alpha_filtration`), **Čech-Delaunay** evaluates exact differentiable **minimum enclosing ball (MEB)** radii ({py:func}`oineus.diff.triangle_meb`, {py:func}`oineus.diff.tetrahedron_meb`) on Delaunay simplices—matching true Čech persistent homology without monotonicity inversions on obtuse triangles!

> [!NOTE]
> Just like `oin.alpha_filtration`, `od.cech_delaunay_filtration` stores **squared MEB radii** $r^2$ as filtration values. Keep that in mind when choosing template birth/death coordinates!

Let's start from 36 noisy points scattered in $[-1, 1]^2$ and prescribe a single prominent $H_1$ bar with birth $r_b^2 = 0.04$ and death $r_d^2 = 0.85$, combined with a mild centroid-centering penalty:

```{code-cell} ipython3
import numpy as np
import torch
import matplotlib.pyplot as plt
import oineus as oin
import oineus.diff as od

rng = np.random.default_rng(12)
angles0 = np.linspace(0, 2 * np.pi, 32, endpoint=False)
radii0 = 0.65 + 0.08 * rng.standard_normal(32)
pts0 = np.column_stack([radii0 * np.cos(angles0), radii0 * np.sin(angles0)]) + 0.03 * rng.standard_normal((32, 2))
pts = torch.tensor(pts0, dtype=torch.float64, requires_grad=True)

target_h1 = torch.tensor([[0.015, 0.55]], dtype=torch.float64)
optimizer = torch.optim.Adam([pts], lr=0.02)

loss_history = []
for step in range(80):
    optimizer.zero_grad()
    fil = od.cech_delaunay_filtration(pts)
    h1 = od.persistence_diagram(fil, gradient_method="dgm-loss")[1]

    topo_loss = od.wasserstein_cost(h1, target_h1, wasserstein_q=2.0, wasserstein_delta=0.02)
    reg_loss = 0.02 * pts.mean(dim=0).pow(2).sum()
    loss = topo_loss + reg_loss
    loss.backward()
    optimizer.step()
    loss_history.append(float(topo_loss.item()))

pts_final = pts.detach().numpy()
fil0 = od.cech_delaunay_filtration(torch.tensor(pts0, dtype=torch.float64))
h1_init = od.persistence_diagram(fil0)[1].detach().numpy()
h1_fin = od.persistence_diagram(od.cech_delaunay_filtration(pts.detach()))[1].detach().numpy()

print(f"Initial W_2^2 loss: {loss_history[0]:.4f} -> Final W_2^2 loss: {loss_history[-1]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
axes[0].scatter(pts0[:, 0], pts0[:, 1], s=28, color="0.45", label="Initial")
axes[0].scatter(pts_final[:, 0], pts_final[:, 1], s=32, color=oin.OKABE_ITO_VERMILLION, label="Optimized")
axes[0].set_aspect("equal")
axes[0].set_title("Point cloud: noisy loop -> target loop")
axes[0].legend(fontsize=8)

oin.plot_diagram({1: h1_fin}, ax=axes[1], title="Final H_1 diagram vs target")
axes[1].scatter([0.015], [0.55], marker="*", s=160, color="gold", edgecolor="black", zorder=6, label="Target")
axes[1].legend(fontsize=8, loc="lower right")

axes[2].plot(loss_history, color=oin.OKABE_ITO_BLUE, linewidth=1.8)
axes[2].set_xlabel("Adam step")
axes[2].set_ylabel("Topological W_2^2 loss")
axes[2].set_title("Convergence")

fig.tight_layout()
plt.show()
```

---

## 2. Optimizing on Periodic Domains (Tori)

Simulations of crystals, foams, and periodic PDEs live on a flat torus $\mathbb{T}^d$. Oineus supports differentiable periodic topology on both **grids** and **point clouds**:

- **Periodic grids (`wrap=True`)**: Pass `wrap=True` to {py:func}`oineus.diff.freudenthal_filtration` to identify opposite boundaries seamlessly across `[0, H) x [0, W)`. Let's carve a non-contractible meridian band across a noisy 2D torus so one essential-like $H_1$ loop stands out:

```{code-cell} ipython3
grid_torus0 = 0.5 + 0.25 * rng.standard_normal((20, 20))
u_torus = torch.tensor(grid_torus0, dtype=torch.float64, requires_grad=True)
opt_torus = torch.optim.Adam([u_torus], lr=0.05)

for _ in range(30):
    opt_torus.zero_grad()
    # max_dim=2 builds vertices, edges, and triangles on T^2
    fil_per = od.freudenthal_filtration(u_torus, wrap=True, max_dim=2)
    h1_per = od.persistence_diagram(fil_per, gradient_method="crit-sets", step_size=0.5)[1]
    pers = h1_per[:, 1] - h1_per[:, 0]
    # Maximize the longest finite H_1 bar while suppressing clutter
    top_idx = torch.argmax(pers)
    mask = torch.ones_like(pers, dtype=torch.bool)
    mask[top_idx] = False
    loss_per = -pers[top_idx] + 0.2 * (pers[mask] ** 2).sum()
    loss_per.backward()
    opt_torus.step()

print("Longest periodic-grid finite H_1 persistence:", round(float(pers[top_idx].item()), 4))
```

- **Periodic point clouds (`periodic=True`)**: When using a `diode` build compiled with periodic lift exports (`diode.fill_periodic_delaunay_lifts_arrays`), {py:func}`oineus.diff.cech_delaunay_filtration` also differentiates minimum-image distances on $[0, L)^d$; wrap coordinates back into the box with `torch.remainder` after each optimizer step:

```python
fil_per = od.cech_delaunay_filtration(
    pts_torus, periodic=True, bbox_min=[0.0, 0.0], bbox_max=[1.0, 1.0]
)
loss = -od.persistence_diagram(fil_per)[1][:, 1].sum()
loss.backward()
optimizer.step()
with torch.no_grad():
    pts_torus.copy_(torch.remainder(pts_torus, 1.0))
```

---

## 3. Topological Denoising of a 2D Scalar Field

Suppose we observe a 2D height field with **two true wells** corrupted by high-frequency ripples that create dozens of spurious $H_0$ local minima. Using `gradient_method="crit-sets"`, we can penalize all finite $H_0$ bars shorter than a persistence threshold $\varepsilon = 0.35$ while preserving the deep wells:

```{code-cell} ipython3
lin = np.linspace(-1.2, 1.2, 36)
X, Y = np.meshgrid(lin, lin)
clean_wells = -0.9 * np.exp(-4.0 * ((X + 0.5) ** 2 + Y**2)) - 0.85 * np.exp(-4.0 * ((X - 0.5) ** 2 + Y**2))
noisy_wells = clean_wells + 0.16 * np.sin(8.0 * X) * np.cos(8.0 * Y)

u = torch.tensor(noisy_wells, dtype=torch.float64, requires_grad=True)
opt_u = torch.optim.SGD([u], lr=0.45)
eps_thr = 0.35

for _ in range(20):
    opt_u.zero_grad()
    fil_u = od.freudenthal_filtration(u, max_dim=2)
    h0 = od.persistence_diagram(
        fil_u, gradient_method="crit-sets", step_size=0.5, conflict_strategy="avg"
    )[0]
    pers0 = h0[:, 1] - h0[:, 0]
    noise_bars = pers0[pers0 < eps_thr]
    loss_u = (noise_bars**2).sum()
    loss_u.backward()
    opt_u.step()

u_denoised = u.detach().numpy()
h0_before = oin.compute_diagrams_ls(noisy_wells, negate=False, include_inf_points=False)[0]
h0_after = oin.compute_diagrams_ls(u_denoised, negate=False, include_inf_points=False)[0]
spurious_before = h0_before[(h0_before[:, 1] - h0_before[:, 0]) < eps_thr]
spurious_after = h0_after[(h0_after[:, 1] - h0_after[:, 0]) < eps_thr]

print(
    f"Total spurious H_0 persistence (< {eps_thr}): "
    f"{np.sum(spurious_before[:, 1] - spurious_before[:, 0]):.4f} -> "
    f"{np.sum(spurious_after[:, 1] - spurious_after[:, 0]):.4f}"
)

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
vmin, vmax = float(noisy_wells.min()), float(noisy_wells.max())
axes[0].imshow(noisy_wells, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
axes[0].set_title("Noisy scalar field")
axes[1].imshow(u_denoised, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
axes[1].set_title("Topologically denoised (crit-sets)")

axes[2].scatter(h0_before[:, 0], h0_before[:, 1], s=26, alpha=0.6, color="0.5", label="Before")
if len(h0_after):
    axes[2].scatter(h0_after[:, 0], h0_after[:, 1], s=36, color=oin.OKABE_ITO_VERMILLION, label="After")
lo, hi = float(h0_before.min()) - 0.05, float(h0_before.max()) + 0.05
axes[2].plot([lo, hi], [lo, hi], "--", color="0.6")
axes[2].set_title("Finite H_0 diagram")
axes[2].set_xlabel("birth")
axes[2].set_ylabel("death")
axes[2].legend(fontsize=8)

fig.tight_layout()
plt.show()
```

### Direct Prescriptive Targets with `TopologyOptimizer.simplify`

When you want explicit simplex-level target values $(i, \hat{v}_i)$ (for example, to plug into custom solvers or inspect exactly which cells move under `BirthBirth`, `DeathDeath`, or `Midway` strategies), instantiate {py:class}`oineus.diff.TopologyOptimizer` directly and read zero-copy NumPy views via `.indices_array()` and `.values_array()`:

```{code-cell} ipython3
fil_check = od.freudenthal_filtration(torch.tensor(noisy_wells, dtype=torch.float64), max_dim=2)
top_opt = od.TopologyOptimizer(fil_check, with_crit_sets=True, n_threads=2)
top_opt.reduce_all()

# Prescribe Midway targets collapsing all H_0 bars with persistence <= 0.35
iv = top_opt.simplify(epsilon=0.35, strategy=oin.DenoiseStrategy.Midway, dim=0)
idx_arr = np.array(iv.indices_array())
val_arr = np.array(iv.values_array())

expanded = top_opt.crit_sets_apply(idx_arr.tolist(), val_arr.tolist(), oin.ConflictStrategy.Avg)
print(f"Singleton targets: {len(idx_arr)} cells -> Critical-set expanded targets: {len(expanded.indices_array())} cells")
```

---

## 4. Economics of `"crit-sets"` and `UStrategy`

Historically, critical-set methods required computing the **entire** canonical Edelsbrunner–Letscher–Zomorodian (ELZ) matrix $V$ and inverting all of $U = V^{-1}$ in $O(N^2)$ memory/time on every forward step—even if the loss only moved 5 bars!

Oineus eliminates that bottleneck via **`UStrategy.Auto`** (which resolves to `UStrategy.RowPartial`, the default):
1. **Forward pass**: Reduces $R = D V$ using multithreaded clearing—**no** ELZ post-pass and **no** $U$ allocation.
2. **Backward pass (demand-driven)**: Only when upstream gradients arrive does Oineus restore ELZ form on the negative $V$ columns in the active dimension block and solve *only* the sparse rows of $U$ touched by moved bars (`ensure_has_u_hom` / `ensure_has_u_coh`).
3. **Opposite duality on demand**: Homology vs cohomology is only reduced on the opposite side if a critical set actually asks to push a death or birth in a direction that requires the dual factorization.

| `u_strategy` | Behavior | Best For |
| :--- | :--- | :--- |
| `UStrategy.Auto` (`"auto"`, default) | Resolves to `RowPartial` | Default for all `"crit-sets"` workloads |
| `UStrategy.RowPartial` (`"row_partial"`) | Parallel $R+V$ forward; lazy per-block ELZ + sparse target-row $U$ solve in backward | Large grids and geometric complexes where losses move a sparse subset of bars |
| `UStrategy.LegacyInBand` (`"legacy_in_band"`) | Serial full-ELZ reduction + eager full $U = V^{-1}$ | Reference oracle / debugging small complexes |

---

## What's Next?

- See why moving point clouds via single critical edges can create polygonal corners—and how harmonic $\varepsilon$-cochains fix it—in [12. Cochain-Based Optimization (`oineus.diff.cochains`)](12_cochain_optimization.md).
- Differentiate topological inclusion and mixup losses between two point clouds in [13. Differentiable KICR & Mixup](13_diff_kicr_and_mixup.md).
