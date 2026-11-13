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

# 10. Differentiable Persistence (`oineus.diff`)

Persistent homology involves discrete combinatorial sorting and $\mathbb{Z}_2$ Gaussian elimination—yet away from exact ties between filtration values, the map from input coordinates or pixel intensities to persistence diagrams is **piecewise smooth**. {py:mod}`oineus.diff` connects Oineus's multithreaded C++ reduction engine to **PyTorch** and **JAX** automatic differentiation.

By the end of this tutorial you will know how to:
- Inspect how {py:class}`oineus.diff.DiffFiltration` pairs a C++ {py:class}`oineus.Filtration` with an autograd-tracked `values` tensor.
- Choose between PyTorch and JAX (including `float32` vs `float64` routing and JAX's eager-boundary rule).
- Understand and visualize the difference between **`gradient_method="dgm-loss"`** (sparse critical-pair gradients) and **`gradient_method="crit-sets"`** (critical-set gradients with {py:class}`oineus.ConflictStrategy`).
- Inspect diagram-space loss gradients with {py:func}`oineus.plot_diagram_gradient`.

---

## 1. How Piecewise-Smooth Autodiff Works (`DiffFiltration`)

Suppose a parameterized object $\theta$ (point coordinates, image pixels, neural network weights) assigns smooth filtration values $v(\theta) \in \mathbb{R}^N$ to the $N$ cells of a complex. Whenever $v(\theta)$ has distinct cell values:

1. **Forward pass (C++)**: Oineus sorts the complex and reduces the boundary/coboundary matrix in C++ to find the integer index pairs $(i_k, j_k)$ in `sorted_id` coordinates.
2. **Gather & Differentiate (PyTorch / JAX)**: Because the integer pairing $(i_k, j_k)$ is locally constant under small perturbations of $v(\theta)$, each finite persistence point is simply a differentiable gather $(v[i_k], v[j_k])$ from `fil.values`!

Every builder in {py:mod}`oineus.diff` returns a {py:class}`oineus.diff.DiffFiltration` wrapping:
- `fil.under_fil`: the underlying C++ {py:class}`oineus.Filtration` (combinatorial cells and sorted order).
- `fil.values`: a 1D PyTorch `Tensor` or JAX `Array` aligned with `sorted_id` indices `0 .. len(fil) - 1`.

```{code-cell} ipython3
import numpy as np
import torch
import matplotlib.pyplot as plt
import oineus as oin
import oineus.diff as od

rng = np.random.default_rng(7)
pts_np = np.column_stack([
    np.cos(np.linspace(0, 2 * np.pi, 24, endpoint=False)),
    np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False)),
]) + 0.08 * rng.standard_normal((24, 2))

pts = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)

# Differentiable Vietoris-Rips filtration up to 2-simplices (for H_0 and H_1)
dfil = od.vr_filtration(pts, max_dim=2, max_diameter=1.8)
print("Underlying C++ filtration:", type(dfil.under_fil).__name__, "size =", len(dfil))
print("Differentiable values:    ", dfil.values.shape, dfil.values.dtype, "requires_grad =", dfil.values.requires_grad)

dgms = od.persistence_diagram(dfil, gradient_method="dgm-loss")
h1 = dgms[1]  # (n_bars, 2) torch.Tensor with columns [birth, death]

# Maximize total H_1 persistence sum(death - birth)
loss = -(h1[:, 1] - h1[:, 0]).sum()
loss.backward()

print(f"H_1 bars: {len(h1)}, ||d(loss)/d(pts)||_2 = {pts.grad.norm().item():.4f}")
```

> [!NOTE]
> Currently, {py:func}`oineus.diff.persistence_diagram` requires `include_inf_points=False` (the default) and returns finite persistence pairs in dimensions `0 .. fil.max_dim - 1`. Make sure your filtration has `max_dim >= k + 1` to differentiate finite $H_k$ bars!

---

## 2. PyTorch & JAX Backend Support Matrix

{py:mod}`oineus.diff` detects the framework automatically from the input tensor (`torch.Tensor` vs `jax.Array`) and returns diagrams and losses in that same framework:

| Component | Functions | PyTorch | JAX |
| :--- | :--- | :---: | :---: |
| **Grid filtrations** | `freudenthal_filtration`, `cube_filtration` | Yes | Yes |
| **Simplicial / geometric filtrations** | `vr_filtration`, `alpha_filtration`, `weak_alpha_filtration` | Yes | Yes |
| **Combinatorial constructions** | `mapping_cylinder_filtration`, `min_filtration` | Yes | Yes |
| **Exact MEB / Čech filtrations** | `cech_delaunay_filtration`, `cech_filtration`, `triangle_meb`, `tetrahedron_meb` | Yes | *PyTorch only* |
| **Diagram & inclusion extractors** | `persistence_diagram`, `kicr_diagrams`, `mixup_barcodes` | Yes | Yes |
| **Diagram distances** | `wasserstein_cost`, `sliced_wasserstein_distance`, `sliced_wasserstein_distance_diag_corrected` | Yes | Yes |
| **Harmonic $\varepsilon$-cochain losses** | `birth_content`, `death_content`, `persistence_content_loss` | Yes | *PyTorch only* |

### Precision (`float32` vs `float64`) and JAX Eager Boundary

- **Automatic dtype dispatch**: Passing `torch.float32` (or default 32-bit JAX arrays) routes C++ filtration construction and reduction through Oineus's `float32` backend (`_f32`), while `torch.float64` (or JAX with `jax_enable_x64=True`) uses `float64`.
- **JAX eager boundary**: Because persistence diagram shapes `(n_bars, 2)` depend on the data, do **not** wrap `od.vr_filtration` or `od.persistence_diagram` inside `@jax.jit` or `jax.vmap`. Instead, JIT-compile your neural network upstream and your smooth post-processing downstream, calling `jax.value_and_grad` across the eager persistence step:

```python
import jax
import jax.numpy as jnp
import oineus.diff as od

def h1_total_persistence_loss(pts_jax: jax.Array) -> jax.Array:
    fil = od.alpha_filtration(pts_jax)
    h1 = od.persistence_diagram(fil, gradient_method="dgm-loss")[1]
    return -jnp.sum(h1[:, 1] - h1[:, 0])

loss_val, grad_pts = jax.value_and_grad(h1_total_persistence_loss)(jnp.asarray(pts_np, dtype=jnp.float32))
```

---

## 3. `"dgm-loss"` vs `"crit-sets"`: Why Critical Sets Matter

{py:func}`oineus.diff.persistence_diagram` offers two backward modes via `gradient_method`:

1. **`gradient_method="dgm-loss"`** *(default)*: Backpropagates solely through the two critical cells $(\sigma_{\text{birth}}, \sigma_{\text{death}})$ of each bar. On a 2D image with 1,000+ pixels, a single gradient step touches **only 2 pixels per bar**! Moving one minimum pixel upward immediately hands the minimum over to an adjacent pixel, forcing tiny learning rates and hundreds of steps to carve or fill a basin.
2. **`gradient_method="crit-sets"`**: Uses the **critical-set method** (Nigmetov & Morozov, *Topological Optimization with Big Steps*, 2024). Given a target shift on a birth or death value (controlled by `step_size`), Oineus uses the reduced $R, V, U$ factorization to identify **every cell whose filtration value would cross the target threshold** during a big step, propagating gradients coherently across the entire basin or ridge!

When multiple bars push the same cell in different directions, `conflict_strategy` ({py:class}`oineus.ConflictStrategy`) resolves overlapping targets:
- `"avg"` (`ConflictStrategy.Avg`, default): averages target displacements on shared cells.
- `"max"` (`ConflictStrategy.Max`): picks the largest-magnitude displacement.
- `"sum"` (`ConflictStrategy.Sum`): sums displacements (standard additive subgradient).
- `"fca"` (`ConflictStrategy.FixCritAvg`): keeps critical cells exact and averages non-critical cells.

Let's compare the pixel-space gradient $|\nabla_{\text{image}} \mathcal{L}|$ of a single step that suppresses spurious $H_0$ noise basins on a 2D scalar field:

```{code-cell} ipython3
# Smooth bowl with noise basins
lin = np.linspace(-1.0, 1.0, 32)
X, Y = np.meshgrid(lin, lin)
field_np = 0.6 * (X**2 + Y**2) + 0.22 * np.sin(5 * X) * np.cos(5 * Y)

grads = {}
for method in ("dgm-loss", "crit-sets"):
    img = torch.tensor(field_np, dtype=torch.float64, requires_grad=True)
    fil_img = od.freudenthal_filtration(img, max_dim=2)
    dgm0 = od.persistence_diagram(
        fil_img,
        gradient_method=method,
        step_size=0.5,
        conflict_strategy="avg",
    )[0]
    # Penalize all finite H_0 bars (spurious local minima)
    loss_h0 = ((dgm0[:, 1] - dgm0[:, 0]) ** 2).sum()
    loss_h0.backward()
    grads[method] = img.grad.detach().numpy()
    nnz = int(np.count_nonzero(np.abs(grads[method]) > 1e-10))
    print(f"method={method:10s} -> non-zero pixel gradients: {nnz:4d} / {img.numel()}")

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
im0 = axes[0].imshow(field_np, origin="lower", cmap="viridis")
axes[0].set_title("Input scalar field")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

for ax, method in zip(axes[1:], ("dgm-loss", "crit-sets")):
    im = ax.imshow(np.abs(grads[method]), origin="lower", cmap="magma")
    ax.set_title(f"|grad| ({method})")
    fig.colorbar(im, ax=ax, fraction=0.046)

fig.tight_layout()
plt.show()
```

Notice the stark contrast: `"dgm-loss"` illuminates isolated critical saddles and minima, whereas `"crit-sets"` distributes gradients across the surrounding sublevel basins so a single optimizer step lifts an entire noisy valley at once.

---

## 4. Visualizing Diagram-Space Gradients (`plot_diagram_gradient`)

Before running an optimization loop, it is often helpful to check **where** your topological loss wants to push each bar in the birth–death plane. Calling `.retain_grad()` on a PyTorch diagram tensor and passing it to {py:func}`oineus.plot_diagram_gradient` overlays quiver arrows directly on the diagram:
- `descent=False` *(default)* draws $+\nabla_{\text{dgm}} \mathcal{L}$ (direction of steepest **increase**).
- `descent=True` draws $-\nabla_{\text{dgm}} \mathcal{L}$ (the **negative-gradient / optimization step** direction).

Let's inspect the negative gradient of {py:func}`oineus.diff.wasserstein_cost` when matching our noisy circle's $H_1$ diagram to a single target bar at `(0.35, 1.55)`:

```{code-cell} ipython3
pts_q = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
dfil_q = od.vr_filtration(pts_q, max_dim=2, max_diameter=1.8)
h1_q = od.persistence_diagram(dfil_q)[1]
h1_q.retain_grad()

target_h1 = torch.tensor([[0.35, 1.55]], dtype=torch.float64)
w2_loss = od.wasserstein_cost(h1_q, target_h1, wasserstein_q=2.0)
w2_loss.backward()

fig, ax = plt.subplots(figsize=(5.2, 4.8))
oin.plot_diagram_gradient(
    h1_q,
    descent=True,
    ax=ax,
    title="Negative gradient of W_2^2 toward target bar (0.35, 1.55)",
)
ax.scatter([0.35], [1.55], marker="*", s=160, color="gold", edgecolor="black", zorder=6, label="Target bar")
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()
```

The dominant $H_1$ point gets pulled straight toward the gold target star `(0.35, 1.55)`, while every spurious short-lived loop gets pushed orthogonally onto the diagonal $y = x$!

---

## What's Next?

- Put differentiable persistence into full optimization loops (point-cloud sculpting, periodic tori, scalar-field denoising, and lazy partial-$U$ economics) in [11. Topological Optimization Loops](11_topological_optimization.md).
- Spread point-cloud gradients smoothly across whole loops using harmonic $\varepsilon$-cochains in [12. Cochain-Based Optimization (`oineus.diff.cochains`)](12_cochain_optimization.md).
