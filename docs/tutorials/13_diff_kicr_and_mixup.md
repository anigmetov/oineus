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

# 13. Differentiable KICR & Mixup

Ordinary persistence diagrams track topological features of a **single** space. When training multi-class embeddings, domain-adaptation models, or coupled geometric configurations, you often want gradients that control **how two spaces interact**:
- Does a subcomplex $L \hookrightarrow K$ preserve or kill cycles? ({py:func}`oineus.diff.kicr_diagrams`)
- Does point cloud $B$ intrude into and prematurely fill the topological holes of point cloud $A$? ({py:func}`oineus.diff.mixup_barcodes`)

Both functions work natively in **PyTorch** and **JAX**.

By the end of this tutorial you will know how to:
- Compute differentiable kernel, image, and cokernel diagrams with {py:func}`oineus.diff.kicr_diagrams` ({py:class}`oineus.diff.KICRDiagrams`).
- Route gradients cleanly through the ambient filtration `K.values`.
- Minimize cross-cloud topological interference (`total_mixup`, `mean_mixup_percentage`) with {py:func}`oineus.diff.mixup_barcodes` ({py:class}`oineus.diff.DiffMixupBarcodes`).

---

## 1. Differentiable Kernel, Image & Cokernel Diagrams (`kicr_diagrams`)

Given an inclusion of filtrations $L \hookrightarrow K$ (where every cell of $L$ is present in $K$ with the same filtration value), {py:func}`oineus.diff.kicr_diagrams` runs C++ {py:class}`oineus.KerImCokReduced` and gathers finite `(birth, death)` pairs from `K.values`:

- `dgms.kernel[dim]`: classes born in $L$ that become boundaries earlier in $K$.
- `dgms.image[dim]`: classes born in $L$ that survive into $K$.
- `dgms.cokernel[dim]`: classes of $K$ not in the image of $L$.

> [!IMPORTANT]
> **Gradient routing rule for `kicr_diagrams(K, L)`**: Every birth and death index of all three families is gathered from **`K.values`**! Even if `L` is passed as a `DiffFiltration`, gradients flow exclusively through `K.values`. Therefore, when `L = VR(A)` and `K = VR(A ∪ B)`, build `K` from `pts_union = torch.cat([A, B], dim=0)` so gradients flow back through `K.values` into **both** `A` and `B`!

Let's verify this on an inclusion $L = \operatorname{VR}(A) \hookrightarrow K = \operatorname{VR}(A \cup B)$ where $A$ is a unit circle and $B$ places points near the origin (creating a **kernel** $H_1$ bar because $A$'s loop is killed earlier in $A \cup B$):

```{code-cell} ipython3
import numpy as np
import torch
import matplotlib.pyplot as plt
import oineus as oin
import oineus.diff as od

rng = np.random.default_rng(21)
theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
A_np = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.04 * rng.standard_normal((24, 2))
B_np = 0.28 * rng.standard_normal((10, 2))  # intruding inside A's hole!

A_t = torch.tensor(A_np, dtype=torch.float64, requires_grad=True)
B_t = torch.tensor(B_np, dtype=torch.float64, requires_grad=True)

# Route both A_t and B_t through K = VR(cat([A_t, B_t]))
pts_union = torch.cat([A_t, B_t], dim=0)
K = od.vr_filtration(pts_union, max_dim=2, max_diameter=2.1)
L = oin.vr_filtration(A_np, max_dim=2, max_diameter=2.1, packed=False)

kicr = od.kicr_diagrams(K, L, kernel=True, image=True, cokernel=True)
print("Families computed:", kicr.families)
print("Image H1 bars:   ", kicr.image[1].detach().numpy().round(4))
print("Kernel H1 bars:  ", kicr.kernel[1].detach().numpy().round(4))

# Penalize image death occurring early -> maximize image H_1 persistence!
im1 = kicr.image[1]
loss_im = -(im1[:, 1] - im1[:, 0]).sum()
loss_im.backward()

print(f"||d(loss)/dA|| = {A_t.grad.norm().item():.4f}, ||d(loss)/dB|| = {B_t.grad.norm().item():.4f}")
```

---

## 2. Differentiable Mixup Barcodes (`oineus.diff.mixup_barcodes`)

When your inclusion is specifically $L = \operatorname{VR}(A) \hookrightarrow K = \operatorname{VR}(A \cup B)$, {py:func}`oineus.diff.mixup_barcodes` automates the union construction, induced matching, and differentiable gather in one call!

It returns a {py:class}`oineus.diff.DiffMixupBarcodes` container where `mb[dim]` is a differentiable `(n_bars, 3)` tensor of `(birth, image_death, death)` triples with four built-in differentiable summary scalars:
- `mb.total_persistence(dim)`: $\sum (d - b)$
- `mb.total_mixup(dim)`: $\sum (d - d')$ — total premature lifetime loss caused by $B$
- `mb.total_mixup_percentage(dim)`: $\sum \frac{d - d'}{d - b}$
- `mb.mean_mixup_percentage(dim)`: scale-invariant mean in $[0, 1]$

### Untangling Two Point Clouds by Minimizing `total_mixup(1)`

Suppose class $A$ is a ring and class $B$ initially sits right inside $A$'s hole, causing heavy $H_1$ topological mixup ($d' \ll d$). Let's hold $A$ fixed and optimize $B$'s coordinates to **minimize `total_mixup(1)`** (plus a gentle tether keeping $B$ coherent):

```{code-cell} ipython3
A_fixed = torch.tensor(A_np, dtype=torch.float64)
B_opt = torch.tensor(B_np + np.array([0.25, 0.15]), dtype=torch.float64, requires_grad=True)

optimizer = torch.optim.Adam([B_opt], lr=0.05)
mixup_hist = []

for step in range(45):
    optimizer.zero_grad()
    mb = od.mixup_barcodes(A_fixed, B_opt, max_dim=1)
    mix_loss = mb.total_mixup(1)
    # Mild cohesion term so B moves out as a compact cluster rather than flying to infinity
    center_B = B_opt.mean(dim=0, keepdim=True)
    cohesion = 0.05 * ((B_opt - center_B) ** 2).sum() + 0.02 * torch.relu(B_opt.norm(dim=1) - 1.8).pow(2).sum()
    loss = mix_loss + cohesion
    loss.backward()
    optimizer.step()
    mixup_hist.append(float(mix_loss.item()))

B_final = B_opt.detach().numpy()
mb_init = oin.mixup_barcodes(A_np, B_np + np.array([0.25, 0.15]), max_dim=1)
mb_fin = oin.mixup_barcodes(A_np, B_final, max_dim=1)

print(
    f"H_1 total mixup: {mb_init.total_mixup(1):.4f} (mean % = {mb_init.mean_mixup_percentage(1):.2%}) -> "
    f"{mb_fin.total_mixup(1):.4f} (mean % = {mb_fin.mean_mixup_percentage(1):.2%})"
)

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
axes[0].scatter(A_np[:, 0], A_np[:, 1], s=34, color=oin.OKABE_ITO_BLUE, label="Cloud A (ring)")
axes[0].scatter(
    B_np[:, 0] + 0.25, B_np[:, 1] + 0.15,
    s=34, marker="x", color="0.45", label="Cloud B (initial inside hole)",
)
axes[0].scatter(B_final[:, 0], B_final[:, 1], s=42, color=oin.OKABE_ITO_VERMILLION, label="Cloud B (optimized)")
axes[0].set_aspect("equal")
axes[0].set_title("Minimizing H_1 Mixup Ejects B from A's Hole")
axes[0].legend(fontsize=8, loc="upper left")

axes[1].plot(mixup_hist, color=oin.OKABE_ITO_VERMILLION, linewidth=2.0)
axes[1].set_xlabel("Adam step")
axes[1].set_ylabel("total_mixup(dim=1)")
axes[1].set_title("H_1 Topological Mixup Decay")

fig.tight_layout()
plt.show()
```

### JAX Equivalent

Because both `od.kicr_diagrams` and `od.mixup_barcodes` use framework-neutral index-gather autodiff, the exact same objective works in JAX via `jax.value_and_grad`:

```python
import jax
import jax.numpy as jnp
import oineus.diff as od

def mixup_loss_jax(B_jax: jax.Array, A_jax: jax.Array) -> jax.Array:
    mb = od.mixup_barcodes(A_jax, B_jax, max_dim=1)
    return mb.total_mixup(1)

val, grad_B = jax.value_and_grad(mixup_loss_jax)(
    jnp.asarray(B_np, dtype=jnp.float32),
    jnp.asarray(A_np, dtype=jnp.float32),
)
```

---

## What's Next?

- Update reduced decompositions dynamically without re-reducing from scratch using vineyards, Luo–Nelson warm starts, and SiRUP cell pruning in [14. Dynamic Persistence, Vineyards & SiRUP](14_dynamic_persistence_and_sirup.md).
