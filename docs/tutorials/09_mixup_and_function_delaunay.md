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

# 9. Mixup Barcodes & Function–Delaunay Bifiltrations

Suppose you have **two point populations $A$ and $B$ in the same ambient space**—for example:
- tumor cells $A$ vs. infiltrating immune cells $B$ in spatial transcriptomics/pathology,
- class-$0$ vs. class-$1$ latent embeddings in a neural network layer,
- galaxies vs. void tracers in cosmology.

Computing persistence of $A$ and $B$ *separately* misses how they **topologically interact**. **Mixup barcodes** (Wagner, Carrenson, Bauer & Lesnick, [arXiv:2402.15058](https://arxiv.org/abs/2402.15058)) track every homology bar $[b, d)$ of $A$ under the inclusion $\operatorname{Filt}(A) \hookrightarrow \operatorname{Filt}(A \cup B)$:

- **$b$ (birth)**: scale at which the feature is born inside $A$,
- **$d'$ (image death)**: scale at which the feature dies inside $A \cup B$ ($b \le d' \le d$),
- **$d$ (original death)**: scale at which the feature dies inside $A$ alone.

Each bar $[b, d)$ splits into two sub-bars:
1. **Image sub-bar $[b, d')$**: scales where the feature survives even in the presence of $B$.
2. **Mixup sub-bar $[d', d)$**: persistence *lost* because points of $B$ bridged or filled the feature of $A$! The **mixup percentage** $\frac{d - d'}{d - b} \in [0, 1]$ measures topological entanglement ($0$ = completely separated, $\approx 1$ = $B$ sits right inside $A$'s holes/gaps).

---

## 1. Vietoris–Rips Mixup Barcodes (`mixup_barcodes`)

{py:func}`oineus.mixup_barcodes` computes Vietoris–Rips mixup triples `(b, d', d)` across dimensions `0..max_dim` and returns a {py:class}`oineus.MixupBarcodes` object. Let's compare two geometric configurations of a circle $A$ and a secondary cloud $B$:
1. **$B_{\text{out}}$**: $B$ sits well outside the circle $A$ $\implies$ zero $H_1$ mixup!
2. **$B_{\text{in}}$**: $B$ is sprinkled right inside the central hole of $A$ $\implies$ $B$ fills the loop early ($d' \ll d$), producing high $H_1$ mixup!

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import oineus as oin

rng = np.random.default_rng(42)
angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
A = np.column_stack([np.cos(angles), np.sin(angles)]) + 0.05 * rng.standard_normal((36, 2))

B_out = np.array([3.5, 0.0]) + 0.25 * rng.standard_normal((12, 2))
B_in = 0.25 * rng.standard_normal((12, 2))

mb_out = oin.mixup_barcodes(A, B_out, max_dim=1, n_threads=4)
mb_in = oin.mixup_barcodes(A, B_in, max_dim=1, n_threads=4)

print("B outside circle -> H1 total mixup:", round(mb_out.total_mixup(1), 4),
      "| H1 total mixup %:", round(100 * mb_out.total_mixup_percentage(1), 2))
print("B inside  circle -> H1 total mixup:", round(mb_in.total_mixup(1), 4),
      "| H1 total mixup %:", round(100 * mb_in.total_mixup_percentage(1), 2))
print("Dominant H1 (birth, image_death, death) with B_in:\n",
      np.round(mb_in[1][np.argmax(mb_in[1][:, 2] - mb_in[1][:, 0])], 4))
```

Let's plot the point configurations alongside the decomposed **Image $[b, d')$** + **Mixup $[d', d)$** $H_1$ barcode:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

axes[0].scatter(A[:, 0], A[:, 1], s=25, color=oin.OKABE_ITO_BLUE, label="A (ring)")
axes[0].scatter(B_out[:, 0], B_out[:, 1], s=30, marker="^", color=oin.OKABE_ITO_VERMILLION, label="B_out")
axes[0].set_aspect("equal")
axes[0].legend(fontsize=8)
axes[0].set_title("Disjoint: 0% H1 mixup")

axes[1].scatter(A[:, 0], A[:, 1], s=25, color=oin.OKABE_ITO_BLUE, label="A (ring)")
axes[1].scatter(B_in[:, 0], B_in[:, 1], s=30, marker="^", color=oin.OKABE_ITO_VERMILLION, label="B_in")
axes[1].set_aspect("equal")
axes[1].legend(fontsize=8)
axes[1].set_title("Entangled: B fills A's hole")

triples = mb_in.in_dimension(1)
order = np.argsort(triples[:, 0])
for y, (b, dp, d) in enumerate(triples[order]):
    axes[2].plot([b, dp], [y, y], lw=6, color=oin.OKABE_ITO_BLUE, solid_capstyle="butt",
                 label="Image [b, d')" if y == 0 else None)
    if d > dp:
        axes[2].plot([dp, d], [y, y], lw=6, color=oin.OKABE_ITO_VERMILLION, solid_capstyle="butt",
                     label="Mixup [d', d)" if y == len(triples) - 1 else None)
axes[2].set_xlabel("VR Diameter")
axes[2].set_yticks(range(len(triples)))
axes[2].set_yticklabels([f"bar {i}" for i in range(len(triples))])
axes[2].legend(loc="lower right", fontsize=8)
axes[2].set_title("H1 Mixup Barcode (B_in)")
fig.tight_layout()
```

---

## 2. Scaling to Thousands of Points with `function_delaunay_bifiltration`

Why not simply compute Alpha filtrations $\operatorname{Alpha}(A)$ and $\operatorname{Alpha}(A \cup B)$ for large 2D/3D point clouds?
Because inserting points of $B$ **changes the Delaunay triangulation**, so $\operatorname{Alpha}(A)$ is *not* a subcomplex of $\operatorname{Alpha}(A \cup B)$!

{py:func}`oineus.function_delaunay_bifiltration` solves this elegantly using the **Function–Delaunay construction** in $\mathbb{R}^{d+1}$ (via `diode`): lifting $A$ to height $z = 0$ and $B$ to height $z = 1$ in $(d+1)$ dimensions produces a **true simplicial inclusion $L \hookrightarrow K$** whose persistence diagrams are algebraically isomorphic to the Čech inclusion $\check{C}(A) \hookrightarrow \check{C}(A \cup B)$, while keeping sparse Delaunay size! You can then pass `(K, L)` directly to {py:func}`oineus.mixup_barcodes_of_filtrations` or {py:func}`oineus.compute_kernel_image_cokernel_reduction`:

```{code-cell} ipython3
# 400 points on a circle (label 0 = A) + 80 interior points (label 1 = B)
theta_big = rng.uniform(0, 2 * np.pi, 400)
A_big = np.column_stack([np.cos(theta_big), np.sin(theta_big)]) + 0.04 * rng.standard_normal((400, 2))
B_big = 0.30 * rng.standard_normal((80, 2))

pts_all = np.vstack([A_big, B_big])
labels = np.array([0] * len(A_big) + [1] * len(B_big))

K_fd, L_fd = oin.function_delaunay_bifiltration(
    pts_all, labels, a_value=0, max_dim=2, squared=True, jitter=1e-7, seed=0, n_threads=4
)
mb_fd = oin.mixup_barcodes_of_filtrations(K_fd, L_fd, max_dim=1, n_threads=4)

print(f"480-point Function-Delaunay sizes -> L (A only): {L_fd.size():,d} | K (A u B): {K_fd.size():,d}")
print(f"H1 total persistence in A: {mb_fd.total_persistence(1):.4f}")
print(f"H1 total mixup caused by B: {mb_fd.total_mixup(1):.4f} ({100 * mb_fd.total_mixup_percentage(1):.1f}%)")
```
