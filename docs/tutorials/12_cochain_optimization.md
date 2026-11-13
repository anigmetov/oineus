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

# 12. Cochain-Based Optimization (`oineus.diff.cochains`)

When you optimize a point cloud's $H_1$ loop using standard singleton persistence gradients (`gradient_method="dgm-loss"`), each step moves **only two edges**: the single critical edge that closes the cycle at birth $b$, and the single critical edge of the triangle that fills the cycle at death $d$. Pushing two vertices per step causes point clouds to kink, trade critical edges abruptly, or stall in polygonal corners.

{py:mod}`oineus.diff.cochains` (also re-exported directly from {py:mod}`oineus.diff`) implements **Weighill & Zhou's birth and death cochain optimization** (*Topological optimization with birth and death cochains*, `arXiv:2603.25575`), spreading gradients smoothly over harmonic $\varepsilon$-neighborhoods of the birth and death thresholds.

By the end of this tutorial you will know how to:
- Compute and visualize {py:class}`oineus.diff.Cochain` supports via {py:func}`oineus.diff.birth_cochain`, {py:func}`oineus.diff.death_cochain`, and {py:func}`oineus.diff.birth_death_cochains`.
- Understand **birth content** $B_\varepsilon$, **death content** $D_\varepsilon$ (including Vietoris–Rips `edge_relaxed` redistribution), and multi-scale {py:func}`oineus.diff.persistence_content_loss`.
- Compare singleton `"dgm-loss"` against multi-$\varepsilon$ cochain optimization on a point-mover benchmark.

---

## 1. Geometry of $\varepsilon$-Birth and $\varepsilon$-Death Cochains

Fix a finite persistence bar $[b, d)$ in degree $k$ and a window width $0 < \varepsilon < \frac{d - b}{2}$:

- **$\varepsilon$-Birth cochain ($\alpha_\varepsilon$ in degree $k$)**: The unique $\ell_2$-minimal real cocycle on $X_{b + \varepsilon}$ representing the born cohomology class and vanishing identically on $X_{b - \varepsilon}$. Around an $H_1$ loop, $\alpha_\varepsilon$ assigns smooth harmonic weights across **all bottlenecks/bridges** near scale $b$, not just the single narrowest gap!
- **$\varepsilon$-Death cochain ($\omega_\varepsilon = \delta \hat{\beta}$ in degree $k + 1$)**: Extends the dying cocycle from $X_{d - \varepsilon}$ to $X_{d + \varepsilon}$ with minimal coboundary norm (a discrete Dirichlet / Laplace-learning problem). Its coboundary $\omega_\varepsilon$ weights the $(k+1)$-simplices spanning the interior of the hole near death $d$.

> [!NOTE]
> **Optional dependency for degree $\ge 1$**: Degree-0 birth/death cochains use exact connected-component indicators and need only SciPy. Degree-1 cochains extract cocycle representatives over $\mathbb{Z}/p$ (`prime=47` by default) via **`dionysus`** (`pip install oineus[topo]`) and lift them to exact integers. Content losses (`birth_content`, `death_content`, `persistence_content_loss`) require PyTorch.

Each {py:class}`oineus.diff.Cochain` stores:
- `degree`: simplex dimension ($k$ for birth, $k+1$ for death),
- `sorted_ids`: global filtration indices into `fil.values`,
- `coeffs`: signed real harmonic coefficients,
- `weights`: normalized $L_1$ magnitudes $|\text{coeffs}| / \|\text{coeffs}\|_1$ (summing to `1.0`),
- `t_K`, `t_L`: the window endpoints $[b-\varepsilon, b+\varepsilon]$ or $[d-\varepsilon, d+\varepsilon]$.

Let's inspect and plot both cochains on a noisy 14-point loop:

```{code-cell} ipython3
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import oineus as oin
import oineus.diff as od

rng = np.random.default_rng(3)
phi = np.sort(rng.uniform(0, 2 * np.pi, 14))
pts0 = np.column_stack([np.cos(phi), np.sin(phi)]) + 0.12 * rng.standard_normal((14, 2))

fil0 = od.vr_filtration(torch.tensor(pts0, dtype=torch.float64), max_dim=2)
b_coch, d_coch = od.birth_death_cochains(fil0, dim=1, bar="longest", eps=0.15, relative_eps=True)

print("Birth cochain:", b_coch)
print("Death cochain:", d_coch)
print("Weights sum to 1:", round(float(b_coch.weights.sum()), 6), round(float(d_coch.weights.sum()), 6))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

# Left: degree-1 birth cochain (weighted edges)
segments, lw = [], []
w_max_b = float(b_coch.weights.max())
for sid, w in zip(b_coch.sorted_ids, b_coch.weights):
    u, v = fil0.under_fil.cell(int(sid)).vertices
    segments.append([pts0[u], pts0[v]])
    lw.append(1.0 + 5.0 * (w / w_max_b))
lc = LineCollection(segments, linewidths=lw, colors=oin.OKABE_ITO_BLUE, alpha=0.85)
axes[0].add_collection(lc)
axes[0].scatter(pts0[:, 0], pts0[:, 1], s=35, color="black", zorder=4)
axes[0].set_aspect("equal")
axes[0].set_title(f"eps-Birth Cochain ({len(b_coch.sorted_ids)} edges)")

# Right: degree-2 death cochain (weighted triangles)
polys, alphas = [], []
w_max_d = float(d_coch.weights.max())
for sid, w in zip(d_coch.sorted_ids, d_coch.weights):
    i, j, k = fil0.under_fil.cell(int(sid)).vertices
    polys.append([pts0[i], pts0[j], pts0[k]])
    alphas.append(0.12 + 0.65 * (w / w_max_d))
for poly, a in zip(polys, alphas):
    axes[1].add_collection(PolyCollection([poly], facecolors=oin.OKABE_ITO_VERMILLION, edgecolors="0.4", linewidths=0.5, alpha=float(a)))
axes[1].scatter(pts0[:, 0], pts0[:, 1], s=35, color="black", zorder=4)
axes[1].set_aspect("equal")
axes[1].set_title(f"eps-Death Cochain ({len(d_coch.sorted_ids)} triangles)")

for ax in axes:
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

fig.tight_layout()
plt.show()
```

---

## 2. Differentiable Content Losses & Edge-Relaxed VR Mode

Holding the cochain weights $w$ fixed (detached) on each step defines smooth linear functionals of the filtration values:

$$B_\varepsilon = \sum_{i \in \operatorname{supp}(\alpha_\varepsilon)} w_i \, v_i, \qquad D_\varepsilon = \sum_{j \in \operatorname{supp}(\omega_\varepsilon)} w_j \, v_j, \qquad D_\varepsilon - B_\varepsilon \xrightarrow{\varepsilon \to 0^+} d - b.$$

- {py:func}`oineus.diff.birth_content`: returns $B_\varepsilon$ as a differentiable scalar `torch.Tensor`.
- {py:func}`oineus.diff.death_content`: returns $D_\varepsilon$. On Vietoris–Rips filtrations (`edge_relaxed=None`, auto-enabled for VR in `dim >= 1`), each support triangle's weight is split equally across its **newly entered facets** in $(d - \varepsilon, d + \varepsilon]$, ensuring every triangle pushes all candidate bottleneck chords rather than just its single current longest edge.
- {py:func}`oineus.diff.persistence_content_loss`: averages $D_\varepsilon - B_\varepsilon$ across a tuple of relative window widths such as `eps=(0.01, 0.05, 0.1)`, sharing a **single** cocycle computation across all scales! **Maximize** `persistence_content_loss(..., mode="persistence")` (i.e. minimize `-persistence_content_loss(...)`) to grow the bar.

> [!IMPORTANT]
> When `relative_eps=True` (the default), `eps` is interpreted as a fraction of the bar length $d - b$, and every entry must satisfy `0 < eps < 0.5` so the birth snapshot $X_{b + \varepsilon(d-b)}$ stays strictly below the cocycle truncation $X_{d - \varepsilon_{\min}(d-b)}$.

---

## 3. Point-Mover Benchmark: Multi-$\varepsilon$ Cochains vs Singleton `"dgm-loss"`

Let's maximize the longest $H_1$ bar of a noisy 10-point circle constrained to the unit disk ($\sum_i \max(0, \|x_i\|_2 - 1)^2$), comparing multi-$\varepsilon$ cochain content (`eps=(0.01, 0.05, 0.15)`) against singleton `"dgm-loss"`:

```{code-cell} ipython3
rng_bench = np.random.default_rng(1)
phi_b = np.sort(rng_bench.uniform(0, 2 * np.pi, 10))
cloud0 = np.column_stack([np.cos(phi_b), np.sin(phi_b)]) + 0.12 * rng_bench.standard_normal((10, 2))

def disk_penalty(pts: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.linalg.norm(pts, dim=1) - 1.0).pow(2).sum()

def true_longest_h1(pts_arr: np.ndarray) -> float:
    h1 = oin.compute_diagrams_vr(pts_arr, max_dim=1, include_inf_points=False)[1]
    return float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0

def run_pointmover(use_cochains: bool, n_steps: int = 120, lr: float = 0.025):
    pts = torch.tensor(cloud0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([pts], lr=lr)
    hist = []
    for _ in range(n_steps):
        opt.zero_grad()
        fil = od.vr_filtration(pts, max_dim=2)
        if use_cochains:
            topo = od.persistence_content_loss(fil, dim=1, bar="longest", eps=(0.01, 0.05, 0.15))
        else:
            dgm1 = od.persistence_diagram(fil, gradient_method="dgm-loss")[1]
            idx = torch.argmax(dgm1[:, 1] - dgm1[:, 0])
            topo = dgm1[idx, 1] - dgm1[idx, 0]
        loss = -topo + disk_penalty(pts)
        loss.backward()
        opt.step()
        hist.append(true_longest_h1(pts.detach().numpy()))
    return hist, pts.detach().numpy()

hist_coch, pts_coch = run_pointmover(use_cochains=True)
hist_sing, pts_sing = run_pointmover(use_cochains=False)

print(
    f"Initial longest H1: {true_longest_h1(cloud0):.4f} | "
    f"Singleton: {hist_sing[-1]:.4f} | Multi-eps Cochains: {hist_coch[-1]:.4f}"
)

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
axes[0].plot(hist_coch, color=oin.OKABE_ITO_VERMILLION, linewidth=2.0, label="Multi-eps cochains")
axes[0].plot(hist_sing, color=oin.OKABE_ITO_BLUE, linestyle="--", linewidth=1.8, label="Singleton (dgm-loss)")
axes[0].set_xlabel("SGD step")
axes[0].set_ylabel("True longest H_1 persistence")
axes[0].set_title("Loop growth trajectory")
axes[0].legend(fontsize=8)

theta = np.linspace(0, 2 * np.pi, 200)
for ax, final_pts, label, col in (
    (axes[1], pts_coch, "Multi-eps Cochains", oin.OKABE_ITO_VERMILLION),
    (axes[2], pts_sing, "Singleton (dgm-loss)", oin.OKABE_ITO_BLUE),
):
    ax.plot(np.cos(theta), np.sin(theta), ":", color="0.65", linewidth=1.2)
    ax.scatter(cloud0[:, 0], cloud0[:, 1], s=26, color="0.65", label="Initial")
    ax.scatter(final_pts[:, 0], final_pts[:, 1], s=42, color=col, label="Final")
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(label)
    ax.legend(fontsize=8, loc="lower right")

fig.tight_layout()
plt.show()
```

Spreading the birth and death forces over multiple $\varepsilon$-windows nudges neighboring vertices simultaneously toward a evenly spaced regular polygon on the unit circle!

---

## What's Next?

- Differentiate topological relationships **between** two complexes or point clouds in [13. Differentiable KICR & Mixup](13_diff_kicr_and_mixup.md).
- Explore dynamic vineyards, warm-started re-reduction, and SiRUP cell pruning in [14. Dynamic Persistence, Vineyards & SiRUP](14_dynamic_persistence_and_sirup.md).
