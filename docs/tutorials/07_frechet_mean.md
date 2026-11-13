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

# 7. Wasserstein Fréchet Means (Barycenters)

Given a collection of persistence diagrams $D_1, \dots, D_m$ (with optional non-negative weights $w_1, \dots, w_m$), their **Wasserstein Fréchet mean** (or $W_2$ barycenter, Turner et al. 2014) is a consensus diagram $\bar{D}$ minimizing the weighted sum of squared $2$-Wasserstein distances:

$$
\mathcal{E}(\bar{D}) = \sum_{i=1}^{m} w_i \, W_2(\bar{D}, D_i)^2.
$$

Because diagram space has Alexandrov curvature bounded from below (non-unique matchings create piecewise-quadratic basins), $\mathcal{E}(\bar{D})$ is **non-convex**: Lloyd-style alternation (match $\bar{D}$ to every $D_i$, then replace each point of $\bar{D}$ by the arithmetic mean of its matched partners/diagonal projections) converges to a local minimum that depends on initialization.

Oineus provides a complete toolkit for escaping poor local minima:
- Multiple **initializers** ({py:class}`oineus.FrechetMeanInit`: `MedoidDiagram`, `Grid`, `RandomDiagram`, `FirstDiagram`, `Custom`),
- **Multistart search** ({py:func}`oineus.frechet_mean_multistart`),
- **Coarse-to-fine progressive persistence scheduling** ({py:func}`oineus.progressive_frechet_mean`, {py:func}`oineus.progressive_frechet_mean_multistart`).

---

## 1. Comparing Initialization Strategies on an Ensemble of Noisy Circles

Let's generate $m = 8$ persistence diagrams from noisy two-ring point clouds (each having two true $H_1$ features plus varying numbers of spurious noise bars near the diagonal) and compare the Fréchet objective {py:func}`oineus.frechet_mean_objective` across initializers:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import oineus as oin

rng = np.random.default_rng(21)


def sample_two_rings_h1(rng: np.random.Generator) -> np.ndarray:
    t1 = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    t2 = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    c1 = 1.0 * np.column_stack([np.cos(t1), np.sin(t1)]) + 0.07 * rng.standard_normal((36, 2))
    c2 = np.array([2.6, 0.0]) + 0.55 * np.column_stack([np.cos(t2), np.sin(t2)]) + 0.05 * rng.standard_normal((24, 2))
    pts = np.vstack([c1, c2])
    return oin.compute_diagrams_alpha(pts, include_inf_points=False).in_dimension(1)


diagrams = [sample_two_rings_h1(rng) for _ in range(8)]
print("Diagram cardinalities:", [len(d) for d in diagrams])

delta = 1e-3
bary_first = oin.frechet_mean(
    diagrams, init_strategy=oin.FrechetMeanInit.FirstDiagram,
    wasserstein_delta=delta, n_threads=4,
)
bary_medoid = oin.frechet_mean(
    diagrams, init_strategy=oin.FrechetMeanInit.MedoidDiagram,
    wasserstein_delta=delta, n_threads=4,
)
bary_grid = oin.frechet_mean(
    diagrams, init_strategy=oin.FrechetMeanInit.Grid,
    grid_n_x_bins=8, grid_n_y_bins=8, wasserstein_delta=delta, n_threads=4,
)

for label, b in [("FirstDiagram", bary_first), ("MedoidDiagram", bary_medoid), ("Grid(8x8)", bary_grid)]:
    obj = oin.frechet_mean_objective(diagrams, b, wasserstein_delta=delta, n_threads=4)
    print(f"{label:15s} -> support={len(b):2d} pts | objective={obj:.6f}")
```

---

## 2. Progressive Multistart Fréchet Means

Why can single-stage Lloyd iteration get stuck when diagrams contain lots of near-diagonal clutter? Early iterations spend matching budget pairing high-persistence candidates inconsistently across clutter points.

{py:func}`oineus.progressive_frechet_mean_multistart` solves this via **coarse-to-fine persistence continuation**:
1. {py:func}`oineus.make_frechet_mean_persistence_schedule` builds a decreasing sequence of persistence thresholds $\tau_0 > \tau_1 > \dots > \tau_K = 0$.
2. At $\tau_0$, only the dominant topological features are active, so matching aligns the macro-structure cleanly across all $m$ diagrams.
3. As $\tau_k$ decreases, newly active bars inject candidate barycenter points ({py:func}`oineus.frechet_mean_newborn_points_from_newly_active`), warm-starting each finer Lloyd stage!

Passing `return_details=True` returns rich convergence diagnostics across starts and stages:

```{code-cell} ipython3
schedule = oin.make_frechet_mean_persistence_schedule(diagrams)
print("Automatic persistence threshold schedule:", [round(t, 4) for t in schedule])

bary_ms, ms_info = oin.frechet_mean_multistart(
    diagrams, starts=("medoid", "grid"),
    grid_n_x_bins=8, grid_n_y_bins=8,
    wasserstein_delta=delta, return_details=True, n_threads=4,
)
bary_prog, prog_info = oin.progressive_frechet_mean_multistart(
    diagrams, starts=("medoid", "second_medoid"),
    wasserstein_delta=delta, return_details=True, n_threads=4,
)

ms_winner = min(ms_info["runs"], key=lambda r: r["objective"])["start"]
prog_winner = min(prog_info["runs"], key=lambda r: r["objective"])["start"]
print(f"Multistart best objective:             {ms_info['objective']:.6f} (winner: {ms_winner})")
print(f"Progressive multistart best objective: {prog_info['objective']:.6f} (winner: {prog_winner}, stages: {len(prog_info['thresholds'])})")
```

Let's visualize the 8 input diagrams overlaid in translucent blue alongside the consensus **Progressive Multistart Fréchet Mean** in vermillion stars:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 5.0))

all_pts = np.vstack(diagrams)
hi = float(all_pts.max()) * 1.08
ax.plot([0, hi], [0, hi], "--", color="0.6", linewidth=1.0)

for idx, dgm in enumerate(diagrams):
    ax.scatter(
        dgm[:, 0], dgm[:, 1], s=22, alpha=0.35, color=oin.OKABE_ITO_BLUE,
        label="Input diagrams (m=8)" if idx == 0 else None,
    )

ax.scatter(
    bary_prog[:, 0], bary_prog[:, 1],
    s=95, marker="*", color=oin.OKABE_ITO_VERMILLION, edgecolor="black", linewidth=0.5, zorder=5,
    label="Progressive Frechet mean",
)
ax.set_xlim(0, hi)
ax.set_ylim(0, hi)
ax.set_aspect("equal")
ax.set_xlabel("Birth (r^2)")
ax.set_ylabel("Death (r^2)")
ax.legend(loc="lower right", fontsize=9)
ax.set_title("Consensus H1 Barycenter of 8 Noisy Two-Ring Clouds")
fig.tight_layout()
```

Notice the two crisp vermillion stars capturing the exact consensus `(birth, death)` of the large ring ($r^2 \approx 1.0$) and smaller ring ($r^2 \approx 0.30$), while transient clutter averages cleanly down near the diagonal!

:::{tip}
Working with superlevel (`negate=True`) persistence diagrams where $\text{birth} > \text{death}$? Pass `domain=oin.DiagramPlaneDomain.BelowDiagonal` (or `DiagramPlaneDomain.Mixed`) to `frechet_mean` and `init_frechet_mean_*`.
:::
