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

# 6. Distances & Matchings

To compare persistence diagrams quantitatively—whether for hypothesis testing, clustering, stability verification, or loss functions—Oineus provides three complementary families of metrics:

1. **Bottleneck ($W_\infty$) distance & matching**: {py:func}`oineus.bottleneck_distance`, {py:func}`oineus.bottleneck_matching` (powered by **Hera**).
2. **$q$-Wasserstein ($W_q$) distance & matching**: {py:func}`oineus.wasserstein_distance`, {py:func}`oineus.wasserstein_matching` (powered by **Hera**).
3. **Sliced Wasserstein & Diagram-Corrected Sliced Wasserstein**: {py:func}`oineus.sliced_wasserstein_distance`, {py:func}`oineus.sliced_wasserstein_distance_diag_corrected`.

---

## 1. Bottleneck & $q$-Wasserstein Distances

Given two persistence diagrams $A$ and $B$ (augmented with countably many copies of the diagonal $\Delta = \{(x, x) : x \in \mathbb{R}\}$), a matching $\gamma: A \cup \Delta \to B \cup \Delta$ pairs points across diagrams or projects unmatched points onto $\Delta$ via {py:func}`oineus.point_to_diagonal` ($(b, d) \mapsto (\frac{b+d}{2}, \frac{b+d}{2})$):

$$
W_q(A, B) = \inf_{\gamma} \left( \sum_{x \in A \cup \Delta} \|x - \gamma(x)\|_p^q \right)^{1/q}, \qquad
W_\infty(A, B) = \inf_{\gamma} \sup_{x \in A \cup \Delta} \|x - \gamma(x)\|_\infty.
$$

Key parameters:
- **`q`**: outer Wasserstein exponent ($q \ge 1.0$, commonly $q = 1.0$ or $q = 2.0$).
- **`internal_p`**: ground norm on $\mathbb{R}^2$ (`np.inf` by default, so diagonal distance is $\frac{|d - b|}{2}$).
- **`delta`**: relative approximation tolerance in Hera (`0.01` = 1% auction/kerber approximation; pass **`delta=0.0`** in `bottleneck_distance` / `bottleneck_matching` for exact geometric bottleneck computation).

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import oineus as oin

dgm_a = np.array([
    [0.10, 1.40],
    [0.35, 1.15],
    [0.70, 0.82],   # low-persistence noise near diagonal
    [0.90, 1.65],
], dtype=np.float64)

dgm_b = np.array([
    [0.18, 1.32],
    [0.42, 1.25],
    [0.95, 1.55],
    [1.20, 1.28],   # low-persistence noise near diagonal
], dtype=np.float64)

b_exact = oin.bottleneck_distance(dgm_a, dgm_b, delta=0.0)
w1 = oin.wasserstein_distance(dgm_a, dgm_b, q=1.0, delta=1e-4)
w2 = oin.wasserstein_distance(dgm_a, dgm_b, q=2.0, delta=1e-4)

print(f"Exact Bottleneck distance (delta=0.0): {b_exact:.4f}")
print(f"1-Wasserstein distance:                {w1:.4f}")
print(f"2-Wasserstein distance:                {w2:.4f}")
```

---

## 2. Inspecting Explicit Matchings (`DiagramMatching` & `BottleneckMatching`)

When you need to know *which* features correspond across two datasets, call {py:func}`oineus.wasserstein_matching` (returns {py:class}`oineus.DiagramMatching`) or {py:func}`oineus.bottleneck_matching` (returns {py:class}`oineus.BottleneckMatching`, a subclass that adds `.longest`):

- `m.finite_to_finite`: `(k, 2)` int64 array of `(idx_a, idx_b)` off-diagonal matches.
- `m.a_to_diagonal` / `m.b_to_diagonal`: 1D int64 arrays of unmatched indices projected to $\Delta$.
- `m.distance` and `m.cost` (`distance ** q`).
- `bm.longest.finite`: list of {py:class}`oineus.FiniteLongestEdge` entries tied for the bottleneck distance (each exposing `.length`, `.idx_a`, `.idx_b`, `.point_a`, `.point_b`).

Let's verify that recomputing the $W_2^2$ cost manually from `m.finite_to_finite`, `m.a_to_diagonal`, and `m.b_to_diagonal` matches `m.cost` to machine precision, and plot both matchings side by side with {py:func}`oineus.vis.plot_matching`:

```{code-cell} ipython3
w2_m = oin.wasserstein_matching(dgm_a, dgm_b, q=2.0, delta=1e-6)
bn_m = oin.bottleneck_matching(dgm_a, dgm_b, delta=0.0)

# Reconstruct W2^2 manually under L_inf ground metric
manual_cost = 0.0
for ia, ib in w2_m.finite_to_finite:
    manual_cost += np.max(np.abs(dgm_a[ia] - dgm_b[ib])) ** 2
for ia in w2_m.a_to_diagonal:
    proj = oin.point_to_diagonal(dgm_a, [ia])[0]
    manual_cost += np.max(np.abs(dgm_a[ia] - proj)) ** 2
for ib in w2_m.b_to_diagonal:
    proj = oin.point_to_diagonal(dgm_b, [ib])[0]
    manual_cost += np.max(np.abs(dgm_b[ib] - proj)) ** 2

assert np.isclose(manual_cost, w2_m.cost, rtol=1e-5)
print("Finite-to-finite pairs:", w2_m.finite_to_finite.tolist())
print("Unmatched A -> diag:", w2_m.a_to_diagonal.tolist(), "| B -> diag:", w2_m.b_to_diagonal.tolist())
print("Bottleneck longest finite edge(s):", [(e.idx_a, e.idx_b, round(e.length, 4)) for e in bn_m.longest.finite])

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
oin.plot_matching(
    dgm_a, dgm_b, w2_m, ax=axes[0],
    plot_diagonal_projections=True, title=f"W2 Matching (dist={w2_m.distance:.3f})",
)
oin.plot_matching(
    dgm_a, dgm_b, bn_m, ax=axes[1],
    highlight_longest=True, plot_diagonal_projections=True,
    title=f"Bottleneck Matching (dist={bn_m.distance:.3f})",
)
fig.tight_layout()
```

---

## 3. Essential Points & `InfKind` Families

When `ignore_inf_points=False` is passed to `wasserstein_matching` or `bottleneck_matching`, essential points (carrying $+\infty$ or $-\infty$ coordinates) are partitioned into the four {py:class}`oineus.InfKind` families:

- `InfKind.INF_DEATH`: `(finite, +inf)` — standard sublevel essential classes,
- `InfKind.NEG_INF_DEATH`: `(finite, -inf)` — superlevel (`negate=True`) essential classes,
- `InfKind.INF_BIRTH` and `InfKind.NEG_INF_BIRTH`.

Two diagrams have a finite distance when `ignore_inf_points=False` **if and only if** they have the exact same number of essential points in every `InfKind` family (in which case 1D optimal sorting along the finite coordinate gives the exact essential matching in `m.essential.inf_death`, etc.):

```{code-cell} ipython3
ess_a = np.vstack([[[0.0, np.inf]], dgm_a])
ess_b = np.vstack([[[0.07, np.inf]], dgm_b])

m_ess = oin.wasserstein_matching(ess_a, ess_b, q=2.0, ignore_inf_points=False)
print("Essential INF_DEATH index pairs:", m_ess.essential.inf_death.tolist())
print("Total W2 distance (including essential shift 0.0 -> 0.07):", round(m_ess.distance, 4))
```

---

## 4. Sliced Wasserstein & Diagram-Corrected Sliced Wasserstein

Exact auction matching scales super-quadratically on dense diagrams. **Sliced Wasserstein** ({py:func}`oineus.sliced_wasserstein_distance`, Carrière et al. 2017) projects diagrams onto $M$ unit directions on the semicircle $[-\frac{\pi}{2}, \frac{\pi}{2}]$, cross-pads each diagram with the other's diagonal projections, and averages 1D sorted $L_1$ distances in $O(M (|A| + |B|) \log(|A| + |B|))$ time!

Oineus also implements **Diagram-Corrected Sliced Wasserstein** ({py:func}`oineus.sliced_wasserstein_distance_diag_corrected`). Why is a diagonal correction needed?
In standard 1D rank-matching, an off-diagonal point $p \in A$ can be paired with the diagonal stand-in of a *distant* point $p' \in B$, charging a long skew distance along the diagonal that true Wasserstein never pays (true Wasserstein simply projects both $p$ and $p'$ orthogonally to $\Delta$!). The diagram-corrected variant re-charges any point-to-diagonal match against **its own** diagonal projection $\operatorname{diag}(p)$ and charges $0$ between two diagonal stand-ins:

```{code-cell} ipython3
# A prominent bar [0.0, 2.0] (diagonal projection at (1.0, 1.0)) vs near-diagonal clutter around 0.4:
# Along slices near the birth axis, standard 1D rank-sorting pairs diagonal stand-ins at ~0.4 against
# the diagonal stand-in at 1.0, charging spurious transport along the diagonal!
dgm_signal = np.array([[0.0, 2.0]], dtype=np.float64)
dgm_clutter = np.array([[0.35, 0.37], [0.40, 0.42]], dtype=np.float64)

w1_true = oin.wasserstein_distance(dgm_signal, dgm_clutter, q=1.0, delta=1e-5)
sw_std = oin.sliced_wasserstein_distance(dgm_signal, dgm_clutter, n_directions=400, seed=0)
sw_corr = oin.sliced_wasserstein_distance_diag_corrected(dgm_signal, dgm_clutter, n_directions=400, seed=0)

print(f"True W1 distance:               {w1_true:.4f}")
print(f"Standard Sliced Wasserstein:    {sw_std:.4f} (inflated by diag-to-diag & skew-diag pairs)")
print(f"Diag-Corrected Sliced Wasser.:  {sw_corr:.4f} (zeros diag-to-diag, uses self-diagonal cost)")
assert sw_corr < sw_std
```
