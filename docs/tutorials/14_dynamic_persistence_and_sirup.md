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

# 14. Dynamic Persistence, Vineyards & SiRUP

When a filtration changes slightly—time-varying scalar fields, iterative optimization steps, local mesh edits, or interactive outlier pruning—re-reducing the entire boundary matrix from scratch wastes almost all prior work. Oineus provides three complementary families of **in-place decomposition updates** on {py:class}`oineus.Decomposition`:

1. **Vineyards & Move Schedules** (`transpose`, `transpose_to`, `move`, `apply_move_schedule`): realize a new filtration order via elementary transpositions or long-range moves (Cohen-Steiner, Edelsbrunner & Morozov 2006).
2. **Luo–Nelson Warm-Start Reduction** (`update_with_permutation`, `update_with_edits`): permute or insert/delete cells in bulk and repair reducedness and triangularity in a fast sweep (Luo & Nelson, `arXiv:2503.13337`).
3. **SiRUP Cell Pruning** ({py:func}`oineus.remove_simplices`, `Decomposition.remove_simplices`): delete any coface-closed union of stars in place while updating both barcodes and representative cycles (Giunti & Lazovskis, `arXiv:2411.12574`).

---

## 1. Preparing a Decomposition for Dynamic Updates (`make_dynamic`)

All dynamic manipulation methods update the factorization $R = D V$ in place, which imposes two setup requirements:
1. Build the decomposition with **`oin.Decomposition(fil, dualize=...)`** (not the fused streaming shortcut `oin.reduce`, which discards $D$ to save memory).
2. Reduce with **`compute_v=True`** (and `use_clearing=False` whenever full $R = D V$ consistency across all columns is desired).

Calling `dcmp.make_dynamic(n_threads=...)` pre-builds row-incidence lookup structures in parallel so subsequent transpositions and moves touch only $O(\text{changed support})$ rather than scanning columns linearly:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import oineus as oin

rng = np.random.default_rng(10)
field0 = rng.random((10, 10))
fil0 = oin.freudenthal_filtration(field0, max_dim=2)

params = oin.ReductionParams()
params.compute_v = True
params.use_clearing = False
params.n_threads = 1

dcmp = oin.Decomposition(fil0, dualize=False)
dcmp.reduce(params)
dcmp.make_dynamic(n_threads=2)

print("Filtration size:", fil0.size(), "| is_dynamic:", dcmp.is_dynamic(), "| sanity_check:", dcmp.sanity_check())
```

---

## 2. Reordering Filtrations: Vineyards vs Move Schedules vs Warm Starts

Suppose we perturb several grid vertices of `field0` to obtain `field1`. Both `fil0` and `fil1` share the exact same combinatorial cells (matched by `cell.uid`), arranged in a new dimension-grouped filtration order.

Every reordering method takes a `new_to_old` list of length $N$ where **`new_to_old[new_pos] = old_pos`**, plus an optional {py:class}`oineus.DecompositionManipStats` accumulator to record transposition counts, column additions in $R$ and $V$, and wall-clock time:

```{code-cell} ipython3
# Perturb 12 pixels on the 10x10 grid
field1 = field0.copy()
for _ in range(12):
    r, c = rng.integers(0, 10, size=2)
    field1[r, c] = rng.random()

fil1 = oin.freudenthal_filtration(field1, max_dim=2)

# Map each cell's new sorted_id in fil1 -> its old sorted_id in fil0 via combinatorial uid:
old_pos_by_uid = {cell.uid: cell.sorted_id for cell in fil0.cells()}
new_to_old = [old_pos_by_uid[cell.uid] for cell in fil1.cells()]

# Ground-truth from-scratch reduction on fil1
dcmp_ref = oin.Decomposition(fil1, dualize=False)
dcmp_ref.reduce(params)
h1_ref = dcmp_ref.diagram(fil1)[1]
h1_ref_sorted = h1_ref[np.lexsort(h1_ref.T)]

for method_name in ("transpose_to", "apply_move_schedule", "update_with_permutation"):
    work = dcmp.clone()
    work.make_dynamic(n_threads=2)
    stats = oin.DecompositionManipStats()
    getattr(work, method_name)(new_to_old, stats)

    h1_upd = work.diagram(fil1)[1]
    h1_upd_sorted = h1_upd[np.lexsort(h1_upd.T)]
    assert np.allclose(h1_upd_sorted, h1_ref_sorted)

    print(
        f"{method_name:25s} | col_adds(R+V) = {stats.n_column_additions():5d} "
        f"(R={stats.n_column_additions_r:4d}, V={stats.n_column_additions_v:4d})"
    )
```

> [!TIP]
> **Which reordering method should you pick?**
> - `transpose(i)` / `transpose_to(new_to_old)`: Best when cells move by 1–2 adjacent positions along a continuous homotopy (classic vineyards).
> - `move(i, j)` / `apply_move_schedule(new_to_old)`: Uses longest-increasing-subsequence scheduling to jump outlier cells directly to their destination instead of bubbling through $O(N)$ adjacent swaps.
> - `update_with_permutation(new_to_old)`: Best for moderate-to-large perturbations—permutes $D, R, V$ in one pass and repairs upper-triangularity (`sanity_check() == True`) with dramatically fewer column operations than elementary swaps!
> - **Cohomology (`dualize=True`)**: Because coboundary matrices reverse filtration indices, pass the reversal-conjugate permutation `[(n - 1) - new_to_old[(n - 1) - k] for k in range(n)]`.

### Cell Insertion & Deletion (`update_with_edits`)

When cells are simultaneously **deleted, reordered, and inserted** (while keeping cells blocked by dimension), pass `-1` in `new_to_old` at every newly inserted position alongside the updated boundary columns and `dim_first` / `dim_last` offsets:

```{code-cell} ipython3
work_edit = dcmp.clone()
D0 = [list(col) for col in work_edit.d_data]
n_old = len(D0)
df, dl = list(work_edit.dim_first), list(work_edit.dim_last)

# Replace the very last triangle with a triangle on the first three edges (or delete it cleanly):
nto_delete_last = list(range(n_old - 1))
D_deleted = D0[: n_old - 1]
dl_deleted = [dl[0], dl[1], dl[2] - 1]

edit_stats = oin.DecompositionManipStats()
work_edit.update_with_edits(nto_delete_last, D_deleted, df, dl_deleted, edit_stats)
print("After deleting top triangle -> len(R):", len(work_edit.r_data), "| sanity_check:", work_edit.sanity_check())
```

---

## 3. SiRUP: Pruning Simplices & Updating Cycles (`remove_simplices`)

Suppose you want to remove a noisy outlier vertex, carve an obstacle out of a mesh, or test how local regions affect homology. Removing a set of cells $S$ leaves a valid subcomplex $K \setminus S$ **if and only if** $S$ is **up-closed** (closed under cofaces: a union of open stars).

{py:func}`oineus.remove_simplices` implements **SiRUP** (Giunti & Lazovskis):
1. Computes the coface closure `L = fil.star_closure(seeds)` automatically when `close_star=True` (the default).
2. Mutates `dcmp` **in place** (`dcmp.remove_simplices(L)`) to update both the homology persistence pairing and the representative cycles in $V$.
3. Returns the matching survivor filtration `new_fil = fil.without_cells(L)`.

Let's triangulate a 2D disk (where $H_1$ is initially trivial!) and punch a hole right through its center by removing the star of the central vertex with `oin.remove_simplices`, watching a brand-new $H_1$ loop appear along with its updated cycle representative:

```{code-cell} ipython3
# Build a 7x7 bowl centered at (3, 3)
coords = np.linspace(-1.0, 1.0, 7)
X, Y = np.meshgrid(coords, coords)
bowl = 0.25 * (X**2 + Y**2) + 1e-3 * X

# Use slim=False so we can inspect vertex coordinates of representative edges directly
fil_disk = oin.freudenthal_filtration(bowl, max_dim=2, slim=False)
dcmp_disk = oin.Decomposition(fil_disk, dualize=False)
dcmp_disk.reduce(params)

print("Finite + infinite H1 bars BEFORE puncturing center:", dcmp_disk.diagram(fil_disk)[1])

# Find the central vertex (row=3, col=3 -> flat vertex id 3*7 + 3 = 24)
center_vid = 3 * 7 + 3
center_sorted_id = next(c.sorted_id for c in fil_disk.cells() if c.dim == 0 and c.vertices == [center_vid])
star_cells = fil_disk.star_closure([center_sorted_id])
print(f"Removing star of center vertex (sorted_id={center_sorted_id}): {len(star_cells)} cells removed")

sirup_stats = oin.DecompositionManipStats()
fil_punctured = oin.remove_simplices(fil_disk, dcmp_disk, [center_sorted_id], stats=sirup_stats)

dgms_after = dcmp_disk.diagram(fil_punctured, include_inf_points=True)
print("H1 bars AFTER SiRUP puncture:", dgms_after[1])

# Extract the essential H1 cycle born at the punctured rim from V!
pts_h1 = dgms_after.in_dimension(1, as_numpy=False)
ess_bar = next(pt for pt in pts_h1 if np.isinf(pt.death))
cycle_col = dcmp_disk.v_data[ess_bar.birth_index]

# Plot the surviving 1-skeleton and the new H1 cycle representative around the puncture
v_xy = np.column_stack([X.ravel(), Y.ravel()])
fig, ax = plt.subplots(figsize=(5.0, 4.8))
for cell in fil_punctured.cells():
    if cell.dim == 1:
        u, v = cell.vertices
        ax.plot([v_xy[u, 0], v_xy[v, 0]], [v_xy[u, 1], v_xy[v, 1]], color="0.82", linewidth=0.9, zorder=1)

for idx in cycle_col:
    u, v = fil_punctured.cell(idx).vertices
    ax.plot(
        [v_xy[u, 0], v_xy[v, 0]], [v_xy[u, 1], v_xy[v, 1]],
        color=oin.OKABE_ITO_VERMILLION, linewidth=3.0, zorder=3,
    )

ax.scatter([0.0], [0.0], marker="x", s=100, color="black", zorder=4, label="Removed center vertex")
ax.set_aspect("equal")
ax.set_title("Updated H_1 Representative Cycle after SiRUP Puncture")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
plt.show()
```

> [!NOTE]
> SiRUP (`remove_simplices`) operates on **homology** decompositions (`dualize=False`) and requires `compute_v=True`. Passing a non-up-closed cell list with `close_star=False` or a cohomology decomposition raises a clear `ValueError` / `RuntimeError`.

---

## What's Next?

- Tune multithreaded reduction, fused streaming `oin.reduce`, compact `slim` / `packed` cell encodings, `float32` dispatch, and `ReductionTimings` in [15. Performance, Compact Encodings & `float32`](15_performance_and_encodings.md).
