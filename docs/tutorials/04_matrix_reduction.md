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

# 4. Matrix Reduction & Decomposition

Under the hood of persistent homology lies a $\mathbb{Z}_2$ matrix factorization:

$$
R = D V, \qquad U = V^{-1},
$$

where:
- $D$ is the $m \times m$ $\mathbb{Z}_2$ boundary matrix (or coboundary matrix $D^\perp$ when `dualize=True`),
- $V$ is upper-triangular with $1$s on the diagonal (recording column additions),
- $R$ is **reduced**: no two nonzero columns share the same lowest nonzero row index $\operatorname{low}(R_{\cdot, j})$.

Each pivot $i = \operatorname{low}(R_{\cdot, j})$ pairs **positive (birth) cell $i$** with **negative (death) cell $j$**. Unpaired zero columns of $R$ whose index is not a pivot of any higher-dimensional column represent **essential (infinite-persistence) homology classes**.

---

## 1. Fused `oin.reduce(...)` vs. Explicit `oin.Decomposition`

Oineus exposes two ways to run matrix reduction:

1. **{py:func}`oineus.reduce`**: a fused single-call entry point (`dcmp = oin.reduce(fil, ...)`). For slim/packed filtrations in diagram-only mode, `oin.reduce` streams boundary columns on the fly without ever materializing `d_data` in RAM!
2. **`dcmp = oin.Decomposition(fil, dualize=False)` + `dcmp.reduce(params)`**: materializes `d_data` upfront, useful when inspecting $D$, verifying $R = D V$ explicitly, or constructing a `Decomposition` directly from a custom sparse column list `Decomposition(d_cols, n_rows)`.

Let's inspect {py:class}`oineus.ReductionParams` and verify the $R = D V \pmod 2$ identity via SciPy sparse matrices:

```{code-cell} ipython3
import numpy as np
import oineus as oin

rng = np.random.default_rng(0)
pts = rng.random((45, 2))
fil = oin.vr_filtration(pts, max_dim=2, max_diameter=0.7, packed=False)

params = oin.ReductionParams(
    n_threads=4,
    use_clearing=True,
    compute_v=True,
    compute_u=False,
)
params.sanity_check = True
dcmp = oin.Decomposition(fil, dualize=False, n_threads=4)
dcmp.reduce(params)

print("is_reduced:", dcmp.is_reduced, "| factorization_valid:", dcmp.factorization_valid)
print("has_matrix_v:", dcmp.has_matrix_v(), "| has_full_matrix_u:", dcmp.has_full_matrix_u())

# Check R == (D @ V) mod 2 using SciPy CSC exports
D_sp = dcmp.d_as_csc()
V_sp = dcmp.v_as_csc()
R_sp = dcmp.r_as_csc()
diff_mod2 = ((D_sp @ V_sp - R_sp).data % 2)
assert len(diff_mod2) == 0 or np.all(diff_mod2 == 0)
print(f"Verified R = D V (mod 2) across {fil.size()} columns! nnz(D)={D_sp.nnz}, nnz(V)={V_sp.nnz}, nnz(R)={R_sp.nnz}")
```

---

## 2. Homology (`dualize=False`) vs. Cohomology (`dualize=True`)

By persistent duality, reducing the anti-transposed **coboundary** matrix $D^\perp$ (`dualize=True`) yields the **exact same persistence diagram** as reducing $D$ (`dualize=False`), while often running dramatically faster on **Vietoris–Rips** filtrations when combined with the Chen–Kerber **clearing** optimization (`use_clearing=True`).

Why? In Vietoris–Rips with `max_dim = k + 1`, the overwhelming majority of columns lie in the top dimension $k + 1$. Under cohomology + clearing, reduced $k$-cochains immediately clear almost all $(k+1)$-columns in $O(1)$ without touching their coboundaries!

```{code-cell} ipython3
pts_vr = rng.random((85, 3))
fil_vr = oin.vr_filtration(pts_vr, max_dim=2, max_diameter=0.85, n_threads=4)

dcmp_hom = oin.reduce(fil_vr, dualize=False, n_threads=4, use_clearing=True)
dcmp_coh = oin.reduce(fil_vr, dualize=True, n_threads=4, use_clearing=True)

dgm_hom = dcmp_hom.diagram(fil_vr)
dgm_coh = dcmp_coh.diagram(fil_vr)

for q in range(2):
    # Sort rows lexicographically to compare unordered multisets of (birth, death) points
    a = dgm_hom[q][np.lexsort((dgm_hom[q][:, 1], dgm_hom[q][:, 0]))]
    b = dgm_coh[q][np.lexsort((dgm_coh[q][:, 1], dgm_coh[q][:, 0]))]
    assert np.allclose(a, b)

print(f"Homology wall time:   {dcmp_hom.timings.reduction_total * 1e3:6.2f} ms")
print(f"Cohomology wall time: {dcmp_coh.timings.reduction_total * 1e3:6.2f} ms")
```

:::{tip}
- For **Vietoris–Rips** complexes, prefer `dualize=True` (the default of {py:func}`oineus.compute_diagrams_vr`).
- For **regular grids** (cubical & Freudenthal) and **Alpha** shapes in 2D/3D, `dualize=False` (the default of {py:func}`oineus.compute_diagrams_ls` and {py:func}`oineus.compute_diagrams_alpha`) is fast—especially on cubical grids where **apparent pairs** (`use_apparent_pairs=None`, auto-enabled) eliminate most columns before reduction even begins!
:::

---

## 3. Canonical ELZ Form (`restore_elz`) & Recovering $U = V^{-1}$

In-band computation of $U = V^{-1}$ during reduction only works serially without clearing (`n_threads=1, use_clearing=False`). How does Oineus obtain $U$ (or just a handful of rows of $U$ for critical-set gradients) **in parallel with clearing enabled**?

1. Run fast parallel clearing reduction with `compute_v=True`.
2. Normalize $V$ into canonical **Edelsbrunner–Letscher–Zomorodian (ELZ)** form via `dcmp.restore_elz()` (or pass `dims_to_restore_elz=[...]` in `ReductionParams`). In ELZ form, off-diagonal entries of $V$ only appear at columns with strictly smaller pivots, guaranteeing that forward substitution terminates cleanly.
3. Recover either **all rows** in a dimension (`dcmp.compute_full_u_rows(fil, dim)`) or **only target-bounded entries of selected rows** (`dcmp.compute_partial_u_rows(fil, rows, bounds, dim)`).

```{code-cell} ipython3
# 1. Complete parallel U via ReductionParams(compute_u=True)
dcmp_full_u = oin.Decomposition(fil, dualize=False, n_threads=4)
dcmp_full_u.reduce(oin.ReductionParams(n_threads=4, use_clearing=True, compute_u=True))
assert dcmp_full_u.has_full_matrix_u()

U_sp = dcmp_full_u.u_as_csr()
I_check = (dcmp_full_u.v_as_csc().tocsr() @ U_sp)
I_check.data %= 2
I_check.eliminate_zeros()
assert I_check.nnz == fil.size() and np.all(I_check.diagonal() == 1)
print(f"Verified V @ U = I (mod 2) via parallel compute_u=True! ({dcmp_full_u.timings.compute_u * 1e3:.2f} ms)")

# 2. Manual ELZ normalization + target-bounded selected rows of U
dcmp_elz = oin.Decomposition(fil, dualize=False, n_threads=4)
dcmp_elz.reduce(oin.ReductionParams(n_threads=4, use_clearing=True, compute_v=True))
print("ELZ violators right after parallel+clearing reduce:", dcmp_elz.n_elz_violators())
dcmp_elz.restore_elz(n_threads=4)
print("ELZ violators after restore_elz():                 ", dcmp_elz.n_elz_violators(), "| is_elz:", dcmp_elz.is_elz())

# Solve only a single row of U in dimension 1 up to a target value bound
sample_edge_row = fil.dim_first[1]
dcmp_elz.compute_partial_u_rows(fil, rows=[sample_edge_row], bounds=[0.5], dim=1, n_threads=2)
print("Partial U row support length:", len(dcmp_elz.u_row(sample_edge_row)))
```

---

## 4. Zero-Persistence Pairs & Index Diagrams

Standard `dcmp.diagram(fil)` filters out **zero-persistence pairs** ($\text{birth} == \text{death}$, caused when a face and its destroying coface enter at the exact same filtration value, as happens constantly in Vietoris–Rips cliques and piecewise-constant grids).

When you need *every* algebraic pairing:

- **`dcmp.zero_pers_diagram(fil)`** extracts exclusively the zero-persistence pairs `(b, b)` in each dimension.
- **`dgms.index_diagram_in_dimension(q)`** returns `(birth_sorted_id, death_sorted_id)` pairs, and `dgms.in_dimension(q, as_numpy=False)` returns {py:class}`oineus.DiagramPoint` objects carrying both `.birth`/`.death` values and `.birth_index`/`.death_index` sorted IDs!

```{code-cell} ipython3
dgms_pos = dcmp.diagram(fil)
dgms_zero = dcmp.zero_pers_diagram(fil)

pts_h1 = dgms_pos.in_dimension(1, as_numpy=False)
print(f"H1 positive-persistence pairs: {len(pts_h1)}")
print(f"H1 zero-persistence pairs:     {len(dgms_zero.in_dimension(1))}")

if pts_h1:
    top_pt = max(pts_h1, key=lambda p: p.persistence)
    print(
        f"Longest H1 DiagramPoint: birth={top_pt.birth:.4f} (sorted_id={top_pt.birth_index}), "
        f"death={top_pt.death:.4f} (sorted_id={top_pt.death_index}), pers={top_pt.persistence:.4f}"
    )
```

---

## 5. Profiling Phases with `ReductionTimings`

Every `reduce()` call populates `dcmp.timings` ({py:class}`oineus.ReductionTimings`) with per-phase wall-clock seconds (`prepare`, `reduce`, `bauer`, `restore_elz`, `copy_back`, `copy_pivots`, `compute_u`, and path-comparable `reduction_total`):

```{code-cell} ipython3
t_info = dcmp_coh.timings
print(
    f"prepare={t_info.prepare*1e3:.2f}ms | reduce={t_info.reduce*1e3:.2f}ms | "
    f"copy_pivots={t_info.copy_pivots*1e3:.2f}ms | total={t_info.reduction_total*1e3:.2f}ms"
)
```
