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

# 16. Ecosystem Interop & Design Boundaries

Oineus is designed to play cleanly with the scientific Python and TDA ecosystems—NumPy, SciPy sparse linear algebra, PyTorch, JAX, `pickle`, **GUDHI**, **Dionysus**, and **`diode`**—while being transparent about its mathematical and data-structure boundaries.

---

## 1. NumPy & SciPy Sparse Matrix Export

Every matrix in {py:class}`oineus.Decomposition` exports directly to `scipy.sparse` without Python loops:
- `dcmp.d_as_csc()`, `dcmp.r_as_csc()`, `dcmp.v_as_csc()`: return `scipy.sparse.csc_matrix` ($D, R, V$ are stored column-wise).
- `dcmp.u_as_csr()`: returns `scipy.sparse.csr_matrix` ($U = V^{-1}$ is stored row-wise).
- {py:func}`oineus.to_scipy_matrix`: converts any column list (such as `fil.boundary_matrix()` or `fil.coboundary_matrix()`) into a `csc_matrix`.
- `fil.get_vertices()`, `fil.get_edges()`, `fil.get_triangles()`, `fil.get_tetrahedra()`, and `fil.get_simplices_as_arr(k)`: return contiguous `(n_k, k + 1)` `uint64` NumPy arrays of vertex IDs in filtration order.

```{code-cell} ipython3
import pickle
import numpy as np
import scipy.sparse as sp
import oineus as oin

rng = np.random.default_rng(9)
pts = rng.random((20, 2))
fil = oin.alpha_filtration(pts)

params = oin.ReductionParams(compute_v=True, compute_u=True, use_clearing=False, n_threads=1)
dcmp = oin.Decomposition(fil, dualize=False)
dcmp.reduce(params)

D = dcmp.d_as_csc()
R = dcmp.r_as_csc()
V = dcmp.v_as_csc()
U = dcmp.u_as_csr()

# Verify R = D @ V (mod 2) and V @ U = I (mod 2) using SciPy sparse arithmetic!
residual_dv = (D @ V - R).data % 2
ident_vu = (V.tocsr() @ U)
ident_vu.data %= 2
ident_vu.eliminate_zeros()

print(f"Shapes -> D: {D.shape}, R: {R.shape}, V: {V.shape}, U: {U.shape}")
print("||D V - R||_mod2 == 0:", bool(np.all(residual_dv == 0)))
print("V @ U == I (mod 2):   ", (ident_vu - sp.eye(fil.size(), format="csr")).nnz == 0)
print("Triangles array shape:", fil.get_triangles().shape)
```

---

## 2. Full `pickle` Serialization (`float64` & `float32`)

All core Oineus objects—filtrations (fat, slim, packed, product, `float64`, and `float32`), cells, `Diagrams`, `Decomposition`, `DiagramMatching`, `BottleneckMatching`, `KerImCokReduced`, `MixupBarcodes`, `ReductionParams`, and `DecompositionManipStats`—implement C++/Python `__getstate__` and `__setstate__` for `pickle`, `multiprocessing`, and `joblib` caching:

```{code-cell} ipython3
dgms = dcmp.diagram(fil)
match = oin.wasserstein_matching(dgms[1], dgms[1], q=2.0)

for label, obj in (
    ("Packed Alpha Filtration", fil),
    ("Reduced Decomposition", dcmp),
    ("Diagrams", dgms),
    ("DiagramMatching", match),
):
    restored = pickle.loads(pickle.dumps(obj))
    print(f"Round-trip pickle OK -> {label:24s} ({type(restored).__name__})")
```

---

## 3. Interoperability with GUDHI & Dionysus

Because `oin.Filtration` accepts `(vertices, value)` pairs and exports `(n_k, k+1)` simplex arrays and filtration values, converting to or from **GUDHI** (`gudhi.SimplexTree`) and **Dionysus** (`dionysus.Filtration`) takes just a few lines:

```{code-cell} ipython3
import gudhi
import dionysus

# 1. GUDHI SimplexTree -> Oineus Filtration
st = gudhi.RipsComplex(points=pts, max_edge_length=0.8).create_simplex_tree(max_dimension=2)
fil_from_gudhi = oin.Filtration([( list( verts ), float(val)) for verts, val in st.get_filtration()])
dgms_oin = oin.reduce(fil_from_gudhi, oin.ReductionParams(n_threads=2)).diagram(fil_from_gudhi, include_inf_points=False)

st.compute_persistence(homology_coeff_field=2)
h1_gudhi = np.asarray(st.persistence_intervals_in_dimension(1))
h1_oin = dgms_oin[1]
assert np.allclose(h1_gudhi[np.lexsort(h1_gudhi.T)], h1_oin[np.lexsort(h1_oin.T)])
print(f"GUDHI <-> Oineus H1 intervals match ({len(h1_oin)} bars)")

# 2. Oineus Filtration -> Dionysus Filtration
dio_fil = dionysus.Filtration([
    dionysus.Simplex(list(fil.cell(i).vertices), float(fil.cell_value_by_sorted_id(i)))
    for i in range(fil.size())
])
dio_fil.sort()
dio_m = dionysus.homology_persistence(dio_fil)
dio_dgms = dionysus.init_diagrams(dio_m, dio_fil)
h1_dio = np.array([[pt.birth, pt.death] for pt in dio_dgms[1] if np.isfinite(pt.death)])
h1_alpha = fil_dgm = dcmp.diagram(fil, include_inf_points=False)[1]
assert np.allclose(h1_dio[np.lexsort(h1_dio.T)], h1_alpha[np.lexsort(h1_alpha.T)], atol=1e-5)
print(f"Oineus <-> Dionysus H1 intervals match ({len(h1_dio)} bars)")
```

---

## 4. Summary of Design Boundaries & Current Limitations

Knowing what Oineus does—and does not—support out of the box avoids surprises when designing pipelines:

| Aspect | Supported in Oineus | Current Boundary / Workaround |
| :--- | :--- | :--- |
| **Coefficient field** | $\mathbb{Z}_2$ across all C++ reductions, KICR, mixup, and critical sets | Odd-prime $\mathbb{Z}/p$ persistent cohomology is used inside `oineus.diff.cochains` (via `dionysus`) for degree-1 harmonic cochains; general $\mathbb{Z}/p$ C++ reduction is not exposed in Python. |
| **Regular grid dimensions** | **1D, 2D, 3D, and 4D** (`Grid_1D`..`Grid_4D`, `freudenthal_filtration`, `cube_filtration`, `compute_diagrams_ls`) | Grids of dimension $\ge 5$ require custom simplicial/cell filtrations (`oin.Filtration`). |
| **Periodic grid boundaries (`wrap=True`)** | Supported in `freudenthal_filtration`, `diff.freudenthal_filtration`, and `compute_diagrams_ls` | `cube_filtration(..., wrap=True)` is not yet implemented (raises `RuntimeError`); use `freudenthal_filtration(..., wrap=True)` for periodic tori. |
| **Vietoris–Rips ambient dimension** | Point clouds in **1D–5D** via fast geometric distance kernels; **arbitrary ambient dimension / metric spaces** via `from_pwdists=True` | For point clouds in $\mathbb{R}^d$ with $d \ge 6$, precompute the `(n, n)` distance matrix (e.g. `scipy.spatial.distance.cdist`) and pass `from_pwdists=True`. |
| **Alpha & Čech-Delaunay ambient dimension** | **2D and 3D** (including weighted and periodic Delaunay via `diode`) | Higher-dimensional Delaunay requires external triangulation passed into `oin.Filtration` + `oin.cech_filtration` / custom values. |
| **Simplex vertex count** | Fat `Simplex` combinatorial UIDs pack up to **14 vertices** (simplex dimension $\le 13$); bit-packed `PackedSimplex` packs into 64/128-bit words | Simplices of dimension $\ge 14$ exceed the 64-bit fat `Simplex` UID layout; use `Decomposition(boundary_cols, n_rows)` directly for very high-dimensional abstract complexes. |
| **Differentiable diagrams (`oineus.diff`)** | Finite persistence diagrams (`include_inf_points=False`) in PyTorch & JAX; `cech_delaunay_filtration` & `cochains` in PyTorch | Essential (`+inf`) points in `oineus.diff.persistence_diagram` raise `NotImplementedError`; in JAX, keep filtration/diagram calls outside `@jax.jit` / `jax.vmap` (eager boundary). |

---

## Where to Go Next

- Browse all 31 runnable scripts in the [Examples Catalog](../examples/index.md).
- Look up any class, enum, method, or parameter in the [API Reference](../api/index.md).
