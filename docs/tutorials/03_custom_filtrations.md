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

# 3. Custom Complexes & Filtration Operations

Beyond regular grids and Euclidean point clouds, Oineus lets you build filtrations over **arbitrary abstract cell complexes**—graphs, meshes, hypergraphs, clique complexes, cubical subsets, or product spaces—and manipulate them with subcomplex filtering, cell deletion, pointwise minima, and SciPy sparse matrix export.

---

## 1. Building Simplicial Filtrations from Scratch

There are three ways to construct a custom simplicial {py:class}`oineus.Filtration`:

1. **`oin.Filtration([oin.Simplex(...), ...], negate=False)`**: explicit {py:class}`oineus.Simplex` objects (with automatic or explicit user `id`s).
2. **`oin.Filtration([([v0, ...], value), ...])`**: lightweight `(vertices, value)` pairs.
3. **{py:func}`oineus.list_to_filtration`**: `[(user_id, [v0, ...], value), ...]` tuples.

:::{important}
Every filtration must form a valid **filtered cell complex**:
1. Every face of a cell in the complex must also belong to the complex.
2. In sublevel order (`negate=False`), no face may have a strictly larger value than its coface ($\text{val}(\tau) \le \text{val}(\sigma)$ for $\tau \subset \sigma$). Ties are broken automatically by Oineus so lower-dimensional faces always precede higher-dimensional cofaces.
:::

Let's build a hollow triangle that enters at $t \in [0, 3]$ and is capped by a 2-simplex at $t = 5.0$:

```{code-cell} ipython3
import numpy as np
import oineus as oin

cells = [
    oin.Simplex(10, [0], 0.0),
    oin.Simplex(20, [1], 0.0),
    oin.Simplex(30, [2], 0.5),
    oin.Simplex(40, [0, 1], 1.0),
    oin.Simplex(50, [1, 2], 2.0),
    oin.Simplex(60, [0, 2], 3.0),   # closes the 1-cycle at t = 3.0
    oin.Simplex(70, [0, 1, 2], 5.0) # fills the 1-cycle at t = 5.0
]
fil = oin.Filtration(cells)
print(fil)
print("Filtration kind:", fil.kind.as_str(), "| max_dim:", fil.max_dim, "| size:", len(fil))

dgms = oin.reduce(fil).diagram(fil, include_inf_points=True)
print("H0 diagram:\n", dgms[0])
print("H1 diagram:\n", dgms[1])
```

The exact same filtration can be constructed succinctly via {py:func}`oineus.list_to_filtration`:

```{code-cell} ipython3
fil_from_list = oin.list_to_filtration([
    (0, [0], 0.0), (1, [1], 0.0), (2, [2], 0.5),
    (3, [0, 1], 1.0), (4, [1, 2], 2.0), (5, [0, 2], 3.0),
    (6, [0, 1, 2], 5.0),
])
assert np.array_equal(dgms[1], oin.reduce(fil_from_list).diagram(fil_from_list)[1])
```

---

## 2. Navigating Indices: `id`, `uid`, and `sorted_id`

When `Filtration` sorts your input cells by `(value, dim)`, each cell retains its input `id` while receiving a `0`-based filtration position `sorted_id`. Let's inspect how `id_by_sorted_id`, `sorted_id_by_id`, and `get_simplices_as_arr` map between them:

```{code-cell} ipython3
for sid in range(fil.size()):
    c = fil.cell(sid)
    print(
        f"sorted_id={sid} | user id={fil.id_by_sorted_id(sid):2d} | "
        f"dim={c.dim} | vertices={c.vertices} | value={c.value:.1f}"
    )

print("\nEdges array (n_1, 2):\n", fil.get_edges())
print("Dim offsets (dim_first, dim_last):", fil.dim_first, fil.dim_last)
```

:::{note}
Oineus stores filtration cells **grouped contiguously by dimension** (`dim_first[d] .. dim_last[d]`), sorted by filtration value within each dimension. When you update cell values in place with `fil.set_values(new_values)` (indexed by `0..N-1` user IDs), Oineus re-sorts each dimension block and updates `sorting_permutation()` automatically.
:::

---

## 3. Custom Cubical & Product Cells (`Cube_*D`, `ProdSimplex`)

You can also construct custom cubical subsets over a {py:class}`oineus.GridDomain_2D` or product complexes made of {py:class}`oineus.ProdSimplex` cells ($\sigma \times \tau$, used internally by mapping cylinders and prisms):

```{code-cell} ipython3
# Build the 4 boundary edges + 4 corner vertices of a single 2D unit square (hollow square -> 1 H1 loop)
dom = oin.GridDomain_2D(2, 2)
square_top = oin.CombinatorialCube_2D([0, 0], [0, 1], dom)

cube_cells = []
for edge in square_top.boundary():
    cube_cells.append(oin.Cube_2D(edge.anchor_vertex, [
        d for d in range(2) if any(v[d] != edge.anchor_vertex[d] for v in edge.vertices)
    ], dom, 1.0))
    for vtx in edge.boundary():
        cube_cells.append(oin.Cube_2D(vtx.anchor_vertex, [], dom, 0.0))

# Deduplicate by combinatorial uid
unique_cubes = list({c.uid: c for c in cube_cells}.values())
fil_hollow_sq = oin.Filtration(unique_cubes)
dgms_sq = oin.reduce(fil_hollow_sq).diagram(fil_hollow_sq, include_inf_points=True)
print("Hollow 2D cube H1 essential bars:", dgms_sq[1])

# Product filtration: segment [0, 1] x vertex [0] via multiply_filtration
seg_fil = oin.Filtration([([0], 0.0), ([1], 0.0), ([0, 1], 1.0)])
prism_fil = oin.multiply_filtration(seg_fil, oin.Simplex([0, 1], 0.5))
print("Product filtration isinstance ProdFiltration:", isinstance(prism_fil, oin.ProdFiltration), "| size:", prism_fil.size())
```

---

## 4. Subfiltrations, Star Closures & `min_filtration`

Oineus provides algebraic operations on filtrations out of the box:

- **`fil.subfiltration(predicate)`**: keeps every cell where `predicate(combinatorial_cell)` is `True` (must form a closed subcomplex).
- **`fil.star_closure(seed_sorted_ids)`**, **`fil.is_up_closed(cells)`**, and **`fil.without_cells(cells_to_remove)`**: Deleting a set of cells from a complex leaves a valid subcomplex if and only if the deleted set is **closed under cofaces** (an up-set / union of open stars). `star_closure` computes that minimal coface closure automatically!
- **{py:func}`oineus.min_filtration`**: Given two filtrations $F_1, F_2$ over compatible simplices, builds their union where shared simplices enter at $\min(v_1(\sigma), v_2(\sigma))$.

```{code-cell} ipython3
# 1. Subfiltration: drop the interior triangle [0, 1, 2] by keeping only dim <= 1 simplices
skeleton_1 = fil.subfiltration(lambda s: s.dim <= 1)
print("1-skeleton size:", skeleton_1.size(), "| H1 diagram:", oin.reduce(skeleton_1).diagram(skeleton_1)[1])

# 2. Star closure & without_cells: remove vertex [2] (sorted_id=2) and all incident edges/triangles
to_remove = fil.star_closure([2])
print("Star closure of sorted_id 2:", to_remove, "| is_up_closed:", fil.is_up_closed(to_remove))
fil_trimmed = fil.without_cells(to_remove)
print("Surviving cells after removing star of vertex [2]:", [c.vertices for c in fil_trimmed.cells()])

# 3. Pointwise minimum of two filtrations on the same triangle
fil_early_cap = oin.Filtration([
    ([0], 0.0), ([1], 0.0), ([2], 0.0),
    ([0, 1], 2.0), ([1, 2], 2.0), ([0, 2], 2.0),
    ([0, 1, 2], 3.5),
])
fil_late_cap = oin.Filtration([
    ([0], 0.0), ([1], 0.0), ([2], 0.0),
    ([0, 1], 1.0), ([1, 2], 1.0), ([0, 2], 1.0),
    ([0, 1, 2], 6.0),
])
fil_min = oin.min_filtration(fil_early_cap, fil_late_cap)
print("min_filtration H1 bar (born at min=1.0, dies at min=3.5):", oin.reduce(fil_min).diagram(fil_min)[1])
```

---

## 5. Exporting Sparse Boundary & Coboundary Matrices to SciPy

Need the raw $\mathbb{Z}_2$ boundary operator $D$ or coboundary operator $D^\perp$ for spectral Laplacians, linear algebra, or custom solvers? Use `fil.boundary_matrix()` + {py:func}`oineus.to_scipy_matrix` (or {py:func}`oineus.get_boundary_matrix` directly from vertex lists):

```{code-cell} ipython3
D_cols = fil.boundary_matrix()
D_csc = oin.to_scipy_matrix(D_cols, shape=(fil.size(), fil.size()))

# Verify fundamental lemma of homology: D^2 = 0 (mod 2)!
D_sq_mod2 = (D_csc @ D_csc).data % 2
print("Boundary CSC matrix shape:", D_csc.shape, "| nnz:", D_csc.nnz)
print("Max entry of (D @ D) mod 2:", int(D_sq_mod2.max()) if len(D_sq_mod2) else 0)
assert np.all(D_sq_mod2 == 0)
```
