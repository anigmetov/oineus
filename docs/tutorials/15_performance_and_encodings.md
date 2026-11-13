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

# 15. Performance, Compact Encodings & `float32`

On a $512^3$ 3D volume or a 5,000-point Vietoris–Rips filtration, simplices easily number in the tens of millions. At that scale, **memory bandwidth**—not arithmetic—dominates runtime. Oineus combines four architectural optimizations so Python workloads run at native C++ speed with minimal RAM:

1. **Compact cell encodings** (`slim=True` on grids, `packed=True` on simplicial complexes).
2. **Single-precision (`float32`) dispatch** when 32-bit filtration values suffice.
3. **Fused streaming reduction (`oin.reduce`)** + **clearing** + **apparent pairs** so boundary columns never need to sit in RAM.
4. **Lock-free parallel chunk reduction** with configurable working-column representations ({py:class}`oineus.ColumnRepr`) and per-phase profiling ({py:class}`oineus.ReductionTimings`).

---

## 1. Fused Streaming `oin.reduce` vs `oin.Decomposition`

There are two ways to reduce a filtration in Oineus:

| Entry Point | Stores `d_data`? | Materializes `r_data`? | Best For |
| :--- | :---: | :---: | :--- |
| **`oin.reduce(fil, params, dualize=...)`** | **No** (streams facets on the fly) | Lazy (keeps compact pivot array until `.r_data` is accessed) | Fast diagram extraction and large-scale batch jobs |
| **`oin.Decomposition(fil, dualize=...)` + `.reduce(params)`** | **Yes** | Yes | Dynamic updates (`transpose`, `update_with_permutation`), `sanity_check()`, or inspecting $D$ |

On complete cubical grids, parallel diagram-only homology also activates **apparent pairs** automatically (`params.use_apparent_pairs=None`), immediately pairing zero-persistence gradient pairs without touching a working column:

```{code-cell} ipython3
import time
import numpy as np
import matplotlib.pyplot as plt
import oineus as oin

rng = np.random.default_rng(42)
vol = rng.random((28, 28, 28))

cfil = oin.cube_filtration(vol, max_dim=3, n_threads=4)
params = oin.ReductionParams(n_threads=4, use_clearing=True, compute_v=False)

dcmp_fused = oin.reduce(cfil, params, dualize=False)
print(f"3D cubical cells: {cfil.size():,}")
print(f"Apparent pairs skipped automatically: {dcmp_fused.n_apparent_pairs():,}")
print(f"Fused reduction wall-clock:           {dcmp_fused.timings.reduction_total * 1e3:.2f} ms")
```

---

## 2. Compact Encodings (`slim=True` & `packed=True`) vs Fat Cells

By default, Oineus's factory builders pick the most memory-compact C++ cell representation:

- **Slim grid cells (`slim=True`, default in `freudenthal_filtration`; always used in `cube_filtration`)**: Encodes each grid simplex/cube in a single integer `(anchor_vertex, permutation_or_span_bits)` and generates boundary facets arithmetic-on-the-fly instead of storing heap-allocated vertex vectors.
- **Bit-packed simplices (`packed=True`, default in `vr_filtration` and `alpha_filtration`)**: Packs $(k+1)$ sorted vertex IDs into a single 64-bit or 128-bit integer word whenever `(max_dim + 1) * ceil(log2(n_vertices))` fits (falling back transparently to fat `Simplex` otherwise).

### When Should You Pass `slim=False` or `packed=False`?

Pass `slim=False` or `packed=False` only when you specifically need a **fat `Simplex` filtration**:
1. Mutating filtration values in place with `fil.set_values(new_values)` on an existing filtration.
2. Passing filtrations into {py:mod}`oineus.diff` (note that `oineus.diff` builders already default to `slim=False` / `packed=False` automatically!).
3. Combining arbitrary user-built `oin.Simplex` filtrations via `mapping_cylinder` or `multiply_filtration`.

Every diagram, KICR reduction, relative diagram, and SiRUP helper accepts both compact and fat encodings identically:

```{code-cell} ipython3
pts = rng.random((120, 3))
vr_packed = oin.vr_filtration(pts, max_dim=2, max_diameter=0.55, packed=True, n_threads=4)
vr_fat = oin.vr_filtration(pts, max_dim=2, max_diameter=0.55, packed=False, n_threads=4)

print("Packed concrete C++ type:", type(vr_packed).__name__, "| isinstance Filtration:", isinstance(vr_packed, oin.Filtration))
print("Fat concrete C++ type:   ", type(vr_fat).__name__, "                  | simplices:", vr_fat.size())
```

---

## 3. Single-Precision Dispatch (`float32` vs `float64`)

Whenever you pass a `np.float32` array (or `torch.float32` / `jnp.float32` tensor in `oineus.diff`), Oineus routes construction and reduction to its compiled `float32` extension (`oineus._f32`), halving the storage footprint of filtration values and persistence diagrams:

```{code-cell} ipython3
vol_small_f64 = rng.random((16, 16, 16), dtype=np.float64)
vol_small_f32 = vol_small_f64.astype(np.float32)

dgm_f64 = oin.compute_diagrams_ls(vol_small_f64, include_inf_points=False, params=oin.ReductionParams(n_threads=4))[1]
dgm_f32 = oin.compute_diagrams_ls(vol_small_f32, include_inf_points=False, params=oin.ReductionParams(n_threads=4))[1]

bn_diff = oin.bottleneck_distance(dgm_f64, dgm_f32.astype(np.float64), delta=0.0)
print("f64 diagram dtype:", dgm_f64.dtype, "| f32 diagram dtype:", dgm_f32.dtype)
print(f"Exact bottleneck distance between f64 and f32 H1 diagrams: {bn_diff:.2e}")
```

---

## 4. Profiling Phases with `ReductionTimings` & Tuning `ColumnRepr`

Every reduced {py:class}`oineus.Decomposition` exposes `dcmp.timings` ({py:class}`oineus.ReductionTimings`) and, when $U$ is computed post-hoc, `dcmp.u_timings` ({py:class}`oineus.UComputeTimings`):
- `timings.prepare`: building the atomic parallel column table (`0.0` on serial `n_threads=1`).
- `timings.reduce`: core column reduction loop.
- `timings.bauer`: filling cleared $V$ columns when `use_clearing=True` and `compute_v=True`.
- `timings.restore_elz`: optional post-pass when `dims_to_restore_elz` is requested.
- `timings.copy_back` / `timings.copy_pivots`: moving parallel results to rest storage.
- `timings.compute_u`: parallel $V^T U^T = I$ solve when `compute_u=True`.
- `timings.reduction_total`: total end-to-end reduction time comparable across all thread counts.

Under `params.advanced` ({py:class}`oineus.ReductionParamsAdvanced`), you can also select the working-column data structure ({py:class}`oineus.ColumnRepr`) and parallel work-stealing `chunk_size`:
- `ColumnRepr.BitTree` *(default)*: hierarchical 64-ary bitset (PHAT `bit_tree_pivot_column`) — fastest all-around $O(1)$ pivot lookup and symmetric difference.
- `ColumnRepr.Full`: flat bitvector + max-heap.
- `ColumnRepr.Heap`: lazy max-heap (compact on ultra-sparse matrices).
- `ColumnRepr.Set`: `std::set` balanced tree.

Let's profile thread scaling and `ReductionTimings` phases on our 3D grid filtration:

```{code-cell} ipython3
vol_bench = rng.random((48, 48, 48), dtype=np.float64)
fil_bench = oin.freudenthal_filtration(vol_bench, max_dim=3, n_threads=4)

thread_counts = [1, 2, 4]
totals_ms = []
for nt in thread_counts:
    p = oin.ReductionParams(n_threads=nt, use_clearing=True, compute_v=False, col_repr=oin.ColumnRepr.BitTree)
    d = oin.reduce(fil_bench, p)
    t = d.timings
    totals_ms.append(t.reduction_total * 1e3)
    print(
        f"n_threads={nt} | total={t.reduction_total*1e3:6.2f} ms "
        f"(prepare={t.prepare*1e3:5.2f} ms, reduce={t.reduce*1e3:6.2f} ms, pivots={t.copy_pivots*1e3:4.2f} ms)"
    )

fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.bar([str(nt) for nt in thread_counts], totals_ms, color=oin.OKABE_ITO_BLUE, width=0.5)
ax.set_xlabel("n_threads")
ax.set_ylabel("reduction_total (ms)")
ax.set_title(f"Parallel Fused Reduction ({fil_bench.size():,} simplices)")
fig.tight_layout()
plt.show()
```

> [!TIP]
> **Quick rule of thumb for Vietoris–Rips vs Grids**:
> - For **Vietoris–Rips** (`vr_filtration`), pass **`dualize=True`** (persistent cohomology, already the default in `compute_diagrams_vr`) + `use_clearing=True`. Coboundary columns of high-codimension VR simplices stay sparse, yielding huge speedups over primal homology.
> - For **Lower-star Grids** (`freudenthal_filtration`, `cube_filtration`) and **Alpha shapes** (`alpha_filtration`), keep **`dualize=False`** (primal homology, the default in `compute_diagrams_ls` and `compute_diagrams_alpha`).

---

## What's Next?

- Export matrices to SciPy, serialize objects with `pickle`, exchange complexes with GUDHI / Dionysus / `diode`, and review Oineus's architectural limits in [16. Ecosystem Interop & Design Boundaries](16_interop_and_limitations.md).
