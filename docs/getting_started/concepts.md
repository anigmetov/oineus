# Core Concepts & Decision Guide

Every workflow in Oineus follows a clean, modular pipeline where you can either call a high-level one-liner or inspect and manipulate each intermediate stage directly.

## The Oineus Pipeline

```text
  Input Data            Filtration             Matrix Reduction        Diagrams & Cycles        Downstream Analysis
┌────────────┐        ┌────────────┐          ┌────────────────┐      ┌─────────────────┐      ┌───────────────────┐
│ Grid array │        │ Filtration │          │ Decomposition  │      │ oin.Diagrams    │      │ Bottleneck / Wass │
│ Point cloud├───────►│ ProdFiltr. ├─────────►│ R = D V, U=V⁻¹ ├─────►│ KICR / Mixup    ├─────►│ Frechet means     │
│ Simplices  │        │ DiffFiltr. │          │ or oin.reduce  │      │ Cycles/Cochains │      │ oineus.vis plots  │
│ Torch/JAX  │        └────────────┘          └────────────────┘      └─────────────────┘      │ PyTorch/JAX grad  │
└────────────┘                                                                                 └───────────────────┘
```

### Two Entry Levels

1. **One-call convenience functions** ({py:func}`oineus.compute_diagrams_ls`, {py:func}`oineus.compute_diagrams_vr`, {py:func}`oineus.compute_diagrams_alpha`, {py:func}`oineus.mixup_barcodes`) build the filtration, run parallel reduction with sensible defaults (`use_clearing=True`), and return diagrams immediately.
2. **Explicit stage objects** ({py:class}`oineus.Filtration` $\to$ {py:class}`oineus.Decomposition` $\to$ `dcmp.diagram(fil)`) give you full control over:
   - homology vs. cohomology (`dualize=False` vs. `dualize=True`),
   - tracking the reduction matrix $V$ (`compute_v=True`) and its inverse $U = V^{-1}$ (`compute_u=True` or on-demand `compute_partial_u_rows`),
   - extracting representative cycles, bounding chains, and cocycles from `r_data` and `v_data`,
   - dynamic updates (vineyards `transpose`/`move`, Luo–Nelson `update_with_permutation`, SiRUP `remove_simplices`),
   - critical-set optimization ({py:class}`oineus.TopologyOptimizer`).

---

## Core Conventions to Keep in Mind

### 1. Indexing: User `id` vs. Filtration `sorted_id`

Every cell in a {py:class}`oineus.Filtration` carries two integer identifiers:

- **`id` (`uid`)**: the cell's identifier prior to filtration sorting (for example, vertex `0..N-1` in a point cloud or user-assigned integer IDs in {py:func}`oineus.list_to_filtration`).
- **`sorted_id`**: the cell's `0`-based position in filtration order (sorted first by value—or descending value when `negate=True`—and then by dimension so faces always precede cofaces).

Columns and rows of boundary matrices (`r_data`, `v_data`, `d_data`) and index persistence diagrams (`dgms.index_diagram_in_dimension(q)`) use **`sorted_id`** (when `dualize=True`, matrix indices are reversed coboundary indices convertible via `dcmp.filtration_index(matrix_idx)`). Use `fil.id_by_sorted_id(sid)` and `fil.sorted_id_by_id(uid)` to translate freely between the two.

### 2. Filtration `max_dim` vs. Diagram `max_dim`

To know when a $k$-dimensional cycle becomes a boundary (dies), the complex **must contain $(k+1)$-cells**:

- **Filtration builders** ({py:func}`oineus.vr_filtration`, `oineus.diff.vr_filtration`, {py:func}`oineus.freudenthal_filtration`) interpret `max_dim` as the **maximum cell dimension** included in the filtration. To obtain accurate $H_k$ deaths, pass `max_dim = k + 1` to `vr_filtration`!
- **High-level diagram functions** ({py:func}`oineus.compute_diagrams_vr`, {py:func}`oineus.compute_diagrams_ls`) interpret `max_dim` as the **maximum homology diagram dimension**: `compute_diagrams_vr(..., max_dim=k)` automatically constructs the `(k + 1)`-skeleton under the hood and trims the output to $H_0 \dots H_k$.

### 3. Filtration Scale Conventions

Different geometric complexes parameterize filtration values differently by mathematical convention:

| Filtration | Value Convention | Relationship to Ball Radius $r$ |
| --- | --- | --- |
| **Vietoris–Rips** (`vr_filtration`, `compute_diagrams_vr`) | Pairwise **diameter** | $2r$ |
| **Čech** (`cech_filtration`, `diff.cech_filtration`) | **Radius** of smallest enclosing ball (MEB) | $r$ (`diff.cech_filtration` uses $r^2$ when `squared=True`, default `True`; check parameter!) |
| **Alpha / Weak Alpha / Čech–Delaunay** (`alpha_filtration`, `compute_diagrams_alpha`, `diff.alpha_filtration`, `diff.cech_delaunay_filtration`) | **Squared radius** of circumsphere / MEB | $r^2$ |

Always compare diagrams built with compatible scale conventions (or take $\sqrt{\cdot}$ / multiply by $2$ before computing distances).

---

## Decision Tables

### Which Filtration Should I Use?

| Input Data | Best Choice | Why / Trade-offs |
| --- | --- | --- |
| 1D–4D image / volume (isotropic topology, critical points on pixels/voxels) | {py:func}`oineus.cube_filtration` | Compact cell count on grids; supports `values_on="vertices"` or `"cells"`. |
| 1D–4D scalar field (simplicial complex or periodic `wrap=True` torus needed) | {py:func}`oineus.freudenthal_filtration` | Purely simplicial ($d!$ top simplices per cube); supports periodic `wrap=True` boundary conditions. |
| Point cloud in $\mathbb{R}^2$ or $\mathbb{R}^3$ (up to $\mathbb{R}^d$, small $d$) | {py:func}`oineus.alpha_filtration` / {py:func}`oineus.compute_diagrams_alpha` | Exact nerve equivalence to union of balls with $O(N)$–$O(N^{\lceil d/2 \rceil})$ Delaunay simplices; supports weights and periodic boxes (`diode`). |
| Differentiable point cloud in $\mathbb{R}^2$ or $\mathbb{R}^3$ | {py:func}`oineus.diff.cech_delaunay_filtration` or {py:func}`oineus.diff.alpha_filtration` | `cech_delaunay_filtration` evaluates closed-form MEBs over Delaunay simplices (also supports periodic torus boxes). |
| High-dimensional point cloud or arbitrary distance matrix | {py:func}`oineus.vr_filtration` / {py:func}`oineus.compute_diagrams_vr` | Metric-only, works in any dimension or `from_pwdists=True`; set `max_diameter` to cap combinatorial blowup. |
| Two point clouds $A, B$ (measuring geometric entanglement / hole-filling) | {py:func}`oineus.mixup_barcodes` or {py:func}`oineus.function_delaunay_bifiltration` | Quantifies how adding $B$ kills features born in $A$. |
| Inclusion $L \hookrightarrow K$ or simplicial map $f: K \to L$ | {py:func}`oineus.compute_kernel_image_cokernel_reduction` / {py:func}`oineus.mapping_cylinder` | Extracts $\ker f_*$, $\operatorname{im} f_*$, and $\operatorname{coker} f_*$ persistence diagrams. |

### Which Diagram Distance Should I Use?

| Goal | Function | Complexity & Differentiability |
| --- | --- | --- |
| Worst-case ($L_\infty$) stability bound or longest matched shift | {py:func}`oineus.bottleneck_distance` / {py:func}`oineus.bottleneck_matching` | Exact (`delta=0.0`) or $(1+\delta)$-approx via Hera. |
| Aggregate $L_q$ discrepancy across all features + explicit matching | {py:func}`oineus.wasserstein_distance` / {py:func}`oineus.wasserstein_matching` | Exact/approx via Hera auction; differentiable wrapper in {py:func}`oineus.diff.wasserstein_cost`. |
| Fast kernel / ML loss over batches of diagrams (PyTorch / JAX / NumPy) | {py:func}`oineus.sliced_wasserstein_distance_diag_corrected` (and `oineus.diff.*`) | $O(M N \log N)$ closed-form 1D projections; smoothly differentiable everywhere; `diag_corrected` removes cross-cardinality diagonal bias. |
| Consensus / average diagram of a collection $\{D_1, \dots, D_m\}$ | {py:func}`oineus.progressive_frechet_mean_multistart` | Coarse-to-fine persistence thresholding + multi-start k-means-style alternation avoids poor local minima. |

### Which Differentiable / Optimization Method Should I Use?

| Scenario | Method | Gradient Support & Behavior |
| --- | --- | --- |
| Fast standard topological regularizer in a neural net or point cloud | `diff.persistence_diagram(..., gradient_method="dgm-loss")` | **Sparse** ($2$ simplices per bar: birth creator & death destroyer). Fastest forward + backward (`R` only, no `V`/`U`). |
| Pushing plateaued scalar fields or taking large steps across critical-value swaps | `diff.persistence_diagram(..., gradient_method="crit-sets")` or {py:class}`oineus.TopologyOptimizer` | **Critical set** (columns of $V$ / selected rows of $U$ up to target threshold). Moves entire basins/plateaus coherently without stalling. |
| Smoothly growing/shrinking a geometric loop in a point cloud without spiky single-edge artifacts | {py:func}`oineus.diff.persistence_content_loss` (`oineus.diff.cochains`) | **Harmonic $\varepsilon$-cochain support** (Weighill & Zhou). Distributes gradient smoothly over all edges/triangles around the hole across multiple $\varepsilon$ scales. |
