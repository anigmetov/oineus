# Tutorials

Every tutorial below is an **executable MyST-NB document**: all code blocks run against Oineus during the documentation build, so every printed number, timing, and Matplotlib figure matches the current library behavior.

We recommend reading {doc}`../getting_started/quickstart` and {doc}`../getting_started/concepts` first, then following whichever track matches your task.

---

## Track A — Filtrations & Complexes

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 1. Grid & Image Filtrations
:link: 01_grid_filtrations
:link-type: doc

Sublevel vs. superlevel (`negate`), Freudenthal vs. Cubical complexes, `values_on="vertices"` vs. `"cells"`, periodic `wrap=True` tori, and 1D–4D volumes.
:::

:::{grid-item-card} 2. Point-Cloud Filtrations
:link: 02_point_cloud_filtrations
:link-type: doc

Vietoris–Rips (`max_dim` vs. diagram `max_dim`, distance matrices), Alpha (weighted, periodic, exact), Weak Alpha, Čech, Čech–Delaunay, and scale conventions.
:::

:::{grid-item-card} 3. Custom Complexes & Filtrations
:link: 03_custom_filtrations
:link-type: doc

`Simplex`, `Cube_*D`, `ProdSimplex`, `list_to_filtration`, `id` vs. `sorted_id`, `subfiltration`, `without_cells`, `star_closure`, `min_filtration`, and SciPy boundary export.
:::
::::

## Track B — Matrix Reduction, Cycles & Visualization

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 4. Matrix Reduction & Decomposition
:link: 04_matrix_reduction
:link-type: doc

Inside $R = D V$ and $U = V^{-1}$, fused `reduce`, homology vs. cohomology (`dualize`), clearing & apparent pairs, canonical ELZ form (`restore_elz`), and zero-persistence diagrams.
:::

:::{grid-item-card} 5. Representative Cycles & Visualization
:link: 05_cycles_and_visualization
:link-type: doc

Extracting homologous cycles and bounding chains from `r_data` / `v_data`, overlaying them with `plot_chain`, hybrid scatter-density `plot_diagram`, `InfKind`, and Okabe–Ito styling.
:::
::::

## Track C — Distances, Matchings & Barycenters

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 6. Distances & Matchings
:link: 06_distances_and_matchings
:link-type: doc

Exact and approximate Bottleneck & $q$-Wasserstein distances, inspecting `DiagramMatching` / `BottleneckMatching` & `InfKind` essential pairings, `plot_matching`, and Sliced Wasserstein.
:::

:::{grid-item-card} 7. Wasserstein Fréchet Means
:link: 07_frechet_mean
:link-type: doc

Computing consensus persistence diagrams (barycenters), navigating non-convex local minima, initializer strategies, multistart, and coarse-to-fine `progressive_frechet_mean_multistart`.
:::
::::

## Track D — Maps, Subcomplexes & Multi-Group Interactions

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 8. Relative Persistence, KICR & Mapping Cylinders
:link: 08_relative_kicr_and_cylinders
:link-type: doc

Relative homology $H_*(K, L)$, kernel / image / cokernel persistence (`KerImCokReduced`), general simplicial maps via `mapping_cylinder`, and Bauer–Lesnick `get_induced_matching`.
:::

:::{grid-item-card} 9. Mixup Barcodes & Function–Delaunay
:link: 09_mixup_and_function_delaunay
:link-type: doc

Measuring how a second point cloud $B$ fills or splits topology in $A$ (`mixup_barcodes`), birth/image-death/death triples, and scaling via `function_delaunay_bifiltration`.
:::
::::

## Track E — Differentiable TDA & Optimization (`oineus.diff`)

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 10. Differentiable Persistence (`oineus.diff`)
:link: 10_differentiable_persistence
:link-type: doc

Reverse-mode autodiff through persistence, PyTorch & JAX backend support matrix, `float32` vs. `float64`, `dgm-loss` vs. `crit-sets` (`ConflictStrategy`), and `plot_diagram_gradient`.
:::

:::{grid-item-card} 11. End-to-End Topological Optimization
:link: 11_topological_optimization
:link-type: doc

Point-cloud sculpting with `wasserstein_cost` & `sliced_wasserstein_distance_diag_corrected`, scalar-field topological denoising, periodic Čech–Delaunay, and `UStrategy` partial-$U$ economics.
:::

:::{grid-item-card} 12. Birth & Death Cochain Optimization
:link: 12_cochain_optimization
:link-type: doc

Weighill–Zhou harmonic $\varepsilon$-birth and $\varepsilon$-death cochains (`oineus.diff.cochains`), `persistence_content_loss`, and why multi-$\varepsilon$ cochain gradients move whole loops smoothly.
:::

:::{grid-item-card} 13. Differentiable KICR & Mixup
:link: 13_diff_kicr_and_mixup
:link-type: doc

Backpropagating through `diff.kicr_diagrams` and `diff.mixup_barcodes` to optimize topological alignment or separation between two populations in PyTorch and JAX.
:::
::::

## Track F — Dynamic Persistence, Performance & Interoperability

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 14. Dynamic Persistence & SiRUP
:link: 14_dynamic_persistence_and_sirup
:link-type: doc

`make_dynamic`, vineyards (`transpose`, `move`, `apply_move_schedule`), Luo–Nelson warm-started re-reduction (`update_with_permutation`, `update_with_edits`), and SiRUP `remove_simplices`.
:::

:::{grid-item-card} 15. Performance, Encodings & Precision
:link: 15_performance_and_encodings
:link-type: doc

Thread scaling (`n_threads`), fused `oineus.reduce`, slim (`slim=True`) and bit-packed (`packed=True`) filtrations, `float32` vs. `float64` memory footprint, and `ReductionTimings`.
:::

:::{grid-item-card} 16. Ecosystem Interop & Design Boundaries
:link: 16_interop_and_limitations
:link-type: doc

Bridging Oineus with NumPy, SciPy sparse, PyTorch, JAX, GUDHI, Dionysus, `diode`, and `pickle`, plus clear scope boundaries ($\mathbb{Z}_2$ coefficients, grid dims 1–4, vertex limits).
:::
::::

```{toctree}
:maxdepth: 1
:hidden:

01_grid_filtrations
02_point_cloud_filtrations
03_custom_filtrations
04_matrix_reduction
05_cycles_and_visualization
06_distances_and_matchings
07_frechet_mean
08_relative_kicr_and_cylinders
09_mixup_and_function_delaunay
10_differentiable_persistence
11_topological_optimization
12_cochain_optimization
13_diff_kicr_and_mixup
14_dynamic_persistence_and_sirup
15_performance_and_encodings
16_interop_and_limitations
```
