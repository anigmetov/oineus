# Runnable Examples Catalog

The repository ships **31 standalone Python scripts** under [`examples/python/`](https://github.com/anigmetov/oineus/tree/master/examples/python). Run any script directly after installing `oineus` (and any optional dependency noted in the table):

```bash
python examples/python/example_vr.py
```

---

## Filtrations & Core Persistence

| Script | Dependencies | Description | Related Tutorial |
| --- | --- | --- | --- |
| `example.py` | `numpy` | Lower-star Freudenthal filtration on a regular grid, matrix reduction, and diagram extraction. | {doc}`../tutorials/01_grid_filtrations` |
| `example_cube.py` | `numpy` | 3D cubical filtration (`values_on` vertices and cells) and persistence diagram computation. | {doc}`../tutorials/01_grid_filtrations` |
| `example_vr.py` | `numpy` | Vietoris–Rips filtration and persistence diagrams on a synthetic point cloud. | {doc}`../tutorials/02_point_cloud_filtrations` |
| `example_manual.py` | `numpy` | Building a custom simplicial filtration manually from {py:class}`oineus.Simplex` cells. | {doc}`../tutorials/03_custom_filtrations` |

## Distances, Matchings & Fréchet Means

| Script | Dependencies | Description | Related Tutorial |
| --- | --- | --- | --- |
| `example_matching.py` | `numpy`, `matplotlib` | Five scenarios covering `wasserstein_matching`, `bottleneck_matching`, tied longest edges, essential `InfKind` matches, and `plot_matching`. | {doc}`../tutorials/06_distances_and_matchings` |
| `example_sliced_wasserstein.py` | `numpy`, `torch` | Sliced Wasserstein and diagram-corrected Sliced Wasserstein distances as fast differentiable losses. | {doc}`../tutorials/06_distances_and_matchings` |
| `example_frechet_mean.py` | `numpy`, `matplotlib` | Single-start, multistart, and progressive Wasserstein Fréchet means (barycenters) of diagram ensembles. | {doc}`../tutorials/07_frechet_mean` |

## Relative Persistence, KICR, Mapping Cylinders & Mixup

| Script | Dependencies | Description | Related Tutorial |
| --- | --- | --- | --- |
| `example_cone.py` | `numpy` | Coning off a subcomplex as a concrete relative-persistence / mapping-cylinder example. | {doc}`../tutorials/08_relative_kicr_and_cylinders` |
| `example_kernel.py` | `numpy` | Kernel, image, and cokernel persistence diagrams via {py:func}`oineus.compute_kernel_image_cokernel_reduction`. | {doc}`../tutorials/08_relative_kicr_and_cylinders` |
| `example_kernel_cyl.py` | `numpy` | Kernel and cokernel persistence for general maps via {py:func}`oineus.mapping_cylinder` and {py:func}`oineus.compute_ker_cok_reduction_cyl`. | {doc}`../tutorials/08_relative_kicr_and_cylinders` |
| `mixup_example.py` | `numpy`, optional `torch` | Wagner et al. mixup barcodes comparing disjoint vs. hole-filling point clouds $A, B$ plus differentiable `oineus.diff.mixup_barcodes`. | {doc}`../tutorials/09_mixup_and_function_delaunay` |

## Visualization & Representative Cycles (`oineus.vis`)

| Script | Dependencies | Description | Related Tutorial |
| --- | --- | --- | --- |
| `test_plot_chain_alpha.py` | `matplotlib`, `diode` | Extracts a 1-cycle representative from `r_data` on an Alpha filtration and overlays it onto a 2D point cloud with `plot_chain`. | {doc}`../tutorials/05_cycles_and_visualization` |
| `test_plot_chain_field.py` | `matplotlib` | Highlights the saddle death cell merging two basins of a 2D scalar field using `plot_chain` on a cubical filtration. | {doc}`../tutorials/05_cycles_and_visualization` |
| `test_plot_chain_basin.py` | `matplotlib` | Visualizes the critical-set basin of attraction (cohomology $V$ column) of a scalar-field peak with `TopologyOptimizer.increase_birth` and `plot_chain`. | {doc}`../tutorials/05_cycles_and_visualization` |
| `test_plot_diagram_gradient.py` | `matplotlib`, `torch` | Computes a differentiable $H_1$ diagram, runs `loss.backward()`, and renders per-point gradient arrows with `plot_diagram_gradient`. | {doc}`../tutorials/10_differentiable_persistence` |
| `visualize_critical_set_vr.py` | `matplotlib` | Plots critical sets of simplices on a Vietoris–Rips filtration under birth/death target shifts. | {doc}`../tutorials/11_topological_optimization` |
| `visualize_critical_set_alpha.py` | `matplotlib`, `diode` | Plots critical sets on a 2D Alpha filtration. | {doc}`../tutorials/11_topological_optimization` |

## Differentiable TDA, Cochains & Benchmarks (`oineus.diff`)

| Script | Dependencies | Description | Related Tutorial |
| --- | --- | --- | --- |
| `example_diff_vr_pts.py` | `torch` | Backpropagating through `oineus.diff.vr_filtration` directly to 2D/3D point coordinates. | {doc}`../tutorials/10_differentiable_persistence` |
| `example_diff_vr_dists.py` | `torch` | Backpropagating through `oineus.diff.vr_filtration` from a differentiable pairwise distance matrix. | {doc}`../tutorials/10_differentiable_persistence` |
| `example_diff_alpha_grad.py` | `torch`, `diode`, `matplotlib` | Side-by-side gradient field comparison of differentiable Alpha vs. Čech–Delaunay filtrations on nested rings. | {doc}`../tutorials/10_differentiable_persistence` |
| `test_diff_diagram.py` | `torch` | Comprehensive walkthrough of `persistence_diagram` comparing `gradient_method="dgm-loss"` vs. `"crit-sets"` across birth/death shifts. | {doc}`../tutorials/10_differentiable_persistence` |
| `example_opt.py` | `torch` | Critical-set topological denoising (`TopologyOptimizer.simplify`) on a scalar grid. | {doc}`../tutorials/11_topological_optimization` |
| `example_opt_vr.py` | `torch` | Critical-set topological optimization on a Vietoris–Rips point-cloud filtration. | {doc}`../tutorials/11_topological_optimization` |
| `example_persistence_optim.py` | `torch` | Minimal PyTorch optimization loop amplifying the dominant $H_1$ feature on a $16 \times 16$ grid using `"crit-sets"`. | {doc}`../tutorials/11_topological_optimization` |
| `example_wasserstein_opt.py` | `torch` | Point-cloud shape optimization driven by `oineus.diff.wasserstein_cost` matching gradients. | {doc}`../tutorials/11_topological_optimization` |
| `example_opt_periodic_cech_delaunay.py` | `torch`, `diode` | Optimizing point positions on a flat 2D torus $[0, 1)^2$ via `cech_delaunay_filtration(..., periodic=True)`. | {doc}`../tutorials/11_topological_optimization` |
| `example_cochains_pointmover.py` | `torch`, `dionysus`, `matplotlib` | Weighill–Zhou multi-$\varepsilon$ `persistence_content_loss` vs. single-simplex `"dgm-loss"` on a noisy circle. | {doc}`../tutorials/12_cochain_optimization` |
| `example_cone_diff.py` | `torch` | Differentiable cone construction via `oineus.diff`. | {doc}`../tutorials/13_diff_kicr_and_mixup` |
| `example_diff_cyl.py` | `torch` | Differentiable mapping cylinder filtration (`oineus.diff.mapping_cylinder_filtration`). | {doc}`../tutorials/13_diff_kicr_and_mixup` |
| `bench_alpha_vs_cd.py` | `torch`, `diode` | Forward & backward timing benchmark comparing `diff.alpha_filtration` vs. `diff.cech_delaunay_filtration` on 3D clouds ($5\text{k}$–$10\text{k}$ points). | {doc}`../tutorials/15_performance_and_encodings` |
| `bench_periodic_cech_delaunay.py` | `torch`, `diode` | Runtime benchmark of periodic Delaunay lifts and `cech_delaunay_filtration(..., periodic=True)`. | {doc}`../tutorials/15_performance_and_encodings` |
