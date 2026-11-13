# Python API Reference

This reference documents **every public class, enum, function, and constant** exported by Oineus across its Python namespaces:

- **`oineus`** — core filtrations, cells & grids, parallel matrix reduction, persistence diagrams, bottleneck/Wasserstein/sliced Wasserstein distances, Fréchet means, relative/KICR persistence, mixup barcodes, and critical-set optimization.
- **`oineus.diff`** (and **`oineus.diff.cochains`**) — reverse-mode differentiable filtrations, diagrams, KICR, mixup, Wasserstein losses, and Weighill–Zhou birth/death cochain content losses for **PyTorch** and **JAX**.
- **`oineus.vis`** — Matplotlib visualization utilities and Okabe–Ito style presets (all plotting functions and style constants are also re-exported directly under `oineus.*`).

:::{tip}
Oineus ships PEP 561 `.pyi` type stubs (`oineus/py.typed`), so IDEs (VS Code / Pylance, PyCharm) and static type checkers (`mypy`, `pyright`) provide full autocomplete and type checking for all C++ and Python APIs—including facade classes like {py:class}`oineus.Filtration` and {py:class}`oineus.Diagrams`.
:::

::::{grid} 1 1 2 2
:gutter: 3
:margin: 2

:::{grid-item-card} Filtrations & Construction
:link: filtrations
:link-type: doc

`Filtration`, `ProdFiltration`, Freudenthal, cubical, Vietoris–Rips, Alpha, Čech, Function–Delaunay, mapping cylinders, and SciPy boundary export.
:::

:::{grid-item-card} Cells & Regular Grids
:link: cells_grids
:link-type: doc

`Simplex`, `ProdSimplex`, `Cube_1D`–`Cube_4D`, combinatorial cell views, and `Grid_1D`–`Grid_4D` domains.
:::

:::{grid-item-card} Matrix Reduction & Dynamic Persistence
:link: decomposition
:link-type: doc

`Decomposition` ($R = DV$, $U = V^{-1}$), fused `reduce`, `ReductionParams`, dynamic vineyards, Luo–Nelson warm starts, and SiRUP `remove_simplices`.
:::

:::{grid-item-card} Diagrams, Distances & Fréchet Means
:link: diagrams_distances
:link-type: doc

`Diagrams`, `DiagramPoint`, Bottleneck, Wasserstein, Sliced Wasserstein, explicit matchings, essential `InfKind` pairs, and Wasserstein barycenters.
:::

:::{grid-item-card} Relative Persistence, KICR & Mixup
:link: kicr_mixup
:link-type: doc

Relative homology, kernel/image/cokernel persistence (`KerImCokReduced`), induced matchings, and two-population `MixupBarcodes`.
:::

:::{grid-item-card} Critical-Set Optimization
:link: optimization
:link-type: doc

`TopologyOptimizer`, `ConflictStrategy`, `DenoiseStrategy`, `IndicesValues`, and target helpers (`get_denoise_target`, `get_nth_persistence`).
:::

:::{grid-item-card} Differentiable TDA (`oineus.diff`)
:link: diff
:link-type: doc

PyTorch & JAX differentiable filtrations, `persistence_diagram`, `kicr_diagrams`, `mixup_barcodes`, Wasserstein losses, and `oineus.diff.cochains`.
:::

:::{grid-item-card} Visualization (`oineus.vis`)
:link: vis
:link-type: doc

`plot_diagram`, `plot_matching`, `plot_diagram_gradient`, `plot_chain`, style builders, and Okabe–Ito palette constants.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

filtrations
cells_grids
decomposition
diagrams_distances
kicr_mixup
optimization
diff
vis
```

## Global Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
