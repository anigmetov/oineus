# API reference

Auto-generated from the installed `oineus` package. Entries are grouped
thematically below; full alphabetical indices live at the end.

```{note}
Many public functions currently have minimal docstrings. Signatures and
types are accurate; prose will improve over time.
```

## Filtration construction

```{eval-rst}
.. currentmodule:: oineus

.. autosummary::
   :toctree: _autosummary

   freudenthal_filtration
   vr_filtration
   cube_filtration
   alpha_filtration
   compute_diagrams_alpha
   list_to_filtration
   mapping_cylinder
   min_filtration
   multiply_filtration
```

## Persistence computation

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   compute_diagrams_ls
   compute_diagrams_vr
   compute_relative_diagrams
   get_boundary_matrix
   is_reduced
```

## Diagrams

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   Diagrams
   DiagramPoint
   IndexDiagramPoint
   DiagramMatching
```

## Distances and matchings

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   bottleneck_distance
   wasserstein_distance
   sliced_wasserstein_distance
   sliced_wasserstein_distance_diag_corrected
   wasserstein_matching
   bottleneck_matching
   point_to_diagonal
   get_permutation_dtv
```

## Fréchet means

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   frechet_mean
   frechet_mean_multistart
   progressive_frechet_mean
   progressive_frechet_mean_multistart
   frechet_mean_objective
   init_frechet_mean_first_diagram
   init_frechet_mean_random_diagram
   init_frechet_mean_medoid_diagram
   init_frechet_mean_diagonal_grid
```

## Core classes

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   Simplex
   CombinatorialSimplex
   ProdSimplex
   Filtration
   ProdFiltration
   Decomposition
   ReductionParams
```

## Kernel / image / cokernel

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   KICRParams
   KerImCokReduced
   KerImCokReducedProd

.. autosummary::
   :toctree: _autosummary

   compute_kernel_image_cokernel_reduction
   compute_ker_cok_reduction_cyl
   get_induced_matching
```

## Topology optimization

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   TopologyOptimizer
   TopologyOptimizerProd
   IndicesValues

.. autosummary::
   :toctree: _autosummary

   get_denoise_target
   get_nth_persistence
   get_ls_wasserstein_matching_target_values
```

## Enums

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ConflictStrategy
   DenoiseStrategy
   DiagramPlaneDomain
   FrechetMeanInit
   VREdge
```

## Utilities

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   to_scipy_matrix
   max_distance
   plot_diagram
   plot_diagram_gradient
   plot_matching
   plot_chain
```

## Differentiable (`oineus.diff`)

```{eval-rst}
.. currentmodule:: oineus.diff

.. autosummary::
   :toctree: _autosummary

   freudenthal_filtration
   vr_filtration
   cube_filtration
   alpha_filtration
   weak_alpha_filtration
   cech_delaunay_filtration
   mapping_cylinder_filtration
   min_filtration
   persistence_diagram
   sliced_wasserstein_distance
   sliced_wasserstein_distance_diag_corrected
   wasserstein_cost

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   DiffFiltration
   TopologyOptimizer
   PersistenceDiagrams
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
