# Diagrams, Distances & Fréchet Means

Persistence diagram containers, one-call diagram shortcuts, Bottleneck / Wasserstein / Sliced Wasserstein distances, explicit optimal matchings (including essential `InfKind` families), and Wasserstein Fréchet means (barycenters).

## Persistence Diagram Containers

```{eval-rst}
.. currentmodule:: oineus

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   Diagrams
   DiagramPoint
   IndexDiagramPoint
   InfKind
```

## One-Call Diagram Computation

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   compute_diagrams_ls
   compute_diagrams_vr
   compute_diagrams_alpha
```

## Distances & Optimal Matchings

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   bottleneck_distance
   wasserstein_distance
   bottleneck_matching
   wasserstein_matching
   point_to_diagonal
   sliced_wasserstein_distance
   sliced_wasserstein_distance_diag_corrected
```

## Matching Result Structures

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   DiagramMatching
   BottleneckMatching
   EssentialMatches
   LongestEdges
   EssentialLongestEdges
   FiniteLongestEdge
   EssentialLongestEdge
```

## Wasserstein Fréchet Means (Barycenters)

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   DiagramPlaneDomain
   FrechetMeanInit

.. autosummary::
   :toctree: _autosummary

   frechet_mean
   frechet_mean_multistart
   progressive_frechet_mean
   progressive_frechet_mean_multistart
   frechet_mean_objective
   make_frechet_mean_persistence_schedule
   frechet_mean_newborn_points_from_newly_active
   init_frechet_mean_first_diagram
   init_frechet_mean_random_diagram
   init_frechet_mean_medoid_diagram
   init_frechet_mean_diagonal_grid
```
