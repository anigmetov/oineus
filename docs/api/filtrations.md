# Filtrations & Complex Construction

Functions and classes in `oineus` for building simplicial, cubical, and product filtrations from regular grids, point clouds, pairwise distances, simplicial maps, or explicit cell lists.

## Filtration Classes & Metadata

```{eval-rst}
.. currentmodule:: oineus

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   Filtration
   ProdFiltration
   FiltrationKind
   VREdge
```

## Grid Filtrations (1D–4D)

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   freudenthal_filtration
   cube_filtration
```

## Point-Cloud & Metric Filtrations

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   vr_filtration
   alpha_filtration
   cech_filtration
   function_delaunay_bifiltration
   max_distance
```

## Custom Complexes, Operations & Boundary Matrices

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   list_to_filtration
   mapping_cylinder
   multiply_filtration
   min_filtration
   get_boundary_matrix
   to_scipy_matrix
```
