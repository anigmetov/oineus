# Visualization (`oineus.vis`)

Publication-ready Matplotlib plotting utilities for persistence diagrams, bottleneck/Wasserstein matchings, per-point optimization gradients, and algebraic cycles/chains on 2D grids and 2D/3D point clouds.

:::{note}
Every function and constant in `oineus.vis` is also re-exported directly on the top-level `oineus` module (for example, `oin.plot_diagram` is identical to `oin.vis.plot_diagram`).
:::

## Main Plotting Functions

```{eval-rst}
.. currentmodule:: oineus.vis

.. autosummary::
   :toctree: _autosummary

   plot_diagram
   plot_matching
   plot_diagram_gradient
   plot_chain
```

## Style Builder Helpers

Each `default_*_style(**overrides)` function returns a fresh copy of the corresponding default Matplotlib keyword-argument dictionary merged with `overrides`, leaving global defaults untouched.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   default_point_style
   default_inf_point_style
   default_inf_line_style
   default_diagonal_style
   default_grid_style
   default_density_style
   default_diagram_a_point_style
   default_diagram_b_point_style
   default_matching_edge_style
   default_longest_edge_style
   default_diagonal_projection_a_style
   default_diagonal_projection_b_style
   default_diagram_gradient_style
   default_point_cloud_style
   default_chain_vertex_style
   default_chain_edge_style
   default_chain_triangle_style
   default_chain_tetrahedron_style
```

## Palette & Default Style Constants

```{eval-rst}
.. autodata:: oineus.vis.OKABE_ITO_BLUE
.. autodata:: oineus.vis.OKABE_ITO_VERMILLION
.. autodata:: oineus.vis.DEFAULT_DENSITY_THRESHOLD
.. autodata:: oineus.vis.DEFAULT_MATCHING_EDGE_QUANTILE
.. autodata:: oineus.vis.DEFAULT_GRADIENT_TOP_K_ARROWS
.. autodata:: oineus.vis.DEFAULT_POINT_STYLE
.. autodata:: oineus.vis.DEFAULT_INF_POINT_STYLE
.. autodata:: oineus.vis.DEFAULT_INF_LINE_STYLE
.. autodata:: oineus.vis.DEFAULT_DIAGONAL_STYLE
.. autodata:: oineus.vis.DEFAULT_GRID_STYLE
.. autodata:: oineus.vis.DEFAULT_DENSITY_STYLE
.. autodata:: oineus.vis.DEFAULT_DIAGRAM_A_POINT_STYLE
.. autodata:: oineus.vis.DEFAULT_DIAGRAM_B_POINT_STYLE
.. autodata:: oineus.vis.DEFAULT_MATCHING_EDGE_STYLE
.. autodata:: oineus.vis.DEFAULT_LONGEST_EDGE_STYLE
.. autodata:: oineus.vis.DEFAULT_DIAGONAL_PROJECTION_A_STYLE
.. autodata:: oineus.vis.DEFAULT_DIAGONAL_PROJECTION_B_STYLE
.. autodata:: oineus.vis.DEFAULT_DIAGRAM_GRADIENT_STYLE
.. autodata:: oineus.vis.DEFAULT_POINT_CLOUD_STYLE
.. autodata:: oineus.vis.DEFAULT_CHAIN_VERTEX_STYLE
.. autodata:: oineus.vis.DEFAULT_CHAIN_EDGE_STYLE
.. autodata:: oineus.vis.DEFAULT_CHAIN_TRIANGLE_STYLE
.. autodata:: oineus.vis.DEFAULT_CHAIN_TETRAHEDRON_STYLE
```
