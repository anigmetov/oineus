# Differentiable TDA (`oineus.diff`)

Autodiff-compatible filtrations, persistence diagrams, kernel/image/cokernel diagrams, mixup barcodes, Wasserstein / Sliced Wasserstein losses, and Weighill–Zhou birth/death cochain content losses (`oineus.diff.cochains`) supporting **PyTorch** (`torch.Tensor`) and **JAX** (`jax.Array`).

## Differentiable Filtrations

```{eval-rst}
.. currentmodule:: oineus.diff

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   DiffFiltration

.. autosummary::
   :toctree: _autosummary

   freudenthal_filtration
   cube_filtration
   vr_filtration
   alpha_filtration
   weak_alpha_filtration
   cech_filtration
   cech_delaunay_filtration
   mapping_cylinder_filtration
   min_filtration
```

## Differentiable Diagrams, KICR & Mixup

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   persistence_diagram
   kicr_diagrams
   mixup_barcodes

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   PersistenceDiagrams
   KICRDiagrams
   KICRFamilyDiagrams
   DiffMixupBarcodes
```

## Differentiable Distances & Losses

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   wasserstein_cost
   sliced_wasserstein_distance
   sliced_wasserstein_distance_diag_corrected
```

## Birth & Death Cochains (`oineus.diff.cochains`)

All symbols below are available both on `oineus.diff.*` and on the submodule `oineus.diff.cochains.*`.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   Cochain

.. autosummary::
   :toctree: _autosummary

   birth_cochain
   death_cochain
   birth_death_cochains
   birth_content
   death_content
   persistence_content_loss
```

## Optimizer Wrapper, Strategies & Geometry Primitives

Note that `oineus.diff` also re-exports {py:class}`oineus.ConflictStrategy` and {py:class}`oineus.DenoiseStrategy` for convenience.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   TopologyOptimizer
   UStrategy

.. autosummary::
   :toctree: _autosummary

   edge_circumradius_sq
   triangle_circumradius_sq
   tetrahedron_circumradius_sq
   triangle_meb
   tetrahedron_meb

.. autodata:: oineus.diff.TORCH_AVAILABLE

   Boolean flag indicating whether ``torch`` could be imported in the current Python environment.
```
