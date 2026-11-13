# Critical-Set Optimization

Low-level C++ topology optimizers implementing the critical-set ("big steps") method of Nigmetov & Morozov (2022) for simplicial, product, and cubical filtrations, together with conflict-resolution and denoising target helpers.

:::{seealso}
For reverse-mode autodiff integration with **PyTorch** and **JAX**, see {doc}`diff` ({py:func}`oineus.diff.persistence_diagram` and {py:class}`oineus.diff.TopologyOptimizer`).
:::

## Optimizer Classes & Target Containers

```{eval-rst}
.. currentmodule:: oineus

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   TopologyOptimizer
   TopologyOptimizerProd
   TopologyOptimizerCube_1D
   TopologyOptimizerCube_2D
   TopologyOptimizerCube_3D
   TopologyOptimizerCube_4D
   IndicesValues
   IndicesValuesProd
```

## Strategies & Target Helpers

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ConflictStrategy
   DenoiseStrategy

.. autosummary::
   :toctree: _autosummary

   get_nth_persistence
   get_denoise_target
```
