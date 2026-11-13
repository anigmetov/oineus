# Matrix Reduction & Dynamic Persistence

Core boundary/coboundary matrix decomposition ($R = D V$ and $U = V^{-1}$), fused persistence reduction, dynamic vineyards, Luo–Nelson warm-started re-reduction, and SiRUP cell removal.

## Decomposition & Reduction Configuration

```{eval-rst}
.. currentmodule:: oineus

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   Decomposition
   ReductionParams
   ReductionParamsAdvanced
   ColumnRepr
   ReductionTimings
   UComputeTimings
   DecompositionManipStats
```

## Reduction, Dynamic Updates & Precision Helpers

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   reduce
   is_reduced
   remove_simplices
   get_permutation
   get_permutation_dtv
   as_real_numpy
```

## Precision Constants

```{eval-rst}
.. autodata:: oineus.REAL_DTYPE
   :annotation: = numpy.dtype('float64')

   Default floating-point NumPy dtype of the compiled Oineus C++ extension.
```
