# Interoperability

Oineus is happy to round-trip data with the rest of the scientific Python
stack. This page collects the conversions that come up most often.

## NumPy

Persistence diagrams are `(n, 2)` `float64` NumPy arrays (or
`float32` if Oineus was built with `OINEUS_PYTHON_REAL=float`). Pass them
to {py:func}`oineus.bottleneck_distance`,
{py:func}`oineus.wasserstein_distance`, and the plotting helpers.

To extract a single dimension from a {py:class}`oineus.Diagrams`:

```{code-block} python
arr  = dgms.in_dimension(1)                     # (n, 2) NumPy array
pts  = dgms.in_dimension(1, as_numpy=False)     # list of DiagramPoint objects
```

## SciPy sparse

The reduction matrices $R$, $V$, $U$, $D$ are accessible as SciPy-style
$\mathbb{F}_2$ sparse matrices:

```{code-block} python
R = dcmp.r_as_csc()    # scipy.sparse.csc_matrix
V = dcmp.v_as_csc()
D = dcmp.d_as_csc()
U = dcmp.u_as_csr()    # u_data is stored row-major
```

Use these for matrix sanity checks ($R = D V$ mod 2), exporting to other
tools, or pulling cycles out by hand.

## PyTorch

The {py:mod}`oineus.diff` subpackage takes `torch.Tensor` inputs and
returns differentiable diagram tensors; see {doc}`differentiable` for
the full pattern. Key conventions:

- Pass `torch.float64` tensors. `float32` inputs are silently up-cast,
  which surprises some downstream code.
- Tensors that carry `requires_grad` keep their gradient flow through the
  filtration construction and reduction.
- The diff API requires `include_inf_points=False` for now.

## GUDHI

Oineus and GUDHI agree on the meaning of a persistence diagram (an
unordered set of `(birth, death)` pairs per dimension), so the
conversion is just NumPy:

```{code-block} python
# Oineus -> GUDHI
gudhi_pairs = [(int(d), tuple(map(float, p)))
               for d in dgms.dims() for p in dgms.in_dimension(d)]

# GUDHI -> Oineus (a single dimension as an (n, 2) numpy array)
import numpy as np
arr = np.array([(b, d) for (_, (b, d)) in gudhi_pairs if _ == 1])
```

For cross-validation we routinely compare Oineus diagrams to GUDHI's
cubical or alpha output on shared inputs; see
`tests/test_fr_dgm_random.py` for an in-tree example using dionysus.

## diode

{py:func}`oineus.compute_diagrams_alpha`,
{py:func}`oineus.diff.alpha_filtration`, and the underlying alpha-shape
filtration builder all depend on `diode` (CGAL bindings) for the Delaunay
construction. Install it via `pip install diode`. The combinatorics come
from diode; the filtration values are recomputed inside Oineus (and, in
the diff case, with autograd attached).

## Pickle

All major Oineus classes are picklable, including the heavy ones:

- {py:class}`oineus.Filtration`, {py:class}`oineus.ProdFiltration`
- {py:class}`oineus.Decomposition`
- {py:class}`oineus.Diagrams`, {py:class}`oineus.DiagramPoint`
- {py:class}`oineus.KerImCokReduced`, {py:class}`oineus.KerImCokReducedProd`
- {py:class}`oineus.ReductionParams`, {py:class}`oineus.KICRParams`
- {py:class}`oineus.TopologyOptimizer`
- {py:class}`oineus.DiagramMatching`, {py:class}`oineus.BottleneckMatching`

This means you can:

- Hand an entire `TopologyOptimizer` (with its cached boundary matrix) to
  a `multiprocessing` worker.
- Cache reduced filtrations to disk with `joblib.Memory`.
- Round-trip a `KerImCokReduced` through any pickle-aware serialization
  layer.

The one common gotcha: a pickle from a build with one
`OINEUS_PYTHON_REAL` (e.g. `double`) is not compatible with a build using
the other (e.g. `float`).

## See also

- {doc}`filtrations` -- how to build a `Filtration` from a NumPy array or
  a Python list.
- {doc}`differentiable` -- the PyTorch path.
- {doc}`performance` -- when to keep intermediate state around and when
  to throw it away.
