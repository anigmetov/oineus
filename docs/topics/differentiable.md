# Differentiable persistence diagrams

The {py:mod}`oineus.diff` subpackage turns the persistence pipeline into a
differentiable PyTorch layer: you feed in a tensor (a point cloud, a
distance matrix, a scalar field, ...), build a filtration whose values
inherit `requires_grad`, ask for the persistence diagram, drop the
diagram into any differentiable loss, and call `loss.backward()`.
Gradients propagate back through the persistence map to the input tensor.

This is **the canonical way to do topology-aware optimization in Oineus**.
Whether the goal is "match a target diagram", "increase the persistence
of this $H_1$ feature", or "remove this near-diagonal noise", the route
is the same: pick a filtration builder from `oineus.diff`, write the loss
on the diagram tensor, and let PyTorch handle the rest. The non-PyTorch
route via {py:class}`oineus.TopologyOptimizer` is also exposed
({doc}`optimization`), but for most users the diff layer is simpler and
strictly more flexible.

## The shared pattern

```{code-block} python
import torch
import oineus.diff as diff

# 1. Build a DiffFiltration over a torch tensor.
fil = diff.X_filtration(input_tensor, ...)

# 2. Extract diagrams. `include_inf_points=False` is currently required.
dgms = diff.persistence_diagram(fil, include_inf_points=False)

# 3. Build any differentiable loss on the diagram tensors.
dgm1 = dgms[1]                       # (n_h1, 2) tensor of (b, d)
loss = (dgm1[:, 1] - dgm1[:, 0]).pow(2).sum()    # total H1 persistence^2

# 4. Backprop.
loss.backward()
print(input_tensor.grad)
```

`dgms` behaves like a `dict[int, Tensor]`. The returned tensors carry
gradient flow back to `fil.values`, which in turn is a function of the
input tensor.

`include_inf_points=False` is the only currently-supported option (the
forward pass throws if you ask for `True`). The standard fix is to cap
`max_diameter` (VR) or pick a `max_dim` (Cech-Delaunay) so the
topological features you care about die finitely; in most loss designs
you do not want essentials anyway.

## Point clouds

### Cech-Delaunay

The differentiable Cech-Delaunay filtration assigns each Delaunay simplex
the radius of its minimum enclosing ball. The combinatorics come from
CGAL via `diode`; the values are recomputed analytically in PyTorch so
gradients flow back to the points.

```{code-block} python
pts = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)
fil = diff.cech_delaunay_filtration(pts)
dgms = diff.persistence_diagram(fil)
```

### Weak alpha

Weak alpha is the cheaper sibling: it uses the same Delaunay
combinatorics but takes each simplex's value to be the squared
circumradius of its longest *edge* (rather than the minimum enclosing
ball of its vertices). Same gradient interface:

```{code-block} python
fil = diff.weak_alpha_filtration(pts)
dgms = diff.persistence_diagram(fil)
```

### Alpha

Differentiable alpha (squared alpha-shape values on Delaunay simplices)
needs `diode` for the combinatorics; the values are reconstructed in
PyTorch:

```{code-block} python
fil = diff.alpha_filtration(pts)
dgms = diff.persistence_diagram(fil)
```

For low-D points, alpha and Cech-Delaunay produce nearly the same H1
above the noise floor; weak alpha is the fastest but does not always
match Cech below the longest-edge threshold. See
`examples/python/bench_alpha_vs_cd.py` for a runtime comparison.

### Vietoris-Rips

Differentiable VR works either from raw points (Euclidean distances
recomputed in PyTorch) or from a precomputed pairwise-distance matrix:

```{code-block} python
fil = diff.vr_filtration(pts, max_dim=2, n_threads=4)
# or, with a pairwise distance matrix that itself requires grad:
fil = diff.vr_filtration(D, from_pwdists=True, max_dim=2)

dgms = diff.persistence_diagram(fil)
```

## Function data on a grid

The Freudenthal and cubical builders both have a `oineus.diff` mirror.
The input is a torch tensor (image / volume), `requires_grad=True`, and
the filtration values are the data values (or their negation).

```{code-block} python
img = torch.tensor(img_np, dtype=torch.float64, requires_grad=True)

# Triangulated grid (Freudenthal):
fil = diff.freudenthal_filtration(img, negate=False)

# Genuine cubical:
fil = diff.cube_filtration(img, negate=False, values_on="vertices")

dgms = diff.persistence_diagram(fil)
loss = dgms[1][:, 1].sum()              # penalize total H1 death-time
loss.backward()
```

This is the right tool for "make my scalar field have prescribed
topology": pick a loss that pushes the diagram where you want it, and
let autograd send the gradient back to the pixel values.

## Topology-aware optimization

The most common reason to reach for differentiable diagrams is to *move*
topology. Two ingredients suffice: a loss that scores how far the current
diagram is from the desired one, and the autograd graph built by
`diff.persistence_diagram`.

### Match a target diagram with sliced Wasserstein

The closed-form Wasserstein distance ({py:func}`oineus.wasserstein_distance`)
is not differentiable on the diagram coordinates. The standard smooth
replacement is sliced Wasserstein:

```{code-block} python
target = torch.tensor(target_dgm, dtype=torch.float64)
loss = diff.sliced_wasserstein_distance(dgms[1], target, n_directions=64)
loss.backward()
```

Sliced Wasserstein averages one-dimensional Wasserstein distances over
`n_directions` random projections. It is fast, smooth, and propagates
gradients to the diagram points (which the closed-form distance does
not). For a worked end-to-end example see
{doc}`../tutorials/03_differentiable_wasserstein_gradients`.

### Match a target diagram with differentiable Wasserstein

When sliced Wasserstein is too coarse -- a small number of high-
persistence features whose individual matchings really matter -- use
{py:func}`oineus.diff.wasserstein_cost` for the true $W_q$ matching with
gradient flow through every matched pair:

```{code-block} python
cost = diff.wasserstein_cost(
    dgms[1], target,
    wasserstein_q=2.0,
    wasserstein_delta=0.05,
    internal_p=float("inf"),
    ignore_inf_points=True,
)
loss = cost ** (1.0 / 2.0)        # the actual W_2 distance
loss.backward()
```

It returns the *cost* $\sum_{\text{pair}} d(p, q)^q$, so the
$W_q$ distance itself is `cost ** (1 / q)`. Internally it calls Hera once
(non-differentiable) to discover the optimal matching, then reconstructs
the cost in torch so gradients flow back to every finite-to-finite,
finite-to-diagonal, and essential-to-essential pair. Trade-offs vs.
sliced Wasserstein:

- More expensive per call (one Hera matching per forward).
- Discontinuous when the optimal matching changes -- the gradient is
  well-defined within a matching region but jumps when the pairing flips.
- Exact on the chosen $q$ and $p$-norm; no Monte Carlo variance.

For training loops with many small steps, sliced Wasserstein is usually
faster overall; for single-shot "shape this diagram" optimizations,
`wasserstein_cost` is the right tool.

### Denoise / sculpt with the critical-set method

When the goal is "make these specific pairs cancel" or "stretch this
specific pair", the autograd-friendly version of the critical-set method
is `gradient_method="crit-sets"`:

```{code-block} python
fil = diff.vr_filtration(pts, max_dim=2)

top_opt = diff.TopologyOptimizer(fil)
dgm = top_opt.compute_diagram(include_inf_points=False)

dim = 1
eps = top_opt.get_nth_persistence(dim, n=2)
indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
critical_sets = top_opt.singletons(indices, values)
crit_indices, crit_values = top_opt.combine_loss(
    critical_sets, oin.ConflictStrategy.Max
)

crit_indices = np.asarray(crit_indices, dtype=np.int64)
crit_values = torch.as_tensor(crit_values)

top_loss = ((fil.values[crit_indices] - crit_values) ** 2).mean()
top_loss.backward()           # gradient flows back to pts
```

The {py:class}`oineus.diff.TopologyOptimizer` keeps a cached boundary
matrix between calls, so iterating this loop is much cheaper than
rebuilding the filtration each step. See
`examples/python/example_diff_vr_pts.py` and
`weak_alpha_expand_loop.py` for full worked examples.

## gradient_method: dgm-loss vs. crit-sets

```{code-block} python
dgms = diff.persistence_diagram(fil, gradient_method="dgm-loss")    # default
dgms = diff.persistence_diagram(fil, gradient_method="crit-sets")
```

- `"dgm-loss"` (default): gradient flows only through the pair of
  simplices `(birth, death)` that defines each diagram point. The
  forward pass reduces a single decomposition; the backward is a cheap
  scatter. Use this for diagram-shape losses (sliced Wasserstein, total
  persistence, distance-to-target).
- `"crit-sets"`: gradient propagates through the **full critical set**
  of each diagram point (every simplex that contributes to the
  homological event). Slower and needs more matrix data on the forward,
  but useful when you want gradients that move the entire causal chain
  of cells rather than just the two extremes. Conflict resolution
  between overlapping critical sets is controlled by
  `conflict_strategy` (`"avg"` (default), `"max"`, `"sum"`, `"fca"`).

For most loss surfaces, `dgm-loss` is the right default.

## Pitfalls

- **Discontinuities of the persistence map.** The persistence map is
  piecewise-smooth but discontinuous when the *pairing* itself changes
  (which simplex births/kills which class). A single `loss.backward()`
  is well-defined wherever the pairing is stable, but successive
  gradient steps can flip the pairing, and the gradient changes abruptly
  across that flip. This is intrinsic, not a bug.
- **Inf points.** `include_inf_points=False` is currently required.
  Choose `max_diameter` (VR) or `max_dim` (Cech-Delaunay) so the
  topological features you care about die finitely; otherwise they will
  not appear in the diagram tensor.
- **Filtration data type.** Pass `torch.float64` tensors. The C++ side
  is double-precision; `float32` inputs are silently up-cast and the
  resulting gradient is float64 too, which can be surprising.
- **Requires-grad on `fil.values`.** Gradients propagate through
  `fil.values`, which is constructed from your input tensor. Anything
  that breaks the autograd graph between `input_tensor` and
  `fil.values` (`.detach()`, `.numpy()`, in-place op on the input)
  silently stops the gradient.

## See also

- {doc}`../tutorials/03_differentiable_wasserstein_gradients` -- worked
  end-to-end example with visualization.
- {doc}`visualization` -- how to overlay the diagram gradient on the
  diagram, useful for diagnosing the optimization.
- {doc}`diagrams_distances` -- the non-differentiable distances and
  matchings.
- {doc}`optimization` -- the low-level, non-PyTorch critical-set
  interface (advanced).
- `examples/python/example_diff_alpha_grad.py`, `example_diff_vr_pts.py`,
  `weak_alpha_expand_loop.py`, `bench_alpha_vs_cd.py` -- runnable demos.
