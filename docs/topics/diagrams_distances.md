# Diagram distances and matchings

Oineus exposes Hera-backed bottleneck and Wasserstein distances and full
matching objects. Inputs are single-dimension persistence diagrams as
$(n, 2)$ NumPy arrays (or `list[DiagramPoint]`). To extract a single
dimension from a {py:class}`oineus.Diagrams` collection, use
`dgms.in_dimension(d)`.

## Bottleneck distance

```{code-block} python
import numpy as np
import oineus as oin

dgm_a = oin.compute_diagrams_vr(pts_a, max_dim=1).in_dimension(1)
dgm_b = oin.compute_diagrams_vr(pts_b, max_dim=1).in_dimension(1)

d = oin.bottleneck_distance(dgm_a, dgm_b, delta=0.01)
```

- `delta` is the relative-error tolerance. The default `0.01` (1 %) is
  usually a good trade-off between speed and accuracy. Pass `delta=0.0`
  to request the exact bottleneck distance (slower).

## Wasserstein distance

```{code-block} python
d = oin.wasserstein_distance(dgm_a, dgm_b,
                             q=2.0, internal_p=np.inf, delta=0.01)
```

- `q` -- the Wasserstein exponent. `q = 1` is canonical Wasserstein-1;
  `q = 2` is the $L^2$ transport cost; large `q` approaches the
  bottleneck distance.
- `internal_p` -- the ground-metric norm in the diagram plane. `np.inf`
  (default) is the standard choice and matches the stability theorems for
  sublevel-set persistence. `internal_p = 2.0` gives Euclidean ground
  cost.
- `delta` -- relative-error tolerance. **Must be strictly positive.**
  Unlike bottleneck, Wasserstein has no exact mode -- the underlying
  algorithm always returns an approximation, controlled by `delta`.
  Tighten to `1e-4` for "publication-quality" answers.

If both diagrams contain points at infinity (essential pairs), they are
matched within their family (positive-inf to positive-inf, etc.).
Mismatched essential counts produce $+\infty$ distance.

## Matching objects

When you need the actual pairing -- which point in $A$ corresponds to
which point in $B$, or which $A$-point is matched to the diagonal -- use
{py:func}`oineus.wasserstein_matching` or
{py:func}`oineus.bottleneck_matching`:

```{code-block} python
m = oin.wasserstein_matching(dgm_a, dgm_b, q=2.0)

m.finite_to_finite   # (k, 2) int array: indices into dgm_a / dgm_b
m.a_to_diagonal      # (k,)   int array: dgm_a points matched to diagonal
m.b_to_diagonal      # (k,)   int array: dgm_b points matched to diagonal
m.essential          # grouped view: per-family essential matches
m.cost               # raw transport cost (sum of |edge|^q)
m.distance           # cost ** (1/q)
```

For bottleneck:

```{code-block} python
mb = oin.bottleneck_matching(dgm_a, dgm_b, delta=0.0)

mb.finite_to_finite      # as above
mb.distance              # the overall bottleneck distance
mb.longest.finite        # edges realizing the max length within the
                         # finite part
mb.longest.essential     # edges realizing the per-family essential max,
                         # split by family
```

`longest.finite` is the (one or more) edges that realize the maximum
length **among finite-to-finite and finite-to-diagonal edges**;
`longest.essential[k]` is the same for essential family $k$. The overall
bottleneck distance equals the larger of the two, so `longest.finite`
realizes the bottleneck only when the finite max exceeds every essential
family max.

`bottleneck_matching` accepts `ignore_inf_points=True` (default) to strip
essential pairs before matching; pass `False` to keep them in the
picture.

## Sliced Wasserstein

For training loops where you want a smooth, fast, differentiable
approximation of $W_2$, use the sliced variant from
{py:mod}`oineus.diff`:

```{code-block} python
import torch
import oineus.diff as diff

dgm_a_t = torch.tensor(dgm_a)
dgm_b_t = torch.tensor(dgm_b)

d = diff.sliced_wasserstein_distance(dgm_a_t, dgm_b_t, n_directions=50)
d.backward()
```

Sliced Wasserstein is the average of one-dimensional Wasserstein
distances over `n_directions` random projections. It is differentiable on
the diagram side, which the closed-form distances above are not. See
{doc}`differentiable`.

## See also

- {doc}`visualization` -- plotting diagrams and matchings.
- {doc}`decomposition` -- where the diagrams come from.
- {doc}`frechet_mean` -- the Wasserstein barycenter problem; uses these
  same distances internally.
- {doc}`differentiable` -- diagrams as differentiable tensors.
- `tests/test_dgm_dist.py`, `tests/test_wass_match.py`,
  `tests/test_bot_match.py` -- canonical test cases that double as
  runnable examples.
