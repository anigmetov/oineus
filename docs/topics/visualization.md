# Plotting

The visualization helpers live in {py:mod}`oineus.vis` and are re-exported
at the top level. They depend only on matplotlib; an optional density
mode picks up `mpl_scatter_density` if installed.

The four entry points are:

- {py:func}`oineus.plot_diagram` -- one or more persistence diagrams.
- {py:func}`oineus.plot_matching` -- a matching between two diagrams.
- {py:func}`oineus.plot_diagram_gradient` -- diagram with gradient
  arrows on each point (for differentiable workflows).
- {py:func}`oineus.plot_chain` -- a homology generator's chain lifted
  back to the data domain.

## A single diagram

```{code-block} python
import matplotlib.pyplot as plt
import oineus as oin

fig, ax = plt.subplots(figsize=(5, 5))
oin.plot_diagram(dgms.in_dimension(1), ax=ax, title="H1")
plt.show()
```

Pass `dgms` directly (a {py:class}`oineus.Diagrams` object), and
`plot_diagram` will draw all dimensions with one color per dim:

```{code-block} python
oin.plot_diagram(dgms, ax=ax)   # H0 in C0, H1 in C1, ...
```

Or a dict for explicit dim-to-array control:

```{code-block} python
oin.plot_diagram({0: dgm_h0, 1: dgm_h1, 2: dgm_h2}, ax=ax)
```

For very large diagrams (the near-diagonal noise band saturates the
scatter), opt into hybrid density mode:

```{code-block} python
oin.plot_diagram(dgm, scatter_only=False, near_diagonal_fraction=0.05)
```

This renders near-diagonal points as a 2D density (high-persistence
features stay as crisp scatter, so they are never aggregated away).

## Multiple diagrams overlaid

Single-color overlays are the right tool for "did the diagram move?":

```{code-block} python
fig, ax = plt.subplots()
oin.plot_diagram(dgm_a, ax=ax, color="C0", title="A (blue) vs B (orange)")
oin.plot_diagram(dgm_b, ax=ax, color="C1")
```

## A matching

```{code-block} python
m = oin.wasserstein_matching(dgm_a, dgm_b, q=2.0)

fig, ax = plt.subplots(figsize=(6, 6))
oin.plot_matching(dgm_a, dgm_b, m, ax=ax,
                  color_dgm_a="C0", color_dgm_b="C1")
```

`plot_matching` dispatches on the matching type: a Wasserstein matching
draws every edge category (finite/finite, A/diagonal, B/diagonal,
essential/essential); a bottleneck matching draws only finite/finite plus
the longest edge highlighted. Two filtering knobs control clutter on
large diagrams:

- `min_persistence` -- drop edges whose endpoints both have persistence
  below this threshold.
- `top_k_pairs` -- keep only the K edges with the largest endpoint
  persistence (auto-defaults to 200 once there are more than 1000 edges).

## A gradient overlay

When you have differentiable diagrams and want to see "how does the loss
move each point", use {py:func}`oineus.plot_diagram_gradient`:

```{code-block} python
oin.plot_diagram_gradient({1: dgm1_np},
                          gradient={1: dgm1_grad_np},
                          ax=ax, descent=True, title="H1 gradient")
```

`descent=True` flips the arrows so they point in the descent direction
(useful when you are visualizing where each point would move to reduce
the loss). See `examples/python/example_diff_alpha_grad.py` for a full
worked example.

## Chain visualization

For lower-star and alpha filtrations there is also
{py:func}`oineus.plot_chain`, which lifts a homology generator's chain
back into the data domain:

```{code-block} python
# After reducing with compute_v=True, the column of V at the death index
# gives a chain that bounds the H1 generator born by the birth simplex.
oin.plot_chain(fil, chain_indices, points=pts, ax=ax)
```

See `examples/python/test_plot_chain_field.py` (cubical / 2D scalar
field) and `test_plot_chain_alpha.py` (alpha / point cloud).

## See also

- {doc}`diagrams_distances` -- where the diagrams and matchings come
  from.
- {doc}`frechet_mean` -- plotting barycenters alongside their inputs.
- {doc}`differentiable` -- the gradient overlay is the diagnostic for
  PyTorch-driven optimization.
