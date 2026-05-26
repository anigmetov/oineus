# A tour of Oineus

This page sketches the main capabilities at a glance. Each box points to a
topic essay or tutorial with the full story.

## The pipeline

```
    data                 filtration              decomposition                diagram
┌──────────┐     build   ┌────────────┐  reduce  ┌───────────────┐  extract  ┌──────────┐
│ array    │   ───────►  │ ordered    │  ─────►  │ R, V, U       │  ─────►   │ (b, d)   │
│ points   │             │ cells      │          │ matrices      │           │ points   │
│ sim. cx. │             │ w/ values  │          │               │           │ by dim   │
└──────────┘             └────────────┘          └───────────────┘           └──────────┘
                                                                                 │
                                                                                 ▼
                                                    ┌──────────────────────────────────────┐
                                                    │ distances · matching · barycenters · │
                                                    │ optimization · backprop              │
                                                    └──────────────────────────────────────┘
```

Everything in Oineus is built on this pipeline. Each stage is exposed as a
first-class object so you can swap pieces in and out.

## Filtrations

| Function | Input | Output |
| --- | --- | --- |
| {py:func}`oineus.freudenthal_filtration` | 1D/2D/3D array | simplicial lower-/upper-star filtration of a triangulated grid |
| {py:func}`oineus.cube_filtration` | 1D/2D/3D array | cubical filtration (values on vertices or top cells) |
| {py:func}`oineus.vr_filtration` | point cloud or pairwise-distance matrix | Vietoris–Rips filtration |
| {py:func}`oineus.compute_diagrams_alpha` | 2D/3D points | alpha-shape filtration via diode |
| {py:func}`oineus.list_to_filtration` | list of `(id, vertices, value)` | user-specified simplicial filtration |

See [filtration types](../topics/filtrations.md) for when to pick which.

## Persistence

```{code-block} python
fil = oineus.vr_filtration(points, max_dim=2, max_diameter=2.0)
dcmp = oineus.Decomposition(fil, dualize=True)
dcmp.reduce(oineus.ReductionParams(n_threads=8, clearing_opt=True))
dgms = dcmp.diagram(fil)
```

Why use `Decomposition` instead of the one-liner {py:func}`oineus.compute_diagrams_vr`?
Because the decomposition object also exposes `v_data`, `u_data_t`, and
`r_data` — useful if you need cycles, representatives, or gradients. See
[inside the decomposition](../topics/decomposition.md).

## Distances and matchings

- {py:func}`oineus.bottleneck_distance` — exact bottleneck via Hera.
- {py:func}`oineus.wasserstein_distance` — q-Wasserstein (any `q >= 1`) with
  tunable internal norm.
- {py:func}`oineus.wasserstein_matching` — the full matching
  (finite↔finite, finite↔diagonal, essential↔essential) as a `DiagramMatching`.
- `oineus.diff.sliced_wasserstein_distance` — differentiable sliced
  Wasserstein.

See [diagram distances](../topics/diagrams_distances.md).

## Map-induced persistence

Oineus has two complementary routes when persistence of a pair $(K, L)$ or
of a simplicial map matters:

1. {py:func}`oineus.compute_kernel_image_cokernel_reduction` — gives you
   {py:class}`oineus.KerImCokReduced` with `kernel_diagrams()`,
   `image_diagrams()`, `cokernel_diagrams()`.
2. {py:func}`oineus.mapping_cylinder` / {py:func}`oineus.compute_ker_cok_reduction_cyl`
   — build the cylinder explicitly, then reduce.

See [kernel/image/cokernel](../topics/kicr.md) and [mapping cylinders](../topics/mapping_cylinder.md).

## Topology optimization

```{code-block} python
opt = oineus.TopologyOptimizer(fil)
dgms = opt.compute_diagram(include_inf_points=False)
singletons = opt.simplify(eps=0.2, strategy=oineus.DenoiseStrategy.BirthBirth, dim=1)
idx_vals = opt.combine_loss(singletons, oineus.ConflictStrategy.Max)
# → use idx_vals to update your filtration values / produce gradients
```

The critical-set method turns "make this H1 bar shorter" into a concrete
list of which cells to move and in which direction. See
[topology optimization](../topics/optimization.md).

## Differentiable pipeline

```{code-block} python
import torch
import oineus.diff as diff

pts = torch.randn(50, 2, requires_grad=True)
fil = diff.vr_filtration(pts, max_dim=1, max_diameter=2.0)
dgms = diff.persistence_diagram(fil, include_inf_points=False)
loss = dgms[1][:, 1].sum()       # penalize total 1D death-time
loss.backward()
print(pts.grad.shape)
```

See [differentiable filtrations](../topics/differentiable.md).

## Barycenters

```{code-block} python
mean = oineus.frechet_mean(list_of_diagrams, max_iter=200)
```

With tricky or noisy collections, use
{py:func}`oineus.frechet_mean_multistart` or
{py:func}`oineus.progressive_frechet_mean`. See [Fréchet mean](../topics/frechet_mean.md).
