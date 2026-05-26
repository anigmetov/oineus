# User guide

A task-oriented tour of Oineus, in the order most users encounter the
problems. Each section is a self-contained snippet with a pointer to the
fuller topic essay. The tour assumes you already know what a persistence
diagram is and what Wasserstein distance does; for a from-scratch
introduction, start with {doc}`tutorials/01_tda_for_beginners`.

## 1. Persistence diagrams of point clouds

For 2D or 3D points, the alpha filtration is the fastest path -- it
produces an order-of-magnitude fewer cells than Vietoris-Rips at the
same resolution:

```{code-block} python
import numpy as np
import oineus as oin

pts = np.load("points.npy")               # (n, 2) or (n, 3)
dgms = oin.compute_diagrams_alpha(pts)
print(dgms.in_dimension(1))
```

For points in any dimension, or for non-Euclidean metrics, use
Vietoris-Rips. Always cap `max_dim` and `max_diameter`:

```{code-block} python
dgms = oin.compute_diagrams_vr(pts, max_dim=2, max_diameter=2.0)
```

More: {doc}`topics/filtrations`.

## 2. Persistence diagrams of function data on a grid

For a 2D image or 3D volume, the one-shot helper builds a Freudenthal
filtration (triangulated grid, sublevel sets) and returns the diagrams:

```{code-block} python
img = np.load("volume.npy")               # 2D or 3D float array
dgms = oin.compute_diagrams_ls(img, negate=False, max_dim=2)
print(dgms.in_dimension(0).shape, dgms.in_dimension(1).shape)
```

Use `negate=True` for superlevel sets. For genuine cubical (rather than
triangulated) complexes, use {py:func}`oineus.cube_filtration` instead;
this matches the GUDHI cubical convention.

More: {doc}`topics/filtrations`.

## 3. Distances between diagrams

Hera-backed bottleneck and q-Wasserstein, single-dimension at a time:

```{code-block} python
dgm_a = dgms_a.in_dimension(1)
dgm_b = dgms_b.in_dimension(1)

d_bot = oin.bottleneck_distance(dgm_a, dgm_b, delta=0.01)
d_w   = oin.wasserstein_distance(dgm_a, dgm_b, q=2.0,
                                 internal_p=np.inf, delta=0.01)
```

`delta` is the relative-error tolerance; the default 1 % is fine for
exploration. Pass `delta=0.0` for the exact bottleneck.

More: {doc}`topics/diagrams_distances`.

## 4. The manual workflow: filtration -> decomposition -> reduce -> diagram

The one-shot helpers collapse four stages into one call. When you need
parallel reduction, $V$/$U$ matrices, cycle representatives, or anything
non-default, build the stages by hand:

```{code-block} python
fil = oin.vr_filtration(pts, max_dim=2, max_diameter=2.0, n_threads=4)

dcmp = oin.Decomposition(fil, dualize=True)

params = oin.ReductionParams(n_threads=8, compute_v=True, clearing_opt=True)
dcmp.reduce(params)

dgms = dcmp.diagram(fil, include_inf_points=True)
```

The `Decomposition` keeps `r_data`, `v_data`, `u_data_t` around for
sparse-matrix inspection or downstream optimization.

More: {doc}`topics/decomposition`.

## 5. Wasserstein barycenter / Fréchet mean

The single-run optimizer is sensitive to its seed. For most inputs the
progressive multistart is the right default -- it walks down a
persistence-threshold schedule so the high-persistence features lock in
before the near-diagonal mass takes over:

```{code-block} python
diagrams = [dgms_i.in_dimension(1) for dgms_i in batch]

bary, details = oin.progressive_frechet_mean_multistart(
    diagrams,
    starts=("medoid", "second_medoid"),
    wasserstein_delta=1e-4,
    max_iter=200,
    return_details=True,
)
print(details["thresholds"])
```

More: {doc}`topics/frechet_mean`.

## 6. Plotting diagrams

The visualization helpers live in {py:mod}`oineus.vis` and are
re-exported at the top level. The defaults (pure scatter, $H_k$ legend,
$+\infty$ line) are good for most diagrams:

```{code-block} python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
oin.plot_diagram(dgms, ax=ax, title="VR persistence")
plt.show()
```

For overlaying two diagrams, pass `color=` per call. For overlaying a
matching, use {py:func}`oineus.plot_matching`. For very large diagrams,
opt into hybrid density rendering with `scatter_only=False`.

More: {doc}`topics/diagrams_distances`.

## 7. Differentiable persistence diagrams

The {py:mod}`oineus.diff` subpackage turns the pipeline into a PyTorch
layer. **This is the canonical way to do topology-aware optimization in
Oineus**: pick a filtration builder from `oineus.diff`, write the loss
on the diagram tensor, call `loss.backward()`. The same four filtration
choices are available (`alpha_filtration`, `weak_alpha_filtration`,
`cech_delaunay_filtration`, `vr_filtration`) plus their function-data
analogues (`freudenthal_filtration`, `cube_filtration`):

```{code-block} python
import torch
import oineus.diff as diff

pts = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)
fil = diff.cech_delaunay_filtration(pts)
dgms = diff.persistence_diagram(fil, include_inf_points=False)

loss = dgms[1][:, 1].sum()    # penalize total H1 death-time
loss.backward()
print(pts.grad.shape)
```

Sliced Wasserstein is the canonical smooth, differentiable distance for
"match a target diagram" losses (`diff.sliced_wasserstein_distance`).

More: {doc}`topics/differentiable`.

## 8. Kernel / image / cokernel diagrams

For a simplicial map encoded as an inclusion $L \hookrightarrow K$,
Oineus computes all three induced persistence modules in one pass:

```{code-block} python
K = [(0, [0], 10.0), (1, [1], 10.0), (2, [2], 10.0), (3, [3], 10.0),
     (4, [0, 1], 10.0), (5, [1, 2], 10.0),
     (6, [0, 3], 10.0), (7, [2, 3], 10.0)]
L = [(0, [0], 10.0), (1, [1], 10.0), (2, [2], 10.0),
     (3, [0, 1], 10.0), (4, [1, 2], 10.0)]

kicr = oin.compute_kernel_image_cokernel_reduction(K, L)
print(kicr.kernel_diagrams().in_dimension(0))
print(kicr.image_diagrams().in_dimension(0))
print(kicr.cokernel_diagrams().in_dimension(0))
```

For maps that are not literal inclusions, build a mapping cylinder
first; see {doc}`topics/mapping_cylinder`.

More: {doc}`topics/kicr`.

## 9. Zero-persistence diagrams

Pairs `(birth, death)` with `birth == death` are filtered out of the
default diagram. They are common on grids with plateaus and on duplicate-
distance VR filtrations. To inspect them, use
`dcmp.zero_pers_diagram(fil)`:

```{code-block} python
dcmp = oin.Decomposition(fil, dualize=True)
dcmp.reduce(oin.ReductionParams(n_threads=4))

regular = dcmp.diagram(fil)            # zero-pers pairs filtered out
zero    = dcmp.zero_pers_diagram(fil)  # zero-pers pairs only
print(zero.in_dimension(0))
```

The {py:class}`oineus.KICRParams.include_zero_persistence` flag is the
analogue for kernel/image/cokernel diagrams.

More: {doc}`topics/decomposition`.

## What's next

The {doc}`topics/index` directory has one short essay per capability the
guide touches on, with deeper API notes and pitfalls. The
{doc}`api/index` is the full auto-generated reference; for the prose
explanations of *why* the API looks the way it does, the topic essays
are the better starting point.
