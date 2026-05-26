(quickstart)=
# Quickstart

A persistence diagram from a 2D scalar field in ten lines:

```{code-block} python
import numpy as np
import oineus

rng = np.random.default_rng(0)
x = np.linspace(-2, 2, 64)
X, Y = np.meshgrid(x, x)
img = np.exp(-(X**2 + Y**2)) + 0.4 * rng.standard_normal(X.shape)

dgms = oineus.compute_diagrams_ls(img, max_dim=1)
print(dgms.in_dimension(0).shape)   # H0: (n0, 2)
print(dgms.in_dimension(1).shape)   # H1: (n1, 2)
```

`compute_diagrams_ls` builds a Freudenthal filtration from `img` (each pixel
is a vertex in a triangulated grid), sweeps the filtration in order of
increasing value, and reports each (birth, death) pair grouped by homology
dimension.

## A persistence diagram from points

For a point cloud, use Vietoris–Rips:

```{code-block} python
theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
points += 0.05 * rng.standard_normal(points.shape)

dgms = oineus.compute_diagrams_vr(points, max_dim=1, max_diameter=2.5)
print(dgms.in_dimension(1))         # expect one prominent H1 bar
```

The longest H1 bar is the circle's one-dimensional hole — everything else is
short-persistence noise.

For points in 2D or 3D, the alpha filtration is much smaller than VR and
gives the same diagrams up to dimension $d-1$ (where $d$ is the ambient
dimension):

```{code-block} python
dgms = oineus.compute_diagrams_alpha(points)
print(dgms.in_dimension(1))         # same prominent H1 bar, fewer cells
```

`compute_diagrams_alpha` requires the optional [`diode`](https://github.com/mrzv/diode)
dependency for the underlying CGAL Delaunay construction.

## Comparing two diagrams

```{code-block} python
dgm_a = dgms.in_dimension(1)
dgm_b = oineus.compute_diagrams_vr(
    points + 0.02 * rng.standard_normal(points.shape),
    max_dim=1, max_diameter=2.5,
).in_dimension(1)

print("bottleneck:", oineus.bottleneck_distance(dgm_a, dgm_b))
print("wasserstein(q=2):", oineus.wasserstein_distance(dgm_a, dgm_b, q=2.0))
```

## What's next

- The [beginners tutorial](../tutorials/01_tda_for_beginners.ipynb) explains
  the pictures behind these numbers.
- The [topics section](../topics/index.md) unpacks how each of these pieces
  works under the hood.
- The [API reference](../api/index.md) lists every public symbol.
