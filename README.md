## About
Oineus is an implementation of shared-memory parallel
computation of persistent homology published in
*D. Morozov and A. Nigmetov.
"Towards lockfree persistent homology."
Proceedings of the 32nd ACM Symposium on Parallelism
in Algorithms and Architectures. 2020.*

Currently it supports computation of lower-star persistence
of scalar functions on regular grids or of user-defined
filtrations (where each simplex needs to be created manually).
It is written in C++ with python bindings (pybind11).

## Compilation

Oineus requires C++17 standard and python3.
`Pybind11` is included as a submodule.
Compilation is standard:

```shell
$ git clone --recurse-submodules git@github.com:anigmetov/oineus.git
$ cd oineus
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```

## Usage

Compiled Oineys python package is located in `[build_directory]/python/bindings`.
Functions are given as NumPy arrays of either np.float32 or np.float64 dtype.

`compute_diagrams_ls` function arguments:
* `data`: 1D, 2D or 3D array with function values on a grid.
* `negate`: if True, compute upper-star persistence, default: False.
* `wrap`: if True, domain is treated as torus (periodic boundary conditions),
    default: False.
* `params`: settings used to run reduction, `params.n_threads` specifies how
many threads to use.
* `max_dim`: maximum dimension to compute diagrams (filtration will
be one dimension higher: to get persistence diagrams in dimension 1,
we need 2-simplices).
* `n_threads`: number of threads to use, default: 1.
* `return`: Diagrams in all dimensions. Diagrams in each dimension will be returned by `in_dimension` function
as 2D numpy arrays `[(b_1, d_1),
         (b_2, d_2), ... ]`.

```python
>>> import numpy as np
>>> import oineus as oin
>>> f = np.random.randn(48, 48, 48)
>>> params = oin.ReductionParams()
>>> params.n_threads = 16
>>> dgms = oin.compute_diagrams_ls(data=f, negate=False, wrap=False, params=params, include_inf_points=True, max_dim=2)
>>> dgm = dgms.in_dimension(0)
```

### Kernel, image and cokernel persistence
Oineus can compute the kernel, image and cokernel persistence diagrams as in ["Persistent Homology for Kernels, Images, and Cokernels"](https://doi.org/10.1137/1.9781611973068.110) by D. Cohen-Steiner, H. Edelsbrunner, D. Morozov. These are calculated using 

## License

Oineus is a free program distributed under modified
BSD license. See legal.txt for details.
