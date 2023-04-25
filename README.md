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
Oineus can compute the kernel, image and cokernel persistence diagrams as in ["Persistent Homology for Kernels, Images, and Cokernels"](https://doi.org/10.1137/1.9781611973068.110) by D. Cohen-Steiner, H. Edelsbrunner, D. Morozov. We first perform the required reductions using `compute_kernel_image_cokernel_diagrams`, which has arguments:
* `K` the simplicial complex with function values, as a list with an element per simplex in the format `[simplex_id, vertices, value]`, where `vertices` is a list containing the ids of the vertices, and value is the value under the function f.
* `L` the simplicial sub-complex with function values, as a list with an element per simplex in the format `[simplex_id, vertices, value]`, where `vertices` is a list containing the ids of the vertices, and value is the value under the function g.
* `L_to_K` a list which maps the cells in L to their corresponding cells in K,
* `params` the parameters you want to use for the reduction: `n_threads`, `kernel`, `image`, `cokernel` are the ones that can be modified. `n_threads` is the number of threads you want to use, and `kernel`, `image`, `cokernel` are boolean values, which default to `False`. Set them to `True` if you want to extract the respective diagrams immediately.
* `return` an object which contains the kernel, image and cokernel diagrams, as well as the reduced matrices.

To obtain the different diagrams, we use `kernel_diagrams`, `image_diagrams`, `cokernel_diagrams`, and then we can use `in_dimension` to get the sepcific diagram in a specific dimension. If a specific set of diagrams as not already been extracted, and you call them, they will be generated rather than an empty return value.

**Note:** aside from the number of threads, all other parameters are set already. 

#### Example
Suppose we have a simplicial complex $K$ with a function $f$ on it, and a subcomplex $L \subset K$ with a function $g$ on it. In this example, $g = f|_L$. We then perform the 5 necessary reductions and compute the persistence diagrams using `compute_kernel_image_cokernel_diagrams`, and then access the 3 sets of diagrams using `kernel()`, `image()`, `cokernel()` respectively. After which we can obtain a diagram in a specific dimension $i$ using `in_dimension(i)`.

```python
>>> import oineus as oin
>>> params = oin.ReaductionParams()
>>> params.n_threads = 4
>>> params.kernel = True
>>> params.image = True
>>> params.cokernel = True
>>> K = [[0, [0], 10], [1,[1],50], [2,[2], 10], [3, [3], 10], [4,[0,1], 50], [5, [1,2], 50], [6,[0,3], 10], [7, [2,3], 10]]
>>> L = [[0, [0], 10], [1,[1],50], [2,[2], 10], [3, [0,1], 50], [4,[1,2],50]]
>>> L_to_K = [0,1,2,4,5]
>>> ker_im_cok_dgms = oin.compute_kernel_image_cokernel_reduction(K, L, L_to_K, params)
>>> ker_dgms = ker_im_cok_dgms.kernel_diagrams()
>>> im_dgms = ker_im_cok_dgms.image_diagrams()
>>> cok_dgms = ker_im_cok_dgms.cokernel_diagrams()
>>> print(ker_dgms.in_dimension(0))
>>> print(im_dgms.in_dimension(0))
>>> print(cok_dgms.in_dimension(0))
```
 
## License

Oineus is a free program distributed under modified
BSD license. See legal.txt for details.
