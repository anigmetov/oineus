# Quick introduction to Oineus

## Simplices and Filtrations

```python
import oineus as oin
import numpy as np
import torch
```

First, let us do everything by hand. If we want to create a filtration,
we need to create simplices first. We have a filtration of a triangle,
and we use `double` to store filtration values (second argument to the constructor):
```python
# vertices
v0 = oin.Simplex([0], 0.1)
v1 = oin.Simplex([1], 0.2)
v2 = oin.Simplex([2], 0.3)

# edges
e1 = oin.Simplex([0, 1], 1.2)
e2 = oin.Simplex([0, 2], 1.4)
e3 = oin.Simplex([1, 2], 2.1)

# triangle
t1 = oin.Simplex([0, 1, 2], 4.0)
```

We now put simplices into a list and create a parallel list
of values, so that simplex `simplices[i]` enters filtration at time `values[i]`.
```python
simplices = [v0,  v1,  v2,  e1,  t1,  e2,  e3]
```
We put simplices of positive dimension in arbitrary order here. 
**Vertices must always appear in the list first, and in the order prescribed by their index**.

Now we create a filtration.

```python
# constructor will sort simplices and assign sorted_ids
fil = oin.Filtration(simplices)

print(fil)
```

Note that:
1. Each simplex has `id`, which equals its index in the list `simplices`. This is precisely why we insist
that vertex `[i]` appears in the `i`-th position in `simplices`: we want the `id` of a vertex
to match its index we use when we create positive-dimensional simplices: `oin.Simplex([0, 2])`
consists of vertices `simplices[0]` and `simplices[2]`.
2. Each simplex has `sorted_id`, which equals its index in the filtration order.
When we ask `fil` for simplices, they will appear in the order determined
by the `sorted_id`.

The constructor of a filtration has some additional arguments:
* `set_ids` is `True` by default, that is why `id`s are overwritten. Set it to `False` to preserve original `id`s.
Caveat: vertex `i` must still have `id == i` and the `id`s must be unique. You can specify
the `id` as the first argument to constructor: `oin.Simplex(3, [0, 1], 0.5)` or assign to it: `sigma.id = 3`.
* `sort_only_by_dimension` is `False` by default. If you know that your simplices are already in the correct
order, you can set it to `True`: the simplices will be arranged by dimension, but the order
of simplices of same dimension will be preserved.

## Common filtrations

### Vietoris-Rips. 

You can create a VR filtration from a point cloud or from a distance matrix.


For point cloud, only dimensions 1, 2 and 3 are supported.
An input can be a NumPy array, a Jax array or a PyTorch tensor.
The shape is `(#points, dimension)`, in other words, each point must be a row
in a matrix.

```python
import numpy as np
import oineus as oin

# create 20 random points in space
np.random.seed(1)
n_points = 20
dim = 3
points = np.random.uniform(size=(n_points, dim))

fil = oin.get_vr_filtration(points, max_dim=3, max_radius=2)
print(fil)
```

The parameters are:
* `points`: coordinates of points in the point cloud.
* `max_dim`: the resulting filtration will contain simplices up to and including `max_dim`.
If you want to compute persistence diagrams in dimension `d`, you need `max_dim >= d+1`.
* `max_radius`: only consider balls up to this radius.

For distance matrix:

```python
import numpy as np
import scipy.spatial

import oineus as oin


# create 20 random points in space
np.random.seed(1)
n_points = 20
dim = 6
points = np.random.uniform(size=(n_points, dim))

distances = scipy.spatial.distance.cdist(points, points, 'euclidean')

fil = oin.get_vr_filtration_from_pwdists(distances, max_dim=3, max_radius=2)
print(fil)
```

### Lower-star filtration.

Lower-star filtrations are supported for functions on a regular D-dimensional grid
for D = 1 , 2, or 3. Function values are represented as an D-dimensional NumPy array.

```python
# create scalar function on 8x8x8 grid
f = np.random.uniform(size=(8, 8, 8))

fil = oin.get_freudenthal_filtration(data=f, max_dim=3)
```

If you want upper-star filtration, set `negate` to `True`:
```python
fil = oin.get_freudenthal_filtration(data=f, negate=True, max_dim=3)
```
If you want periodic boundary conditions (D-dimensional torus instead
of D-dimensional cube), set `wrap` to `True`:
```python
fil = oin.get_freudenthal_filtration(data=f, wrap=True, max_dim=3)
```

## Persistence Diagrams

Persistence diagram is computed from `R=DV, RU=D` decomposition.
In fact, we only need the `R` matrix to read off the persistence pairing,
but other matrices are needed in topological optimization.
The corresponding class
is called `VRUDecomposition`. When we create it, we must specify whether we want homology
(`dualize=False`) or cohomology (`dualize=True`).

```python
# no cohomology
dualize = False
# create VRU decomposition object, does not perform reduction yet
dcmp = oin.Decomposition(fil, dualize)
```

In order to perform reduction, we need to set parameters.
This is done through a single object of class `ReductionParams` that encapsulates all
of these parameters. 

```python
rp = oin.ReductionParams()
```
Some of the parameters are:
* `rp.clearing_opt` whether you want to use clearing optimization.
* `rp.n_threads`: number of threads to use, default is 1.
* `rp.compute_v`: whether you want to compute the `V` matrix. `True` by default.
* `rp.compute_u`: whether you want to compute the `U` matrix. `False` by default.
This cannot be done in multi-threaded mode, so the reduction will return an error, if `n_threads > 1`
and this option is set.

```python
rp.compute_u = rp.compute_v = True
rp.n_threads = 16
# perform reduction
dcmp.reduce(rp)

# now we can acess V, R and U
# indices are sorted_ids of simplices == indices in fil.cells()
V = dcmp.v_data
print(f"Example of a V column: {V[-1]}, this chain contains cells:")

simplices = fil.simplices()
for sigma_idx in V[-1]:
    print(simplices[sigma_idx])
```

Now we can ask for a diagram. The `diagram` methods
uses the critical values from the filtration that was used to construct it to get
the values of simplices and returns diagrams in all dimensions. By default,
diagrams include points at infinity. If we only want the finite part,
we can specify that by `include_inf_points`.
```python
dgms = dcmp.diagram(fil, include_inf_points=False)
```
To get diagram in one specific dimension, we can subscript
the object or call the `in_dimension` method.
Diagram will be returned as a NumPy array of shape `(n, 2)`

```python
dim=2
dgm_2 = dcmp.diagram(fil).in_dimension(dim)
# or
dgm_2 = dcmp.diagram(fil)[dim]

assert type(dgm_2) is np.ndarray
```
Now, e.g. the birth coordinates are simply `dgm_2[:, 0]`.

If we want to know the peristence pairing, that is, which 
birth and death simplex gave us this particular point,
we can use `index_diagram_in_dimension`.
```python
dim=2
ind_dgm_2 = dcmp.diagram(fil).index_diagram_in_dimension(dim)
```
It is also a NymPy array (of integral type).
We can get the zero persistence diagram:
```
z_dgm_2 = dcmp.zero_pers_diagram(fil).in_dimension(dim)
```
Finally, sometimes it can be more convenient to have not a NumPy
array, but as `list` of diagram points that have birth/death values
and birth/death indices as members. For this, the `in_dimension`
method has the second argument, `as_numpy`, which is `True` by default,
but we can set it to `False`:

```python
dgm = dcmp.diagram(fil).in_dimension(dim, False)
for p in dgm:
    print(p.birth, p.death, p.birth_index, p.death_index)
```

How to map this back to filtration? Let us take a look
at a single point in the index diagram:
```python
sigma_sorted_idx, tau_sorted_idx = ind_dgm_2[0, :]
```
`sigma_sorted_idx` is the index of the birth simplex (triangle) in filtration order.
`tau_sorted_idx` is the index of the death simplex (tetrahedron) in filtration order.
There are many ways to get the original simplex:
* `sigma = fil.get_simplex(sigma_sorted_idx)` will return a simplex itself. So, `fil.get_simplex`
takes the `sorted_id` of a simplex and just accesses the vector of simplices at this index,
so it is cheap.
* `sigma_idx = fil.get_id_by_sorted_id(sigma_sorted_idx)` will return the `id` of `sigma`.
Recall that, by default, it is the index of `sigma` in the original list of simplices that was used to create the filtration.
This is convenient, if you have a parallel array of some information, one entry per simplex,
which you want to access.


## Topology Optimization

Topology optimization is performed by the `TopologyOptimizer` class.
It is created from a filtration.

```python
opt = oin.TopologyOptimizer(fil)
```
In order to specify the target (where some points
in the diagram should go), we use indices and values.

Let us consider an example. Here is a helper function to generate data:
```python
def sample_data(num_points: int=100, noise_std_dev=0.1):
    # sample points from the unit circle and add noise
    # num_points: number of points to sample
    # noise_std_dev: standard deviation of Gaussian noise
    # return points as differentiable torch tensor
    np.random.seed(1)

    angles = np.random.uniform(low=0, high=2*np.pi, size=num_points)
    x = np.cos(angles)
    y = np.sin(angles)

    x += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
    y += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)

    pts = np.vstack((x, y)).T
    pts = torch.Tensor(pts)
    pts.requires_grad_(True)

    return pts
```
Suppose that we want to push all but the first n-1 points
in the PD in dimension dim to move to the diagonal.
The corresponding loss function is computed by this function:
```python
def topological_loss(pts: torch.Tensor, dim: int=1, n: int=2):
    pts_as_numpy = pts.clone().detach().numpy().astype(np.float64)
    fil, longest_edges = oin.get_vr_filtration_and_critical_edges(pts_as_numpy, max_dim=2, max_radius=9.0, n_threads=1)
    
    top_opt = oin.TopologyOptimizer(fil)

    eps = top_opt.get_nth_persistence(dim, n)

    indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)

    critical_sets = top_opt.singletons(indices, values)
    crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)

    crit_indices = np.array(crit_indices, dtype=np.int32)

    crit_edges = longest_edges[crit_indices, :]
    crit_edges_x, crit_edges_y = crit_edges[:, 0], crit_edges[:, 1]

    crit_values = torch.Tensor(crit_values)
    if len(crit_edges_x) > 0:
        top_loss = torch.sum(torch.abs(torch.sum((pts[crit_edges_x, :] - pts[crit_edges_y, :])**2, axis=1) - crit_values ** 2))
    else:
        top_loss = torch.zeros(())
        top_loss.requires_grad_(True)
    return top_loss
```
First, we need to convert `pts` to a NumPy array, because that is the type
that `oin.get_vr_filtration_and_critical_edges` expects as the first argument.
Then we create the `TopologyOptimizer` object that provides access to all
optimization-related functions.
```python
    eps = top_opt.get_nth_persistence(dim, n)
```
Since we want to preserve all but the first `n-1` points of the diagram,
we compute the persistence of the `n`-th point in the diagram. All points
with persistence at most `eps` will be driven to the diagonal.
There are 3 natural choices: a point `(b, d)` can be moved to 
`(b, b)`, `(d, d)` or `((b+d)/2, (b+d)/2)`.
They correspond to 3 members of the enum `oin.DenoiseStrategy`: `BirthBirth`, `DeathDeath`, `Midway`.

```python
    indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
```
`indices` and `values` encode what we call the *matching loss*.  
It is easier to explain the meaning of this by example.
Say, we want to move two points of the persistence diagram, `(b_1, d_1)` and `(b_2, d_2)` to the destinations
`(target_b_1, target_d_1)` and `(target_b_2, target_d_2)` respectively.
Recall that `b_1` is the filtration value of some simplex in filtration,
say, `sigma_1`. Similarly, `d_1` corresponds to `sigma_2`, `b_2` corresponds
to `sigma_3` and `d_2` corresponds to `sigma_4`.
In this case, `indices = [i_1, i_2, i_3, i_4]` and 
`values = [target_b_1, target_d_1, target_b_2, target_d_2]`,
where `i_1` is the index (`sorted_id`) of `sigma_1` in the filtration, `i_2` is
the index of `sigma_2`, etc.


Note that each pair, like `(i_2, target_d_1)` defines
the *singleton loss*. The line
```python
    critical_sets = top_opt.singletons(indices, values)
```
computes all the critical sets at once. `critical_sets`
is a `list` of the same length as `indices`.
Each element of the list is a pair `(value, simplex_indices)`,
where `simplex_indices` contains the critical set
(indices of simplices in filtration order) and
`value` is the target value that should be assigned to all of them.

There can be conflicts: different terms in the matching loss
can send the same simplex to different values.
In order to resolve them, we use the function `combine_loss`:
```python
    crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)
```
The meaning of the output is the same as in `simplify`:
`crit_indices` contains the indices of simplices and `crit_values`
contains the values that we want these simplices to have.
The conflicts are resolved according to the `oin.ConflictStrategy` enum:
`Max` means that we choose the value that is the farthest one from the current
filtration value of the given simplex, `Avg` means that we take the average.


In this example we use Vietoris--Rips. The simplex
`crit_indices[k]` has the longest edge, and `crit_values[k]` is the length
that we want this edge to have. It remains to express
this in the differentiable (known to Torch) way.

First, for each critical simplex, let us extract
the endpoints of its longest edge.
```python
    # convert from list of ints to np.array, so that subscription works
    crit_indices = np.array(crit_indices, dtype=np.int32)
    
    # we get only the edges of the critical simplices
    crit_edges = longest_edges[crit_indices, :]
    # split them into start and end points
    # for critical simplex sigma that appears in position k in crit_indices,
    # crit_edges_x[k] and crit_edges_y[k] give the indices (in pts)
    # of the endpoints of its longest edge.
    crit_edges_x, crit_edges_y = crit_edges[:, 0], crit_edges[:, 1]
```

```python
    top_loss = torch.sum(torch.abs(torch.sum((pts[crit_edges_x, :] - pts[crit_edges_y, :])**2, axis=1) - crit_values ** 2))
```

The expression `torch.sum((pts[crit_edges_x, :] - pts[crit_edges_y, :])**2, axis=1)`
performs summation over all coordinates, so the resulting tensor
contains the squared lengths of critical edges computed in a differentiable
way. Do not take the square root of these lenghts,
this will lead to NaN-s in your gradients (see https://github.com/pytorch/pytorch/issues/15506,
but this is not Torch-specific, Jax has the same behavior).
Instead, use the squares of target lengths, as in this example.



Remarks:

1. We used `simplify` just for convenience. In principle,
the *matching loss* can be defined by the user manually,
it is really just two arrays (lists): simplices and which filtration
value we want them to have, `indices` and `values`.
2. Another convenience function that `TopologyOptimizer`
provides is `match`, it computes the loss that corresponds
to Wasserstein matching to a template diagram.
3. If you do not want to use the critical set method,
but only backpropagate through the critical simplices,
just use `indices` and `values` in place of `crit_indices`, `crit_values`
when computing the edge endpoints and the squared lengths.
This is what the paper calls `diagram method`.
4. The real code contains `if` statement: when there are no critical simplices
the loss should be 0 (say, `eps` was too small). However, `torch` does not handle
this corner case as we need, so we return a differentiable zero tensor manually.
5. If you need to work with high-dimensional data, `Oineus` will only accept the distance
matrix to compute VR filtration and longest edges, not the points
themselves. To be able to differentiate the distances, use `torch.cdist` or analogues
to compute your distances in the differentiable way.



## Kernel, image and cokernel persistence
Oineus can compute the kernel, image and cokernel persistence diagrams as in ["Persistent Homology for Kernels, Images, and Cokernels"](https://doi.org/10.1137/1.9781611973068.110) by D. Cohen-Steiner, H. Edelsbrunner, D. Morozov. We first perform the required reductions using `compute_kernel_image_cokernel_diagrams`, which has arguments:
* `K` the simplicial complex with function values, as a list with an element per simplex in the format `[simplex_id, vertices, value]`, where `vertices` is a list containing the ids of the vertices, and value is the value under the function f.
* `L` the simplicial sub-complex with function values, as a list with an element per simplex in the format `[simplex_id, vertices, value]`, where `vertices` is a list containing the ids of the vertices, and value is the value under the function g.
* `L_to_K` a list which maps the cells in L to their corresponding cells in K,
* `n_threads` the number of threads you want to use,
* `return` an object which contains the kernel, image and cokernel diagrams, as well as the reduced matrices.

To obtain the different diagrams, we use `kernel()`, `image()`, `cokernel()`, and then we can use `in_dimension` to get the sepcific diagram in a specific dimension.

**Note:** aside from the number of threads, all other parameters are set already. 

#### Example
Suppose we have a simplicial complex $K$ with a function $f$ on it, and a subcomplex $L \subset K$ with a function $g$ on it. In this example, $g = f|_L$. We then perform the 5 necessary reductions and compute the persistence diagrams using `compute_kernel_image_cokernel_diagrams`, and then access the 3 sets of diagrams using `kernel()`, `image()`, `cokernel()` respectively. After which we can obtain a diagram in a specific dimension $i$ using `in_dimension(i)`.

```python
import oineus as oin
n_threads = 4
K = [[0, [0], 10], [1,[1],50], [2,[2], 10], [3, [3], 10], [4,[0,1], 50], [5, [1,2], 50], [6,[0,3], 10], [7, [2,3], 10]]
L = [[0, [0], 10], [1,[1],50], [2,[2], 10], [3, [0,1], 50], [4,[1,2],50]]
L_to_K = [0,1,2,4,5]
ker_im_cok_dgms = oin.compute_kernel_image_cokernel_diagrams(K, L, L_to_K, n_threads)
ker_dgms = ker_im_cok_dgms.kernel()
im_dgms = ker_im_cok_dgms.image()
cok_dgms = ker_im_cok_dgms.cokernel()
ker_dgms.in_dimension(0)
```
 
