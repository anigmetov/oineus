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

fil = oin.vr_filtration(points, max_dim=3, max_diameter=2)
print(fil)
```

The parameters are:
* `points`: coordinates of points in the point cloud.
* `max_dim`: the resulting filtration will contain simplices up to and including `max_dim`.
If you want to compute persistence diagrams in dimension `d`, you need `max_dim >= d+1`. Default value:
dimension of the points.
* `max_diameter`: only consider simplices up to this diameter. Default value:
a minimax that guarantees contractibility after that value, as in Ripser.

For distance matrix you use the same function, just
specify the parameter `from_pwdists` to be True.
In this case, `max_dim` must be supplied.

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

fil = oin.vr_filtration(distances, from_pwdists=True, max_dim=3)
print(fil)
```

### Lower-star filtration.

Lower-star filtrations are supported for functions on a regular D-dimensional grid
for D = 1 , 2, or 3. Function values are represented as an D-dimensional NumPy array.

```python
# create scalar function on 8x8x8 grid
f = np.random.uniform(size=(8, 8, 8))

fil = oin.freudenthal_filtration(data=f, max_dim=3)
```

If you want upper-star filtration, set `negate` to `True`:
```python
fil = oin.freudenthal_filtration(data=f, negate=True)
```
If you want periodic boundary conditions (D-dimensional torus instead
of D-dimensional cube), set `wrap` to `True`:
```python
fil = oin.freudenthal_filtration(data=f, wrap=True)
```
If your data is D-dimensional, but you only need lower-dimensional
simplices, you can specify `max_dim` parameter (it defaults to `data.ndim`):
```python
fil = oin.freudenthal_filtration(data=f, max_dim=2)
```
Note that it is the dimension of maximal simplex in the filtration;
if you want to compute persistence diagram in dimension k,
you need to have simplices of dimension k+1 (negative ones).

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
* `sigma = fil.simplex(sigma_sorted_idx)` will return a simplex itself. So, `fil.simplex`
takes the `sorted_id` of a simplex and just accesses the vector of simplices at this index,
so it is cheap.
* `sigma_idx = fil.id_by_sorted_id(sigma_sorted_idx)` will return the `id` of `sigma`.
Recall that it is the index of `sigma` in the original list of simplices that was used to create the filtration.
This is convenient, if you have a parallel array of some information, one entry per simplex,
which you want to access.


## Topology Optimization


There are two ways to perform topological optimization.  We consider two examples, both use the following function to generate differentiable data.


```python
def sample_data(num_points: int=50, noise_std_dev=0.1):
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

pts = sample_data()
```


We also set up a PyTorch optimizer that acts on the points:


```python
opt = torch.optim.SGD([pts], lr=0.1)
```


### Differentiable filtrations


This method requires `eagerpy` package. It provides wrappers around tensors from PyTorch, Jax and Tensorflow.  Differentiable filtrations are defined in oineus.diff subpackage.  To create a differentiable Vietoris-Rips, we can use the function `oineus.diff.vr_filtration` with essentially the same signature as `oineus.vr_filtration`.


```python
import oineus as oin
import oineus.diff


fil = oin.diff.vr_filtration(pts)
```


Now `fil` is an object that contains the standard filtration (accessible as `fil.under_fil`) and a differentiable tensor of critical values `fil.values`.  We also need to create a TopologyOptimizer object:


```python
top_opt = oin.diff.TopologyOptimizer(fil)
```


Suppose that we want to keep only the most persistent point in the 1-dimensional persistence diagram, and all other points should go to the diagonal vertically (that is, a point (b, d) ideally should go to point (b, b)).


```python
dim = 1
n = 2
dgm = top_opt.compute_diagram(include_inf_points=False)
# eps is the threshold: all points whose persistence does not exceed eps will move
# if we want to remove all but one, the most persistent point, eps must be the
# second biggest persistence
eps = top_opt.get_nth_persistence(dim, n)
```


TopologyOptimizer provides a helper function `simplify` which returns a tuple of indices and values. This is key concept: this tuple encodes prescribed targets for individual diagram points. The interpretation is: for every `i`, simplex `fil[indices[i]]` must take new critical value `values[i]`. In this case, we want to modify only the death value, so every simplex in `indices` will be a negative simplex that produces an off-diagonal point in the diagram.


```python
indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
```


Now, if we want to use the traditional method of back-propagation through a persistence diagram, we simply need to define the topological loss and use standard PyTorch approach like that:


```python
# zeros gradients on pts
opt.zero_grad()

top_loss = torch.mean((fil.values[indices] - values) ** 2)

# populates gradients on pts from top_loss
top_loss.backward()

# updates pts using gradient descent
opt.step()
```

However, if we want to operate through a bigger set of simplices at once, we can do the following (see 'Topological Optimization with Big Steps' for details):



```python
# returns a list of critical sets for each singleton loss defined by indices[i], values[i]
critical_sets = top_opt.singletons(indices, values)

# resolve conflicts between critical sets
crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)

# convert lists to something torch can understand
crit_indices = np.array(crit_indices, dtype=np.int32)
crit_values = torch.Tensor(crit_values)

opt.zero_grad()
top_loss = torch.mean((fil.values[crit_indices] - crit_values) ** 2)
top_loss.backward()
opt.step()
```


### Manual


This is a more complicated way, which gives you more control, you do all the steps by hand.  The corresponding loss function is computed by this function:
```python
def topological_loss(pts: torch.Tensor, dim: int=1, n: int=2):
    pts_as_numpy = pts.clone().detach().numpy().astype(np.float64)
    # we need critical edges for each simplex, so we specify
    # with_critical_edges
    fil, longest_edges = oin.vr_filtration(pts_as_numpy, with_critical_edges=True)

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


First, we need to convert `pts` to a NumPy array, because that is the type that `oin.vr_filtration` expects as the first argument.  Then we create the `TopologyOptimizer` object that provides access to all optimization-related functions.


```python
    eps = top_opt.get_nth_persistence(dim, n)
```


Since we want to preserve all but the first `n-1` points of the diagram, we compute the persistence of the `n`-th point in the diagram. All points with persistence at most `eps` will be driven to the diagonal.  There are 3 natural choices: a point `(b, d)` can be moved to `(b, b)`, `(d, d)` or `((b+d)/2, (b+d)/2)`.  They correspond to 3 members of the enum `oin.DenoiseStrategy`: `BirthBirth`, `DeathDeath`, `Midway`.
```python
    indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
```
`indices` and `values` encode what we call the *matching loss*.  It is easier to explain the meaning of this by example.  Say, we want to move two points of the persistence diagram, `(b_1, d_1)` and `(b_2, d_2)` to the destinations `(target_b_1, target_d_1)` and `(target_b_2, target_d_2)` respectively.  Recall that `b_1` is the filtration value of some simplex in filtration, say, `sigma_1`. Similarly, `d_1` corresponds to `sigma_2`, `b_2` corresponds to `sigma_3` and `d_2` corresponds to `sigma_4`.  In this case, `indices = [i_1, i_2, i_3, i_4]` and `values = [target_b_1, target_d_1, target_b_2, target_d_2]`, where `i_1` is the index (`sorted_id`) of `sigma_1` in the filtration, `i_2` is the index of `sigma_2`, etc.


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


There can be conflicts: different terms in the matching loss can send the same simplex to different values.  In order to resolve them, we use the function `combine_loss`:
```python
    crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)
```
The meaning of the output is the same as in `simplify`: `crit_indices` contains the indices of simplices and `crit_values` contains the values that we want these simplices to have.  The conflicts are resolved according to the `oin.ConflictStrategy` enum: `Max` means that we choose the value that is the farthest one from the current filtration value of the given simplex, `Avg` means that we take the average.


In this example we use Vietoris--Rips. The simplex `crit_indices[k]` has the longest edge, and `crit_values[k]` is the length that we want this edge to have. It remains to express this in the differentiable (known to Torch) way.


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

The expression `torch.sum((pts[crit_edges_x, :] - pts[crit_edges_y, :])**2, axis=1)` performs summation over all coordinates, so the resulting tensor contains the squared lengths of critical edges computed in a differentiable way. Do not take the square root of these lenghts, this will lead to NaN-s in your gradients (see https://github.com/pytorch/pytorch/issues/15506, but this is not Torch-specific, Jax has the same behavior).  Instead, either use the squares of target lengths, as in this example, or add a small number before taking the square root.


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
* `params` specifies which of the three components (image, kernel, cokernel)
    you need. Type: KICRParams.
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
ker_im_cok_dgms = oin.compute_kernel_image_cokernel_diagrams(K, L)
# by default, all 3 diagrams are computed
ker_dgms = ker_im_cok_dgms.kernel()
im_dgms = ker_im_cok_dgms.image()
cok_dgms = ker_im_cok_dgms.cokernel()
ker_dgms.in_dimension(0)
# if you only want, e.g., kernel:
params = oin.KICRParams()
params.image = params.cokernel = False
ker_im_cok_dgms = oin.compute_kernel_image_cokernel_diagrams(K, L, params)
```
