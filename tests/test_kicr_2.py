import oineus as oin
from icecream import ic

v0 = oin.Simplex([0], 0.0)
v1 = oin.Simplex([1], 0.0)
v2 = oin.Simplex([2], 0.0)
v3 = oin.Simplex([3], 0.0)

eps = 0.2
d = 2.0

e1 = oin.Simplex([0, 1], eps)
e2 = oin.Simplex([2, 3], eps)
e3 = oin.Simplex([1, 2], d)
e4 = oin.Simplex([0, 3], d+2*eps)

fil_K = oin.Filtration([v0, v1, v2, v3, e1, e2, e3, e4])
fil_L = oin.Filtration([v0, v3, e4])


params = oin.KICRParams()
params.n_threads = 1
params.codomain = True
params.kernel = True
params.image = True
params.cokernel = True
params.sanity_check = True
params.verbose = False

kicr = oin.KerImCokReduced(fil_K, fil_L, params)
ic(kicr.kernel_diagrams().in_dimension(0))
ic(kicr.cokernel_diagrams().in_dimension(0))
ic(kicr.image_diagrams().in_dimension(0))

# ic| kicr.kernel_diagrams().in_dimension(0): array([[2. , 2.4]])
# ic| kicr.cokernel_diagrams().in_dimension(0): array([[0. , 0.2],
#                                                      [0. , 0.2]])
# ic| kicr.image_diagrams().in_dimension(0): array([[ 0.,  2.],
#                                                   [ 0., inf]])
