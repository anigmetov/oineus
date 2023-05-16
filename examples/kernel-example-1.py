import oineus as oin

params=oin.ReductionParams()
params.n_threads=4
params.kernel=True
params.image=True
params.cokernel=True

K = [ [0, [0], 10], [1,[1],10], [2,[2], 10], [3, [3], 10], [4,[0,1], 10], [5, [1,2], 10], [6,[0,3], 10], [7, [2,3], 10] ]
L = [ [0, [0], 10], [1,[1],10], [2,[2], 10], [3, [0,1], 10], [4,[1,2],10] ]
IdMapping = [0,1,2,4,5]

kicr = oin.compute_kernel_image_cokernel_reduction(K, L, IdMapping, params)

kernel_dgms = kicr.kernel_diagrams()
kernel_dgms = kicr.image_diagrams()
cokernel_dgms = kicr.cokernel_diagrams()
print(kernel_dgms.in_dimension(0))
print(image_dgms.in_dimension(0))
print(cokernel_dgms.in_dimension(0))


