import oineus as oin

params=oin.ReductionParams()
params.n_threads=4
params.kernel=True
params.image=True
params.cokernel=True

K = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [3], 0], [4,[0,1], 60], [5, [1,2], 70], [6,[0,3], 30], [7, [2,3], 40] ]
L = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [0,1], 60], [4,[1,2],70] ]
IdMapping = [0,1,2,4,5]

kicr = oin.compute_kernel_image_cokernel_reduction(K, L, IdMapping, params)

kernel_dgms = kicr.kernel_diagrams()
image_dgms = kicr.image_diagrams()
cokernel_dgms = kicr.cokernel_diagrams()
print(kernel_dgms.in_dimension(0))
print(image_dgms.in_dimension(0))
print(cokernel_dgms.in_dimension(0))


