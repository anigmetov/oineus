import oineus as oin

n_threads = 4


K = [ [0, [0], 10], [1,[1],50], [2,[2], 10], [3, [3], 10], [4,[0,1], 50], [5, [1,2], 50], [6,[0,3], 10], [7, [2,3], 10] ]
L = [ [0, [0], 10], [1,[1],50], [2,[2], 10], [3, [0,1], 50], [4,[1,2],50] ]
IdMapping = [0,1,2,4,5]
params= oineus.ReductionParams()
params.kernel=True
params.image=True
params.cokernel=True
params.verbose=True
params.n_threads=32

kicr = oineus.compute_kernel_image_cokernel_reduction(K, L, L_to_K, params)