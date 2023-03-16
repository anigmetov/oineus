import oineus as oin

params = oin.ReductionParams()
params.n_threads = 4
params.clearing_opt = False


K = [ [0, [0], 10], [1,[1],50], [2,[2], 10], [3, [3], 10], [4,[0,1], 50], [5, [1,2], 50], [6,[0,3], 10], [7, [2,3], 10] ]
L = [ [0, [0], 10], [1,[1],50], [2,[2], 10], [3, [0,1], 50], [4,[1,2],50] ]
IdMapping = [0,1,2,4,5]

kicr = oin.compute_kernel_image_cokernel_diagrams(K, L,IdMapping, params.n_threads)

kicr.get_kernel_diagrams()
