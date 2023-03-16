import oineus as oin

n_threads = 4


K = [ [0, [0], 10], [1,[1],10], [2,[2], 10], [3, [3], 10], [4,[0,1], 10], [5, [1,2], 10], [6,[0,3], 10], [7, [2,3], 10] ]
L = [ [0, [0], 10], [1,[1],10], [2,[2], 10], [3, [0,1], 10], [4,[1,2],10] ]
IdMapping = [0,1,2,4,5]

kicr = oin.compute_kernel_image_cokernel_diagrams(K, L,IdMapping, n_threads)

ker = kicr.get_kernel_diagrams()

im = kicr.get_image_diagrams()

cok = kicr.get_cokernel_diagrams()
