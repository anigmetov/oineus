import oineus as oin

n_threads = 5


K = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [3], 0], [4,[0,1], 60], [5, [1,2], 70], [6,[0,3], 30], [7, [2,3], 40] ]
L = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [0,1], 60], [4,[1,2],70] ]
IdMapping = [0,1,2,4,5]

kicr = oin.compute_kernel_image_cokernel_diagrams(K, L,IdMapping, n_threads)

ker = kicr.kernel()

im = kicr.image()

cok = kicr.cokernel()
