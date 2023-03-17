import oineus as oin

n_threads = 5


K = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [3], 50], [4,[4], 15], [5, [5], 12], [6,[0,1], 50], [7, [1,2], 60], [8,[2,3], 70], [9, [3,4], 80], [10, [0,5], 30], [11,[4,5], 20]]
L = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [3], 50], [4,[4], 15], [5,[0,1], 50], [6, [1,2], 60], [7,[2,3], 70], [8, [3,4], 80] ]
IdMapping = [0,1,2,3,4,6,7,8,9]


kicr = oin.compute_kernel_image_cokernel_diagrams(K, L,IdMapping, n_threads)


ker = kicr.kernel()

im = kicr.image()

cok = kicr.cokernel()
