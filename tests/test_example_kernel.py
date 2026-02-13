import oineus as oin

K = [ [0, [0], 10], [1,[1],10], [2,[2], 10], [3, [3], 10], [4,[0,1], 10], [5, [1,2], 10], [6,[0,3], 10], [7, [2,3], 10] ]
L = [ [0, [0], 10], [1,[1],10], [2,[2], 10], [3, [0,1], 10], [4,[1,2],10] ]

kicr = oin.compute_kernel_image_cokernel_reduction(K, L)

kernel_dgms = kicr.kernel_diagrams()
image_dgms = kicr.image_diagrams()
cokernel_dgms = kicr.cokernel_diagrams()

print(kernel_dgms.in_dimension(0))
print(image_dgms.in_dimension(0))
print(cokernel_dgms.in_dimension(0))


