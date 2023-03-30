import oineus

params= oineus.ReductionParams()
params.kernel=True
params.n_threads=32

kicr = oineus.compute_kernel_image_cokernel_reduction([[0,[0], 10], [1, [1], 30], [2, [2], 40], [3, [3], 0], [4, [0,1], 70], [5, [1,2], 60], [6, [0,3], 40], [7, [2,3], 50]], [[0,[0], 10.], [1, [1], 30], [2, [2], 20], [3, [0,1], 70], [4, [1,2], 60]], [0,1,2,4,5], params)
print(kicr)