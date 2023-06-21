import oineus as oin
import math

def test_kernel_1():
	params=oin.ReductionParams()
	params.n_threads=4
	params.kernel=True
	params.image=True
	params.cokernel=True
	K = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [3], 50], [4,[4], 15], [5, [5], 12], [6,[0,1], 50], [7, [1,2], 60], [8,[2,3], 70], [9, [3,4], 80], [10, [0,5], 30], [11,[4,5], 20]]
	L = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [3], 50], [4,[4], 15], [5,[0,1], 50], [6, [1,2], 60], [7,[2,3], 70], [8, [3,4], 80] ]
	IdMapping = [0,1,2,3,4,6,7,8,9]
	kicr = oin.compute_kernel_image_cokernel_reduction(K, L, IdMapping, params)
	errors = False
	if (kicr.kernel_diagrams().in_dimension(0) != [[30., 80.],[20., math.inf]]).any():
		print("Error in kernel_test_1: kernel diagram in dimenion 0 incorrect.")
		errors = True
	if (kicr.kernel_diagrams().in_dimension(1) != []).any():
		print("Error in kernel_test_1: kernel diagram in dimenion 1 incorrect.")
		errors = True
	if (kicr.cokernel_diagrams().in_dimension(0) != [[12., 20.]]).any():
		print("Error in kernel_test_1: cokernel diagram in dimenion 0 incorrect.")
		errors = True
	if (kicr.cokernel_diagrams().in_dimension(1) != [[80., math.inf]]).any():
		print("Error in kernel_test_1: cokernel diagram in dimenion 1 incorrect.")
		errors = True
	if (kicr.image_diagrams().in_dimension(0) != [[15., 30.],[20., 60.],[50., 70.], [10., math.inf]]).any():
		print("Error in kernel_test_1: image diagram in dimenion 0 incorrect.")
		errors = True
	if (kicr.image_diagrams().in_dimension(1) != []).any():
		print("Error in kernel_test_1: image diagram in dimenion 0 incorrect.")
		errors = True
	if errors:
		return 1
	else:
		return 0
 

def test_kernel_2():
	params=oin.ReductionParams()
	params.n_threads=4
	params.kernel=True
	params.image=True
	params.cokernel=True
	K=[[0,[0], 10], [1, [1], 30], [2, [2], 10], [3, [3], 0], [4, [0,1], 30], [5, [1,2], 30], [6, [0,3], 10], [7, [2,3], 10]]
	L=[[0,[0], 10.], [1, [1], 30], [2, [2], 10], [3, [0,1], 30], [4, [1,2], 30]] 
	IdMapping=[0,1,2,4,5]
	kicr = oin.compute_kernel_image_cokernel_reduction(K, L, IdMapping, params)
	errors = False
	if (kicr.kernel_diagrams().in_dimension(0) != [[10., 30.]]).any():
		print("Error in kernel_test_1: kernel diagram in dimenion 0 incorrect.")
		errors = True
	if (kicr.kernel_diagrams().in_dimension(1) != []).any():
		print("Error in kernel_test_1: kernel diagram in dimenion 1 incorrect.")
		errors = True
	if (kicr.cokernel_diagrams().in_dimension(0) != [[0., 10.]]).any():
		print("Error in kernel_test_1: cokernel diagram in dimenion 0 incorrect.")
		errors = True
	if (kicr.cokernel_diagrams().in_dimension(1) != [[30., math.inf]]).any():
		print("Error in kernel_test_1: cokernel diagram in dimenion 1 incorrect.")
		errors = True
	if (kicr.image_diagrams().in_dimension(0) != [[10., math.inf]]).any():
		print("Error in kernel_test_1: image diagram in dimenion 0 incorrect.")
		errors = True
	if (kicr.image_diagrams().in_dimension(1) != []).any():
		print("Error in kernel_test_1: image diagram in dimenion 0 incorrect.")
		errors = True
	if errors:
		return 1
	else:
		return 0

