import oineus as oin
import math

def test_kernel_1():
    params=oin.KICRParams()
    params.n_threads=4
    params.kernel=True
    params.image=True
    params.cokernel=True
    K = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [3], 50], [4,[4], 15], [5, [5], 12], [6,[0,1], 50], [7, [1,2], 60], [8,[2,3], 70], [9, [3,4], 80], [10, [0,5], 30], [11,[4,5], 20]]
    L = [ [0, [0], 10], [1,[1],50], [2,[2], 20], [3, [3], 50], [4,[4], 15], [5,[0,1], 50], [6, [1,2], 60], [7,[2,3], 70], [8, [3,4], 80] ]
    IdMapping = [0,1,2,3,4,6,7,8,9]
    K = oin.list_to_filtration(K)
    L = oin.list_to_filtration(L)
    kicr = oin.KerImCokReduced(K, L, params)
    # print(kicr.kernel_diagrams().in_dimension(0))
    # assert (kicr.kernel_diagrams().in_dimension(0) == [[30., 80.],[20., math.inf]]).all()
    assert (len(kicr.kernel_diagrams().in_dimension(1)) == 0)
    assert (kicr.cokernel_diagrams().in_dimension(0) == [[12., 20.]]).all()
    assert (kicr.cokernel_diagrams().in_dimension(1) == [[80., math.inf]]).all()
    assert (kicr.image_diagrams().in_dimension(0) == [[15., 30.],[20., 60.],[50., 70.], [10., math.inf]]).all()
    assert (len(kicr.image_diagrams().in_dimension(1)) == 0)

def test_kernel_2():
    params=oin.KICRParams()
    params.n_threads=4
    params.kernel=True
    params.image=True
    params.cokernel=True
    K=[[0,[0], 10], [1, [1], 30], [2, [2], 10], [3, [3], 0], [4, [0,1], 30], [5, [1,2], 30], [6, [0,3], 10], [7, [2,3], 10]]
    L=[[0,[0], 10.], [1, [1], 30], [2, [2], 10], [3, [0,1], 30], [4, [1,2], 30]]
    K = oin.list_to_filtration(K)
    L = oin.list_to_filtration(L)
    kicr = oin.KerImCokReduced(K, L, params)
    assert (kicr.kernel_diagrams().in_dimension(0) == [[10., 30.]]).all()
    assert (len(kicr.kernel_diagrams().in_dimension(1)) == 0)
    assert (kicr.cokernel_diagrams().in_dimension(0) == [[0., 10.]]).all()
    assert (kicr.cokernel_diagrams().in_dimension(1) == [[30., math.inf]]).all()
    assert (kicr.image_diagrams().in_dimension(0) == [[10., math.inf]]).all()
    assert (len(kicr.image_diagrams().in_dimension(1)) == 0)


if __name__ == "__main__":
    test_kernel_1()
    test_kernel_2()

