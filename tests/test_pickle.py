import numpy as np
import pickle

import oineus as oin


def test_cells():
    vs = [0, 1, 2]
    sigma = oin.CombinatorialSimplex(vs)
    p_sigma = pickle.dumps(sigma)
    sigma_back = pickle.loads(p_sigma)
    assert sigma == sigma_back

    sigma_val = oin.Simplex(vs, 1.0)
    p_sigma_val = pickle.dumps(sigma_val)
    sigma_val_back = pickle.loads(p_sigma_val)
    assert sigma_val == sigma_val_back


    dom = oin.GridDomain_2D(10, 10)
    p_dom = pickle.dumps(dom)
    dom_back = pickle.loads(p_dom)
    assert dom == dom_back

    cube = oin.CombinatorialCube_2D([1, 2], [0], dom)
    p_cube = pickle.dumps(cube)
    cube_back = pickle.loads(p_cube)
    assert cube == cube_back



def test_filtrations():
    n_pts = 20
    dim = 3
    pts = np.random.normal(0.0, 1.0, n_pts * dim).reshape(n_pts, dim)
    vr_fil = oin.vr_filtration(pts)
    cells = vr_fil.cells()
    a = pickle.dumps(cells)
    b = pickle.loads(a)
    p_vr_fil = pickle.dumps(vr_fil)
    vr_fil_back = pickle.loads(p_vr_fil)

    assert vr_fil_back == vr_fil

