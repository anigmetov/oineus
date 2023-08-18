#!/usr/bin/env python3

import numpy as np
import oineus as oin

def dion_dgm_to_numpy(dion_dgm):
    return np.array([[p.birth, p.death] for p in dion_dgm ]).astype(np.float32)

def prepare_data():
    import dionysus as dion
    top_dim = 3
    for negate in [True, False]:
        a = np.load("vertebra_32x32x32_float32.npy")
        # compute diagrams with Dionysis
        fil_us = dion.fill_freudenthal(a, reverse=negate)
        p = dion.homology_persistence(fil_us)
        dion_dgms = dion.init_diagrams(p, fil_us)
        for dim in range(top_dim):
            dgm = dion_dgm_to_numpy(dion_dgms[dim])
            if negate:
                dgm[dgm == np.inf] = -np.inf
            if negate:
                np.save(f"dgm_vertebra_32_neg_{dim}.npy", dgm)
            else:
                np.save(f"dgm_vertebra_32_{dim}.npy", dgm)


def sort_numpy_array(a: np.ndarray):
    assert a.ndim == 2
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def test_vertebra():
    top_dim = 3
    wrap = False
    a = np.load("vertebra_32x32x32_float32.npy")
    for dualize in [True, False]:
        for n_threads in [1, 7]:
            for negate in [False, True]:
                for clearing_opt in [True]:
                    print(f"{dualize=}, {negate=}, {n_threads=}, {clearing_opt=}")
                    rp = oin.ReductionParams()
                    rp.n_threads = n_threads
                    rp.clearing_opt = clearing_opt
                    oin_dgms = oin.compute_diagrams_ls(a, negate, wrap, top_dim-1, rp, include_inf_points=True, dualize=dualize)
                    for dim in range(top_dim):
                        if negate:
                            dion_dgm_fname = f"dgm_vertebra_32_neg_{dim}.npy"
                        else:
                            dion_dgm_fname = f"dgm_vertebra_32_{dim}.npy"
                        dion_dgm = np.load(dion_dgm_fname)
                        oin_dgm = oin_dgms[dim]

                        dion_dgm = sort_numpy_array(dion_dgm)
                        oin_dgm = sort_numpy_array(oin_dgm)
                        assert dion_dgm.shape == oin_dgm.shape

                        dion_dgm[dion_dgm == np.inf] = 0.0
                        dion_dgm[dion_dgm == -np.inf] = 0.0
                        oin_dgm[oin_dgm == np.inf] = 0.0
                        oin_dgm[oin_dgm == -np.inf] = 0.0

                        assert np.linalg.norm(dion_dgm - oin_dgm) < 0.00001



if __name__ == "__main__":
    test_vertebra()