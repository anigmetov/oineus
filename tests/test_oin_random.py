#!python3

import numpy as np
import oineus as oin
import dionysus as dion

def dion_dgm_to_numpy(dion_dgm, include_inf_points=True):
    return np.array([[p.birth, p.death] for p in dion_dgm if include_inf_points or np.abs(p.birth) < np.inf and np.abs(p.death) < np.inf]).astype(np.float32)

def helper_test_random(n, dualize, negate, n_threads, seed=1, top_dim=3, compute_v=False, compute_u=False):
    # no wrap in Dionysus
    wrap = False

    # generate random grid data
    np.random.seed(seed)
    a = np.random.randn(n ** 3).reshape((n, n, n))

    rp = oin.ReductionParams()
    rp.n_threads = n_threads
    rp.compute_v = compute_v
    rp.compute_u = compute_u
    # compute diagrams with Oineus
    oin_dgms = oin.compute_diagrams_ls(a, negate, wrap, top_dim-1, rp, include_inf_points=True, dualize=dualize)

    # compute diagrams with Dionysis
    fil_us = dion.fill_freudenthal(a, reverse=negate)
    p = dion.homology_persistence(fil_us)
    dion_dgms = dion.init_diagrams(p, fil_us)

    dist = 0.0

    for dim in range(top_dim):
        # convert Oineus diagram to Dionysus format
        oin_dgm = dion.Diagram(oin_dgms[dim])
        dion_dgm = dion_dgms[dim]
        dion_dgm = dion_dgm_to_numpy(dion_dgm, True)
        if negate:
            dion_dgm[dion_dgm == np.inf] = -np.inf
        dion_dgm = dion.Diagram(dion_dgm)
        dist += dion.bottleneck_distance(oin_dgm, dion_dgm)

    assert(dist < 0.001)


def test_random():
    for n in [4, 8, 17]:
        for dualize in [True, False]:
            for negate in [True, False]:
                for compute_u in [True, False]:
                    for compute_v in [True, False]:
                        for n_threads in [1, 4, 7]:
                            if n_threads > 1 and compute_u:
                                continue
                            helper_test_random(n=n, negate=negate, n_threads=n_threads, dualize=dualize, compute_v=compute_v, compute_u=compute_u)


if __name__ == "__main__":
    test_random()
