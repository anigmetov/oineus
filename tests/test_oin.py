#!/usr/bin/env python3

import argh
import argparse
import numpy as np

import oineus as oin
import dionysus as dion


def compare_diagrams(n=16, negate=False, n_threads=1, seed=1, top_dim=2):
    # no wrap in Dionysus
    wrap = False

    # generate random grid data
    np.random.seed(seed)
    a = np.random.randn(n ** 3).reshape((n, n, n))

    params = oin.ReductionParams()
    params.n_threads = n_threads

    # compute diagrams with Oineus
    oin_dgms = oin.compute_diagrams_ls(a, negate, wrap, top_dim, params, include_inf_points=True, dualize=False)

    # compute diagrams with Dionysis
    fil_us = dion.fill_freudenthal(a, reverse=negate)
    p = dion.homology_persistence(fil_us)
    dion_dgms = dion.init_diagrams(p, fil_us)

    dist = 0.0

    for dim in range(top_dim):
        # convert Oineus diagram to Dionysus format
        oin_dgm = dion.Diagram(oin_dgms[dim])
        dion_dgm = dion_dgms[dim]
        dist += dion.bottleneck_distance(oin_dgm, dion_dgm)

    print("total dist: ", dist)
    assert(dist < 0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    argh.add_commands(parser, [compare_diagrams])
    argh.dispatch(parser)
