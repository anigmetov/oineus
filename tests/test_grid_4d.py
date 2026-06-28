"""4D grid support: Freudenthal + cubical filtrations on a 4D grid.

4D grids were previously unsupported (fr_displacements threw for D>=4 and
OINEUS_MAX_CUBE_DIM was 3). This checks the end-to-end Python pipeline for D=4:

- the slim (anchor,type) Freudenthal filtration (_FreudenthalFiltration_4D) is
  built by default and reduces to the SAME diagrams as the fat universal-Simplex
  path (the oracle) -- the C++ side already proves the boundary matrices match
  column-for-column (tests_freudenthal_cell.cpp); this is the diagram-level check;
- the 4D cubical filtration (_CubeFiltration_4D) builds, reduces, and has the
  right number of essential H0 classes (one per connected component = 1 for a
  full grid);
- the facade dispatches optimizer / KICR classes for the 4D filtrations.

Self-contained: uses the fat Freudenthal path as the ground-truth oracle, so it
runs without dionysus / gudhi.
"""

import numpy as np
import pytest
import oineus as oin


def sort_finite(dgm):
    d = np.array(dgm, dtype=np.float64).reshape(-1, 2)
    d[d == np.inf] = 1e9
    d[d == -np.inf] = -1e9
    return d[np.lexsort((d[:, 1], d[:, 0]))]


def dgms_close(a, b, tol=1e-9):
    a, b = sort_finite(a), sort_finite(b)
    return a.shape == b.shape and (a.size == 0 or np.allclose(a, b, atol=tol))


def reduce_to_dgms(fil, dualize, n_threads, top_dim):
    dcmp = oin.Decomposition(fil, dualize)
    rp = oin.ReductionParams()
    rp.n_threads = n_threads
    dcmp.reduce(rp)
    dgms = dcmp.diagram(fil, include_inf_points=True)
    return [np.asarray(dgms.in_dimension(d)) for d in range(top_dim)]


@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_freudenthal_4d_slim_matches_fat(negate, dualize, n_threads):
    # small asymmetric 4D grid catches stride / dimension bugs
    a = np.random.default_rng(42).random((5, 4, 4, 3)).astype(np.float64)
    dim = 4

    fil_slim = oin.freudenthal_filtration(a, negate=negate, max_dim=dim, slim=True, n_threads=n_threads)
    fil_fat = oin.freudenthal_filtration(a, negate=negate, max_dim=dim, slim=False, n_threads=n_threads)

    assert type(fil_slim).__name__ == "_FreudenthalFiltration_4D"
    assert type(fil_fat).__name__ == "_Filtration"
    assert fil_slim.size() == fil_fat.size()

    slim_dgms = reduce_to_dgms(fil_slim, dualize, n_threads, dim)
    fat_dgms = reduce_to_dgms(fil_fat, dualize, n_threads, dim)
    for d in range(dim):
        assert dgms_close(slim_dgms[d], fat_dgms[d]), f"4D slim != fat in dim {d}"


def test_cube_4d_builds_and_reduces():
    a = np.random.default_rng(7).random((5, 4, 4, 3)).astype(np.float64)
    fil = oin.cube_filtration(a)
    assert type(fil).__name__ == "_CubeFiltration_4D"

    dgms = reduce_to_dgms(fil, dualize=False, n_threads=1, top_dim=4)
    # a full 4D grid is connected -> exactly one essential (infinite-death) H0 class
    h0 = dgms[0]
    n_essential = int(np.sum(np.isinf(h0[:, 1]))) if h0.size else 0
    assert n_essential == 1


def test_cube_and_freudenthal_4d_agree_on_components():
    # cubical and Freudenthal triangulate the same data differently, but H0 essential
    # count (number of connected components) must agree -- a coarse cross-check
    a = np.random.default_rng(11).random((4, 4, 4, 4)).astype(np.float64)

    cube_h0 = reduce_to_dgms(oin.cube_filtration(a), False, 1, 1)[0]
    fr_h0 = reduce_to_dgms(oin.freudenthal_filtration(a, max_dim=4), False, 1, 1)[0]

    cube_ess = int(np.sum(np.isinf(cube_h0[:, 1]))) if cube_h0.size else 0
    fr_ess = int(np.sum(np.isinf(fr_h0[:, 1]))) if fr_h0.size else 0
    assert cube_ess == fr_ess == 1


def test_facade_dispatch_4d():
    # the Python facade must route 4D filtrations to their 4D optimizer classes
    a = np.random.default_rng(3).random((4, 4, 4, 4)).astype(np.float64)

    fr = oin.freudenthal_filtration(a, max_dim=4, slim=True)
    opt_fr = oin.TopologyOptimizer(fr)
    assert type(opt_fr).__name__ == "TopologyOptimizerFreudenthal_4D"

    cube = oin.cube_filtration(a)
    opt_cube = oin.TopologyOptimizer(cube)
    assert type(opt_cube).__name__ == "TopologyOptimizerCube_4D"
