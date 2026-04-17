import pickle

import oineus as oin


def test_vredge_api():
    edge = oin.VREdge(1)
    edge.x = 2
    edge.y = 3

    assert edge[0] == 2
    assert edge[1] == 3
    _ = repr(edge)


def test_reduction_params_api():
    params = oin.ReductionParams()
    params.n_threads = 1
    params.chunk_size = 16
    params.write_dgms = False
    params.sort_dgms = False
    params.clearing_opt = True
    params.acq_rel = False
    params.print_time = False
    params.elapsed = 0.0
    params.compute_v = True
    params.compute_u = True
    params.restore_elz = False
    params.do_sanity_check = False
    params.elapsed_restore_elz = 0.0
    params.elapsed_copy_back = 0.0
    params.elapsed_copy_pivots = 0.0
    params.verbose = False

    _ = repr(params)

    params_alt = oin.ReductionParams(n_threads=2, chunk_size=32, clearing_opt=True, compute_v=False, compute_u=False, restore_elz=False, verbose=False)
    _ = repr(params_alt)

    params_back = pickle.loads(pickle.dumps(params))
    assert params_back.n_threads == params.n_threads


def test_kicr_params_api():
    params = oin.KICRParams()
    params.codomain = True
    params.kernel = True
    params.image = True
    params.cokernel = True
    params.include_zero_persistence = True
    params.verbose = False
    params.sanity_check = False
    params.n_threads = 1

    _ = params.params_f
    _ = params.params_g
    _ = params.params_ker
    _ = params.params_im
    _ = params.params_cok
    _ = repr(params)

    params_back = pickle.loads(pickle.dumps(params))
    assert params_back.n_threads == params.n_threads


def test_denoise_strategy_enum():
    _ = oin.DenoiseStrategy.BirthBirth.as_str()
    _ = oin.DenoiseStrategy.DeathDeath.as_str()
    _ = oin.DenoiseStrategy.Midway.as_str()


def test_conflict_strategy_enum():
    _ = oin.ConflictStrategy.Max.as_str()
    _ = oin.ConflictStrategy.Avg.as_str()
    _ = oin.ConflictStrategy.Sum.as_str()
    _ = oin.ConflictStrategy.FixCritAvg.as_str()


def test_diagram_plane_domain_enum():
    _ = oin.DiagramPlaneDomain.AboveDiagonal.as_str()
    _ = oin.DiagramPlaneDomain.BelowDiagonal.as_str()
    _ = oin.DiagramPlaneDomain.Mixed.as_str()


def test_frechet_mean_init_enum():
    _ = oin.FrechetMeanInit.Custom.as_str()
    _ = oin.FrechetMeanInit.FirstDiagram.as_str()
    _ = oin.FrechetMeanInit.MedoidDiagram.as_str()
    _ = oin.FrechetMeanInit.RandomDiagram.as_str()
    _ = oin.FrechetMeanInit.Grid.as_str()
