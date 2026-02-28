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
