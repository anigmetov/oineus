import pickle

import pytest

import oineus as oin


def test_vredge_api():
    edge = oin.VREdge(2, 3)

    assert edge[0] == 2
    assert edge[1] == 3
    _ = repr(edge)

    default_y = oin.VREdge(4)
    assert default_y.x == 4
    assert default_y.y == 0


def test_reduction_params_api():
    params = oin.ReductionParams()
    params.n_threads = 1
    params.advanced.chunk_size = 16
    params.use_clearing = True
    params.compute_v = True
    params.compute_u = True
    params.advanced.dims_to_restore_elz = []
    params.sanity_check = False
    params.verbose = False

    _ = repr(params)

    params_alt = oin.ReductionParams(n_threads=2, chunk_size=32, use_clearing=True, compute_v=False, compute_u=False, dims_to_restore_elz=[], verbose=False)
    _ = repr(params_alt)

    params_back = pickle.loads(pickle.dumps(params))
    assert params_back.n_threads == params.n_threads


def test_reduction_params_ctor_defaults_agree():
    # the kwargs ctor must default every field to the C++ in-class defaults,
    # i.e. agree with the no-arg ctor (they used to silently differ)
    default = oin.ReductionParams()
    kwargs_default = oin.ReductionParams(verbose=False)
    assert kwargs_default.n_threads == default.n_threads
    assert kwargs_default.advanced.chunk_size == default.advanced.chunk_size
    assert kwargs_default.use_clearing == default.use_clearing
    assert kwargs_default.compute_v == default.compute_v
    assert kwargs_default.compute_u == default.compute_u
    assert kwargs_default.advanced.col_repr == default.advanced.col_repr
    assert kwargs_default.verbose == default.verbose
    assert kwargs_default == default


def test_reduction_params_advanced():
    # chunk_size / col_repr / dims_to_restore_elz live in the nested
    # advanced sub-struct: flat attribute access is a clean break (loud
    # AttributeError), in-place mutation goes through params.advanced,
    # and ==, repr, pickle all include the nested block
    params = oin.ReductionParams()

    for name in ("chunk_size", "col_repr", "dims_to_restore_elz"):
        with pytest.raises(AttributeError):
            getattr(params, name)

    params.advanced.chunk_size = 64
    params.advanced.col_repr = oin.ColumnRepr.Full
    params.advanced.dims_to_restore_elz = [0, 1]
    assert params.advanced.chunk_size == 64
    assert params.advanced.col_repr == oin.ColumnRepr.Full
    assert list(params.advanced.dims_to_restore_elz) == [0, 1]

    # whole-object assignment of a standalone ReductionParamsAdvanced
    params2 = oin.ReductionParams()
    params2.advanced = oin.ReductionParamsAdvanced(chunk_size=32)
    assert params2.advanced.chunk_size == 32

    # == distinguishes params differing only in advanced
    a, b = oin.ReductionParams(), oin.ReductionParams()
    assert a == b
    b.advanced.chunk_size += 1
    assert a != b
    assert a.advanced != b.advanced

    # pickle roundtrip carries the nested block (outer and standalone)
    back = pickle.loads(pickle.dumps(params))
    assert back == params
    assert back.advanced.chunk_size == 64
    assert back.advanced.col_repr == oin.ColumnRepr.Full
    assert list(back.advanced.dims_to_restore_elz) == [0, 1]
    adv_back = pickle.loads(pickle.dumps(params.advanced))
    assert adv_back == params.advanced

    # repr contains the nested advanced block
    r = repr(params)
    assert "advanced = ReductionParamsAdvanced(" in r
    assert "chunk_size = 64" in r
    assert "col_repr = Full" in r
    assert repr(params.advanced).startswith("ReductionParamsAdvanced(")


def test_reduction_params_renamed_field_aliases():
    # old field names keep working as read/write aliases of the new ones
    params = oin.ReductionParams()

    params.clearing_opt = False
    assert params.use_clearing is False
    params.use_clearing = True
    assert params.clearing_opt is True

    params.apparent_opt = True
    assert params.use_apparent_pairs is True
    params.use_apparent_pairs = False
    assert params.apparent_opt is False

    params.do_sanity_check = True
    assert params.sanity_check is True
    params.sanity_check = False
    assert params.do_sanity_check is False

    # repr shows the new names only
    r = repr(params)
    assert "use_clearing" in r
    assert "use_apparent_pairs" in r
    assert "sanity_check" in r
    assert "clearing_opt" not in r
    assert "apparent_opt" not in r
    assert "do_sanity_check" not in r
    assert r.startswith("ReductionParams(")


def test_kicr_params_repr_uses_attribute_names():
    # repr labels must match the actual attribute names (kernel, image, ...)
    r = repr(oin.KICRParams())
    assert "kernel = " in r
    assert "image = " in r
    assert "cokernel = " in r
    assert "codomain = " in r
    assert "compute_kernel" not in r


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
