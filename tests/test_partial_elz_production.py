import pickle

import numpy as np
import pytest
import torch

import oineus
import oineus.diff as oin_diff


def circle_filtration(n=24, seed=12):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([
        np.cos(angles) + rng.normal(0, 0.04, n),
        np.sin(angles) + rng.normal(0, 0.04, n),
    ])
    tensor = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    return tensor, oin_diff.vr_filtration(tensor, max_dim=2)


def h1_pair(fil):
    params = oineus.ReductionParams(compute_v=False, n_threads=4)
    decmp = oineus.reduce(fil.under_fil, params, False)
    diagram = decmp.diagram(fil.under_fil, include_inf_points=False)
    indices = diagram.index_diagram_in_dimension(1, as_numpy=True)
    values = diagram.in_dimension(1, as_numpy=True)
    if indices.shape[0] == 0:
        pytest.skip("no H1 point")
    persistence = values[:, 1] - values[:, 0]
    i = int(np.argmax(persistence))
    return indices[i].astype(np.int64), values[i]


def test_role_aware_birth_only_does_not_reduce_homology():
    _, fil = circle_filtration()
    (birth_idx, _), (birth, death) = h1_pair(fil)
    opt = oin_diff.TopologyOptimizer(fil, n_threads=4)

    target = birth + 0.2 * (death - birth)
    opt.crit_sets_apply_typed(
        [int(birth_idx)], [float(target)], [], [],
        oineus.ConflictStrategy.Avg,
    )

    assert opt.is_coh_built
    assert not opt.is_hom_built
    coh = opt.cohomology_decomposition_ref()
    assert coh.negative_v_elz_in_dim(1)
    assert coh.n_computed_u_rows == 0
    assert coh.n_valid_u_rows == 0
    assert not coh.has_matrix_u()


@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize(
    "direction",
    ["birth-increase", "birth-decrease", "death-increase", "death-decrease"],
)
def test_direct_crit_sets_apply_matches_full_elz_oracle(negate, direction):
    rng = np.random.default_rng(4)
    data = torch.tensor(
        rng.uniform(0, 1, size=(8, 8)),
        dtype=torch.float64,
        requires_grad=True,
    )
    fil = oin_diff.freudenthal_filtration(
        data, negate=negate, max_dim=2, slim=False, n_threads=1,
    )
    params = oineus.ReductionParams(compute_v=False, n_threads=4)
    pairing = oineus.reduce(fil.under_fil, params, False)
    diagram = pairing.diagram(fil.under_fil, include_inf_points=False)
    indices = diagram.index_diagram_in_dimension(0, as_numpy=True)
    if indices.shape[0] == 0:
        pytest.skip("no finite H0 point")
    birth_idx, death_idx = map(int, indices[0])
    values = fil.values.detach().cpu().numpy()

    increase_delta = -0.1 if negate else 0.1
    if direction.startswith("birth"):
        index = birth_idx
    else:
        index = death_idx
    delta = increase_delta if direction.endswith("increase") else -increase_delta
    target = float(values[index] + delta)

    lazy = oin_diff.TopologyOptimizer(fil, n_threads=4)
    oracle = oin_diff.TopologyOptimizer(
        fil, n_threads=1,
        u_strategy=oineus._oineus.UStrategy.LegacyInBand,
    )
    got = lazy.crit_sets_apply(
        [index], [target], oineus.ConflictStrategy.Avg,
    )
    expected = oracle.crit_sets_apply(
        [index], [target], oineus.ConflictStrategy.Avg,
    )

    got_pairs = sorted(zip(got.indices_array().tolist(), got.values_array().tolist()))
    expected_pairs = sorted(zip(
        expected.indices_array().tolist(), expected.values_array().tolist(),
    ))
    assert got_pairs == expected_pairs

    if direction.startswith("death"):
        decmp = lazy.homology_decomposition_ref()
        geom_dim = 1
    else:
        decmp = lazy.cohomology_decomposition_ref()
        geom_dim = 0
    assert decmp.negative_v_elz_in_dim(geom_dim)
    expects_u = direction in {"birth-decrease", "death-increase"}
    assert decmp.n_computed_u_rows == int(expects_u)
    assert decmp.n_valid_u_rows == 0


def test_zero_target_does_no_lazy_work():
    _, fil = circle_filtration()
    (_, death_idx), (_, death) = h1_pair(fil)
    opt = oin_diff.TopologyOptimizer(fil, n_threads=4)

    result = opt.crit_sets_apply_typed(
        [], [], [int(death_idx)], [float(death)],
        oineus.ConflictStrategy.Avg,
    )

    assert not opt.is_hom_built
    assert not opt.is_coh_built
    assert len(result.indices_array()) == 0


def test_selected_u_rows_are_exact_and_never_expand_to_full_dimension():
    _, fil = circle_filtration()
    (_, death_idx), (_, death) = h1_pair(fil)
    opt = oin_diff.TopologyOptimizer(fil, n_threads=4)
    opt.ensure_hom_reduced()
    hom = opt.homology_decomposition_ref()
    dim_size = hom.dim_last[2] - hom.dim_first[2] + 1

    # Repetition would have tripped the old density heuristic. The production
    # path deduplicates the one actual row and still performs one selected solve.
    repeated_rows = [int(death_idx)] * dim_size
    repeated_bounds = [float(death + 1.0)] * dim_size
    opt.ensure_has_u_hom(1, repeated_rows, repeated_bounds)

    assert hom.negative_v_elz_in_dim(2)
    assert hom.n_computed_u_rows == 1
    assert hom.n_valid_u_rows == 0
    assert not hom.has_full_matrix_u()
    with pytest.raises(RuntimeError, match="complete U matrix"):
        hom.u_as_csr()


def test_bounded_solve_of_every_row_does_not_claim_complete_u():
    fil = oineus.Filtration([
        oineus.Simplex([0], 0.0),
        oineus.Simplex([1], 0.1),
        oineus.Simplex([2], 0.2),
        oineus.Simplex([3], 0.3),
    ])
    dcmp = oineus.Decomposition(fil, dualize=False)
    dcmp.reduce(oineus.ReductionParams(
        compute_v=True, compute_u=False, use_clearing=False, n_threads=1,
    ))
    rows = list(range(fil.size()))
    dcmp.compute_partial_u_rows(
        fil, rows=rows, bounds=[-1.0] * fil.size(), dim=0, cmp="above",
    )

    assert dcmp.has_matrix_u()
    assert dcmp.n_computed_u_rows == fil.size()
    assert dcmp.n_valid_u_rows == 0
    assert not dcmp.has_full_matrix_u()


def test_partial_state_pickle_and_full_operations_are_guarded():
    _, fil = circle_filtration(n=30, seed=7)
    (_, death_idx), (_, death) = h1_pair(fil)
    opt = oin_diff.TopologyOptimizer(fil, n_threads=4)
    opt.crit_sets_apply_typed(
        [], [], [int(death_idx)], [float(death + 1.0)],
        oineus.ConflictStrategy.Avg,
    )
    hom = opt.homology_decomposition_ref()
    assert hom.negative_v_elz_in_dim(2)
    assert hom.n_computed_u_rows == 1
    assert hom.n_valid_u_rows == 0

    restored = pickle.loads(pickle.dumps(hom))
    assert restored.factorization_valid == hom.factorization_valid
    assert restored.negative_v_elz_in_dim(2)
    assert restored.n_computed_u_rows == hom.n_computed_u_rows
    assert restored.n_valid_u_rows == hom.n_valid_u_rows
    assert list(restored.u_row(int(death_idx))) == list(hom.u_row(int(death_idx)))

    if not hom.factorization_valid:
        with pytest.raises(RuntimeError, match="R = D V"):
            _ = hom.r_data
        with pytest.raises(RuntimeError, match="R = D V"):
            hom.compute_full_u_rows(fil.under_fil, dim=2)


def test_torch_backward_restores_only_target_dimension():
    points, fil = circle_filtration()
    diagrams = oin_diff.persistence_diagram(
        fil, gradient_method="crit-sets", dualize=True, n_threads=4,
    )
    h1 = diagrams.in_dimension(1)
    if h1.shape[0] == 0:
        pytest.skip("no H1 point")

    i = int(torch.argmax(h1[:, 1] - h1[:, 0]))
    target = h1[i, 1].detach() - 0.2 * (h1[i, 1] - h1[i, 0]).detach()
    ((h1[i, 1] - target) ** 2).backward()

    top = diagrams._top_opt
    hom = top.homology_decomposition_ref()
    assert hom.negative_v_elz_in_dim(2)
    assert not hom.negative_v_elz_in_dim(0)
    assert not hom.negative_v_elz_in_dim(1)
    assert hom.n_computed_u_rows == 0
    assert hom.n_valid_u_rows == 0
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


def test_reduce_all_recomputes_complete_u_after_selected_only_storage():
    fil = oineus.Filtration([
        oineus.Simplex([0], 0.0),
        oineus.Simplex([1], 0.1),
        oineus.Simplex([0, 1], 0.2),
    ])
    opt = oineus.TopologyOptimizer(fil, n_threads=4)
    opt.ensure_hom_reduced()
    hom = opt.homology_decomposition_ref()

    opt.ensure_has_u_hom(0, [2], [1.0])

    assert hom.factorization_valid
    assert hom.has_matrix_u()
    assert not hom.has_full_matrix_u()
    assert hom.n_computed_u_rows == 1
    assert hom.n_valid_u_rows == 0

    opt.reduce_all()

    coh = opt.cohomology_decomposition_ref()
    for decmp in (hom, coh):
        assert decmp.has_full_matrix_u()
        assert decmp.has_matrix_v()
        assert decmp.n_valid_u_rows == fil.size()
        assert decmp.n_elz_violators(n_threads=4) == 0
        assert decmp.timings.compute_u > 0.0
        uv = np.asarray((decmp.u_as_csr() @ decmp.v_as_csc()).todense()) % 2
        assert np.array_equal(uv, np.eye(fil.size()))


def test_reduce_all_recomputes_complete_non_elz_parallel_u():
    rng = np.random.default_rng(23)
    fil = oineus.freudenthal_filtration(
        np.ascontiguousarray(rng.random((10, 10))),
    )
    opt = oineus.TopologyOptimizer(fil, n_threads=4)
    opt.ensure_hom_built()
    hom = opt.homology_decomposition_ref()
    hom.reduce(oineus.ReductionParams(
        compute_u=True, use_clearing=True, n_threads=4,
    ))

    assert hom.has_full_matrix_u()
    assert hom.n_elz_violators(n_threads=4) > 0

    opt.reduce_all()

    assert hom.has_full_matrix_u()
    assert hom.has_matrix_v()
    assert hom.n_elz_violators(n_threads=4) == 0
    uv = np.asarray((hom.u_as_csr() @ hom.v_as_csc()).todense()) % 2
    assert np.array_equal(uv, np.eye(fil.size()))
