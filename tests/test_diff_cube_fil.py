"""Tests for oineus.diff.cube_filtration and its interaction with TopologyOptimizer."""
import numpy as np
import pytest

import oineus as oin
import oineus.diff as od

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _random_grid(rng, shape):
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float64)


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 3)])
@pytest.mark.parametrize("values_on", ["vertices", "cells"])
@pytest.mark.parametrize("negate", [False, True])
def test_values_match_under_fil(shape, values_on, negate):
    """diff.values[i] must equal under_fil.cube_value_by_sorted_id(i) for every cube."""
    data = _random_grid(np.random.default_rng(0), shape)
    df = od.cube_filtration(data, negate=negate, values_on=values_on, max_dim=len(shape))
    for i in range(df.size()):
        assert df.under_fil.cube_value_by_sorted_id(i) == pytest.approx(float(df.values[i]))


@pytest.mark.parametrize("shape", [(7,), (4, 5), (3, 4, 3)])
@pytest.mark.parametrize("values_on", ["vertices", "cells"])
@pytest.mark.parametrize("negate", [False, True])
def test_fil_equivalent_to_non_diff(shape, values_on, negate):
    """The underlying filtration is the same (value sequence) as the non-diff cube_filtration."""
    data = _random_grid(np.random.default_rng(1), shape)
    fil_nd = oin.cube_filtration(data, negate=negate, values_on=values_on, max_dim=len(shape))
    df = od.cube_filtration(data, negate=negate, values_on=values_on, max_dim=len(shape))

    assert fil_nd.size() == df.size()
    nd_values = sorted(fil_nd.cube_value_by_sorted_id(i) for i in range(fil_nd.size()))
    df_values = sorted(float(df.values[i]) for i in range(df.size()))
    for a, b in zip(nd_values, df_values):
        assert a == pytest.approx(b)


def test_wrap_rejected():
    data = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(RuntimeError, match="wrap"):
        od.cube_filtration(data, wrap=True)


def test_unsupported_ndim_rejected():
    data = np.zeros((2, 2, 2, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match="ndim"):
        od.cube_filtration(data)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_autograd_sum_of_values():
    """Summing all cube values should produce gradients equal to per-element critical counts."""
    data = torch.tensor(_random_grid(np.random.default_rng(2), (4, 4)), requires_grad=True)
    df = od.cube_filtration(data, max_dim=2)
    loss = df.values.sum()
    loss.backward()

    # Sanity: total gradient == number of cubes (each cube contributes 1 to exactly one data element)
    assert float(data.grad.sum()) == pytest.approx(df.size())


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_autograd_finite_difference():
    """Finite-difference check against autograd on a small 2D grid."""
    rng = np.random.default_rng(3)
    data_np = _random_grid(rng, (3, 3))
    data = torch.tensor(data_np, requires_grad=True)
    df = od.cube_filtration(data, max_dim=2)
    loss = (df.values ** 2).sum()
    loss.backward()
    grad_auto = data.grad.clone().detach().numpy()

    eps = 1e-6
    grad_fd = np.zeros_like(data_np)
    for idx in np.ndindex(data_np.shape):
        base = data_np.copy()
        base[idx] += eps
        hi = float((od.cube_filtration(base, max_dim=2).values ** 2).sum())
        base[idx] -= 2 * eps
        lo = float((od.cube_filtration(base, max_dim=2).values ** 2).sum())
        grad_fd[idx] = (hi - lo) / (2 * eps)

    np.testing.assert_allclose(grad_auto, grad_fd, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_topology_optimizer_dispatch(ndim):
    """TopologyOptimizer must pick the correct cube-specific C++ class by ndim."""
    shape = tuple(3 + i for i in range(ndim))
    data = torch.tensor(_random_grid(np.random.default_rng(4), shape), requires_grad=True)
    df = od.cube_filtration(data, max_dim=ndim)
    opt = od.TopologyOptimizer(df)
    expected_cls_name = f"TopologyOptimizerCube_{ndim}D"
    assert type(opt.under_opt).__name__ == expected_cls_name

    dgms = opt.compute_diagram(include_inf_points=True)
    assert dgms.in_dimension(0).shape[1] == 2


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_topology_optimizer_simplify_runs():
    """simplify(epsilon) should run end-to-end on a cubical filtration."""
    data = torch.tensor(_random_grid(np.random.default_rng(5), (5, 5)), requires_grad=True)
    df = od.cube_filtration(data, max_dim=2)
    opt = od.TopologyOptimizer(df)
    iv = opt.simplify(epsilon=0.1, strategy=oin.DenoiseStrategy.BirthBirth, dim=0)
    # IndicesValues is a (indices, values) pair; lengths must agree.
    assert len(list(iv[0])) == len(list(iv[1]))
