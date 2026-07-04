"""Tests for oineus.diff.mixup_barcodes (differentiable mixup barcodes).

Checks, per framework (torch and jax, float64):

* value parity with the non-differentiable oin.mixup_barcodes -- same
  triples and statistics up to the O(eps) shift of the differentiable VR
  values (tests use a tiny eps so the tolerance is tight), and identical
  index triples;
* gradient correctness against central finite differences, through
  triples of both degrees and through every statistic, with gradients
  reaching both A and B;
* torch and jax gradients agree;
* edge cases: empty B (zero mixup, no gradient into d' beyond d), empty
  A, empty degrees (statistics are differentiable zero constants),
  input-type guards.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import oineus as oin
import oineus.diff as od

EPS = 1e-12  # VR smoothing; tiny so diff values match non-diff closely


def make_clouds(seed=7, n_A=7, n_B=3, d=2):
    rng = np.random.default_rng(seed)
    return rng.random((n_A, d)), rng.random((n_B, d))


def sorted_by_rows(a):
    # row-lexicographic sort, dtype-preserving (works for the float triples
    # and the int64 index triples alike)
    a = np.asarray(a)
    if a.size == 0:
        return a
    return a[np.lexsort(a.T[::-1])]


def total_loss(dmb):
    return (dmb.total_mixup(0) + dmb.total_mixup(1)
            + dmb.total_mixup_percentage(0) + dmb.total_mixup_percentage(1)
            + dmb.mean_mixup_percentage(0) + dmb.mean_mixup_percentage(1))


def loss_value_np(A_np, B_np):
    a = torch.tensor(A_np, dtype=torch.float64)
    b = torch.tensor(B_np, dtype=torch.float64) if B_np is not None else None
    return float(total_loss(od.mixup_barcodes(a, b, max_dim=1, eps=EPS)))


def test_values_match_nondiff_torch():
    A_np, B_np = make_clouds()
    dmb = od.mixup_barcodes(torch.tensor(A_np, dtype=torch.float64),
                            torch.tensor(B_np, dtype=torch.float64),
                            max_dim=1, eps=EPS)
    mb = oin.mixup_barcodes(A_np, B_np, max_dim=1)
    for dim in (0, 1):
        t = dmb[dim].detach().numpy()
        nt = mb.in_dimension(dim)
        # both are gathers of the same cells' values; the diff values carry
        # the sqrt(dist^2 + eps) shift, negligible for eps = 1e-12
        assert t.shape == nt.shape
        assert np.allclose(sorted_by_rows(t), sorted_by_rows(nt), atol=1e-5)
        assert np.array_equal(sorted_by_rows(dmb.index_triples_in_dimension(dim)),
                              sorted_by_rows(mb.index_triples_in_dimension(dim)))
        assert float(dmb.total_mixup(dim)) == pytest.approx(mb.total_mixup(dim), abs=1e-5)
        assert float(dmb.total_mixup_percentage(dim)) == pytest.approx(
            mb.total_mixup_percentage(dim), abs=1e-4)
        assert float(dmb.mean_mixup_percentage(dim)) == pytest.approx(
            mb.mean_mixup_percentage(dim), abs=1e-4)


def test_torch_gradients_match_finite_differences():
    A_np, B_np = make_clouds()
    A = torch.tensor(A_np, requires_grad=True, dtype=torch.float64)
    B = torch.tensor(B_np, requires_grad=True, dtype=torch.float64)
    loss = total_loss(od.mixup_barcodes(A, B, max_dim=1, eps=EPS))
    loss.backward()
    assert A.grad.abs().sum() > 0
    assert B.grad.abs().sum() > 0  # image deaths depend on B

    h = 1e-6
    for arr, grad, is_A in ((A_np, A.grad.numpy(), True), (B_np, B.grad.numpy(), False)):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                p, m = arr.copy(), arr.copy()
                p[i, j] += h
                m[i, j] -= h
                fp = loss_value_np(p if is_A else A_np, p if not is_A else B_np)
                fm = loss_value_np(m if is_A else A_np, m if not is_A else B_np)
                fd = (fp - fm) / (2 * h)
                assert grad[i, j] == pytest.approx(fd, abs=1e-6), (is_A, i, j)


def test_torch_and_jax_agree():
    A_np, B_np = make_clouds(seed=17)
    A = torch.tensor(A_np, requires_grad=True, dtype=torch.float64)
    B = torch.tensor(B_np, requires_grad=True, dtype=torch.float64)
    tval = total_loss(od.mixup_barcodes(A, B, max_dim=1, eps=EPS))
    tval.backward()

    def f(a, b):
        return total_loss(od.mixup_barcodes(a, b, max_dim=1, eps=EPS))

    ja, jb = jnp.asarray(A_np), jnp.asarray(B_np)
    jval = f(ja, jb)
    ga, gb = jax.grad(f, argnums=(0, 1))(ja, jb)
    assert float(jval) == pytest.approx(float(tval.detach()), abs=1e-10)
    assert np.allclose(A.grad.numpy(), np.asarray(ga), atol=1e-9)
    assert np.allclose(B.grad.numpy(), np.asarray(gb), atol=1e-9)


def test_jax_triples_match_nondiff():
    A_np, B_np = make_clouds(seed=23)
    dmb = od.mixup_barcodes(jnp.asarray(A_np), jnp.asarray(B_np), max_dim=1, eps=EPS)
    mb = oin.mixup_barcodes(A_np, B_np, max_dim=1)
    for dim in (0, 1):
        assert np.allclose(sorted_by_rows(np.asarray(dmb[dim])),
                           sorted_by_rows(mb.in_dimension(dim)), atol=1e-5)


def test_empty_B():
    A_np, _ = make_clouds()
    A = torch.tensor(A_np, requires_grad=True, dtype=torch.float64)
    dmb = od.mixup_barcodes(A, None, max_dim=1, eps=EPS)
    for dim in (0, 1):
        t = dmb[dim]
        assert torch.equal(t[:, 1], t[:, 2])  # d' == d
        assert float(dmb.total_mixup(dim)) == 0.0
    # total persistence still differentiates through dgm(A)
    dmb.total_persistence(0).backward()
    assert A.grad.abs().sum() > 0

    B0 = torch.empty((0, 2), dtype=torch.float64)
    dmb0 = od.mixup_barcodes(torch.tensor(A_np, dtype=torch.float64), B0, max_dim=1, eps=EPS)
    assert float(dmb0.total_mixup(0)) == 0.0


def test_empty_A_and_empty_degree():
    B = torch.rand((3, 2), dtype=torch.float64)
    dmb = od.mixup_barcodes(torch.empty((0, 2), dtype=torch.float64), B, max_dim=1, eps=EPS)
    for dim in (0, 1):
        assert dmb[dim].shape == (0, 3)
        z = dmb.total_mixup(dim)
        assert isinstance(z, torch.Tensor) and float(z) == 0.0
        assert float(dmb.mean_mixup_percentage(dim)) == 0.0

    # non-empty A, but an empty degree-2 barcode: statistics are zero constants
    A_np, B_np = make_clouds(seed=3, n_A=5, n_B=2)
    dmb2 = od.mixup_barcodes(torch.tensor(A_np, dtype=torch.float64),
                             torch.tensor(B_np, dtype=torch.float64), max_dim=2, eps=EPS)
    assert dmb2[2].shape == (0, 3)
    assert float(dmb2.total_mixup(2)) == 0.0
    assert float(dmb2.mean_mixup_percentage(2)) == 0.0


def test_float32_inputs():
    # torch's default dtype: the whole pipeline (K, L, KICR) must run in
    # float32 and gradients must flow
    A_np, B_np = make_clouds(seed=31)
    A = torch.tensor(A_np, dtype=torch.float32, requires_grad=True)
    B = torch.tensor(B_np, dtype=torch.float32, requires_grad=True)
    dmb = od.mixup_barcodes(A, B, max_dim=1)
    assert dmb[0].dtype == torch.float32
    loss = dmb.total_mixup(0) + dmb.total_mixup(1)
    loss.backward()
    assert A.grad.abs().sum() > 0 and B.grad.abs().sum() > 0
    mb = oin.mixup_barcodes(A_np, B_np, max_dim=1)
    expected = mb.total_mixup(0) + mb.total_mixup(1)
    assert float(loss.detach()) == pytest.approx(expected, abs=5e-3)


def test_input_guards():
    A_np, B_np = make_clouds()
    with pytest.raises(TypeError):
        od.mixup_barcodes(A_np, B_np)  # numpy input: use oin.mixup_barcodes
    with pytest.raises(TypeError):
        od.mixup_barcodes(torch.tensor(A_np), jnp.asarray(B_np))  # mixed frameworks
    with pytest.raises(KeyError):
        od.mixup_barcodes(torch.tensor(A_np, dtype=torch.float64),
                          torch.tensor(B_np, dtype=torch.float64),
                          max_dim=0, eps=EPS).in_dimension(1)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
