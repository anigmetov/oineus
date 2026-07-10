"""Tests for differentiable kernel/image/cokernel diagrams (oineus.diff.kicr).

Four layers:

1. Index-mapping unit test on a hand-traced complex (the second fixture of
   test_kicr.py / test_kicr_reference.py): pins down that every index-diagram
   entry of every family is a sorted id of the FULL filtration K -- in
   particular the kernel death cell, which lives in L, is identified by its
   K sorted id, not its L sorted id -- and that the gather backward scatters
   exactly onto those entries.
2. Value oracle: the diff forward values equal the finite points of the
   non-diff KICR diagrams, per family and dimension, on the hand-derived
   fixtures and on random VR complexes with a subcomplex; torch and jax,
   float64 and float32 routing.
3. Gradient oracle: central finite differences per family through the full
   points -> VR -> KICR pipeline (torch and jax), plus torch-vs-jax gradient
   equality on identical inputs.
4. Edge cases: empty family diagrams, K == L, disabled families,
   include_inf_points, input validation.

Skips gracefully when torch and/or jax are not installed; the pure-numpy
index-mapping tests always run.
"""

import numpy as np
import pytest

import oineus as oin
import oineus.diff as od
from oineus.diff import pd_core
from oineus._dtype import REAL_DTYPE

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

BACKENDS = [
    pytest.param("torch", marks=pytest.mark.skipif(not HAS_TORCH, reason="requires torch")),
    pytest.param("jax", marks=pytest.mark.skipif(not HAS_JAX, reason="requires jax")),
]

FAMILIES = ("kernel", "image", "cokernel")

# FD tolerances tied to the compiled Real, mirroring test_diff_grad.py
if REAL_DTYPE == np.float32:
    TORCH_DTYPE = torch.float32 if HAS_TORCH else None
    EPS = 1e-3
    ATOL = 1e-2
    RTOL = 1e-2
    GRAD_NONZERO_SQ = 1e-6
else:
    TORCH_DTYPE = torch.float64 if HAS_TORCH else None
    EPS = 1e-6
    ATOL = 1e-5
    RTOL = 1e-5
    GRAD_NONZERO_SQ = 1e-10


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

# hand-derived (K, L) pairs from test_kicr.py / test_kicr_reference.py
KNOWN = [
    (
        [[0, [0], 10], [1, [1], 50], [2, [2], 20], [3, [3], 50], [4, [4], 15], [5, [5], 12],
         [6, [0, 1], 50], [7, [1, 2], 60], [8, [2, 3], 70], [9, [3, 4], 80], [10, [0, 5], 30], [11, [4, 5], 20]],
        [[0, [0], 10], [1, [1], 50], [2, [2], 20], [3, [3], 50], [4, [4], 15],
         [5, [0, 1], 50], [6, [1, 2], 60], [7, [2, 3], 70], [8, [3, 4], 80]],
    ),
    (
        [[0, [0], 10], [1, [1], 30], [2, [2], 10], [3, [3], 0], [4, [0, 1], 30],
         [5, [1, 2], 30], [6, [0, 3], 10], [7, [2, 3], 10]],
        [[0, [0], 10.], [1, [1], 30], [2, [2], 10], [3, [0, 1], 30], [4, [1, 2], 30]],
    ),
    (
        [[0, [0], 0.0], [1, [1], 0.0], [2, [0, 1], 1.0]],
        [[0, [0], 0.0], [1, [1], 0.0]],
    ),
    (
        [[0, [0], 0.0], [1, [1], 0.0], [2, [2], 0.0], [3, [3], 0.0], [4, [2, 3], 0.5], [5, [0, 1], 1.0]],
        [[0, [0], 0.0], [1, [1], 0.0], [2, [2], 0.0], [3, [3], 0.0], [4, [2, 3], 0.5]],
    ),
]


def _known_filtrations(idx):
    K_list, L_list = KNOWN[idx]
    K = oin.list_to_filtration([list(c) for c in K_list])
    L = oin.list_to_filtration([list(c) for c in L_list])
    return K, L


def _np(t):
    if HAS_TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _tensor(x_np, backend, requires_grad=False):
    if backend == "torch":
        return torch.tensor(x_np, requires_grad=requires_grad)
    return jnp.asarray(x_np)


def _as_diff(fil, backend):
    """Wrap a plain filtration in a DiffFiltration whose values tensor is
    read off the filtration itself (so gathered values match the non-diff
    diagrams exactly)."""
    vals = np.array([fil.cell_value_by_sorted_id(i) for i in range(fil.size())],
                    dtype=np.float64)
    return od.DiffFiltration(fil, _tensor(vals, backend, requires_grad=(backend == "torch")))


def _nondiff_finite(kicr, family, dim):
    """Finite points of a non-diff KICR family diagram, insertion order."""
    arr = np.asarray(getattr(kicr, family + "_diagrams")().in_dimension(dim, as_numpy=True))
    arr = arr.reshape(-1, 2)
    return arr[np.isfinite(arr[:, 1])]


def _remove_top_uids(K, frac, seed):
    """uids of a coface-closed removal set: a random subset of top cells."""
    md = K.max_dim
    tops = [i for i in range(K.size()) if K.cell(i).dim == md]
    rng = np.random.default_rng(seed)
    rm = [t for t in tops if rng.random() < frac]
    if not rm:
        rm = [tops[-1]]
    return [K.cell(i).uid for i in rm]


def _star_removal_uids(K, seed):
    """uids of the closed star of the first vertex plus a fixed subset of
    triangles. Removing lower-dimensional cells (not just top cells) puts
    cells of every dimension into K - L, so all three families acquire
    finite points."""
    star = list(K.star_closure([0]))  # cell 0 is a vertex (dim-major order)
    tops = [i for i in range(K.size()) if K.cell(i).dim == 2]
    rng = np.random.default_rng(seed)
    rm = sorted(set(star) | {t for t in tops if rng.random() < 0.4})
    return [K.cell(i).uid for i in rm]


def _sub_by_uids(K, uids):
    """The subfiltration of K with the cells of the given uids removed.
    uids are combinatorial, hence stable under value perturbations -- the
    FD tests rebuild the same L from perturbed points."""
    return K.without_cells([K.sorted_id_by_uid(u) for u in uids])


def _vr_star_fixture(n=6, seed=1):
    pts_np = np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 2)).astype(REAL_DTYPE)
    K = oin.vr_filtration(pts_np, max_dim=2, max_diameter=10.0, packed=False)
    return pts_np, _star_removal_uids(K, seed)


def _family_loss(dgms, family, dims=(0, 1)):
    """Total squared persistence of one family over the given dims;
    framework-agnostic (works on torch tensors and jax arrays)."""
    fam = getattr(dgms, family)
    total = None
    for dim in dims:
        d = fam[dim]
        s = ((d[:, 1] - d[:, 0]) ** 2).sum()
        total = s if total is None else total + s
    return total


def _kicr_pipeline_loss(pts, family, removed_uids):
    """points -> diff VR -> KICR diagrams -> squared-persistence loss."""
    Kd = od.vr_filtration(pts, max_dim=2, max_diameter=10.0, n_threads=1)
    L = _sub_by_uids(Kd.under_fil, removed_uids)
    return _family_loss(od.kicr_diagrams(Kd, L), family)


def _fd_grad(f, x_np, eps=EPS):
    """Central-difference gradient of scalar function ``f(x_np) -> float``."""
    g = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        base = x_np.copy()
        base[idx] += eps
        hi = float(f(base))
        base[idx] -= 2 * eps
        lo = float(f(base))
        g[idx] = (hi - lo) / (2 * eps)
    return g


def _assert_grad_nonzero(*grads):
    for g in grads:
        assert float(np.sum(np.asarray(g) ** 2)) > GRAD_NONZERO_SQ, \
            "gradient is (numerically) zero -- test would pass trivially"


# ---------------------------------------------------------------------------
# 1. index mapping: hand-traced complex (no framework needed)
# ---------------------------------------------------------------------------

def test_index_mapping_hand_traced():
    """Pin the (family, birth, death) -> K sorted id mapping on KNOWN[1].

    K: square 0-1-2-3 (vertices 10, 30, 10, 0; edges 01/12 at 30, 03/23
    at 10); L: the path 0-1-2. Hand trace (verified against the CEHM
    read-off in kernel.h):

    - kernel dim 0: the class [v0]-[v2] of H0(L) dies in H0(K) when the
      path 0-3-2 completes; birth cell = edge [2,3] (the later of [0,3],
      [2,3] in K order, a K-only cell), death cell = edge [1,2] (an L
      cell; enters the diagram as sorted_L_to_sorted_K_[tau]).
    - cokernel dim 0: vertex [3] (K-only, value 0) is born in coker and
      dies at edge [0,3] (value 10).
    - image dim 0: only zero-persistence pairs ([2],[2,3]) and
      ([1],[0,1]), dropped by default.
    """
    K, L = _known_filtrations(1)

    # oineus sorts dimension-major, by value within dimension; pin the order
    order = {tuple(K.cell(i).vertices): i for i in range(K.size())}
    assert order == {(3,): 0, (0,): 1, (2,): 2, (1,): 3,
                     (0, 3): 4, (2, 3): 5, (0, 1): 6, (1, 2): 7}

    fwd = pd_core.kicr_forward(K, L, kernel=True, image=True, cokernel=True,
                               include_zero_persistence=False, n_threads=1)

    assert fwd.index_dgms["kernel"][0].tolist() == [[5, 7]]
    assert fwd.index_dgms["cokernel"][0].tolist() == [[0, 4]]
    assert fwd.index_dgms["image"][0].tolist() == []

    # the kernel death cell [1,2] lives in L: its index is its K sorted id
    # (7), not its L sorted id (4)
    assert L.sorted_id_by_uid(K.cell(7).uid) == 4
    # the kernel birth cell [2,3] is in K only
    L_uids = {L.cell(i).uid for i in range(L.size())}
    assert K.cell(5).uid not in L_uids

    # values follow the indices through K
    vals = [K.cell_value_by_sorted_id(i) for i in range(K.size())]
    assert (vals[5], vals[7]) == (10.0, 30.0)   # kernel point
    assert (vals[0], vals[4]) == (0.0, 10.0)    # cokernel point


def test_index_mapping_zero_persistence():
    K, L = _known_filtrations(1)
    fwd = pd_core.kicr_forward(K, L, kernel=True, image=True, cokernel=True,
                               include_zero_persistence=True, n_threads=1)
    # the two zero-persistence image pairs: ([2], [2,3]) at 10 and
    # ([1], [0,1]) at 30, in insertion (K) order
    assert fwd.index_dgms["image"][0].tolist() == [[2, 5], [3, 6]]
    # kernel/cokernel have no zero-persistence pairs here
    assert fwd.index_dgms["kernel"][0].tolist() == [[5, 7]]
    assert fwd.index_dgms["cokernel"][0].tolist() == [[0, 4]]


@pytest.mark.parametrize("backend", BACKENDS)
def test_hand_traced_values_and_backward(backend):
    """Forward values on KNOWN[1] plus the exact scatter of the backward:
    d(sum(kernel dgm))/d(values) is +1 exactly at the two kernel index
    entries (K sorted ids 5 and 7)."""
    K, L = _known_filtrations(1)
    vals_np = np.array([K.cell_value_by_sorted_id(i) for i in range(K.size())])

    # forward values, checked on concrete (non-traced) tensors
    dgms = od.kicr_diagrams(_as_diff(K, backend), L)
    np.testing.assert_array_equal(_np(dgms.kernel[0]), [[10.0, 30.0]])
    np.testing.assert_array_equal(_np(dgms.cokernel[0]), [[0.0, 10.0]])
    assert _np(dgms.image[0]).shape == (0, 2)

    def build_loss(v):
        return od.kicr_diagrams(od.DiffFiltration(K, v), L).kernel[0].sum()

    if backend == "torch":
        v = torch.tensor(vals_np, requires_grad=True)
        build_loss(v).backward()
        grad = v.grad.detach().numpy()
    else:
        grad = np.asarray(jax.grad(build_loss)(jnp.asarray(vals_np)))

    expected = np.zeros_like(vals_np)
    expected[5] = expected[7] = 1.0
    np.testing.assert_array_equal(grad, expected)


# ---------------------------------------------------------------------------
# 2. value oracle: diff forward == non-diff KICR diagrams (finite points)
# ---------------------------------------------------------------------------

def _assert_matches_nondiff(diff_dgms, kicr, max_dim,
                            families=FAMILIES):
    for family in families:
        fam = getattr(diff_dgms, family)
        for dim in range(max_dim):
            got = _np(fam[dim])
            exp = _nondiff_finite(kicr, family, dim)
            assert got.shape == exp.shape, f"{family} dim {dim}"
            np.testing.assert_allclose(got, exp, atol=0, rtol=0,
                                       err_msg=f"{family} dim {dim}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("idx", range(len(KNOWN)))
def test_values_match_nondiff_on_known(backend, idx):
    K, L = _known_filtrations(idx)
    diff_dgms = od.kicr_diagrams(_as_diff(K, backend), L)
    kicr = oin.compute_kernel_image_cokernel_reduction(K, L)
    _assert_matches_nondiff(diff_dgms, kicr, K.max_dim)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("include_zero", [False, True])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_values_match_nondiff_on_random_vr(backend, seed, include_zero, n_threads):
    pts = np.random.default_rng(seed).random((10, 3))
    K = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=False)
    L = _sub_by_uids(K, _remove_top_uids(K, 0.5, seed))

    diff_dgms = od.kicr_diagrams(_as_diff(K, backend), L,
                                 include_zero_persistence=include_zero,
                                 n_threads=n_threads)

    params = oin.KICRParams(kernel=True, image=True, cokernel=True)
    params.include_zero_persistence = include_zero
    kicr = oin.compute_kernel_image_cokernel_reduction(K, L, params)
    _assert_matches_nondiff(diff_dgms, kicr, K.max_dim)

    # guard against a trivially-empty comparison
    n_points = sum(len(getattr(diff_dgms, fam).index_diagram_in_dimension(d))
                   for fam in FAMILIES for d in range(K.max_dim))
    assert n_points > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_values_match_nondiff_star_removal(backend):
    """L missing cells of every dimension: all three families nonempty."""
    pts_np, uids = _vr_star_fixture(n=7, seed=1)
    K = oin.vr_filtration(pts_np, max_dim=2, max_diameter=10.0, packed=False)
    L = _sub_by_uids(K, uids)
    diff_dgms = od.kicr_diagrams(_as_diff(K, backend), L)
    kicr = oin.compute_kernel_image_cokernel_reduction(K, L)
    _assert_matches_nondiff(diff_dgms, kicr, K.max_dim)
    for fam in FAMILIES:
        assert sum(len(getattr(diff_dgms, fam).index_diagram_in_dimension(d))
                   for d in range(K.max_dim)) > 0, f"{fam} unexpectedly empty"


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_dtypes_and_values(backend):
    """float32 points build a float32 filtration (routed to the _f32
    backend when compiled in); diagrams keep the tensor dtype and agree
    with the non-diff KICR diagrams of the same filtration."""
    pts32 = np.random.default_rng(7).uniform(-1.0, 1.0, (7, 2)).astype(np.float32)
    pts = _tensor(pts32, backend, requires_grad=True)
    Kd = od.vr_filtration(pts, max_dim=2, max_diameter=10.0, n_threads=1)
    assert _np(Kd.values).dtype == np.float32
    L = _sub_by_uids(Kd.under_fil, _star_removal_uids(Kd.under_fil, 1))

    diff_dgms = od.kicr_diagrams(Kd, L)
    kicr = oin.compute_kernel_image_cokernel_reduction(Kd.under_fil, L)
    for family in FAMILIES:
        fam = getattr(diff_dgms, family)
        for dim in range(Kd.under_fil.max_dim):
            got = _np(fam[dim])
            assert got.dtype == np.float32
            exp = _nondiff_finite(kicr, family, dim)
            assert got.shape == exp.shape, f"{family} dim {dim}"
            # gathered tensor values differ from the C++ filtration values
            # by the diff VR eps regularization (sqrt(d^2 + 1e-6)) and
            # float32 round-off
            np.testing.assert_allclose(got, exp, atol=1e-3,
                                       err_msg=f"{family} dim {dim}")


# ---------------------------------------------------------------------------
# 3. gradient oracle: finite differences and torch-vs-jax
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
@pytest.mark.parametrize("family", FAMILIES)
def test_torch_gradient_matches_finite_difference(family):
    pts_np, uids = _vr_star_fixture(n=6, seed=1)
    pts = torch.tensor(pts_np, dtype=TORCH_DTYPE, requires_grad=True)
    _kicr_pipeline_loss(pts, family, uids).backward()
    grad_auto = pts.grad.detach().numpy()
    _assert_grad_nonzero(grad_auto)

    def f(x):
        t = torch.tensor(x, dtype=TORCH_DTYPE)
        return float(_kicr_pipeline_loss(t, family, uids))

    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not HAS_JAX, reason="requires jax")
@pytest.mark.parametrize("family", FAMILIES)
def test_jax_gradient_matches_finite_difference(family):
    pts_np, uids = _vr_star_fixture(n=6, seed=1)
    loss = lambda x: _kicr_pipeline_loss(x, family, uids)
    grad_auto = np.asarray(jax.grad(loss)(jnp.asarray(pts_np)))
    _assert_grad_nonzero(grad_auto)

    def f(x):
        return float(loss(jnp.asarray(x)))

    np.testing.assert_allclose(grad_auto, _fd_grad(f, pts_np), atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not (HAS_TORCH and HAS_JAX), reason="requires torch and jax")
@pytest.mark.parametrize("family", FAMILIES)
def test_torch_vs_jax_gradients_equal(family):
    pts_np, uids = _vr_star_fixture(n=6, seed=1)
    pts_np = pts_np.astype(np.float64)

    pts_t = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
    _kicr_pipeline_loss(pts_t, family, uids).backward()
    g_torch = pts_t.grad.detach().numpy()

    g_jax = np.asarray(jax.grad(
        lambda x: _kicr_pipeline_loss(x, family, uids))(jnp.asarray(pts_np)))

    _assert_grad_nonzero(g_torch)
    np.testing.assert_allclose(g_jax, g_torch, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 4. edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_k_equals_l(backend):
    """K == L: kernel and cokernel have no finite points; image equals the
    finite part of the non-diff image diagram; backward runs fine through
    a loss that touches the empty families."""
    pts = np.random.default_rng(5).random((8, 2))
    K = oin.vr_filtration(pts, max_dim=2, max_diameter=10.0, packed=False)
    Kd = _as_diff(K, backend)
    dgms = od.kicr_diagrams(Kd, K)

    for fam in ("kernel", "cokernel"):
        for dim in range(K.max_dim):
            assert _np(getattr(dgms, fam)[dim]).shape == (0, 2), f"{fam} dim {dim}"

    kicr = oin.compute_kernel_image_cokernel_reduction(K, K)
    _assert_matches_nondiff(dgms, kicr, K.max_dim, families=("image",))
    assert len(dgms.image.index_diagram_in_dimension(0)) > 0

    vals_np = _np(Kd.values)

    def build_loss(v):
        d = od.kicr_diagrams(od.DiffFiltration(K, v), K)
        # touch the empty kernel: contributes exactly zero
        return _family_loss(d, "image", dims=(0,)) + d.kernel[0].sum()

    if backend == "torch":
        v = torch.tensor(vals_np, requires_grad=True)
        build_loss(v).backward()
        grad = v.grad.detach().numpy()
    else:
        grad = np.asarray(jax.grad(build_loss)(jnp.asarray(vals_np)))
    _assert_grad_nonzero(grad)


@pytest.mark.parametrize("backend", BACKENDS)
def test_disabled_family_raises(backend):
    K, L = _known_filtrations(1)
    dgms = od.kicr_diagrams(_as_diff(K, backend), L, kernel=False)
    assert dgms.families == ("image", "cokernel")
    with pytest.raises(RuntimeError, match="kernel"):
        dgms.kernel
    # the other families still work
    assert _np(dgms.image[0]).shape == (0, 2)
    np.testing.assert_array_equal(_np(dgms.cokernel[0]), [[0.0, 10.0]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_families_disabled_raises(backend):
    K, L = _known_filtrations(1)
    with pytest.raises(ValueError, match="at least one"):
        od.kicr_diagrams(_as_diff(K, backend), L,
                         kernel=False, image=False, cokernel=False)


@pytest.mark.parametrize("backend", BACKENDS)
def test_include_inf_points_raises(backend):
    K, L = _known_filtrations(1)
    with pytest.raises(NotImplementedError):
        od.kicr_diagrams(_as_diff(K, backend), L, include_inf_points=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_l_as_diff_filtration_is_unwrapped(backend):
    """Passing L as a DiffFiltration uses only its underlying filtration."""
    K, L = _known_filtrations(1)
    L_diff = _as_diff(L, backend)
    dgms = od.kicr_diagrams(_as_diff(K, backend), L_diff)
    np.testing.assert_array_equal(_np(dgms.kernel[0]), [[10.0, 30.0]])


def test_plain_filtration_as_k_raises():
    K, L = _known_filtrations(1)
    with pytest.raises(TypeError, match="DiffFiltration"):
        od.kicr_diagrams(K, L)


def test_numpy_values_raise():
    K, L = _known_filtrations(1)
    vals = np.array([K.cell_value_by_sorted_id(i) for i in range(K.size())])
    with pytest.raises(TypeError, match="torch.Tensor or a jax array"):
        od.kicr_diagrams(od.DiffFiltration(K, vals), L)


@pytest.mark.parametrize("backend", BACKENDS)
def test_index_diagram_accessor_returns_copy(backend):
    """Mutating the array returned by index_diagram_in_dimension must not
    corrupt the internal pairing (which the torch backward scatters
    through)."""
    K, L = _known_filtrations(1)
    dgms = od.kicr_diagrams(_as_diff(K, backend), L)
    idx = dgms.kernel.index_diagram_in_dimension(0)
    idx[:] = 0
    assert dgms.kernel.index_diagram_in_dimension(0).tolist() == [[5, 7]]


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_gather_diagram_empty_torch_backprops():
    """An empty diagram is a valid zero loss: the empty gather must keep
    grad_fn so backward yields zero grads instead of raising."""
    from oineus.diff.kicr import gather_diagram

    v = torch.rand(5, dtype=torch.float64, requires_grad=True)
    dgm = gather_diagram(v, np.zeros((0, 2), dtype=np.int64), "torch")
    assert tuple(dgm.shape) == (0, 2)
    assert dgm.dtype == v.dtype

    dgm.sum().backward()
    assert v.grad is not None
    assert (v.grad == 0).all()

    # (0, 3) mixup triples keep their shape too
    assert tuple(gather_diagram(v, np.zeros((0, 3), dtype=np.int64), "torch").shape) == (0, 3)


@pytest.mark.skipif(not HAS_JAX, reason="requires jax")
def test_gather_diagram_empty_jax_backprops():
    from oineus.diff.kicr import gather_diagram

    idx = np.zeros((0, 2), dtype=np.int64)
    vals = jnp.arange(5.0)
    dgm = gather_diagram(vals, idx, "jax")
    assert tuple(dgm.shape) == (0, 2)
    assert dgm.dtype == vals.dtype

    grad = jax.grad(lambda v: gather_diagram(v, idx, "jax").sum())(vals)
    assert (np.asarray(grad) == 0).all()


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_mixup_empty_input_torch_backprops():
    # mixup_barcodes routes empty inputs through the gather_diagram empty
    # path; the zero loss must backprop to a (0, d) grad, not raise
    A = torch.zeros((0, 2), dtype=torch.float64, requires_grad=True)
    mb = od.mixup_barcodes(A, None, max_dim=1)
    loss = sum(mb[d].sum() for d in range(2))
    loss.backward()
    assert tuple(A.grad.shape) == (0, 2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_repr_is_informative(backend):
    K, L = _known_filtrations(1)
    dgms = od.kicr_diagrams(_as_diff(K, backend), L)
    r = repr(dgms)
    assert "kernel" in r and "image" in r and "cokernel" in r
    assert "KICRFamilyDiagrams" in repr(dgms.kernel)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
