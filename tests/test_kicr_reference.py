"""Ground-truth tests for kernel/image/cokernel persistence (KICR).

oineus computes KICR with the optimized, templated KerImCokReduced (kernel.h). The
only prior coverage was four hand-derived examples in test_kicr.py and "slim/packed
== fat" parity tests -- nothing validated the fat path itself beyond those four, and
the cube path (which has no fat counterpart, since cube_filtration always returns the
slim cube) had only a smoke test.

This file adds an independent oracle: a deliberately naive, dense Z/2 reference
implementation of the Cohen-Steiner-Edelsbrunner-Harer-Morozov algorithm
("Persistent Homology for Kernels, Images, and Cokernels", SODA 2009). The reference
shares no code with kernel.h -- it does plain column reduction on Python sets and
reads the diagrams off with the paper's case rules -- so it catches implementation
bugs (indexing, the image/kernel/cokernel matrix juggling, the reductions, clearing,
the parallel paths) in the optimized version. Its correctness is human-auditable: the
reduction is textbook, and each KICR step is a short transcription of the paper.

It is first BOOTSTRAPPED against the four hand-derived diagrams in test_kicr.py (the
original author's ground truth); once it reproduces those, it serves as the oracle for
new fat examples, the cube path, and the slim/packed paths. The boundary matrices and
the cell metadata (dim, value, uid, L<->K correspondence) are taken from oineus -- the
boundary matrix and the filtration sort order are validated elsewhere against
dionysus/gudhi -- so what is under test here is purely the KICR algorithm.
"""
import math

import numpy as np
import pytest

import oineus as oin

INF = math.inf


# ---------------------------------------------------------------------------
# Naive dense Z/2 primitives (foolproof; columns are sets of row indices)
# ---------------------------------------------------------------------------

def _reduce(columns):
    """Standard left-to-right Z/2 column reduction. Returns (R, V) with R = D V.

    columns: list of iterables of row indices. R[j], V[j] are sets of row indices;
    Z/2 column addition is set symmetric difference. low(col) = max(col).
    """
    R = [set(c) for c in columns]
    V = [{j} for j in range(len(columns))]
    low_to_col = {}
    for j in range(len(columns)):
        while R[j]:
            lo = max(R[j])
            if lo in low_to_col:
                k = low_to_col[lo]
                R[j] ^= R[k]
                V[j] ^= V[k]
            else:
                low_to_col[lo] = j
                break
    return R, V


def _low(col):
    return max(col) if col else -1


def _ordinary_diagram(R, dims, vals, infinity):
    """Ordinary persistence read-off from a reduced boundary matrix R.

    A column j is positive iff R[j] is empty (gives birth); negative columns pair
    (low(R[j]), j). Unpaired positive columns are essential.
    """
    out = {}
    paired_birth = set()
    for j in range(len(R)):
        if R[j]:
            i = _low(R[j])
            paired_birth.add(i)
            b, d, dim = vals[i], vals[j], dims[i]
            if b != d:
                out.setdefault(dim, []).append((b, d))
    for i in range(len(R)):
        if not R[i] and i not in paired_birth:
            out.setdefault(dims[i], []).append((vals[i], infinity))
    return out


# ---------------------------------------------------------------------------
# Naive reference KICR (CEHM algorithm; independent of kernel.h)
# ---------------------------------------------------------------------------

def kicr_reference(K, L, include_zero_persistence=False):
    """Reference kernel/image/cokernel diagrams for the inclusion L -> K.

    K, L: oineus filtrations (any cell encoding) with L a subcomplex of K. Returns a
    dict family -> {dim -> sorted list of (birth, death)} for the five families
    'domain', 'codomain', 'kernel', 'image', 'cokernel', matching oineus's
    conventions (birth/death values from K for ker/im/cok; L for domain; K for
    codomain; +inf for essential; zero-persistence finite points dropped by default).
    """
    m, n = K.size(), L.size()
    DK = K.boundary_matrix()
    DL = L.boundary_matrix()
    dimK = [K.cell(i).dim for i in range(m)]
    valK = [K.cell_value_by_sorted_id(i) for i in range(m)]
    dimL = [L.cell(i).dim for i in range(n)]
    valL = [L.cell_value_by_sorted_id(i) for i in range(n)]
    inf = K.infinity()

    # L <-> K correspondence by combinatorial uid (the E.1 translation handles slim/packed)
    sL2K = [K.sorted_id_by_uid(L.cell(i).uid) for i in range(n)]
    sK2L = [None] * m
    for i in range(n):
        sK2L[sL2K[i]] = i
    in_L = [sK2L[i] is not None for i in range(m)]

    # Step 1: reduce the two boundary matrices
    R_f, V_f = _reduce(DK)
    R_g, V_g = _reduce(DL)

    # the "L first, then K-L" row order (both blocks keep the K filtration order)
    new_to_old = sorted(range(m), key=lambda i: (0 if in_L[i] else 1, i))
    old_to_new = [0] * m
    for r, o in enumerate(new_to_old):
        old_to_new[o] = r

    # Step 2: image matrix = D_f with rows reindexed to the new order, columns unchanged
    D_im = [[old_to_new[x] for x in col] for col in DK]
    R_im, V_im = _reduce(D_im)

    # positive in R_f iff positive in R_im (paper Observation (ii))
    def positive(i):
        return not R_im[i]

    # Step 3: kernel matrix = the cycle columns of V_im (positive cells), rows reindexed
    ker_cols = []
    K_to_ker_col = [None] * m
    for i in range(m):
        if positive(i):
            K_to_ker_col[i] = len(ker_cols)
            ker_cols.append([old_to_new[x] for x in V_im[i]])
    R_ker, _ = _reduce(ker_cols)

    # Step 4: cokernel matrix = D_f with each L-cycle column replaced by V_g (reindexed L->K)
    D_cok = [set(c) for c in DK]
    for i in range(m):
        li = sK2L[i]
        if li is None or R_g[li]:
            continue
        D_cok[i] = {sL2K[x] for x in V_g[li]}
    R_cok, _ = _reduce(D_cok)

    # ---- read-off ----
    image = {}
    kernel = {}
    cokernel = {}

    # IMAGE deaths: tau negative in R_f, low_im(tau) in the L block
    im_matched = set()
    for tau in range(m):
        if positive(tau):
            continue
        lo = _low(R_im[tau])
        if lo >= n:
            continue
        birth_idx = new_to_old[lo]
        im_matched.add(birth_idx)
        b, d, dim = valK[birth_idx], valK[tau], dimK[birth_idx]
        if b != d or include_zero_persistence:
            image.setdefault(dim, []).append((b, d))
    # IMAGE births (essential): sigma in L, positive in R_g, unmatched
    for sl in range(n):
        birth_idx = sL2K[sl]
        if birth_idx in im_matched or R_g[sl]:
            continue
        image.setdefault(dimK[birth_idx], []).append((valK[birth_idx], inf))

    # KERNEL deaths: tau in L, negative in R_g, positive in R_f
    ker_matched = set()
    for sl in range(n):
        if not R_g[sl]:
            continue
        death_idx = sL2K[sl]
        if not positive(death_idx):
            continue
        col = K_to_ker_col[death_idx]
        if col is None:
            continue
        birth_idx = new_to_old[_low(R_ker[col])]
        ker_matched.add(birth_idx)
        b, d, dim = valK[birth_idx], valK[death_idx], dimK[death_idx] - 1
        if b != d or include_zero_persistence:
            kernel.setdefault(dim, []).append((b, d))
    # KERNEL births (essential): sigma in K-L, negative in R_f, low_im in L block, unmatched
    for sigma in range(m):
        if in_L[sigma] or sigma in ker_matched or positive(sigma):
            continue
        if _low(R_im[sigma]) >= n:
            continue
        kernel.setdefault(dimK[sigma] - 1, []).append((valK[sigma], inf))

    # COKERNEL deaths: tau negative in R_f, low_im(tau) in the K-L block
    cok_matched = set()
    for tau in range(m):
        if positive(tau):
            continue
        if _low(R_im[tau]) < n:
            continue
        if not R_cok[tau]:
            continue
        birth_idx = _low(R_cok[tau])
        cok_matched.add(birth_idx)
        b, d, dim = valK[birth_idx], valK[tau], dimK[birth_idx]
        if b != d or include_zero_persistence:
            cokernel.setdefault(dim, []).append((b, d))
    # COKERNEL births (essential): sigma positive in R_f, (in K-L or negative in R_g), unmatched
    for sigma in range(m):
        if sigma in cok_matched or not positive(sigma):
            continue
        if in_L[sigma] and not R_g[sK2L[sigma]]:
            continue
        cokernel.setdefault(dimK[sigma], []).append((valK[sigma], inf))

    domain = _ordinary_diagram(R_g, dimL, valL, L.infinity())
    codomain = _ordinary_diagram(R_f, dimK, valK, inf)

    return {"domain": domain, "codomain": codomain,
            "kernel": kernel, "image": image, "cokernel": cokernel}


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------

def _canon(points, infinity=INF):
    """Sort a list of (birth, death) pairs, mapping the filtration infinity to +inf."""
    out = []
    for b, d in points:
        out.append((float(b), INF if d == infinity or d == INF else float(d)))
    return sorted(out)


def _ref_dim(ref_family, dim):
    return _canon(ref_family.get(dim, []))


def _oin_dim(dgms, dim):
    arr = np.asarray(dgms.in_dimension(dim), dtype=float).reshape(-1, 2)
    return _canon([tuple(r) for r in arr])


def assert_kicr_matches_reference(K, L, params=None, max_dim=None, families=("kernel", "image", "cokernel")):
    """Assert oineus KICR(K, L) equals the naive reference, family x dimension."""
    if params is None:
        params = oin.KICRParams(kernel=True, image=True, cokernel=True, codomain=True)
    kicr = oin.compute_kernel_image_cokernel_reduction(K, L, params)
    ref = kicr_reference(K, L, include_zero_persistence=params.include_zero_persistence)
    if max_dim is None:
        max_dim = K.max_dim
    getter = {"kernel": kicr.kernel_diagrams, "image": kicr.image_diagrams,
              "cokernel": kicr.cokernel_diagrams, "domain": kicr.domain_diagrams,
              "codomain": kicr.codomain_diagrams}
    inf = K.infinity()
    for fam in families:
        dgms = getter[fam]()
        for d in range(max_dim + 1):
            try:
                got = _oin_dim(dgms, d)
            except IndexError:
                got = []
            exp = _canon([(b, d2) for (b, d2) in ref[fam].get(d, [])], inf)
            assert got == exp, f"{fam} dim {d}: oineus {got} != reference {exp}"
    return kicr


# ---------------------------------------------------------------------------
# 1. BOOTSTRAP: the reference must reproduce the four hand-derived diagrams
# ---------------------------------------------------------------------------

# (K_list, L_list, expected kernel/cokernel/image by dim) from test_kicr.py
_KNOWN = [
    # test_kernel_1
    (
        [[0, [0], 10], [1, [1], 50], [2, [2], 20], [3, [3], 50], [4, [4], 15], [5, [5], 12],
         [6, [0, 1], 50], [7, [1, 2], 60], [8, [2, 3], 70], [9, [3, 4], 80], [10, [0, 5], 30], [11, [4, 5], 20]],
        [[0, [0], 10], [1, [1], 50], [2, [2], 20], [3, [3], 50], [4, [4], 15],
         [5, [0, 1], 50], [6, [1, 2], 60], [7, [2, 3], 70], [8, [3, 4], 80]],
        {"kernel": {0: [(30., 80.)]},
         "cokernel": {0: [(12., 20.)], 1: [(80., INF)]},
         "image": {0: [(15., 30.), (20., 60.), (50., 70.), (10., INF)]}},
    ),
    # test_kernel_2
    (
        [[0, [0], 10], [1, [1], 30], [2, [2], 10], [3, [3], 0], [4, [0, 1], 30],
         [5, [1, 2], 30], [6, [0, 3], 10], [7, [2, 3], 10]],
        [[0, [0], 10.], [1, [1], 30], [2, [2], 10], [3, [0, 1], 30], [4, [1, 2], 30]],
        {"kernel": {0: [(10., 30.)]},
         "cokernel": {0: [(0., 10.)], 1: [(30., INF)]},
         "image": {0: [(10., INF)]}},
    ),
    # test_kernel_0d_domain_vertices_only
    (
        [[0, [0], 0.0], [1, [1], 0.0], [2, [0, 1], 1.0]],
        [[0, [0], 0.0], [1, [1], 0.0]],
        {"kernel": {0: [(1.0, INF)]}},
    ),
    # test_kernel_0d_persistent_bar_with_shared_edge
    (
        [[0, [0], 0.0], [1, [1], 0.0], [2, [2], 0.0], [3, [3], 0.0], [4, [2, 3], 0.5], [5, [0, 1], 1.0]],
        [[0, [0], 0.0], [1, [1], 0.0], [2, [2], 0.0], [3, [3], 0.0], [4, [2, 3], 0.5]],
        {"kernel": {0: [(1.0, INF)]}},
    ),
]


@pytest.mark.parametrize("idx", range(len(_KNOWN)))
def test_reference_reproduces_known_diagrams(idx):
    K_list, L_list, expected = _KNOWN[idx]
    K = oin.list_to_filtration([list(c) for c in K_list])
    L = oin.list_to_filtration([list(c) for c in L_list])
    ref = kicr_reference(K, L)
    for fam, by_dim in expected.items():
        for d, pts in by_dim.items():
            assert _ref_dim(ref[fam], d) == _canon(pts), f"{fam} dim {d}"
        # dimensions not listed must be empty
        for d in set(ref[fam]) - set(by_dim):
            assert _ref_dim(ref[fam], d) == [], f"{fam} dim {d} expected empty, got {ref[fam][d]}"


# Also confirm oineus itself agrees with the (now-validated) reference on the four
# known examples -- this is the bridge: reference == hand truth (above) and
# oineus == reference (here), for every family incl. domain/codomain.
@pytest.mark.parametrize("idx", range(len(_KNOWN)))
def test_oineus_matches_reference_on_known(idx):
    K_list, L_list, _ = _KNOWN[idx]
    K = oin.list_to_filtration([list(c) for c in K_list])
    L = oin.list_to_filtration([list(c) for c in L_list])
    assert_kicr_matches_reference(K, L,
        families=("kernel", "image", "cokernel", "domain", "codomain"))


# ---------------------------------------------------------------------------
# 2. Non-trivial cross-checks: oineus KICR == reference, every encoding
# ---------------------------------------------------------------------------

def _remove_top_subset(K, frac, seed):
    """Subcomplex L = K with a coface-closed subset of its top-dim cells removed.

    Removing (any subset of) max-dimensional cells is always coface-closed, so L is a
    valid subcomplex; filling those cells back in K creates non-trivial kernel/image/
    cokernel classes for the inclusion L -> K.
    """
    md = K.max_dim
    tops = [i for i in range(K.size()) if K.cell(i).dim == md]
    rng = np.random.default_rng(seed)
    mask = rng.random(len(tops)) < frac
    to_remove = [t for t, m in zip(tops, mask) if m]
    if not to_remove:                       # ensure a non-trivial inclusion
        to_remove = [tops[-1]]
    return K.without_cells(to_remove)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_kicr_fat_vr_matches_reference(seed, n_threads):
    pts = np.ascontiguousarray(np.random.default_rng(seed).random((11, 3)))
    K = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=False)
    L = _remove_top_subset(K, frac=0.5, seed=seed)
    params = oin.KICRParams(kernel=True, image=True, cokernel=True, codomain=True)
    params.n_threads = n_threads
    assert_kicr_matches_reference(K, L, params,
        families=("kernel", "image", "cokernel", "domain", "codomain"))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kicr_fat_vr_matches_reference_with_zero_persistence(seed):
    pts = np.ascontiguousarray(np.random.default_rng(seed + 10).random((9, 2)))
    K = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=False)
    L = _remove_top_subset(K, frac=0.7, seed=seed)
    params = oin.KICRParams(kernel=True, image=True, cokernel=True, codomain=True)
    params.include_zero_persistence = True
    assert_kicr_matches_reference(K, L, params)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_kicr_cube_matches_reference(seed, n_threads):
    # cube_filtration has no fat counterpart (always slim cube), so the reference is
    # the only correctness oracle for cube KICR.
    a = np.random.default_rng(seed).random((5, 5))
    K = oin.cube_filtration(a, max_dim=2)
    assert type(K).__name__.startswith("_CubeFiltration")
    L = _remove_top_subset(K, frac=0.5, seed=seed)
    params = oin.KICRParams(kernel=True, image=True, cokernel=True, codomain=True)
    params.n_threads = n_threads
    assert_kicr_matches_reference(K, L, params,
        families=("kernel", "image", "cokernel", "domain", "codomain"))


@pytest.mark.parametrize("dim,shape", [(2, (6, 7)), (3, (4, 4, 5))])
def test_kicr_slim_freudenthal_matches_reference(dim, shape):
    a = np.random.default_rng(dim).random(shape)
    K = oin.freudenthal_filtration(a, max_dim=dim)          # default -> slim
    assert type(K).__name__ == f"_FreudenthalFiltration_{dim}D"
    L = _remove_top_subset(K, frac=0.5, seed=dim)
    assert_kicr_matches_reference(K, L,
        families=("kernel", "image", "cokernel", "domain", "codomain"))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kicr_packed_vr_matches_reference(seed):
    pts = np.ascontiguousarray(np.random.default_rng(seed + 20).random((12, 3)))
    K = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0)  # default -> packed
    assert type(K).__name__ == "_PackedSimplexFiltration_64"
    L = _remove_top_subset(K, frac=0.5, seed=seed)
    assert_kicr_matches_reference(K, L,
        families=("kernel", "image", "cokernel", "domain", "codomain"))


# ---------------------------------------------------------------------------
# 3. Dispatch: mismatched / unsupported inputs rejected cleanly
# ---------------------------------------------------------------------------

def test_kicr_rejects_mismatched_encoding():
    # K packed, L fat: dispatch picks the packed KICR class (on type(K)), whose ctor is
    # bound for one concrete filtration type -> a fat L is a clean nanobind TypeError,
    # never a silent miscompute.
    pts = np.ascontiguousarray(np.random.default_rng(0).random((10, 3)))
    K = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0)              # packed
    L_fat = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=False)
    L_fat = L_fat.without_cells([L_fat.size() - 1])
    with pytest.raises(TypeError):
        oin.compute_kernel_image_cokernel_reduction(K, L_fat)


def test_kicr_rejects_unsupported_type():
    with pytest.raises(TypeError):
        oin.compute_kernel_image_cokernel_reduction(object(), object())
