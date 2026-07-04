"""Mixup barcodes: geometric-topological interaction of two point clouds.

Implements the mixup barcode of Wagner, Arustamyan, Wheeler, Bubenik,
"Mixup Barcodes: Quantifying Geometric-Topological Interactions between
Point Clouds" (arXiv:2402.15058, SoCG 2026).

Setting: two finite point clouds A and B in the same ambient space, the
inclusion of Vietoris-Rips filtrations L = VR(A) -> K = VR(A u B) with a
common range of scales. For each bar [b, d) of the persistence barcode of
L in degree k there is a unique bar [b, d') with the same birth (index)
in the image persistence barcode of the induced map
H_k(L) -> H_k(K) (Bauer-Lesnick induced matching, matched by birth
index; oineus computes the image diagram with the
Cohen-Steiner-Edelsbrunner-Harer-Morozov reduction). Since boundaries of
K can only kill classes of L earlier, b <= d' <= d. The paper packages
this as the *mixup triple* (b, d', d), splitting each persistence bar
into

* the image sub-bar  [b, d')  -- the part of the bar that survives the
  inclusion of B, and
* the mixup sub-bar  [d', d)  -- the "premature death": how much the
  lifetime of the class shortens once the points of B are present.

Summary statistics (per homological degree, over the finite bars):

* mixup(t) = d - d' for a triple t = (b, d', d)
* total mixup = sum of mixups
  (= total persistence of L minus total image persistence)
* mixup percentage of t = (d - d') / (d - b)
* total mixup percentage = sum of the mixup percentages
* mean mixup percentage = their mean, a scale-invariant number in [0, 1]

Zero-persistence bars of L (b == d in filtration value) are dropped, as
in the authors' reference implementation (Ripser-based). Essential bars
are kept in the result object but excluded from the statistics; with the
default truncation (enclosing radius of A) the only essential bar is the
one surviving connected component, whose mixup is zero anyway.

The differentiable variant lives in oineus.diff.mixup_barcodes.
"""

import numpy as np

__all__ = ["MixupBarcodes", "mixup_barcodes", "mixup_barcodes_of_filtrations"]

# death sentinel used by the C++ index diagrams for points at infinity
INVALID_INDEX = np.iinfo(np.uint64).max


def compute_mixup_triples(kicr, fil_K, fil_L, max_dim):
    """Match domain and image index diagrams of a KICR reduction into mixup triples.

    kicr is a KerImCokReduced computed for the inclusion fil_L -> fil_K with
    image diagrams requested. Returns a dict

        dim -> (finite_vals, finite_idx, essential_vals)

    for dim in 0..max_dim, where finite_vals is an (n, 3) float array of
    (birth, image_death, death) triples over the finite positive-persistence
    bars of the domain diagram, finite_idx is the aligned (n, 3) int64 array
    of sorted ids in fil_K (birth cell, image death cell, death cell; for a
    zero-length image sub-bar the birth cell doubles as the image death cell,
    mirroring the paper's (b, b, d) convention for unmatched bars), and
    essential_vals is an (m, 3) float array for the essential bars of the
    domain (death = +inf; image death may be finite if the filtration was
    truncated below the actual death value).

    The matching is by birth cell, as in the paper: the domain birth cell
    (a cell of L) is located in K by its uid, and the image bar born at that
    cell -- which exists and is unique -- supplies the premature death. A
    finite image pair whose birth and death values coincide is dropped by the
    C++ diagram construction; for the matching this is equivalent to an
    image sub-bar of length zero, so the triple degenerates to (b, b, d).
    """
    dom_dgms = kicr.domain_diagrams()
    im_dgms = kicr.image_diagrams()

    # the C++ diagrams only carry dimensions up to the max cell dimension of
    # the respective filtration; degrees above L's have no bars at all
    avail_dim = int(fil_L.max_dim) if fil_L.size() > 0 else -1

    out = {}
    for dim in range(max_dim + 1):
        if dim > avail_dim:
            out[dim] = (np.empty((0, 3), dtype=np.float64),
                        np.empty((0, 3), dtype=np.int64),
                        np.empty((0, 3), dtype=np.float64))
            continue
        dom_val = np.asarray(dom_dgms.in_dimension(dim), dtype=np.float64).reshape(-1, 2)
        dom_idx = np.asarray(dom_dgms.index_diagram_in_dimension(dim, as_numpy=True),
                             dtype=np.uint64).reshape(-1, 2)
        im_val = np.asarray(im_dgms.in_dimension(dim), dtype=np.float64).reshape(-1, 2)
        im_idx = np.asarray(im_dgms.index_diagram_in_dimension(dim, as_numpy=True),
                            dtype=np.uint64).reshape(-1, 2)

        # image bars keyed by the K sorted id of their birth cell
        im_by_birth = {}
        for (b_idx, d_idx), (b_val, d_val) in zip(im_idx, im_val):
            im_by_birth[int(b_idx)] = (float(b_val), float(d_val), int(d_idx))

        finite_vals, finite_idx, essential_vals = [], [], []
        n_matched = 0
        for (b_L, d_L), (b_val, d_val) in zip(dom_idx, dom_val):
            b_K = int(fil_K.sorted_id_by_uid(fil_L.cell(int(b_L)).uid))
            im_row = im_by_birth.get(b_K)
            if im_row is not None:
                n_matched += 1
                im_b_val, im_d_val, im_d_idx = im_row
                if not np.isclose(im_b_val, b_val, rtol=1e-9, atol=1e-12):
                    raise RuntimeError(
                        f"mixup: domain and image bars born at the same cell have "
                        f"different birth values ({b_val} vs {im_b_val}); K and L "
                        f"must carry the same values on shared cells")
            if int(d_L) == INVALID_INDEX:
                # essential domain bar; its image bar is essential too, unless the
                # filtration is truncated and the image class dies below the cutoff
                im_death = np.inf
                if im_row is not None and im_row[2] != int(INVALID_INDEX):
                    im_death = im_row[1]
                essential_vals.append((float(b_val), im_death, np.inf))
                continue
            d_K = int(fil_K.sorted_id_by_uid(fil_L.cell(int(d_L)).uid))
            if im_row is None:
                # the image pair born at b_K had zero persistence and was dropped:
                # image sub-bar is empty, premature death at birth (paper's (b, b, d))
                finite_vals.append((float(b_val), float(b_val), float(d_val)))
                finite_idx.append((b_K, b_K, d_K))
            else:
                im_b_val, im_d_val, im_d_idx = im_row
                if im_d_idx == int(INVALID_INDEX) or im_d_val > d_val * (1 + 1e-9) + 1e-12:
                    raise RuntimeError(
                        f"mixup: image death {im_d_val} exceeds domain death {d_val} "
                        f"in dim {dim}; the filtration orders of K and L are "
                        f"inconsistent on ties")
                finite_vals.append((float(b_val), min(float(im_d_val), float(d_val)), float(d_val)))
                finite_idx.append((b_K, im_d_idx, d_K))

        # every image bar must be consumed by a domain bar (the induced matching
        # is onto the image barcode); a leftover signals inconsistent orderings
        if n_matched != len(im_by_birth):
            raise RuntimeError(
                f"mixup: {len(im_by_birth) - n_matched} image bars in dim {dim} "
                f"were not matched by any domain bar; the filtration orders of "
                f"K and L are inconsistent on ties")

        finite_vals = np.asarray(finite_vals, dtype=np.float64).reshape(-1, 3)
        finite_idx = np.asarray(finite_idx, dtype=np.int64).reshape(-1, 3)
        essential_vals = np.asarray(essential_vals, dtype=np.float64).reshape(-1, 3)

        # deterministic order: by birth, then death, then image death, then birth cell
        if len(finite_vals):
            order = np.lexsort((finite_idx[:, 0], finite_vals[:, 1],
                                finite_vals[:, 2], finite_vals[:, 0]))
            finite_vals, finite_idx = finite_vals[order], finite_idx[order]
        if len(essential_vals):
            essential_vals = essential_vals[np.lexsort((essential_vals[:, 1],
                                                        essential_vals[:, 0]))]

        out[dim] = (finite_vals, finite_idx, essential_vals)
    return out


class MixupBarcodes:
    """Mixup barcodes of an inclusion L -> K, one barcode per homological degree.

    The mixup barcode in degree k is a collection of triples (b, d', d), one
    per finite positive-persistence bar [b, d) of the persistence barcode of
    L, where [b, d') is the matching bar of the image persistence barcode of
    H_k(L) -> H_k(K) (matched by birth cell). The persistence bar splits into
    the image sub-bar [b, d') and the mixup sub-bar [d', d); always
    b <= d' <= d.

    Accessors (all return copies):

    * in_dimension(k) / self[k] -- (n, 3) array of (birth, image_death, death)
    * persistence_barcode(k), image_sub_barcode(k), mixup_sub_barcode(k)
      -- (n, 2) column pairs of the triples
    * index_triples_in_dimension(k) -- (n, 3) int64 sorted ids in K of the
      (birth, image death, death) cells; for an empty image sub-bar the birth
      cell is repeated in the image-death slot
    * essential_in_dimension(k) -- (m, 3) triples of essential bars of L
      (death = +inf); excluded from all statistics

    Statistics (per degree, over the finite triples; 0.0 when there are none):

    * total_persistence(k) = sum(d - b)
    * total_mixup(k) = sum(d - d')
    * mixup_percentages(k) = (d - d') / (d - b), per bar
    * total_mixup_percentage(k) = their sum
    * mean_mixup_percentage(k) = their mean, scale-invariant, in [0, 1]

    Instances are plain-numpy containers: picklable and cheap to copy.
    """

    def __init__(self, finite, finite_idx, essential, max_dim):
        self._finite = finite
        self._finite_idx = finite_idx
        self._essential = essential
        self.max_dim = max_dim

    @classmethod
    def from_triples(cls, triples, max_dim):
        """Build from the dict returned by compute_mixup_triples."""
        finite = {d: t[0] for d, t in triples.items()}
        finite_idx = {d: t[1] for d, t in triples.items()}
        essential = {d: t[2] for d, t in triples.items()}
        return cls(finite, finite_idx, essential, max_dim)

    @classmethod
    def empty(cls, max_dim):
        """An all-empty result (e.g. for an empty point cloud A)."""
        finite = {d: np.empty((0, 3), dtype=np.float64) for d in range(max_dim + 1)}
        finite_idx = {d: np.empty((0, 3), dtype=np.int64) for d in range(max_dim + 1)}
        essential = {d: np.empty((0, 3), dtype=np.float64) for d in range(max_dim + 1)}
        return cls(finite, finite_idx, essential, max_dim)

    def _check_dim(self, dim):
        if dim not in self._finite:
            raise KeyError(f"no mixup barcode in dimension {dim}; "
                           f"available: {sorted(self._finite)}")

    def in_dimension(self, dim):
        """Finite mixup triples in the given degree: (n, 3) array of
        (birth, image_death, death) rows with birth <= image_death <= death."""
        self._check_dim(dim)
        return self._finite[dim].copy()

    def __getitem__(self, dim):
        return self.in_dimension(dim)

    def __contains__(self, dim):
        return dim in self._finite

    def keys(self):
        return sorted(self._finite)

    def index_triples_in_dimension(self, dim):
        """Sorted ids in K of the (birth, image death, death) cells of the
        finite triples, aligned with in_dimension(dim). When the image
        sub-bar is empty the birth cell doubles as the image death cell."""
        self._check_dim(dim)
        return self._finite_idx[dim].copy()

    def essential_in_dimension(self, dim):
        """Essential bars of L as (m, 3) triples (birth, image_death, +inf).
        The image death is +inf as well unless the filtration was truncated
        below the actual death of the image class. Not part of statistics."""
        self._check_dim(dim)
        return self._essential[dim].copy()

    def persistence_barcode(self, dim):
        """Finite positive-persistence bars [b, d) of L: columns 0, 2 of the triples."""
        return self.in_dimension(dim)[:, [0, 2]]

    def image_sub_barcode(self, dim):
        """Image sub-bars [b, d'): columns 0, 1 of the triples."""
        return self.in_dimension(dim)[:, [0, 1]]

    def mixup_sub_barcode(self, dim):
        """Mixup sub-bars [d', d): columns 1, 2 of the triples."""
        return self.in_dimension(dim)[:, [1, 2]]

    def total_persistence(self, dim):
        """Sum of d - b over the finite triples."""
        self._check_dim(dim)
        t = self._finite[dim]
        return float(np.sum(t[:, 2] - t[:, 0]))

    def total_mixup(self, dim):
        """Sum of the mixups d - d' over the finite triples. Equals the total
        persistence of L minus the total image persistence."""
        self._check_dim(dim)
        t = self._finite[dim]
        return float(np.sum(t[:, 2] - t[:, 1]))

    def mixup_percentages(self, dim):
        """Per-bar mixup percentages (d - d') / (d - b), an (n,) array."""
        self._check_dim(dim)
        t = self._finite[dim]
        return (t[:, 2] - t[:, 1]) / (t[:, 2] - t[:, 0])

    def total_mixup_percentage(self, dim):
        """Sum of the per-bar mixup percentages."""
        return float(np.sum(self.mixup_percentages(dim)))

    def mean_mixup_percentage(self, dim):
        """Mean of the per-bar mixup percentages, in [0, 1]; 0.0 if the
        barcode is empty (no interaction)."""
        p = self.mixup_percentages(dim)
        return float(np.mean(p)) if len(p) else 0.0

    def __repr__(self):
        sizes = {d: len(self._finite[d]) for d in sorted(self._finite)}
        tm = {d: round(self.total_mixup(d), 6) for d in sorted(self._finite)}
        return f"MixupBarcodes(max_dim={self.max_dim}, bars={sizes}, total_mixup={tm})"


def mixup_barcodes_of_filtrations(K, L, max_dim=None, n_threads=1):
    """Mixup barcodes of the inclusion of filtrations L -> K.

    K, L: oineus filtrations with L a subcomplex of K (every cell of L is
    present in K, identified by uid) carrying the same filtration values on
    shared cells, and with consistent orders on ties (both are automatic for
    the point-cloud facade mixup_barcodes). Cell encodings of K and L must
    match (both fat, or both packed with the same uid layout -- note that
    packed VR filtrations built from different numbers of points have
    incompatible uids, so the facade uses the fat encoding).

    max_dim: largest homological degree to report; default K.max_dim - 1
    (bars in degree K.max_dim cannot die for lack of higher cells).
    n_threads: parallelism of the underlying reductions.

    Returns a MixupBarcodes object.
    """
    from . import compute_kernel_image_cokernel_reduction
    from . import _oineus

    if K.negate or L.negate:
        raise ValueError("mixup barcodes are defined for sublevel (negate=False) filtrations")

    if max_dim is None:
        max_dim = max(int(K.max_dim) - 1, 0)

    params = _oineus.KICRParams()
    params.kernel = False
    params.image = True
    params.cokernel = False
    params.codomain = False
    params.include_zero_persistence = False
    params.n_threads = max(1, int(n_threads))

    kicr = compute_kernel_image_cokernel_reduction(K, L, params)
    triples = compute_mixup_triples(kicr, K, L, max_dim)
    return MixupBarcodes.from_triples(triples, max_dim)


def mixup_barcodes(A, B, max_dim=1, max_diameter=None, n_threads=1):
    """Mixup barcodes of the point cloud A included into A u B.

    Builds the Vietoris-Rips filtrations L = VR(A) and K = VR(A u B) over a
    common range of scales and computes the mixup barcode of L -> K in
    degrees 0..max_dim (see the module docstring and MixupBarcodes for the
    definitions). The result quantifies how the presence of the points of B
    shortens the lifetime of each topological feature of A: overlap /
    separation in degree 0, encirclement in degree 1, surrounding in the
    codimension-one degree.

    Parameters
    ----------
    A : (n_A, d) array
        The observed point cloud whose features are tracked.
    B : (n_B, d) array or None
        The point cloud mixed into A. May be empty or None, in which case
        every mixup sub-bar is empty and all mixup statistics are zero.
    max_dim : int
        Largest homological degree; simplices up to dimension max_dim + 1
        are enumerated. Default 1.
    max_diameter : float or None
        Common truncation threshold for both filtrations. Default: the
        enclosing radius of A, which is enough for every finite bar of
        VR(A) -- and hence every image death -- to be present; only the
        essential connected-component bar remains at infinity.
    n_threads : int
        Parallelism of the underlying reductions.

    Returns a MixupBarcodes object.

    Note: total-mixup(A, B') <= total-mixup(A, B) for B' a subset of B
    (subsampling property, Lemma in the paper), so B may be subsampled to
    speed things up at the price of a weaker signal only.
    """
    from . import vr_filtration, max_distance

    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    if B is None:
        B = np.empty((0, A.shape[1] if A.ndim == 2 else 0), dtype=np.float64)
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64))

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D point arrays of shape (n, d)")
    if len(A) and len(B) and A.shape[1] != B.shape[1]:
        raise ValueError(
            f"A and B must live in the same ambient space, got d={A.shape[1]} and d={B.shape[1]}")
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative")

    if len(A) == 0:
        return MixupBarcodes.empty(max_dim)

    if max_diameter is None:
        # the enclosing radius of A: VR(A) is a cone (hence acyclic) above it,
        # so every finite bar of L, and therefore every image death, is below
        max_diameter = float(max_distance(A)) if len(A) >= 2 else 0.0

    # fat cells: packed uids depend on the number of points, so packed L and K
    # built from different point counts would not share uids
    L = vr_filtration(A, max_dim=max_dim + 1, max_diameter=max_diameter,
                      packed=False, n_threads=n_threads)
    if len(B) == 0:
        K = L
    else:
        union = np.vstack([A, B])
        K = vr_filtration(union, max_dim=max_dim + 1, max_diameter=max_diameter,
                          packed=False, n_threads=n_threads)

    return mixup_barcodes_of_filtrations(K, L, max_dim=max_dim, n_threads=n_threads)
