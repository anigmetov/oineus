"""compute_partial_u_rows refuses to run when V is not known to be in
ELZ form. The check reads a flag maintained by the reduction drivers
(set by restore_elz() and by reduce_serial without clearing); a full
is_elz() walk per call would be O(matrix) and too expensive in the
gradient loop. The flag-based check fires whenever the reduction
path didn't go through one of those two entry points -- regardless
of whether the matrix happens to be ELZ in this particular case."""

import numpy as np
import pytest
import torch

import oineus as oin
import oineus.diff as oin_diff


def test_compute_partial_u_rows_rejects_non_elz_v():
    rng = np.random.default_rng(2)
    pts_np = rng.uniform(-1, 1, size=(20, 2)).astype(np.float64)
    pts = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    under_fil = fil.under_fil

    # Parallel reduction with clearing on; restore_elz not requested.
    # The driver's inline ELZ-restoration loop does not set the flag,
    # and we don't call restore_elz() afterwards, so partial-U must
    # reject this decomposition.
    dcmp = oin.Decomposition(under_fil, dualize=False)
    rp = oin.ReductionParams()
    rp.compute_v = True
    rp.compute_u = False
    rp.clearing_opt = True
    rp.dims_to_restore_elz = []
    rp.n_threads = 2
    dcmp.reduce(rp)

    with pytest.raises(RuntimeError, match="ELZ"):
        dcmp.compute_partial_u_rows(under_fil, rows=[0], bounds=[1.0],
                                    dim=1, cmp="above", n_threads=1)


def test_compute_partial_u_rows_accepts_after_explicit_restore_elz():
    """Explicit restore_elz() sets the flag and unblocks partial-U."""
    rng = np.random.default_rng(3)
    pts_np = rng.uniform(-1, 1, size=(20, 2)).astype(np.float64)
    pts = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    under_fil = fil.under_fil

    dcmp = oin.Decomposition(under_fil, dualize=False)
    rp = oin.ReductionParams()
    rp.compute_v = True
    rp.compute_u = False
    rp.clearing_opt = True
    rp.dims_to_restore_elz = []
    rp.n_threads = 2
    dcmp.reduce(rp)
    dcmp.restore_elz(n_threads=1)

    # Pick a row that actually belongs to dim 1 (an edge) to avoid
    # the dim-mismatch hang in the row solver.
    row = int(dcmp.dim_first[1])
    dcmp.compute_partial_u_rows(under_fil, rows=[row], bounds=[1.0e9],
                                dim=1, cmp="above", n_threads=1)
