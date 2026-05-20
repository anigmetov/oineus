"""include_inf_points=True is Phase 2; persistence_diagram must raise
NotImplementedError until Phase 2 lands."""

import numpy as np
import pytest
import torch

import oineus.diff as oin_diff


def test_persistence_diagram_inf_points_raises():
    rng = np.random.default_rng(0)
    pts_np = rng.uniform(-1, 1, size=(8, 2)).astype(np.float64)
    pts = torch.tensor(pts_np, dtype=torch.float64, requires_grad=True)
    fil = oin_diff.vr_filtration(pts, max_dim=1)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        oin_diff.persistence_diagram(fil, include_inf_points=True)
