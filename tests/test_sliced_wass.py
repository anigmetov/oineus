"""Pytest tests for sliced Wasserstein distance on differentiable diagrams."""

import numpy as np
import pytest

from oineus._dtype import REAL_DTYPE

# Check if PyTorch is available (optional dependency)
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_DTYPE = torch.float32 if REAL_DTYPE == np.float32 else torch.float64
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DTYPE = None

# Tight tolerance for "should be exactly zero / symmetric" checks.
ABS_TOL = 1e-10 if REAL_DTYPE == np.float64 else 1e-5

# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

import oineus.diff as oin_diff


def test_sliced_wasserstein_basic():
    """Test basic sliced Wasserstein computation."""
    dgm1 = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE, requires_grad=True)
    dgm2 = torch.tensor([[0.2, 1.2], [0.6, 2.1]], dtype=TORCH_DTYPE, requires_grad=True)

    dist = oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=50)

    assert dist.item() >= 0, "Distance must be non-negative"


def test_sliced_wasserstein_identical():
    """Test that distance between identical diagrams is zero."""
    dgm = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]], dtype=TORCH_DTYPE)
    dist = oin_diff.sliced_wasserstein_distance(dgm, dgm.clone(), n_directions=50)

    assert dist.item() == pytest.approx(0.0, abs=ABS_TOL)


def test_sliced_wasserstein_empty():
    """Test handling of empty diagrams."""
    empty = torch.zeros((0, 2), dtype=TORCH_DTYPE)
    dgm = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE)

    dist_empty_empty = oin_diff.sliced_wasserstein_distance(empty, empty, n_directions=10)
    assert dist_empty_empty.item() == pytest.approx(0.0)

    dist_empty_dgm = oin_diff.sliced_wasserstein_distance(empty, dgm, n_directions=10)
    assert dist_empty_dgm.item() > 0


def test_sliced_wasserstein_symmetry():
    """Test that distance is symmetric."""
    dgm1 = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 1.5]], dtype=TORCH_DTYPE)
    dgm2 = torch.tensor([[0.2, 1.2], [0.6, 2.1]], dtype=TORCH_DTYPE)

    torch.manual_seed(42)
    dist_12 = oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=100)

    torch.manual_seed(42)
    dist_21 = oin_diff.sliced_wasserstein_distance(dgm2, dgm1, n_directions=100)

    assert dist_12.item() == pytest.approx(dist_21.item(), abs=ABS_TOL)


def test_sliced_wasserstein_differentiable():
    """Test that gradients flow back through sliced Wasserstein."""
    dgm1 = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]], dtype=TORCH_DTYPE, requires_grad=True)
    dgm2 = torch.tensor([[0.2, 1.2], [0.6, 2.1], [1.1, 3.1]], dtype=TORCH_DTYPE, requires_grad=True)

    torch.manual_seed(42)
    dist = oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=50)
    dist.backward()

    assert dgm1.grad is not None
    assert dgm2.grad is not None
    assert dgm1.grad.norm().item() > 0
    assert dgm2.grad.norm().item() > 0


def test_sliced_wasserstein_gradient_descent():
    """Test that gradient descent reduces distance."""
    target = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE)
    dgm = torch.tensor([[0.5, 1.5], [1.0, 2.5]], dtype=TORCH_DTYPE, requires_grad=True)

    optimizer = torch.optim.SGD([dgm], lr=0.01)

    torch.manual_seed(42)
    initial_dist = oin_diff.sliced_wasserstein_distance(dgm, target, n_directions=50).item()

    for _ in range(10):
        optimizer.zero_grad()
        torch.manual_seed(42)
        dist = oin_diff.sliced_wasserstein_distance(dgm, target, n_directions=50)
        dist.backward()
        optimizer.step()

    torch.manual_seed(42)
    final_dist = oin_diff.sliced_wasserstein_distance(dgm, target, n_directions=50).item()

    assert final_dist < initial_dist


def test_sliced_wasserstein_with_essential_points():
    """Test handling of essential points (infinite deaths)."""
    dgm1 = torch.tensor([[0.0, 1.0], [0.5, float('inf')]], dtype=TORCH_DTYPE, requires_grad=True)
    dgm2 = torch.tensor([[0.2, 1.2], [0.6, float('inf')]], dtype=TORCH_DTYPE, requires_grad=True)

    dist = oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=50, ignore_inf_points=False)

    assert dist.item() >= 0
    dist.backward()
    assert dgm1.grad is not None


def test_sliced_wasserstein_ignore_inf_points():
    """Test that ignore_inf_points flag works."""
    dgm1 = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, float('inf')]], dtype=TORCH_DTYPE)
    dgm2 = torch.tensor([[0.2, 1.2], [0.6, 2.1], [1.2, float('inf')]], dtype=TORCH_DTYPE)

    torch.manual_seed(42)
    dist_with_inf = oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=50, ignore_inf_points=False)

    torch.manual_seed(42)
    dist_without_inf = oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=50, ignore_inf_points=True)

    assert dist_with_inf.item() >= dist_without_inf.item() - 1e-6


def test_essential_cardinality_mismatch():
    """Test that mismatched essential point cardinalities raise an error."""
    dgm1 = torch.tensor([[0.0, 1.0], [0.5, float('inf')]], dtype=TORCH_DTYPE)
    dgm2 = torch.tensor([[0.2, 1.2], [0.6, float('inf')], [1.0, float('inf')]], dtype=TORCH_DTYPE)

    with pytest.raises(ValueError, match="Essential point cardinalities must match"):
        oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=10, ignore_inf_points=False)


def test_diag_corrected_basic():
    """Test diagonal-corrected version basic functionality."""
    dgm1 = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE, requires_grad=True)
    dgm2 = torch.tensor([[0.2, 1.2], [0.6, 2.1]], dtype=TORCH_DTYPE, requires_grad=True)

    dist = oin_diff.sliced_wasserstein_distance_diag_corrected(dgm1, dgm2, n_directions=50)

    assert dist.item() >= 0
    dist.backward()
    assert dgm1.grad is not None


def test_compare_standard_vs_corrected():
    """Compare standard and diagonal-corrected versions."""
    dgm1 = torch.tensor([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]], dtype=TORCH_DTYPE)
    dgm2 = torch.tensor([[0.2, 1.2], [0.6, 2.1]], dtype=TORCH_DTYPE)

    torch.manual_seed(42)
    dist_standard = oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=100)

    torch.manual_seed(42)
    dist_corrected = oin_diff.sliced_wasserstein_distance_diag_corrected(dgm1, dgm2, n_directions=100)

    assert dist_standard.item() >= 0
    assert dist_corrected.item() >= 0


def test_diag_corrected_empty_diagram():
    """Test diagonal-corrected version with empty diagrams."""
    empty = torch.zeros((0, 2), dtype=TORCH_DTYPE)
    dgm = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=TORCH_DTYPE)

    # Both empty
    dist_empty_empty = oin_diff.sliced_wasserstein_distance_diag_corrected(empty, empty, n_directions=10)
    assert dist_empty_empty.item() == pytest.approx(0.0)

    # One empty
    torch.manual_seed(42)
    dist_empty_dgm = oin_diff.sliced_wasserstein_distance_diag_corrected(empty, dgm, n_directions=50)
    assert dist_empty_dgm.item() > 0

    # Reverse order
    torch.manual_seed(42)
    dist_dgm_empty = oin_diff.sliced_wasserstein_distance_diag_corrected(dgm, empty, n_directions=50)
    assert dist_dgm_empty.item() > 0
