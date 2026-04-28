#!/usr/bin/env python3
"""Demo of plot_diagram_gradient.

Builds a tiny VR filtration, computes a differentiable persistence diagram,
defines a quadratic loss on its H1 points, backpropagates, and saves a PNG
showing the gradient vector field on the diagram.

Run as a script (saves diagram_gradient.png in the cwd) or via pytest.
"""
import os

import pytest

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import oineus as oin  # noqa: E402
import oineus.diff as oin_diff  # noqa: E402


def _make_points():
    rng = np.random.default_rng(0)
    n_points = 30
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    pts = pts + 0.05 * rng.standard_normal(pts.shape)
    return torch.tensor(pts, dtype=torch.float64, requires_grad=True)


def _compute_diagram_with_grad():
    pts = _make_points()
    fil = oin_diff.vr_filtration(pts, max_dim=2)
    dgms = oin_diff.persistence_diagram(fil, dualize=True)
    dgm1 = dgms[1]
    dgm1.retain_grad()
    target_persistence = 1.0
    loss = ((dgm1[:, 1] - dgm1[:, 0]) - target_persistence).pow(2).sum()
    loss.backward()
    return dgm1


def _save_plot(dgm1, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    oin.plot_diagram_gradient(
        dgm1,
        ax=ax,
        descent=True,
        title="H1 gradient (descent direction)",
    )
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def test_plot_diagram_gradient(tmp_path):
    dgm1 = _compute_diagram_with_grad()
    assert dgm1.grad is not None
    out_path = tmp_path / "diagram_gradient.png"
    _save_plot(dgm1, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


if __name__ == "__main__":
    dgm1 = _compute_diagram_with_grad()
    out = os.path.join(os.getcwd(), "diagram_gradient.png")
    _save_plot(dgm1, out)
    print(f"Saved {out}")
