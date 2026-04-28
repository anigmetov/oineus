#!/usr/bin/env python3
"""Demo of oineus.plot_chain on an alpha (Cech-Delaunay) filtration.

Builds an alpha filtration over a noisy circle in 2D, finds the H1 pair
with longest persistence, extracts its cycle representative as a chain of
edges from the reduced boundary matrix R, and saves a PNG showing the
cycle highlighted on top of the point cloud.

Run as a script (saves plot_chain_alpha_demo.png next to the script) or
as a pytest case (saves into tmp_path).
"""
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# diode is required for alpha filtrations; gate the test on it.
pytest.importorskip("diode")

import oineus as oin  # noqa: E402


def _circle_with_interior(n_circle=24, n_interior=40, noise=0.05, seed=0):
    """Noisy unit circle plus uniformly-scattered interior points.

    The interior points do not participate in the H1 cycle, so they make the
    cycle visually pop against the broader point cloud.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    circle = circle + noise * rng.standard_normal(circle.shape)

    interior = rng.uniform(-0.55, 0.55, (n_interior, 2))
    return np.vstack([circle, interior])


def _alpha_filtration_and_cycle(points):
    """Build an alpha filtration over `points`, run reduction, and return
    (fil, longest_h1_cycle_chain). The cycle is a list of edge sorted-ids
    from the column of R that kills the longest H1 feature.
    """
    fil = oin._alpha_shapes_filtration(points, n_threads=1)

    dcmp = oin.Decomposition(fil, dualize=False)
    params = oin.ReductionParams()
    params.compute_v = True
    dcmp.reduce(params)

    dgms = dcmp.diagram(fil, include_inf_points=False)
    h1 = dgms.in_dimension(1, as_numpy=False)
    if not h1:
        raise RuntimeError("No finite H1 pairs in this alpha filtration.")

    # Pick the H1 pair with longest persistence.
    best = max(h1, key=lambda p: p.death - p.birth)
    death_idx = int(best.death_index)

    # Column death_idx of R is the boundary of V[death_idx]: the H1 cycle
    # representative as a sum of edges. r_data[j] is column j as a list of
    # row indices, which are filtration sorted_ids of edges (dim-1 cells).
    cycle_edges = list(dcmp.r_data[death_idx])
    return fil, cycle_edges, best


def _save_demo(out_path):
    points = _circle_with_interior()
    fil, cycle, h1_pair = _alpha_filtration_and_cycle(points)

    dcmp = oin.Decomposition(fil, dualize=False)
    params = oin.ReductionParams()
    params.compute_v = True
    dcmp.reduce(params)
    dgms = dcmp.diagram(fil, include_inf_points=False)

    fig, (ax_pts, ax_dgm) = plt.subplots(1, 2, figsize=(11, 5.5))

    persistence = h1_pair.death - h1_pair.birth
    point_style = oin.default_point_cloud_style()
    point_style["s"] = 18.0
    point_style["c"] = "tab:gray"
    oin.plot_chain(
        points,
        fil,
        cycle,
        ax=ax_pts,
        point_style=point_style,
        title=f"Alpha filtration: longest H1 cycle (persistence = {persistence:.3f})",
    )

    oin.plot_diagram(dgms, ax=ax_dgm, title="Persistence diagram (alpha)")
    ax_dgm.scatter(
        [h1_pair.birth], [h1_pair.death],
        s=120, facecolors="none", edgecolors="tab:orange",
        linewidths=2.0, zorder=4, label="cycle plotted at left",
    )
    ax_dgm.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def test_plot_chain_alpha(tmp_path):
    out_path = tmp_path / "plot_chain_alpha_demo.png"
    _save_demo(str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_chain_alpha_demo.png")
    _save_demo(out)
    print(f"Saved {out}")
