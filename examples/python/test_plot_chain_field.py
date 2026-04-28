#!/usr/bin/env python3
"""Demo of oineus.plot_chain on a 2D scalar field with a cubical filtration.

Builds a synthetic 2D function with two clear local minima ("basins"),
runs the cubical lower-star filtration, finds the H0 pair with longest
persistence (the merging of the two basins), and highlights the merging
saddle's death simplex on top of the imshow background.
"""
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import oineus as oin  # noqa: E402


def _two_basin_field(shape=(48, 64)):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx1, cy1 = 0.30 * W, 0.50 * H
    cx2, cy2 = 0.72 * W, 0.50 * H
    g1 = np.exp(-((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2 * (0.10 * W) ** 2))
    g2 = np.exp(-((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2 * (0.10 * W) ** 2))
    field = -(g1 + 0.85 * g2)
    rng = np.random.default_rng(3)
    field += 0.01 * rng.standard_normal(field.shape)
    return field


def _save_demo(out_path):
    field = _two_basin_field()

    fil = oin.cube_filtration(field, max_dim=2)
    dcmp = oin.Decomposition(fil, dualize=False)
    params = oin.ReductionParams()
    params.compute_v = True
    dcmp.reduce(params)

    dgms = dcmp.diagram(fil, include_inf_points=False)
    h0 = dgms.in_dimension(0, as_numpy=False)
    if not h0:
        raise RuntimeError("No finite H0 pairs in this filtration.")

    longest = max(h0, key=lambda p: p.death - p.birth)
    death_idx = int(longest.death_index)
    # Column death_idx of R is the boundary chain that V[death_idx] kills:
    # for an H0 pair of dim-1 saddles, this is the pair of birth vertices
    # being merged. We additionally include the death edge itself so the
    # saddle is visible on the plot.
    chain = list(dcmp.r_data[death_idx]) + [death_idx]

    fig, (ax_field, ax_dgm) = plt.subplots(1, 2, figsize=(12, 5))

    persistence = longest.death - longest.birth
    oin.plot_chain(
        field,
        fil,
        chain,
        ax=ax_field,
        title=f"Cubical filtration: longest H0 merge (persistence={persistence:.3f})",
    )
    ax_field.set_xlabel("j (column)")
    ax_field.set_ylabel("i (row)")

    oin.plot_diagram(dgms, ax=ax_dgm, title="Persistence diagram (cubical)")
    ax_dgm.scatter(
        [longest.birth], [longest.death],
        s=140, facecolors="none", edgecolors="tab:orange",
        linewidths=2.0, zorder=4, label="merge highlighted at left",
    )
    ax_dgm.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def test_plot_chain_field(tmp_path):
    out_path = tmp_path / "plot_chain_field_demo.png"
    _save_demo(str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_chain_field_demo.png")
    _save_demo(out)
    print(f"Saved {out}")
