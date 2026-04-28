#!/usr/bin/env python3
"""Visualize the CSM critical set ("basin") of a peak in a 2D scalar field.

Setup: a 2D field with two peaks. Build the lower-star Freudenthal
filtration on ``-field`` so peaks become H0 birth simplices; the longest
finite H0 pair corresponds to the secondary peak (the dominant peak is
essential -- it never dies).

To "squash" that secondary peak we want to *raise* its birth value, which
is exactly what ``TopologyOptimizer.increase_birth`` computes a critical
set for. The returned indices are the cells whose values would also need
to move when raising the peak's value while respecting the elder rule --
in dim 0 those indices coincide with the *basin of attraction* of the
peak (every vertex currently in the peak's connected component up to the
saddle).

Internally this reduces to the **column of V in the cohomology
decomposition**: see ``include/oineus/top_optimizer.h:535`` for
``increase_birth``, which reads ``decmp_coh_.v_data.at(index_in_matrix(...))``.
"""
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import oineus as oin  # noqa: E402


def _two_peak_field(shape=(48, 64)):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    g1 = np.exp(-((xx - 0.30 * W) ** 2 + (yy - 0.50 * H) ** 2) / (2 * (0.10 * W) ** 2))
    g2 = 0.85 * np.exp(-((xx - 0.72 * W) ** 2 + (yy - 0.50 * H) ** 2) / (2 * (0.10 * W) ** 2))
    return (g1 + g2).astype(np.float64)


def _save_demo(out_path):
    field = _two_peak_field()

    # Lower-star sublevel filtration of -field puts peaks first.
    fil = oin.freudenthal_filtration(-field, max_dim=2)

    # Diagram + the longest finite H0 pair (the secondary peak).
    dcmp = oin.Decomposition(fil, dualize=False)
    params = oin.ReductionParams()
    params.compute_v = True
    dcmp.reduce(params)
    dgms = dcmp.diagram(fil, include_inf_points=True)
    h0 = dgms.in_dimension(0, as_numpy=False)
    finite_h0 = [p for p in h0 if np.isfinite(p.death)]
    longest = max(finite_h0, key=lambda p: p.death - p.birth)
    birth_idx = int(longest.birth_index)
    death_idx = int(longest.death_index)

    # Critical set for "increase the birth value of the peak"
    # (equivalent to "squash the peak down toward the saddle").
    optimizer = oin.TopologyOptimizer(fil)
    optimizer.reduce_all()
    crit_indices = list(optimizer.increase_birth(birth_idx))

    # The death cell (saddle edge) is shown alongside so we can read the
    # boundary of the basin off the picture.
    chain = list(set(crit_indices) | {death_idx})

    fig, (ax_field, ax_dgm) = plt.subplots(1, 2, figsize=(12, 5))

    persistence = longest.death - longest.birth
    n_basin_vertices = sum(1 for i in crit_indices if fil[i].dim == 0)
    oin.plot_chain(
        field,
        fil,
        chain,
        ax=ax_field,
        title=(f"Squash-peak critical set ({n_basin_vertices} basin "
               f"vertices, persistence={persistence:.3f})"),
        source_kind="field",
    )
    ax_field.set_xlabel("j (column)")
    ax_field.set_ylabel("i (row)")

    oin.plot_diagram(dgms, ax=ax_dgm, title="Persistence diagram (Freudenthal on -field)")
    ax_dgm.scatter(
        [longest.birth], [longest.death],
        s=140, facecolors="none", edgecolors="tab:orange",
        linewidths=2.0, zorder=4, label="peak we are squashing",
    )
    ax_dgm.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def test_plot_chain_basin(tmp_path):
    out_path = tmp_path / "plot_chain_basin_demo.png"
    _save_demo(str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_chain_basin_demo.png")
    _save_demo(out)
    print(f"Saved {out}")
