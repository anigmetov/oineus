"""Gallery of plot_matching styling presets (W1 Wasserstein matching).

Loads NPT.285200.d1 + NVT.1000000.d1 from the repo root, computes the W1
(q=1) Wasserstein matching between them, and renders six presets exploring
the matching-edge style space. The auto-cap to top-200 edges by endpoint
persistence (new default in plot_matching) keeps every cell readable.

The two diagrams use the new P11b/G1.5-era oineus defaults: A = Okabe-Ito
blue, B = Okabe-Ito vermillion. We focus the gallery on the edge style.

Outputs land in tests/eval_matching_styles_out/.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator
import mpl_scatter_density  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))

import oineus  # noqa: E402
from oineus.matching import point_to_diagonal  # noqa: E402

OUT_DIR = REPO_ROOT / "tests" / "eval_matching_styles_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----- presets --------------------------------------------------------------

def render_M0(ax, dgm_a, dgm_b, m):
    """Current oineus defaults: gray, lw=0.8, alpha=0.5 (top-200 auto-cap)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax)
    ax.set_title("M0 -- defaults  (gray lw=0.8 alpha=0.5)")


def render_M1(ax, dgm_a, dgm_b, m):
    """Subtle: thinner, lower alpha, light gray."""
    style = oineus.default_matching_edge_style()
    style.update({"linewidth": 0.5, "alpha": 0.4})
    oineus.plot_matching(
        dgm_a, dgm_b, m, ax=ax,
        ordinary_edge_style=style,
        match_color="0.55",
        top_k_pairs=200,
    )
    ax.set_title("M1 -- subtle  (light gray lw=0.5 alpha=0.4)")


def render_M2(ax, dgm_a, dgm_b, m):
    """Bolder: thicker, higher alpha, dark gray."""
    style = oineus.default_matching_edge_style()
    style.update({"linewidth": 1.0, "alpha": 0.65})
    oineus.plot_matching(
        dgm_a, dgm_b, m, ax=ax,
        ordinary_edge_style=style,
        match_color="0.30",
        top_k_pairs=200,
    )
    ax.set_title("M2 -- bolder  (dark gray lw=1.0 alpha=0.65)")


def render_M3(ax, dgm_a, dgm_b, m):
    """Hue contrast: Okabe-Ito green, lw=0.8, alpha=0.7."""
    style = oineus.default_matching_edge_style()
    style.update({"linewidth": 0.8, "alpha": 0.7})
    oineus.plot_matching(
        dgm_a, dgm_b, m, ax=ax,
        ordinary_edge_style=style,
        match_color="#009E73",  # Okabe-Ito green
        top_k_pairs=200,
    )
    ax.set_title("M3 -- hue contrast  (green lw=0.8 alpha=0.7)")


def render_M4(ax, dgm_a, dgm_b, m):
    """Edges coloured by length (matching cost). Bypasses plot_matching --
    plot_matching does not currently expose a per-edge cmap; this preset
    demonstrates what a future ``edge_cmap`` knob would look like."""
    # Render the two diagrams via plot_diagram so the new defaults apply.
    oineus.plot_diagram(dgm_a, ax=ax, color=oineus.OKABE_ITO_BLUE)
    oineus.plot_diagram(dgm_b, ax=ax, color=oineus.OKABE_ITO_VERMILLION)

    # Build the top-200 longest edges (by endpoint persistence).
    pairs = []
    for ia, ib in m.finite_to_finite:
        pa = dgm_a[int(ia)]
        pb = dgm_b[int(ib)]
        pairs.append((pa, pb))
    if len(m.a_to_diagonal) > 0:
        idxs = list(m.a_to_diagonal)
        projs = point_to_diagonal(dgm_a, indices=idxs)
        for li, ia in enumerate(idxs):
            pa = dgm_a[int(ia)]
            pairs.append((pa, projs[li]))
    if len(m.b_to_diagonal) > 0:
        idxs = list(m.b_to_diagonal)
        projs = point_to_diagonal(dgm_b, indices=idxs)
        for li, ib in enumerate(idxs):
            pb = dgm_b[int(ib)]
            pairs.append((pb, projs[li]))

    seg_arr = np.asarray(pairs, dtype=float)
    pers_a = np.abs(seg_arr[:, 0, 1] - seg_arr[:, 0, 0])
    pers_b = np.abs(seg_arr[:, 1, 1] - seg_arr[:, 1, 0])
    rank = np.maximum(pers_a, pers_b)
    if rank.size > 200:
        keep = np.argpartition(rank, -200)[-200:]
    else:
        keep = np.arange(rank.size)
    seg_arr = seg_arr[keep]
    lengths = np.hypot(
        seg_arr[:, 1, 0] - seg_arr[:, 0, 0],
        seg_arr[:, 1, 1] - seg_arr[:, 0, 1],
    )

    lc = LineCollection(seg_arr, cmap="magma", linewidths=0.9, alpha=0.75)
    lc.set_array(lengths)
    ax.add_collection(lc)
    cb = ax.figure.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("edge length", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_title("M4 -- edges coloured by length (magma)")


def render_M5(ax, dgm_a, dgm_b, m):
    """Annotation: few thick gray edges, top-30 only."""
    style = oineus.default_matching_edge_style()
    style.update({"linewidth": 1.5, "alpha": 0.85})
    oineus.plot_matching(
        dgm_a, dgm_b, m, ax=ax,
        ordinary_edge_style=style,
        match_color="0.20",
        top_k_pairs=30,
    )
    ax.set_title("M5 -- annotation  (top-30 thick dark gray)")


PRESETS = [
    ("M0", render_M0),
    ("M1", render_M1),
    ("M2", render_M2),
    ("M3", render_M3),
    ("M4", render_M4),
    ("M5", render_M5),
]


# ----- driver ---------------------------------------------------------------

def _square(ax, dgms):
    coords = np.concatenate(dgms)
    finite = coords[np.isfinite(coords).all(axis=1)]
    lo = float(min(finite[:, 0].min(), finite[:, 1].min()))
    hi = float(max(finite[:, 0].max(), finite[:, 1].max()))
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def main():
    print(f"Output: {OUT_DIR}")
    dgm_a = np.load(REPO_ROOT / "NPT.285200.dat.all.nonperiodic.full.d1.npy")
    dgm_b = np.load(REPO_ROOT / "NVT.1000000.dat.all.nonperiodic.full.d1.npy")
    print(f"  dgm_a (NPT d1): {dgm_a.shape}")
    print(f"  dgm_b (NVT d1): {dgm_b.shape}")

    print("  Computing W1 (q=1) Wasserstein matching...")
    m = oineus.wasserstein_matching(dgm_a, dgm_b, q=1.0)
    print(f"  matching: {len(m.finite_to_finite)} f-f, "
          f"{len(m.a_to_diagonal)} a->diag, "
          f"{len(m.b_to_diagonal)} b->diag")

    for preset_id, render_fn in PRESETS:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        render_fn(ax, dgm_a, dgm_b, m)
        _square(ax, [dgm_a, dgm_b])
        out = OUT_DIR / f"{preset_id}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.name}")

    # 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.0))
    for (preset_id, render_fn), ax in zip(PRESETS, axes.flat):
        render_fn(ax, dgm_a, dgm_b, m)
        _square(ax, [dgm_a, dgm_b])
    fig.suptitle("plot_matching -- W1 (q=1) edge style presets",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_DIR / "grid.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("  wrote grid.png")


if __name__ == "__main__":
    main()
