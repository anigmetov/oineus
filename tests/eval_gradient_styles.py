"""Gallery of plot_diagram_gradient styling presets, two gradient fields per cell.

Loads NPT.285200.d1 + NVT.1000000.d1 from the repo root, computes both the
W1 (q=1) and W2 (q=2) Wasserstein matchings between them, derives the
per-point gradient of the matching cost w.r.t. dgm_a (NPT side), and
overlays both gradient fields on a single diagram for each preset:

  * W1 grad in Okabe-Ito green
  * W2 grad in Okabe-Ito vermillion

Presets compared (2x2 grid):

  G0   - current oineus defaults (auto-cap top-200, default green; vermillion
         for the second field) -- baseline
  G1   - paper-quality thin: shaft 0.003, head 3.5/4.5, alpha 0.8
  G1.5 - in-between: shaft 0.0045, head 3.5/5.0, alpha 0.95
  G2   - annotation: shaft 0.006, head 4.0/6.0, top-30 only

Outputs land in tests/eval_gradient_styles_out/.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import mpl_scatter_density  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))

import oineus  # noqa: E402
from oineus.matching import point_to_diagonal  # noqa: E402

OUT_DIR = REPO_ROOT / "tests" / "eval_gradient_styles_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----- gradient computation ------------------------------------------------

def _wasserstein_grad(dgm_a: np.ndarray, dgm_b: np.ndarray, *, q: float):
    """grad[i] = a_i - matched_point_i (steepest INCREASE direction).

    For points matched to the diagonal the matched point is the diagonal
    projection of a_i.
    """
    m = oineus.wasserstein_matching(dgm_a, dgm_b, q=q)
    grad = np.zeros_like(dgm_a)
    for ia, ib in m.finite_to_finite:
        grad[int(ia)] = dgm_a[int(ia)] - dgm_b[int(ib)]
    if len(m.a_to_diagonal) > 0:
        idxs = list(m.a_to_diagonal)
        projs = point_to_diagonal(dgm_a, indices=idxs)
        for local_i, ia in enumerate(idxs):
            grad[int(ia)] = dgm_a[int(ia)] - projs[local_i]
    return grad


# ----- presets --------------------------------------------------------------

def _quiver_kwargs(*, width, headwidth, headlength, alpha):
    return {
        "width": width,
        "headwidth": headwidth,
        "headlength": headlength,
        "alpha": alpha,
        "angles": "xy",
        "scale_units": "xy",
        "scale": 4.0,
    }


def _overlay_two_grads(
    ax, dgm_a, grad_w1, grad_w2,
    *, top_k, qkw,
    plot_points: bool,
):
    """Render the diagram + two gradient fields.

    The first call to plot_diagram_gradient draws the underlying diagram via
    plot_diagram (so our new default grid + P11b style applies). The second
    skips it (plot_points=False) and just adds the second quiver layer.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        oineus.plot_diagram_gradient(
            dgm_a, grad_w1, ax=ax,
            top_k_arrows=top_k,
            grad_color="#009E73",  # Okabe-Ito green
            quiver_style=qkw,
            plot_points=plot_points,
        )
        oineus.plot_diagram_gradient(
            dgm_a, grad_w2, ax=ax,
            top_k_arrows=top_k,
            grad_color=oineus.OKABE_ITO_VERMILLION,
            quiver_style=qkw,
            plot_points=False,  # already drawn by the first call
        )


def render_G0(ax, dgm_a, grad_w1, grad_w2):
    """Current oineus defaults; second field gets vermillion."""
    _overlay_two_grads(
        ax, dgm_a, grad_w1, grad_w2,
        top_k=200,
        qkw={"angles": "xy", "scale_units": "xy", "scale": 4.0,
             "alpha": 0.85, "width": 0.004,
             "headwidth": 3.5, "headlength": 5.0},
        plot_points=True,
    )
    ax.set_title("G0 -- defaults  (W1 green, W2 vermillion)")


def render_G1(ax, dgm_a, grad_w1, grad_w2):
    """Paper-quality thin: shaft 0.003, head 3.5/4.5, alpha 0.8."""
    _overlay_two_grads(
        ax, dgm_a, grad_w1, grad_w2,
        top_k=200,
        qkw=_quiver_kwargs(width=0.003, headwidth=3.5, headlength=4.5, alpha=0.8),
        plot_points=True,
    )
    ax.set_title("G1 -- thin/paper  (top-200, shaft 0.003)")


def render_G1_5(ax, dgm_a, grad_w1, grad_w2):
    """In-between: shaft 0.0045, head 3.5/5.0, alpha 0.95."""
    _overlay_two_grads(
        ax, dgm_a, grad_w1, grad_w2,
        top_k=200,
        qkw=_quiver_kwargs(width=0.0045, headwidth=3.5, headlength=5.0, alpha=0.95),
        plot_points=True,
    )
    ax.set_title("G1.5 -- mid  (top-200, shaft 0.0045, alpha 0.95)")


def render_G2(ax, dgm_a, grad_w1, grad_w2):
    """Annotation style: top-30 thick."""
    _overlay_two_grads(
        ax, dgm_a, grad_w1, grad_w2,
        top_k=30,
        qkw=_quiver_kwargs(width=0.006, headwidth=4.0, headlength=6.0, alpha=0.95),
        plot_points=True,
    )
    ax.set_title("G2 -- annotation  (top-30 thick)")


PRESETS = [
    ("G0",   render_G0),
    ("G1",   render_G1),
    ("G1_5", render_G1_5),
    ("G2",   render_G2),
]


# ----- driver --------------------------------------------------------------

def _square(ax, dgm):
    finite = dgm[np.isfinite(dgm).all(axis=1)]
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
    grad_w1 = _wasserstein_grad(dgm_a, dgm_b, q=1.0)
    print("  Computing W2 (q=2) Wasserstein matching...")
    grad_w2 = _wasserstein_grad(dgm_a, dgm_b, q=2.0)
    mw1 = np.hypot(grad_w1[:, 0], grad_w1[:, 1])
    mw2 = np.hypot(grad_w2[:, 0], grad_w2[:, 1])
    print(f"  |grad_w1|: max={mw1.max():.3f}, median={np.median(mw1):.4f}")
    print(f"  |grad_w2|: max={mw2.max():.3f}, median={np.median(mw2):.4f}")

    # Per-cell PNGs at 300 dpi
    for preset_id, render_fn in PRESETS:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        render_fn(ax, dgm_a, grad_w1, grad_w2)
        _square(ax, dgm_a)
        out = OUT_DIR / f"{preset_id}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.name}")

    # 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 11.0))
    for (preset_id, render_fn), ax in zip(PRESETS, axes.flat):
        render_fn(ax, dgm_a, grad_w1, grad_w2)
        _square(ax, dgm_a)
    fig.suptitle(
        "plot_diagram_gradient -- W1 (green) + W2 (vermillion) overlay",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    grid_path = OUT_DIR / "grid.png"
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {grid_path.name}")


if __name__ == "__main__":
    main()
