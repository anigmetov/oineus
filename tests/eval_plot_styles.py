"""Gallery of persistence-diagram styling presets.

Renders the same sample data (NPT.285200.dat.all.nonperiodic.full.d{1,2}.npy
in the repo root) under nine single-diagram presets (P0-P8) and six
overlay presets (OvP0-OvP5) so the user can pick the visual that should
become the new oineus default.

No oineus library code is modified by this script. The overlay presets
that need color-keyed density (not currently exposed by plot_diagram) call
the existing _add_density_artist helper directly, which already supports
the color= -> single-hue-fade-to-transparent mode.

Usage:
    PIP_DEPS_DIR=$TMPDIR/oineus-gallery-deps
    PYTHONPATH=$PIP_DEPS_DIR:bindings/python \
        MPLCONFIGDIR=$TMPDIR/mpl-cache \
        python3 tests/eval_plot_styles.py

Outputs land in tests/eval_plot_styles_out/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import mpl_scatter_density  # noqa: F401  (registers the projection)

try:
    import cmcrameri.cm as cmc
    HAS_CMCRAMERI = True
except ImportError:
    HAS_CMCRAMERI = False

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))

from oineus.vis._matplotlib import plot_diagram, _add_density_artist  # noqa: E402

OUT_DIR = REPO_ROOT / "tests" / "eval_plot_styles_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito 8-color colorblind-safe categorical palette (Wong 2011)
OKABE_ITO = {
    "orange":      "#E69F00",
    "sky":         "#56B4E9",
    "green":       "#009E73",
    "yellow":      "#F0E442",
    "blue":        "#0072B2",
    "vermillion":  "#D55E00",
    "purple":      "#CC79A7",
    "black":       "#000000",
}

# "paper" frame rcParams -- applied via rc_context so user mpl session is
# not polluted. Matches what would go behind a future style="paper" kwarg.
PAPER_RC = {
    "figure.figsize":      (4.6, 4.6),
    "figure.dpi":          120,
    "savefig.dpi":         220,
    "savefig.bbox":        "tight",
    "axes.linewidth":      0.7,
    "axes.grid":           True,
    "grid.linestyle":      ":",
    "grid.linewidth":      0.5,
    "grid.color":          "0.85",
    "grid.alpha":          1.0,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.major.size":    4.0,
    "ytick.major.size":    4.0,
    "xtick.minor.size":    2.5,
    "ytick.minor.size":    2.5,
    "xtick.major.width":   0.7,
    "ytick.major.width":   0.7,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":           10,
    "axes.labelsize":      11,
    "axes.titlesize":      11,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.frameon":      False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
}


def _data_path(dim: int) -> Path:
    return REPO_ROOT / f"NPT.285200.dat.all.nonperiodic.full.d{dim}.npy"


def _load(dim: int) -> np.ndarray:
    arr = np.load(_data_path(dim))
    return arr


def _square_axes(ax, lo: float, hi: float, pad_frac: float = 0.04):
    span = hi - lo
    pad = pad_frac * span if span > 0 else 1.0
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def _data_extent(dgm: np.ndarray) -> tuple[float, float]:
    finite = dgm[np.isfinite(dgm).all(axis=1)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(min(finite[:, 0].min(), finite[:, 1].min()))
    hi = float(max(finite[:, 0].max(), finite[:, 1].max()))
    return lo, hi


# ----- single-diagram presets ----------------------------------------------

def render_P0(ax, dgm, *, dim_label):
    """Baseline: oineus current defaults verbatim."""
    plot_diagram({1: dgm}, ax=ax, dim_label_fmt=dim_label)
    ax.set_title("P0 — current defaults")


def render_P1(ax, dgm, *, dim_label):
    """Tightened scatter: smaller, lower alpha, white edge halo."""
    plot_diagram(
        {1: dgm}, ax=ax, dim_label_fmt=dim_label,
        point_style={"marker": "o", "s": 8.0, "alpha": 0.55,
                     "edgecolors": "white", "linewidths": 0.3},
    )
    ax.set_title("P1 — tightened scatter")


def _density_preset(ax, dgm, *, cmap, threshold=2_000, near_frac=0.04,
                    title, point_color=None):
    plot_diagram(
        {1: dgm}, ax=ax,
        color=point_color,  # only applies to outlier scatter
        cmap=cmap,
        density_threshold=threshold,
        near_diagonal_fraction=near_frac,
        density_style={"cmap": cmap, "dpi": 100, "downres_factor": 2},
        point_style={"marker": "o", "s": 14.0, "alpha": 0.9,
                     "edgecolors": "white", "linewidths": 0.4},
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
        dim_label_fmt="H{dim}",
    )
    ax.set_title(title)


def render_P2(ax, dgm, **kw):
    _density_preset(ax, dgm, cmap="magma", title="P2 — density · magma")


def render_P3(ax, dgm, **kw):
    _density_preset(ax, dgm, cmap="magma_r", title="P3 — density · magma_r",
                    point_color=OKABE_ITO["vermillion"])


def render_P4(ax, dgm, **kw):
    cmap = cmc.batlow if HAS_CMCRAMERI else "magma"
    label = "batlow" if HAS_CMCRAMERI else "magma (batlow missing)"
    _density_preset(ax, dgm, cmap=cmap, title=f"P4 — density · {label}")


def render_P5(ax, dgm, **kw):
    """Hexbin alternative -- doesn't go through plot_diagram."""
    finite = dgm[np.isfinite(dgm).all(axis=1)]
    lo, hi = _data_extent(dgm)
    span = hi - lo if hi > lo else 1.0
    ax.hexbin(
        finite[:, 0], finite[:, 1],
        gridsize=42, bins="log", cmap="magma",
        extent=(lo, hi, lo, hi),
        mincnt=1, linewidths=0.0,
    )
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.7, alpha=0.85, zorder=3)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P5 — hexbin · log count · magma")


def render_P6(ax, dgm, **kw):
    """Scatter only, paper frame, Okabe-Ito blue."""
    plot_diagram(
        {1: dgm}, ax=ax, color=OKABE_ITO["blue"],
        use_density=False,
        point_style={"marker": "o", "s": 10.0, "alpha": 0.6,
                     "edgecolors": "white", "linewidths": 0.3},
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
        dim_label_fmt="H{dim}",
    )
    ax.set_title("P6 — scatter · Okabe-Ito blue")


def render_P7(ax, dgm, **kw):
    """Recommended: hybrid density + crisp outliers, magma."""
    _density_preset(
        ax, dgm, cmap="magma", title="P7 — hybrid · magma (recommended)",
        point_color=OKABE_ITO["vermillion"],
    )


def render_P8(ax, dgm, **kw):
    """P7 variant with receding-density cmap (magma_r)."""
    _density_preset(
        ax, dgm, cmap="magma_r", title="P8 — hybrid · magma_r (receding)",
        point_color=OKABE_ITO["vermillion"],
    )


def render_P9(ax, dgm, **kw):
    """Color-keyed density: fade-to-transparent in a single hue.

    Empty axes regions stay white (background shows through), so we get
    the clean look of P3/P8 without tinting the whole plot. This is what
    plot_diagram would do post-extension when called with color= and
    density triggers."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9 — color-keyed density (recommended)")


def render_P10(ax, dgm, **kw):
    """P9 variant in Okabe-Ito blue."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density(
        ax, dgm, color=OKABE_ITO["blue"], lo=lo, hi=hi,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P10 — color-keyed · blue")


# ----- P6 variants: stronger scatter ---------------------------------------

def render_P6a(ax, dgm, **kw):
    """P6 with alpha bumped from 0.6 -> 0.85, same sky blue."""
    plot_diagram(
        {1: dgm}, ax=ax, color=OKABE_ITO["sky"],
        use_density=False,
        point_style={"marker": "o", "s": 10.0, "alpha": 0.85,
                     "edgecolors": "white", "linewidths": 0.3},
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
        dim_label_fmt="H{dim}",
    )
    ax.set_title("P6a — scatter · sky blue · α=0.85")


def render_P6b(ax, dgm, **kw):
    """P6 with saturated Okabe-Ito blue (#0072B2), alpha unchanged."""
    plot_diagram(
        {1: dgm}, ax=ax, color=OKABE_ITO["blue"],
        use_density=False,
        point_style={"marker": "o", "s": 10.0, "alpha": 0.6,
                     "edgecolors": "white", "linewidths": 0.3},
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
        dim_label_fmt="H{dim}",
    )
    ax.set_title("P6b — scatter · saturated blue · α=0.6")


def render_P6c(ax, dgm, **kw):
    """P6 with saturated blue + bumped alpha."""
    plot_diagram(
        {1: dgm}, ax=ax, color=OKABE_ITO["blue"],
        use_density=False,
        point_style={"marker": "o", "s": 10.0, "alpha": 0.8,
                     "edgecolors": "white", "linewidths": 0.3},
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
        dim_label_fmt="H{dim}",
    )
    ax.set_title("P6c — scatter · saturated blue · α=0.8")


def render_P6d(ax, dgm, **kw):
    """P6 with vermillion -- saturation comparison vs blue."""
    plot_diagram(
        {1: dgm}, ax=ax, color=OKABE_ITO["vermillion"],
        use_density=False,
        point_style={"marker": "o", "s": 10.0, "alpha": 0.8,
                     "edgecolors": "white", "linewidths": 0.3},
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
        dim_label_fmt="H{dim}",
    )
    ax.set_title("P6d — scatter · vermillion · α=0.8")


# ----- P9 variants: fix the "whitish core" --------------------------------

def _color_density_with_norm(ax, dgm, *, color, lo, hi, near_frac=0.04,
                             gamma=0.5, dpi=100, downres=2,
                             alpha_max=1.0):
    """Like _draw_color_keyed_density but with explicit norm/dpi controls."""
    import matplotlib.colors as mcolors
    from mpl_scatter_density import ScatterDensityArtist

    finite = dgm[np.isfinite(dgm).all(axis=1)]
    span = hi - lo if hi > lo else 1.0
    near_thr = near_frac * span
    pers = finite[:, 1] - finite[:, 0]
    near_mask = pers <= near_thr
    near = finite[near_mask]
    far = finite[~near_mask]

    if near.shape[0] > 0:
        norm = mcolors.PowerNorm(gamma=gamma)
        artist = ScatterDensityArtist(
            ax, near[:, 0], near[:, 1],
            norm=norm, color=color,
            dpi=dpi, downres_factor=downres,
        )
        ax.add_artist(artist)
        if alpha_max < 1.0:
            artist.set_alpha(alpha_max)
    if far.shape[0] > 0:
        ax.scatter(
            far[:, 0], far[:, 1], c=color,
            s=14.0, alpha=0.9,
            edgecolors="white", linewidths=0.4, zorder=4,
        )


def render_P9a(ax, dgm, **kw):
    """P9 with γ=1.0 (linear): centres saturate harder than periphery."""
    lo, hi = _data_extent(dgm)
    _color_density_with_norm(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        gamma=1.0, dpi=100, downres=2,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9a — color-keyed · γ=1.0 (linear)")


def render_P9b(ax, dgm, **kw):
    """P9 with γ=0.7 (compromise: a bit of low-end boost, less periphery halo)."""
    lo, hi = _data_extent(dgm)
    _color_density_with_norm(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        gamma=0.7, dpi=100, downres=2,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9b — color-keyed · γ=0.7")


def render_P9c(ax, dgm, **kw):
    """P9 with very high histogram DPI -- grain disappears, centres fill in."""
    lo, hi = _data_extent(dgm)
    _color_density_with_norm(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        gamma=0.5, dpi=300, downres=1,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9c — color-keyed · dpi=300, no downres")


def _gaussian_smoothed_density(ax, dgm, *, color, lo, hi,
                               near_frac=0.04, grid=400, sigma=2.5,
                               alpha_max=1.0):
    """KDE-flavored: 2D histogram smoothed with Gaussian, rendered as imshow.

    Continuous-looking density, no per-pixel grain. ``sigma`` is in grid
    cells, so the perceptual smoothing scales with the grid resolution.
    """
    from scipy.ndimage import gaussian_filter
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    finite = dgm[np.isfinite(dgm).all(axis=1)]
    span = hi - lo if hi > lo else 1.0
    near_thr = near_frac * span
    pers = finite[:, 1] - finite[:, 0]
    near = finite[pers <= near_thr]
    far = finite[pers > near_thr]

    pad = 0.04 * span
    edges = np.linspace(lo - pad, hi + pad, grid + 1)
    hist, _, _ = np.histogram2d(near[:, 0], near[:, 1], bins=[edges, edges])
    smoothed = gaussian_filter(hist.T, sigma=sigma)
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()

    rgba = to_rgba(color)
    cmap = LinearSegmentedColormap.from_list(
        f"fade_{color}",
        [(rgba[0], rgba[1], rgba[2], 0.0),
         (rgba[0], rgba[1], rgba[2], alpha_max)],
        N=256,
    )
    ax.imshow(
        smoothed, origin="lower", extent=(lo - pad, hi + pad, lo - pad, hi + pad),
        cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear",
        zorder=1,
    )
    if far.shape[0] > 0:
        ax.scatter(
            far[:, 0], far[:, 1], c=color,
            s=14.0, alpha=0.9,
            edgecolors="white", linewidths=0.4, zorder=4,
        )


def render_P9d(ax, dgm, **kw):
    """Gaussian-smoothed 2D histogram (no mpl_scatter_density grain)."""
    lo, hi = _data_extent(dgm)
    _gaussian_smoothed_density(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        grid=400, sigma=2.5,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9d — gaussian-smoothed density")


def render_P9e(ax, dgm, **kw):
    """Gaussian-smoothed with tighter sigma + cream backdrop axes."""
    lo, hi = _data_extent(dgm)
    ax.set_facecolor("#FFFBE5")  # cream wash like magma_r low-end
    _gaussian_smoothed_density(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        grid=400, sigma=1.8,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9e — gauss-smoothed · cream backdrop")


# ----- P9 outlier-scatter edge fixes (the white-halo problem) --------------

def _draw_color_keyed_density_v2(ax, dgm, *, color, lo, hi,
                                 near_frac=0.04, alpha_max=1.0,
                                 outlier_edge="white",
                                 outlier_edge_lw=0.4,
                                 outlier_alpha=0.9,
                                 outlier_size=14.0):
    """Color-keyed density bulk + scatter outliers with configurable edge."""
    finite = dgm[np.isfinite(dgm).all(axis=1)]
    span = hi - lo if hi > lo else 1.0
    near_thr = near_frac * span
    pers = finite[:, 1] - finite[:, 0]
    near_mask = pers <= near_thr
    near = finite[near_mask]
    far = finite[~near_mask]

    if near.shape[0] > 0:
        artist = _add_density_artist(
            ax, near[:, 0], near[:, 1],
            color=color,
            style={"dpi": 100, "downres_factor": 2},
        )
        if alpha_max < 1.0:
            artist.set_alpha(alpha_max)
    if far.shape[0] > 0:
        kwargs = dict(
            c=color, s=outlier_size, alpha=outlier_alpha,
            linewidths=outlier_edge_lw, zorder=4,
        )
        if outlier_edge is not None:
            kwargs["edgecolors"] = outlier_edge
        ax.scatter(far[:, 0], far[:, 1], **kwargs)


def _darken(color: str, amount: float = 0.55) -> tuple:
    """Multiply RGB channels by ``amount`` to get a darker shade."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r * amount, g * amount, b * amount)


def render_P9f(ax, dgm, **kw):
    """P9 with NO outlier edge -- markers blend smoothly when overlapping."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v2(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        outlier_edge=None, outlier_edge_lw=0.0,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9f — outliers · no edge")


def render_P9g(ax, dgm, **kw):
    """P9 with darker-shade-of-color edges -- adds weight without lightening."""
    lo, hi = _data_extent(dgm)
    dark = _darken(OKABE_ITO["vermillion"], 0.55)
    _draw_color_keyed_density_v2(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        outlier_edge=dark, outlier_edge_lw=0.5,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9g — outliers · darker-shade edge")


def render_P9h(ax, dgm, **kw):
    """P9 with thin black edge (low lw) -- traditional but reads well."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v2(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        outlier_edge="black", outlier_edge_lw=0.2,
        outlier_alpha=0.85,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9h — outliers · thin black edge")


def render_P9i(ax, dgm, **kw):
    """P9 with smaller markers (s=8) + white edge -- halo less dominant."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v2(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        outlier_edge="white", outlier_edge_lw=0.3,
        outlier_size=8.0,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9i — outliers · smaller marker (s=8) + white edge")


def render_P9j(ax, dgm, **kw):
    """P9 outliers: no edge + lower alpha so overlap accumulates colour."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v2(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        outlier_edge=None, outlier_edge_lw=0.0,
        outlier_alpha=0.55, outlier_size=18.0,
    )
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("P9j — outliers · no edge + α=0.55 + s=18")


# ----- P9g iterations: smaller markers + denser-looking density ------------

def _draw_color_keyed_density_v3(ax, dgm, *, color, lo, hi,
                                 near_frac=0.04,
                                 # density knobs
                                 density_kind="hist",  # "hist" or "smooth"
                                 hist_dpi=100, hist_downres=2,
                                 hist_gamma=0.5,
                                 smooth_grid=400, smooth_sigma=2.5,
                                 alpha_max=1.0,
                                 # outlier knobs
                                 outlier_edge=None,
                                 outlier_edge_lw=0.0,
                                 outlier_alpha=0.9,
                                 outlier_size=14.0):
    """Like v2 but with a density_kind switch and full set of knobs."""
    import matplotlib.colors as mcolors
    from mpl_scatter_density import ScatterDensityArtist

    finite = dgm[np.isfinite(dgm).all(axis=1)]
    span = hi - lo if hi > lo else 1.0
    near_thr = near_frac * span
    pers = finite[:, 1] - finite[:, 0]
    near_mask = pers <= near_thr
    near = finite[near_mask]
    far = finite[~near_mask]

    if near.shape[0] > 0:
        if density_kind == "hist":
            norm = mcolors.PowerNorm(gamma=hist_gamma)
            artist = ScatterDensityArtist(
                ax, near[:, 0], near[:, 1],
                norm=norm, color=color,
                dpi=hist_dpi, downres_factor=hist_downres,
            )
            ax.add_artist(artist)
            if alpha_max < 1.0:
                artist.set_alpha(alpha_max)
        elif density_kind == "smooth":
            from scipy.ndimage import gaussian_filter
            from matplotlib.colors import LinearSegmentedColormap, to_rgba

            pad = 0.04 * span
            edges = np.linspace(lo - pad, hi + pad, smooth_grid + 1)
            hist, _, _ = np.histogram2d(
                near[:, 0], near[:, 1], bins=[edges, edges]
            )
            smoothed = gaussian_filter(hist.T, sigma=smooth_sigma)
            if smoothed.max() > 0:
                smoothed = smoothed / smoothed.max()
            # Mask cells whose centre lies strictly below the diagonal
            # (death < birth) -- otherwise the gaussian smear bleeds into
            # the unphysical region.
            centres = 0.5 * (edges[:-1] + edges[1:])
            xx, yy = np.meshgrid(centres, centres)
            below_diag = yy < xx
            smoothed = np.where(below_diag, np.nan, smoothed)
            rgba = to_rgba(color)
            cmap = LinearSegmentedColormap.from_list(
                f"fade_{color}",
                [(rgba[0], rgba[1], rgba[2], 0.0),
                 (rgba[0], rgba[1], rgba[2], alpha_max)],
                N=256,
            )
            cmap.set_bad((0, 0, 0, 0))
            ax.imshow(
                smoothed, origin="lower",
                extent=(lo - pad, hi + pad, lo - pad, hi + pad),
                cmap=cmap, vmin=0.0, vmax=1.0,
                interpolation="bilinear", zorder=1,
            )
        else:
            raise ValueError(density_kind)

    if far.shape[0] > 0:
        kwargs = dict(
            c=color, s=outlier_size, alpha=outlier_alpha,
            linewidths=outlier_edge_lw, zorder=4,
        )
        if outlier_edge is not None:
            kwargs["edgecolors"] = outlier_edge
        ax.scatter(far[:, 0], far[:, 1], **kwargs)


def _set_axes(ax, lo, hi, title):
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    _square_axes(ax, lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)


_DARK_VERM = _darken("#D55E00", 0.55)


def render_P9k(ax, dgm, **kw):
    """P9g + s=8: same edge color but smaller markers."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v3(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        outlier_edge=_DARK_VERM, outlier_edge_lw=0.45,
        outlier_size=8.0,
    )
    _set_axes(ax, lo, hi, "P9k — P9g + s=8")


def render_P9l(ax, dgm, **kw):
    """P9k + linear-norm density (γ=1.0): high-density cells max out faster."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v3(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        hist_gamma=1.0,
        outlier_edge=_DARK_VERM, outlier_edge_lw=0.45,
        outlier_size=8.0,
    )
    _set_axes(ax, lo, hi, "P9l — P9k + density γ=1.0")


def render_P9m(ax, dgm, **kw):
    """P9k + gaussian-smoothed density (DROPPED -- crosses the diagonal).

    Kept as a stub so other indices don't shift; rendered as a placeholder
    label so it's obvious in any old grids that this preset was retired.
    """
    ax.text(0.5, 0.5, "P9m\nretired\n(crosses diagonal)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=14, color="0.5")
    ax.set_axis_off()


def render_P9n(ax, dgm, **kw):
    """P9k + coarser histogram (downres=6): bigger cells, more counts each."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v3(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        hist_dpi=100, hist_downres=6, hist_gamma=0.5,
        outlier_edge=_DARK_VERM, outlier_edge_lw=0.45,
        outlier_size=8.0,
    )
    _set_axes(ax, lo, hi, "P9n — P9k + coarser hist (downres=6)")


def render_P9o(ax, dgm, **kw):
    """P9k + coarser hist + linear γ: most aggressive density saturation."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v3(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        hist_dpi=100, hist_downres=6, hist_gamma=1.0,
        outlier_edge=_DARK_VERM, outlier_edge_lw=0.45,
        outlier_size=8.0,
    )
    _set_axes(ax, lo, hi, "P9o — P9k + downres=6 + γ=1.0")


def render_P9p(ax, dgm, **kw):
    """P9k + gauss-smoothed σ=1.2 (tighter, denser band stays narrow).

    Below-diagonal bleed is masked (death < birth cells dropped)."""
    lo, hi = _data_extent(dgm)
    _draw_color_keyed_density_v3(
        ax, dgm, color=OKABE_ITO["vermillion"], lo=lo, hi=hi,
        density_kind="smooth", smooth_grid=400, smooth_sigma=1.2,
        outlier_edge=_DARK_VERM, outlier_edge_lw=0.45,
        outlier_size=8.0,
    )
    _set_axes(ax, lo, hi, "P9p — P9k + gauss-smoothed σ=1.2 (tight)")


# ----- P11: pure-scatter variants (architectural pick: plot_diagram = scatter)

def _scatter_only(ax, dgm, *, color, edge, edge_lw, size, alpha,
                  lo=None, hi=None, title=""):
    """Pure scatter render -- no density, no near/far split."""
    finite = dgm[np.isfinite(dgm).all(axis=1)]
    if lo is None or hi is None:
        lo_, hi_ = _data_extent(dgm)
        lo = lo if lo is not None else lo_
        hi = hi if hi is not None else hi_
    kwargs = dict(c=color, s=size, alpha=alpha,
                  linewidths=edge_lw, zorder=4)
    if edge is not None:
        kwargs["edgecolors"] = edge
    ax.scatter(finite[:, 0], finite[:, 1], **kwargs)
    _set_axes(ax, lo, hi, title)


def render_P11(ax, dgm, **kw):
    """Pure scatter, vermillion + dark-shade edge (P9p aesthetic, no density)."""
    lo, hi = _data_extent(dgm)
    _scatter_only(
        ax, dgm, color=OKABE_ITO["vermillion"],
        edge=_DARK_VERM, edge_lw=0.45,
        size=8.0, alpha=0.85,
        lo=lo, hi=hi,
        title="P11 — pure scatter · vermillion · s=8 · dark edge",
    )


def render_P11a(ax, dgm, **kw):
    """P11 with no edge -- baseline pure-color scatter."""
    lo, hi = _data_extent(dgm)
    _scatter_only(
        ax, dgm, color=OKABE_ITO["vermillion"],
        edge=None, edge_lw=0.0,
        size=8.0, alpha=0.85,
        lo=lo, hi=hi,
        title="P11a — pure scatter · vermillion · no edge",
    )


def render_P11b(ax, dgm, **kw):
    """P11 in Okabe-Ito saturated blue (alternate hue)."""
    lo, hi = _data_extent(dgm)
    blue = OKABE_ITO["blue"]
    dark_blue = _darken(blue, 0.55)
    _scatter_only(
        ax, dgm, color=blue,
        edge=dark_blue, edge_lw=0.45,
        size=8.0, alpha=0.85,
        lo=lo, hi=hi,
        title="P11b — pure scatter · blue · s=8 · dark edge",
    )


def render_P11c(ax, dgm, **kw):
    """P11 with smaller markers (s=5) -- very dense diagrams."""
    lo, hi = _data_extent(dgm)
    _scatter_only(
        ax, dgm, color=OKABE_ITO["vermillion"],
        edge=_DARK_VERM, edge_lw=0.35,
        size=5.0, alpha=0.85,
        lo=lo, hi=hi,
        title="P11c — pure scatter · vermillion · s=5 · dark edge",
    )


SINGLE_PRESETS = [
    ("P0", render_P0, False),  # (id, fn, paper-frame?)
    ("P1", render_P1, False),
    ("P2", render_P2, True),
    ("P3", render_P3, True),
    ("P4", render_P4, True),
    ("P5", render_P5, True),
    ("P6", render_P6, True),
    ("P7", render_P7, True),
    ("P8", render_P8, True),
    ("P9", render_P9, True),
    ("P10", render_P10, True),
    # P6 strengthened variants
    ("P6a", render_P6a, True),
    ("P6b", render_P6b, True),
    ("P6c", render_P6c, True),
    ("P6d", render_P6d, True),
    # P9 fix-the-whitish-core variants
    ("P9a", render_P9a, True),
    ("P9b", render_P9b, True),
    ("P9c", render_P9c, True),
    ("P9d", render_P9d, True),
    ("P9e", render_P9e, True),
    # P9 outlier-edge fixes (the actual user complaint -- white halo around
    # clustered scatter dots)
    ("P9f", render_P9f, True),
    ("P9g", render_P9g, True),
    ("P9h", render_P9h, True),
    ("P9i", render_P9i, True),
    ("P9j", render_P9j, True),
    # P9g iterations: smaller markers + denser-looking density bulk
    ("P9k", render_P9k, True),
    ("P9l", render_P9l, True),
    ("P9m", render_P9m, True),
    ("P9n", render_P9n, True),
    ("P9o", render_P9o, True),
    ("P9p", render_P9p, True),
    # Pure scatter -- the proposed plot_diagram default (no density)
    ("P11",  render_P11,  True),
    ("P11a", render_P11a, True),
    ("P11b", render_P11b, True),
    ("P11c", render_P11c, True),
]


# ----- overlay (2-diagram) presets ----------------------------------------

def _make_synthetic_generated(dgm_real: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Mimic a "generated vs ground-truth" comparison.

    Adds Gaussian noise to deaths and randomly drops 5% of points so the
    overlay has clear-but-not-identical structure.
    """
    rng = np.random.default_rng(seed)
    n = dgm_real.shape[0]
    keep = rng.random(n) > 0.05
    dgm = dgm_real[keep].copy()
    pers = dgm[:, 1] - dgm[:, 0]
    sigma = 0.20 * pers.mean()
    dgm[:, 1] = dgm[:, 1] + rng.normal(0.0, sigma, size=dgm.shape[0])
    dgm[:, 1] = np.maximum(dgm[:, 1], dgm[:, 0])  # keep death >= birth
    return dgm


def _overlay_axes_setup(ax, lo, hi):
    _square_axes(ax, lo, hi)
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1],
            color="0.55", lw=0.7, alpha=0.85, zorder=2)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")


def render_OvP0(ax, dgm_a, dgm_b):
    """Current defaults, two scatters overlaid via two plot_diagram calls."""
    plot_diagram({1: dgm_a}, ax=ax, color=OKABE_ITO["vermillion"],
                 dim_label_fmt="real H{dim}")
    plot_diagram({1: dgm_b}, ax=ax, color=OKABE_ITO["blue"],
                 dim_label_fmt="gen H{dim}")
    ax.set_title("OvP0 — current defaults (overlay)")
    ax.legend(fontsize=8, loc="lower right")


def render_OvP1(ax, dgm_a, dgm_b):
    """Tightened scatter overlay, distinct Okabe-Ito colors."""
    style = {"marker": "o", "s": 9.0, "alpha": 0.55,
             "edgecolors": "white", "linewidths": 0.3}
    plot_diagram(
        {1: dgm_a}, ax=ax, color=OKABE_ITO["vermillion"],
        use_density=False, point_style=style, dim_label_fmt="real H{dim}",
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
    )
    plot_diagram(
        {1: dgm_b}, ax=ax, color=OKABE_ITO["blue"],
        use_density=False, point_style=style, dim_label_fmt="gen H{dim}",
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
    )
    ax.set_title("OvP1 — scatter · vermillion + blue")
    ax.legend(fontsize=8, loc="lower right")


def _draw_color_keyed_density(ax, dgm, *, color, lo, hi, near_frac=0.04,
                              alpha_max=1.0):
    """Render dgm as color-keyed density bulk + crisp outlier scatter.

    This is the path the proposed `plot_diagram` extension would take when
    the caller passes both color= AND density triggers. We do it inline
    here to demonstrate the result without modifying library code.

    ``alpha_max`` caps the opacity of the densest pixels. Set <1.0 in
    overlay scenarios so two stacked rasters can blend visibly.
    """
    finite = dgm[np.isfinite(dgm).all(axis=1)]
    span = hi - lo if hi > lo else 1.0
    near_thr = near_frac * span
    pers = finite[:, 1] - finite[:, 0]
    near_mask = pers <= near_thr
    near = finite[near_mask]
    far = finite[~near_mask]

    if near.shape[0] > 0:
        artist = _add_density_artist(
            ax, near[:, 0], near[:, 1],
            color=color,
            style={"dpi": 100, "downres_factor": 2},
        )
        if alpha_max < 1.0:
            artist.set_alpha(alpha_max)
    if far.shape[0] > 0:
        ax.scatter(
            far[:, 0], far[:, 1], c=color,
            s=14.0, alpha=0.9,
            edgecolors="white", linewidths=0.4, zorder=4,
        )


def render_OvP2(ax, dgm_a, dgm_b):
    """Color-keyed density+density with alpha cap so overlap blends visibly."""
    lo_a, hi_a = _data_extent(dgm_a)
    lo_b, hi_b = _data_extent(dgm_b)
    lo, hi = min(lo_a, lo_b), max(hi_a, hi_b)
    _overlay_axes_setup(ax, lo, hi)

    _draw_color_keyed_density(ax, dgm_a, color=OKABE_ITO["vermillion"],
                              lo=lo, hi=hi, alpha_max=0.55)
    _draw_color_keyed_density(ax, dgm_b, color=OKABE_ITO["blue"],
                              lo=lo, hi=hi, alpha_max=0.55)
    from matplotlib.patches import Patch
    proxies = [
        Patch(color=OKABE_ITO["vermillion"], alpha=0.55, label="real H1"),
        Patch(color=OKABE_ITO["blue"], alpha=0.55, label="gen H1"),
    ]
    ax.legend(handles=proxies, fontsize=8, loc="lower right")
    ax.set_title("OvP2 — density · color-keyed (α=0.55)")


def render_OvP3(ax, dgm_a, dgm_b):
    """OvP1-style scatter overlay -- recommended for 2-diagram comparison.

    For overlay-with-density, OvP2 is the alternative; but small-marker
    scatter+scatter is in practice clearer for typical sizes (<= 50k each).
    """
    render_OvP1(ax, dgm_a, dgm_b)
    ax.set_title("OvP3 — scatter · recommended for overlay")


def render_OvP4(ax, dgm_a, dgm_b):
    """KDE filled-contour overlay. Per-diagram single-hue contour set."""
    from scipy.stats import gaussian_kde

    lo_a, hi_a = _data_extent(dgm_a)
    lo_b, hi_b = _data_extent(dgm_b)
    lo, hi = min(lo_a, lo_b), max(hi_a, hi_b)
    pad = 0.04 * (hi - lo)
    grid_x, grid_y = np.mgrid[lo - pad:hi + pad:200j, lo - pad:hi + pad:200j]
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])

    def _draw(dgm, color):
        finite = dgm[np.isfinite(dgm).all(axis=1)]
        if finite.shape[0] < 5:
            return
        kde = gaussian_kde(finite.T, bw_method=0.10)
        density = kde(positions).reshape(grid_x.shape)
        levels = np.linspace(density.max() * 0.10, density.max(), 5)
        # Build a single-hue alpha-ramp colormap from the diagram color.
        from matplotlib.colors import LinearSegmentedColormap, to_rgba
        rgba = to_rgba(color)
        cmap = LinearSegmentedColormap.from_list(
            f"fade_{color}",
            [(rgba[0], rgba[1], rgba[2], 0.0),
             (rgba[0], rgba[1], rgba[2], 0.85)],
            N=256,
        )
        ax.contourf(grid_x, grid_y, density, levels=levels,
                    cmap=cmap, antialiased=True)

    _draw(dgm_a, OKABE_ITO["vermillion"])
    _draw(dgm_b, OKABE_ITO["blue"])
    _overlay_axes_setup(ax, lo, hi)
    from matplotlib.patches import Patch
    proxies = [
        Patch(color=OKABE_ITO["vermillion"], alpha=0.7, label="real H1"),
        Patch(color=OKABE_ITO["blue"], alpha=0.7, label="gen H1"),
    ]
    ax.legend(handles=proxies, fontsize=8, loc="lower right")
    ax.set_title("OvP4 — KDE contour overlay")


def render_OvP5(ax, dgm_a, dgm_b):
    """Bad-case: cmap density (defaults) for one + foreign-color scatter
    for the other. Shows why the existing 'don't overlay density' policy
    was conservative -- and why color-keying fixes it."""
    plot_diagram(
        {1: dgm_a}, ax=ax,
        density_threshold=2_000, cmap="viridis",
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
        dim_label_fmt="real H{dim}",
    )
    plot_diagram(
        {1: dgm_b}, ax=ax, color=OKABE_ITO["vermillion"],
        use_density=False,
        point_style={"marker": "o", "s": 10.0, "alpha": 0.55,
                     "edgecolors": "white", "linewidths": 0.3},
        diagonal_style={"linestyle": "-", "color": "0.55",
                        "alpha": 0.85, "linewidth": 0.7},
        dim_label_fmt="gen H{dim}",
    )
    ax.set_title("OvP5 — bad case (cmap density vs scatter)")
    ax.legend(fontsize=8, loc="lower right")


OVERLAY_PRESETS = [
    ("OvP0", render_OvP0),
    ("OvP1", render_OvP1),
    ("OvP2", render_OvP2),
    ("OvP3", render_OvP3),
    ("OvP4", render_OvP4),
    ("OvP5", render_OvP5),
]


# ----- driver ---------------------------------------------------------------

def _render_single_preset(preset_id, render_fn, paper_frame, dgm, dim, out_path):
    rc = PAPER_RC if paper_frame else {}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        render_fn(ax, dgm, dim_label=f"H{dim}")
        if paper_frame:
            lo, hi = _data_extent(dgm)
            _square_axes(ax, lo, hi)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _render_overlay_preset(preset_id, render_fn, dgm_a, dgm_b, out_path):
    with mpl.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        render_fn(ax, dgm_a, dgm_b)
        lo_a, hi_a = _data_extent(dgm_a)
        lo_b, hi_b = _data_extent(dgm_b)
        _square_axes(ax, min(lo_a, lo_b), max(hi_a, hi_b))
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def render_grid_single(dgm, dim, out_path, *, presets=None, title_suffix=""):
    """Render a grid of presets. ``presets`` defaults to SINGLE_PRESETS but
    callers can pass a focused subset."""
    presets = presets if presets is not None else SINGLE_PRESETS
    rc = PAPER_RC.copy()
    n = len(presets)
    cols = 4 if n > 6 else 3
    rows = (n + cols - 1) // cols
    fig_w = 4.5 * cols
    fig_h = 4.5 * rows
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                                 squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for (preset_id, render_fn, paper_frame), ax in zip(
            presets, axes.flat
        ):
            ax.set_visible(True)
            render_fn(ax, dgm, dim_label=f"H{dim}")
            lo, hi = _data_extent(dgm)
            _square_axes(ax, lo, hi)
        fig.suptitle(
            f"Single-diagram presets{title_suffix} · d{dim} "
            f"({dgm.shape[0]} pts)",
            fontsize=14, y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def render_grid_overlay(dgm_a, dgm_b, out_path):
    rc = PAPER_RC.copy()
    rc["figure.figsize"] = (13.5, 9.5)
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.5))
        for (preset_id, render_fn), ax in zip(OVERLAY_PRESETS, axes.flat):
            render_fn(ax, dgm_a, dgm_b)
            lo_a, hi_a = _data_extent(dgm_a)
            lo_b, hi_b = _data_extent(dgm_b)
            _square_axes(ax, min(lo_a, lo_b), max(hi_a, hi_b))
        fig.suptitle(
            f"Overlay presets · real d1 ({dgm_a.shape[0]} pts) "
            f"vs synthetic-generated ({dgm_b.shape[0]} pts)",
            fontsize=14, y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    print(f"Output: {OUT_DIR}")
    if not HAS_CMCRAMERI:
        print("  (cmcrameri missing -- P4 will fall back to magma)")

    dgm_d1 = _load(1)
    dgm_d2 = _load(2)
    print(f"  d1 loaded: {dgm_d1.shape}")
    print(f"  d2 loaded: {dgm_d2.shape}")

    dgm_gen = _make_synthetic_generated(dgm_d1, seed=0)
    print(f"  synthetic generated: {dgm_gen.shape}")

    # Per-cell PNGs
    for preset_id, render_fn, paper_frame in SINGLE_PRESETS:
        for dim, dgm in [(1, dgm_d1), (2, dgm_d2)]:
            out = OUT_DIR / f"{preset_id}_d{dim}.png"
            _render_single_preset(preset_id, render_fn, paper_frame,
                                  dgm, dim, out)
            print(f"  wrote {out.name}")

    for preset_id, render_fn in OVERLAY_PRESETS:
        out = OUT_DIR / f"{preset_id}.png"
        _render_overlay_preset(preset_id, render_fn, dgm_d1, dgm_gen, out)
        print(f"  wrote {out.name}")

    # Full grid (all 20 presets)
    render_grid_single(dgm_d1, 1, OUT_DIR / "grid_d1.png")
    print(f"  wrote grid_d1.png")
    render_grid_single(dgm_d2, 2, OUT_DIR / "grid_d2.png")
    print(f"  wrote grid_d2.png")

    # Focused sub-grids for the two areas the user is iterating on
    p6_focus = [
        ("P6", render_P6, True),
        ("P6a", render_P6a, True),
        ("P6b", render_P6b, True),
        ("P6c", render_P6c, True),
        ("P6d", render_P6d, True),
    ]
    render_grid_single(
        dgm_d1, 1, OUT_DIR / "grid_P6_focus_d1.png",
        presets=p6_focus, title_suffix=" · P6 strengthened",
    )
    print("  wrote grid_P6_focus_d1.png")

    p9_focus = [
        ("P9", render_P9, True),
        ("P9a", render_P9a, True),
        ("P9b", render_P9b, True),
        ("P9c", render_P9c, True),
        ("P9d", render_P9d, True),
        ("P9e", render_P9e, True),
    ]
    render_grid_single(
        dgm_d1, 1, OUT_DIR / "grid_P9_focus_d1.png",
        presets=p9_focus, title_suffix=" · P9 density-rendering variants",
    )
    print("  wrote grid_P9_focus_d1.png")

    p9_edge_focus = [
        ("P9", render_P9, True),
        ("P9f", render_P9f, True),
        ("P9g", render_P9g, True),
        ("P9h", render_P9h, True),
        ("P9i", render_P9i, True),
        ("P9j", render_P9j, True),
    ]
    render_grid_single(
        dgm_d1, 1, OUT_DIR / "grid_P9_edge_focus_d1.png",
        presets=p9_edge_focus, title_suffix=" · P9 outlier-edge fixes",
    )
    print("  wrote grid_P9_edge_focus_d1.png")

    p9_density_focus = [
        ("P9g", render_P9g, True),
        ("P9k", render_P9k, True),
        ("P9l", render_P9l, True),
        ("P9m", render_P9m, True),
        ("P9n", render_P9n, True),
        ("P9o", render_P9o, True),
        ("P9p", render_P9p, True),
    ]
    render_grid_single(
        dgm_d1, 1, OUT_DIR / "grid_P9_density_focus_d1.png",
        presets=p9_density_focus,
        title_suffix=" · P9g iter (smaller markers + denser bulk)",
    )
    print("  wrote grid_P9_density_focus_d1.png")

    p11_focus = [
        ("P6c",  render_P6c,  True),  # for comparison: scatter saturated blue
        ("P11",  render_P11,  True),
        ("P11a", render_P11a, True),
        ("P11b", render_P11b, True),
        ("P11c", render_P11c, True),
        ("P9p",  render_P9p,  True),  # the density-mode pick, side-by-side
    ]
    render_grid_single(
        dgm_d1, 1, OUT_DIR / "grid_P11_focus_d1.png",
        presets=p11_focus,
        title_suffix=" · pure-scatter (proposed plot_diagram default)",
    )
    print("  wrote grid_P11_focus_d1.png")

    render_grid_overlay(dgm_d1, dgm_gen, OUT_DIR / "grid_overlay.png")
    print(f"  wrote grid_overlay.png")


if __name__ == "__main__":
    main()
