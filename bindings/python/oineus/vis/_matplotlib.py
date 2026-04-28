"""Matplotlib implementations of the oineus plotting helpers.

The four user-facing entry points are ``plot_diagram``, ``plot_matching``,
``plot_diagram_gradient`` and ``plot_chain``. Backend-agnostic helpers live
in ``_common``; default style dicts in ``_styles``.
"""
from __future__ import annotations

import typing
import warnings

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

try:
    import mpl_scatter_density  # noqa: F401
    _HAS_MPL_SCATTER_DENSITY = True
except Exception:
    _HAS_MPL_SCATTER_DENSITY = False

from . import _common
from . import _styles
from ._common import (
    _array_diagram,
    _build_diagram_arrays,
    _coerce_chain,
    _coerce_diagram_with_grad,
    _compute_plot_limits,
    _id_to_grid,
    _point_coords_for_edge,
    _resolve_color,
    _resolve_style,
    _shift_for_log,
    _split_near_diagonal,
    _to_dim_diagrams,
)
from ._styles import (
    DEFAULT_DENSITY_STYLE,
    DEFAULT_DENSITY_THRESHOLD,
    DEFAULT_GRADIENT_TOP_K_ARROWS,
    DEFAULT_MATCHING_EDGE_QUANTILE,
    default_chain_edge_style,
    default_chain_tetrahedron_style,
    default_chain_triangle_style,
    default_chain_vertex_style,
    default_density_style,
    default_diagonal_style,
    default_diagonal_projection_a_style,
    default_diagonal_projection_b_style,
    default_diagram_a_point_style,
    default_diagram_b_point_style,
    default_diagram_gradient_style,
    default_inf_line_style,
    default_longest_edge_style,
    default_matching_edge_style,
    default_point_cloud_style,
    default_point_style,
)


# ---------------------------------------------------------------------------
# mpl_scatter_density bridge
# ---------------------------------------------------------------------------

def _require_scatter_density():
    if not _HAS_MPL_SCATTER_DENSITY:
        raise ImportError(
            "Density rendering requires the mpl_scatter_density package. "
            "Install it via `pip install mpl-scatter-density`, or disable "
            "density rendering with use_density=False / "
            "density_threshold=<larger value>."
        )


def _add_density_artist(ax, x, y, *, color=None, style=None, norm=None):
    """Attach a ScatterDensityArtist to a regular matplotlib Axes.

    Returns the artist. Uses ``ScatterDensityArtist`` directly so the caller
    is not forced to construct the Axes with ``projection='scatter_density'``.

    ``color`` selects the monochromatic-fade-to-transparent rendering (used
    when overlaying multiple diagrams in the same plot, e.g. matching). When
    ``color`` is ``None`` the artist uses ``style['cmap']``.
    """
    _require_scatter_density()
    from mpl_scatter_density import ScatterDensityArtist
    import matplotlib.colors as mcolors

    if norm is None:
        norm = mcolors.PowerNorm(gamma=0.5)

    kwargs = dict(style if style is not None else DEFAULT_DENSITY_STYLE)
    if color is not None:
        kwargs["color"] = color
        kwargs.pop("cmap", None)
    artist = ScatterDensityArtist(ax, x, y, norm=norm, **kwargs)
    ax.add_artist(artist)
    return artist


# ---------------------------------------------------------------------------
# plot_diagram
# ---------------------------------------------------------------------------

def plot_diagram(
    diagrams,
    ax=None,
    *,
    color=None,
    log_x: bool = False,
    log_y: bool = False,
    title: typing.Optional[str] = None,
    suptitle: typing.Optional[str] = None,
    axis_bounds: typing.Optional[typing.Mapping[str, float]] = None,
    dims: typing.Optional[typing.Iterable[int]] = None,
    max_dimension: typing.Optional[int] = None,
    use_density: bool = True,
    density_threshold: int = DEFAULT_DENSITY_THRESHOLD,
    near_diagonal_fraction: float = 0.03,
    density_cmap: str = "viridis",
    density_style: typing.Optional[dict] = None,
    inf_line_margin: float = 0.05,
    point_style: typing.Optional[dict] = None,
    diagonal_style: typing.Optional[dict] = None,
    inf_line_style: typing.Optional[dict] = None,
    dim_label_fmt: str = "H{dim}",
):
    """Plot one or more persistence diagrams.

    Above ``density_threshold`` finite points the bulk near the diagonal is
    rendered as a 2D density (via mpl_scatter_density) while points further
    than ``near_diagonal_fraction`` of the axis range from the diagonal are
    still drawn as crisp scatter -- so high-persistence (topologically
    meaningful) features are never aggregated.

    Style-dict kwargs (``point_style``, ``diagonal_style``, ``inf_line_style``,
    ``density_style``) default to copies of the module-level ``DEFAULT_*_STYLE``
    dicts. Per-dim colouring is overridden via ``color`` (dict[int, color]
    or list).
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plot_diagram.")

    point_style = _resolve_style(point_style, default_point_style)
    diagonal_style = _resolve_style(diagonal_style, default_diagonal_style)
    inf_line_style = _resolve_style(inf_line_style, default_inf_line_style)

    bounds = {} if axis_bounds is None else dict(axis_bounds)
    dgms = _to_dim_diagrams(diagrams, dims=dims, max_dimension=max_dimension)
    dims_sorted = sorted(dgms.keys())

    finite_by_dim = {}
    pos_inf_birth_by_dim = {}
    neg_inf_birth_by_dim = {}
    all_finite_births = []
    all_finite_deaths = []
    all_births_for_limits = []

    for dim in dims_sorted:
        arr = dgms[dim]
        births = arr[:, 0] if arr.shape[0] else np.empty((0,), dtype=float)
        deaths = arr[:, 1] if arr.shape[0] else np.empty((0,), dtype=float)

        finite_mask = np.isfinite(births) & np.isfinite(deaths)
        pos_inf_mask = np.isfinite(births) & (np.isposinf(deaths) | np.isnan(deaths))
        neg_inf_mask = np.isfinite(births) & np.isneginf(deaths)

        finite_births = births[finite_mask]
        finite_deaths = deaths[finite_mask]
        pos_inf_births = births[pos_inf_mask]
        neg_inf_births = births[neg_inf_mask]

        finite_by_dim[dim] = (finite_births, finite_deaths)
        pos_inf_birth_by_dim[dim] = pos_inf_births
        neg_inf_birth_by_dim[dim] = neg_inf_births

        if finite_births.size:
            all_finite_births.append(finite_births)
            all_finite_deaths.append(finite_deaths)
            all_births_for_limits.append(finite_births)
        if pos_inf_births.size:
            all_births_for_limits.append(pos_inf_births)
        if neg_inf_births.size:
            all_births_for_limits.append(neg_inf_births)

    all_finite_births = (
        np.concatenate(all_finite_births) if all_finite_births else np.empty((0,), dtype=float)
    )
    all_finite_deaths = (
        np.concatenate(all_finite_deaths) if all_finite_deaths else np.empty((0,), dtype=float)
    )
    all_births_for_limits = (
        np.concatenate(all_births_for_limits) if all_births_for_limits else np.empty((0,), dtype=float)
    )

    any_pos_inf = any(pos_inf_birth_by_dim[d].size > 0 for d in dims_sorted)
    any_neg_inf = any(neg_inf_birth_by_dim[d].size > 0 for d in dims_sorted)

    if all_finite_deaths.size:
        y_min = float(np.min(all_finite_deaths))
        y_max = float(np.max(all_finite_deaths))
        y_span = y_max - y_min
    elif all_births_for_limits.size:
        y_min = float(np.min(all_births_for_limits))
        y_max = float(np.max(all_births_for_limits))
        y_span = y_max - y_min
    else:
        y_min = -1.0
        y_max = 1.0
        y_span = 2.0
    if y_span <= 0.0:
        y_span = max(abs(y_max), abs(y_min), 1.0)

    if "ymax" in bounds:
        inf_y_pos = 0.9 * float(bounds["ymax"])
    else:
        inf_y_pos = y_max + inf_line_margin * y_span

    if "ymin" in bounds:
        inf_y_neg = 0.9 * float(bounds["ymin"])
    else:
        inf_y_neg = y_min - inf_line_margin * y_span

    if all_births_for_limits.size:
        x_span = float(np.ptp(all_births_for_limits))
    else:
        x_span = 1.0
    if x_span <= 0.0:
        x_span = 1.0

    near_thr = near_diagonal_fraction * max(x_span, y_span)

    near_x_parts = []
    near_y_parts = []
    near_mask_by_dim = {}
    for dim in dims_sorted:
        births, deaths = finite_by_dim[dim]
        if births.size == 0:
            near_mask = np.zeros((0,), dtype=bool)
        else:
            near_mask = np.abs(deaths - births) <= near_thr
            if np.any(near_mask):
                near_x_parts.append(births[near_mask])
                near_y_parts.append(deaths[near_mask])
        near_mask_by_dim[dim] = near_mask

    near_x = np.concatenate(near_x_parts) if near_x_parts else np.empty((0,), dtype=float)
    near_y = np.concatenate(near_y_parts) if near_y_parts else np.empty((0,), dtype=float)

    use_density_plot = use_density and all_finite_births.size >= density_threshold and near_x.size > 0
    if use_density_plot:
        _require_scatter_density()

    if ax is None:
        _, ax = plt.subplots()

    y_values_for_shift = all_finite_deaths
    if any_pos_inf:
        y_values_for_shift = np.concatenate([y_values_for_shift, np.asarray([inf_y_pos])])
    if any_neg_inf:
        y_values_for_shift = np.concatenate([y_values_for_shift, np.asarray([inf_y_neg])])

    x_shift = _shift_for_log(all_births_for_limits, log_x)
    y_shift = _shift_for_log(y_values_for_shift, log_y)

    if use_density_plot:
        density_style_resolved = _resolve_style(density_style, default_density_style)
        density_style_resolved.setdefault("cmap", density_cmap)
        _add_density_artist(
            ax,
            near_x + x_shift,
            near_y + y_shift,
            style=density_style_resolved,
        )

    # When a per-dim color override is supplied, it wins over point_style's "c".
    base_scatter_kwargs = dict(point_style)
    base_color = base_scatter_kwargs.pop("c", None)

    for dim_idx, dim in enumerate(dims_sorted):
        dim_color = _resolve_color(color, dim, dim_idx)
        effective_c = dim_color if dim_color is not None else base_color

        scatter_kwargs = dict(base_scatter_kwargs)
        if effective_c is not None:
            scatter_kwargs["c"] = effective_c

        label = dim_label_fmt.format(dim=dim)
        births, deaths = finite_by_dim[dim]
        if births.size:
            mask = ~near_mask_by_dim[dim] if use_density_plot else np.ones_like(births, dtype=bool)
            if np.any(mask):
                ax.scatter(
                    births[mask] + x_shift,
                    deaths[mask] + y_shift,
                    label=label,
                    **scatter_kwargs,
                )

        label_for_inf = label if births.size == 0 else None

        pos_inf_births = pos_inf_birth_by_dim[dim]
        if pos_inf_births.size:
            ax.scatter(
                pos_inf_births + x_shift,
                np.full_like(pos_inf_births, inf_y_pos + y_shift),
                label=label_for_inf,
                **scatter_kwargs,
            )
            label_for_inf = None

        neg_inf_births = neg_inf_birth_by_dim[dim]
        if neg_inf_births.size:
            ax.scatter(
                neg_inf_births + x_shift,
                np.full_like(neg_inf_births, inf_y_neg + y_shift),
                label=label_for_inf,
                **scatter_kwargs,
            )

    if any_pos_inf:
        ax.axhline(inf_y_pos + y_shift, **inf_line_style)
    if any_neg_inf:
        ax.axhline(inf_y_neg + y_shift, **inf_line_style)

    if all_births_for_limits.size or all_finite_deaths.size or any_pos_inf or any_neg_inf:
        if all_births_for_limits.size:
            x_vals = all_births_for_limits + x_shift
        else:
            x_vals = np.asarray([0.0 + x_shift, 1.0 + x_shift])
        if all_finite_deaths.size:
            y_vals = all_finite_deaths + y_shift
        else:
            y_vals = np.asarray([0.0 + y_shift, 1.0 + y_shift])
        lo = min(float(np.min(x_vals)), float(np.min(y_vals)))
        hi = max(float(np.max(x_vals)), float(np.max(y_vals)))
        if any_pos_inf:
            hi = max(hi, float(inf_y_pos + y_shift))
        if any_neg_inf:
            lo = min(lo, float(inf_y_neg + y_shift))
        if hi <= lo:
            hi = lo + 1.0
        ax.plot([lo, hi], [lo, hi], **diagonal_style)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    x_left = None if "xmin" not in bounds else float(bounds["xmin"]) + x_shift
    x_right = None if "xmax" not in bounds else float(bounds["xmax"]) + x_shift
    y_bottom = None if "ymin" not in bounds else float(bounds["ymin"]) + y_shift
    y_top = None if "ymax" not in bounds else float(bounds["ymax"]) + y_shift

    if x_left is not None or x_right is not None:
        ax.set_xlim(left=x_left, right=x_right)
    if y_bottom is not None or y_top is not None:
        ax.set_ylim(bottom=y_bottom, top=y_top)

    ax.set_xlabel("birth" if x_shift == 0 else f"birth (shifted by +{x_shift:.3g})")
    ax.set_ylabel("death" if y_shift == 0 else f"death (shifted by +{y_shift:.3g})")

    if title is not None:
        ax.set_title(title)
    if suptitle is not None:
        ax.figure.suptitle(suptitle)

    if len(dims_sorted) > 1:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), title="dimension")

    return ax


# ---------------------------------------------------------------------------
# plot_diagram_gradient
# ---------------------------------------------------------------------------

def plot_diagram_gradient(
    diagram,
    gradient=None,
    *,
    ax=None,
    dims: typing.Optional[typing.Iterable[int]] = None,
    descent: bool = False,
    plot_points: bool = True,
    use_density: bool = True,
    density_threshold: int = DEFAULT_DENSITY_THRESHOLD,
    top_k_arrows: typing.Optional[int] = None,
    log_x: bool = False,
    log_y: bool = False,
    title: typing.Optional[str] = None,
    axis_bounds: typing.Optional[typing.Mapping[str, float]] = None,
    inf_line_margin: float = 0.05,
    quiver_style: typing.Optional[dict] = None,
    point_style: typing.Optional[dict] = None,
    diagonal_style: typing.Optional[dict] = None,
    inf_line_style: typing.Optional[dict] = None,
    density_style: typing.Optional[dict] = None,
    dim_label_fmt: str = "H{dim}",
):
    """Plot a gradient vector field on top of a persistence diagram.

    For every diagram point at ``(birth, death)`` an arrow with components
    ``(d/dbirth, d/ddeath)`` is drawn at that point. Useful for inspecting
    where an optimization step would move each persistence pair when
    minimizing or maximizing a topology-aware loss.

    Inputs:
        diagram: One of
            - ``torch.Tensor`` of shape ``(n, 2)``,
            - ``numpy.ndarray`` of shape ``(n, 2)``,
            - native ``oineus.Diagrams``,
            - differentiable ``oineus.diff.PersistenceDiagrams``,
            - ``dict[int, ndarray | torch.Tensor]``.
        gradient: Same shape/structure as ``diagram``, or ``None``. When
            ``None`` and the diagram is torch-backed, the gradient is pulled
            from each tensor's ``.grad``. For non-torch inputs it is
            required and must mirror the diagram's per-dimension layout.
        descent: If ``True``, plot ``-grad`` (the descent direction). The
            default plots the gradient as-is (steepest *increase*).
        plot_points: If ``True`` (default), the underlying diagram is drawn
            via ``plot_diagram`` before the arrows are overlaid.

    Inf-death points are skipped (arrows for those rows are dropped). The
    ``quiver_style`` kwarg accepts any ``Axes.quiver`` keyword; the default
    uses ``angles='xy', scale_units='xy', scale=1.0`` so that ``(vx, vy)``
    is interpreted in data coordinates -- the natural convention given
    that diagram coordinates and gradient components share units.

    Above ``density_threshold`` total points the underlying scatter is
    rendered as density (inherited from ``plot_diagram``) and arrows are
    restricted to the top ``top_k_arrows`` points by gradient magnitude.
    When ``top_k_arrows`` is ``None`` it defaults to
    ``DEFAULT_GRADIENT_TOP_K_ARROWS`` whenever the threshold is exceeded;
    set it explicitly to apply the cap below the threshold too.
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plot_diagram_gradient.")

    quiver_style = _resolve_style(quiver_style, default_diagram_gradient_style)

    points_by_dim, grad_by_dim = _coerce_diagram_with_grad(diagram, gradient, dims)
    if not points_by_dim:
        raise ValueError("No diagram points to plot.")

    sign = -1.0 if descent else 1.0

    finite_by_dim: typing.Dict[int, typing.Tuple[np.ndarray, np.ndarray]] = {}
    for dim, pts in points_by_dim.items():
        g = grad_by_dim[dim]
        if pts.shape[0] != g.shape[0]:
            raise ValueError(
                f"Diagram and gradient row counts disagree for dim {dim}: "
                f"{pts.shape[0]} vs {g.shape[0]}."
            )
        finite_mask = np.isfinite(pts).all(axis=1) & np.isfinite(g).all(axis=1)
        finite_by_dim[dim] = (pts[finite_mask], sign * g[finite_mask])

    total_finite = sum(pts.shape[0] for pts, _ in finite_by_dim.values())
    density_active = use_density and total_finite >= density_threshold

    if plot_points:
        finite_dgms = {dim: pts for dim, (pts, _) in finite_by_dim.items()}
        ax = plot_diagram(
            finite_dgms,
            ax=ax,
            log_x=log_x,
            log_y=log_y,
            title=title,
            axis_bounds=axis_bounds,
            inf_line_margin=inf_line_margin,
            point_style=point_style,
            diagonal_style=diagonal_style,
            inf_line_style=inf_line_style,
            density_style=density_style,
            use_density=use_density,
            density_threshold=density_threshold,
            dim_label_fmt=dim_label_fmt,
        )
    elif ax is None:
        _, ax = plt.subplots()

    effective_top_k = top_k_arrows
    if effective_top_k is None and density_active:
        effective_top_k = DEFAULT_GRADIENT_TOP_K_ARROWS

    if effective_top_k is not None:
        dim_arrs, local_arrs, mag_arrs = [], [], []
        for dim, (pts, g) in finite_by_dim.items():
            n = pts.shape[0]
            if n == 0:
                continue
            dim_arrs.append(np.full(n, dim, dtype=np.int64))
            local_arrs.append(np.arange(n, dtype=np.int64))
            mag_arrs.append(np.hypot(g[:, 0], g[:, 1]))
        if dim_arrs:
            dim_concat = np.concatenate(dim_arrs)
            local_concat = np.concatenate(local_arrs)
            mag_concat = np.concatenate(mag_arrs)
            if mag_concat.size > effective_top_k:
                keep = np.argpartition(mag_concat, -effective_top_k)[-effective_top_k:]
                keep_dim = dim_concat[keep]
                keep_local = local_concat[keep]
                new_finite = {}
                for dim, (pts, g) in finite_by_dim.items():
                    mask = np.zeros(pts.shape[0], dtype=bool)
                    mask[keep_local[keep_dim == dim]] = True
                    new_finite[dim] = (pts[mask], g[mask])
                finite_by_dim = new_finite

    finite_births_parts = [pts[:, 0] for pts, _ in finite_by_dim.values() if pts.size]
    finite_deaths_parts = [pts[:, 1] for pts, _ in finite_by_dim.values() if pts.size]
    all_births = np.concatenate(finite_births_parts) if finite_births_parts else np.empty((0,), dtype=float)
    all_deaths = np.concatenate(finite_deaths_parts) if finite_deaths_parts else np.empty((0,), dtype=float)
    x_shift = _shift_for_log(all_births, log_x)
    y_shift = _shift_for_log(all_deaths, log_y)

    for dim in sorted(finite_by_dim.keys()):
        pts, grads = finite_by_dim[dim]
        if pts.shape[0] == 0:
            continue
        ax.quiver(
            pts[:, 0] + x_shift,
            pts[:, 1] + y_shift,
            grads[:, 0],
            grads[:, 1],
            **quiver_style,
        )

    return ax


# ---------------------------------------------------------------------------
# plot_matching
# ---------------------------------------------------------------------------

def plot_matching(
    dgm_a,
    dgm_b,
    matching,
    ax=None,
    *,
    plot_finite_to_finite: typing.Optional[bool] = None,
    plot_a_to_diagonal: typing.Optional[bool] = None,
    plot_b_to_diagonal: typing.Optional[bool] = None,
    plot_essential: typing.Optional[bool] = None,
    highlight_longest: typing.Optional[bool] = None,
    plot_points: bool = True,
    plot_diagonal_projections: bool = False,
    plot_diagonal: bool = True,
    use_density: bool = True,
    density_threshold: int = DEFAULT_DENSITY_THRESHOLD,
    near_diagonal_fraction: float = 0.03,
    edge_quantile: float = DEFAULT_MATCHING_EDGE_QUANTILE,
    density_style: typing.Optional[dict] = None,
    dgm_a_point_style: typing.Optional[dict] = None,
    dgm_b_point_style: typing.Optional[dict] = None,
    ordinary_edge_style: typing.Optional[dict] = None,
    longest_edge_style: typing.Optional[dict] = None,
    diagonal_style: typing.Optional[dict] = None,
    diagonal_projection_a_style: typing.Optional[dict] = None,
    diagonal_projection_b_style: typing.Optional[dict] = None,
    inf_line_style: typing.Optional[dict] = None,
    dgm_a_label: str = "Diagram A",
    dgm_b_label: str = "Diagram B",
    title: typing.Optional[str] = None,
    axis_bounds: typing.Optional[typing.Mapping[str, float]] = None,
    inf_line_margin: float = 0.05,
):
    """Plot a matching between two persistence diagrams.

    Dispatches on ``matching`` type: for Wasserstein (``DiagramMatching``)
    all edge categories are drawn by default; for ``BottleneckMatching`` only
    finite-to-finite edges are drawn and the longest edge(s) are overlaid in
    the highlight style.

    ``dgm_a`` and ``dgm_b`` must be 2D numpy arrays (one homology dimension).

    Above ``density_threshold`` total points the diagram is rendered as a
    density background (near-diagonal points only) plus crisp scatter for
    high-persistence outliers. Ordinary matching edges are filtered to those
    with length above the ``edge_quantile``-th quantile, since most edges in
    a large matching are short noise-to-noise pairs that obscure the
    informative tail. The ``highlight_longest`` overlay (always drawn for
    bottleneck matchings) is unaffected.
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plot_matching.")

    # Avoid circular import
    from ..matching import (
        BottleneckMatching,
        DiagramMatching,
        point_to_diagonal,
    )

    if not isinstance(matching, DiagramMatching):
        raise TypeError("matching must be a DiagramMatching or BottleneckMatching instance.")

    is_bottleneck = isinstance(matching, BottleneckMatching)

    # Type-aware category-flag defaults
    if plot_finite_to_finite is None:
        plot_finite_to_finite = True
    if plot_a_to_diagonal is None:
        plot_a_to_diagonal = not is_bottleneck
    if plot_b_to_diagonal is None:
        plot_b_to_diagonal = not is_bottleneck
    if plot_essential is None:
        plot_essential = False
    if highlight_longest is None:
        highlight_longest = is_bottleneck

    # Resolve style dicts
    dgm_a_point_style = _resolve_style(dgm_a_point_style, default_diagram_a_point_style)
    dgm_b_point_style = _resolve_style(dgm_b_point_style, default_diagram_b_point_style)
    ordinary_edge_style = _resolve_style(ordinary_edge_style, default_matching_edge_style)
    longest_edge_style = _resolve_style(longest_edge_style, default_longest_edge_style)
    diagonal_style = _resolve_style(diagonal_style, default_diagonal_style)
    diagonal_projection_a_style = _resolve_style(
        diagonal_projection_a_style, default_diagonal_projection_a_style)
    diagonal_projection_b_style = _resolve_style(
        diagonal_projection_b_style, default_diagonal_projection_b_style)
    inf_line_style = _resolve_style(inf_line_style, default_inf_line_style)

    bounds = {} if axis_bounds is None else dict(axis_bounds)

    dgm_a = _build_diagram_arrays(dgm_a)
    dgm_b = _build_diagram_arrays(dgm_b)

    # Collect finite coordinates for layout
    def _finite_parts(dgm):
        if dgm.shape[0] == 0:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
        finite = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
        return dgm[finite, 0], dgm[finite, 1]

    a_fin_b, a_fin_d = _finite_parts(dgm_a)
    b_fin_b, b_fin_d = _finite_parts(dgm_b)

    # Essential births / deaths for layout (we want the plot to include them).
    a_pos_inf_b = dgm_a[np.isfinite(dgm_a[:, 0]) & np.isposinf(dgm_a[:, 1]), 0] if dgm_a.size else np.empty((0,))
    b_pos_inf_b = dgm_b[np.isfinite(dgm_b[:, 0]) & np.isposinf(dgm_b[:, 1]), 0] if dgm_b.size else np.empty((0,))
    a_neg_inf_b = dgm_a[np.isfinite(dgm_a[:, 0]) & np.isneginf(dgm_a[:, 1]), 0] if dgm_a.size else np.empty((0,))
    b_neg_inf_b = dgm_b[np.isfinite(dgm_b[:, 0]) & np.isneginf(dgm_b[:, 1]), 0] if dgm_b.size else np.empty((0,))
    a_pos_inf_d = dgm_a[np.isposinf(dgm_a[:, 0]) & np.isfinite(dgm_a[:, 1]), 1] if dgm_a.size else np.empty((0,))
    b_pos_inf_d = dgm_b[np.isposinf(dgm_b[:, 0]) & np.isfinite(dgm_b[:, 1]), 1] if dgm_b.size else np.empty((0,))
    a_neg_inf_d = dgm_a[np.isneginf(dgm_a[:, 0]) & np.isfinite(dgm_a[:, 1]), 1] if dgm_a.size else np.empty((0,))
    b_neg_inf_d = dgm_b[np.isneginf(dgm_b[:, 0]) & np.isfinite(dgm_b[:, 1]), 1] if dgm_b.size else np.empty((0,))

    (x_min, x_max, y_min, y_max, x_span, y_span,
     inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg) = _compute_plot_limits(
        np.concatenate([a_fin_b, b_fin_b]) if a_fin_b.size or b_fin_b.size else np.empty((0,)),
        np.concatenate([a_fin_d, b_fin_d]) if a_fin_d.size or b_fin_d.size else np.empty((0,)),
        extra_xs=[a_pos_inf_b, b_pos_inf_b, a_neg_inf_b, b_neg_inf_b],
        extra_ys=[a_pos_inf_d, b_pos_inf_d, a_neg_inf_d, b_neg_inf_d],
        inf_line_margin=inf_line_margin,
    )

    any_pos_inf_d = (a_pos_inf_b.size + b_pos_inf_b.size) > 0
    any_neg_inf_d = (a_neg_inf_b.size + b_neg_inf_b.size) > 0
    any_pos_inf_b = (a_pos_inf_d.size + b_pos_inf_d.size) > 0
    any_neg_inf_b = (a_neg_inf_d.size + b_neg_inf_d.size) > 0

    if ax is None:
        _, ax = plt.subplots()

    # Decide whether to switch to density mode for the bulk near the diagonal.
    n_finite_total = a_fin_b.size + b_fin_b.size
    use_density_plot = (
        use_density
        and plot_points
        and n_finite_total >= density_threshold
    )
    if use_density_plot:
        _require_scatter_density()

    near_thr = near_diagonal_fraction * max(x_span, y_span)

    # Diagonal
    if plot_diagonal:
        lo = min(x_min, y_min)
        hi = max(x_max, y_max)
        if any_pos_inf_d:
            hi = max(hi, inf_y_pos)
        if any_neg_inf_d:
            lo = min(lo, inf_y_neg)
        if any_pos_inf_b:
            hi = max(hi, inf_x_pos)
        if any_neg_inf_b:
            lo = min(lo, inf_x_neg)
        if hi <= lo:
            hi = lo + 1.0
        ax.plot([lo, hi], [lo, hi], **diagonal_style)

    # Inf lines
    if any_pos_inf_d:
        ax.axhline(inf_y_pos, **inf_line_style)
    if any_neg_inf_d:
        ax.axhline(inf_y_neg, **inf_line_style)
    if any_pos_inf_b:
        ax.axvline(inf_x_pos, **inf_line_style)
    if any_neg_inf_b:
        ax.axvline(inf_x_neg, **inf_line_style)

    # Diagram points: split near-diagonal bulk (density when enabled) from
    # outliers (always scatter so high-persistence features stay crisp).
    def _draw_diagram_points(dgm, point_style, label):
        if dgm.shape[0] == 0:
            return
        coords = np.array([
            _point_coords_for_edge(dgm, i, inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
            for i in range(dgm.shape[0])
        ])
        if not use_density_plot:
            ax.scatter(coords[:, 0], coords[:, 1], label=label, **point_style)
            return
        finite_mask = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        finite = coords[finite_mask]
        non_finite = coords[~finite_mask]
        near_b, near_d, far_b, far_d = _split_near_diagonal(
            finite[:, 0], finite[:, 1], near_thr)
        if near_b.size:
            _add_density_artist(
                ax, near_b, near_d,
                color=point_style.get("c"),
                style=_resolve_style(density_style, default_density_style),
            )
        scatter_label = label if (far_b.size or non_finite.size) else None
        if far_b.size:
            ax.scatter(far_b, far_d, label=scatter_label, **point_style)
            scatter_label = None
        if non_finite.size:
            ax.scatter(non_finite[:, 0], non_finite[:, 1], label=scatter_label, **point_style)

    if plot_points:
        _draw_diagram_points(dgm_a, dgm_a_point_style, dgm_a_label)
        _draw_diagram_points(dgm_b, dgm_b_point_style, dgm_b_label)

    # Gather all edges to draw, grouped by category.
    ordinary_segments: list = []

    if plot_finite_to_finite:
        for ia, ib in matching.finite_to_finite:
            pa = _point_coords_for_edge(dgm_a, ia, inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
            pb = _point_coords_for_edge(dgm_b, ib, inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
            ordinary_segments.append((pa, pb))

    diag_proj_a_coords = []
    diag_proj_b_coords = []

    if plot_a_to_diagonal and len(matching.a_to_diagonal) > 0:
        projs = point_to_diagonal(dgm_a, indices=matching.a_to_diagonal)
        for local_i, ia in enumerate(matching.a_to_diagonal):
            pa = _point_coords_for_edge(dgm_a, int(ia), inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
            pproj = (float(projs[local_i, 0]), float(projs[local_i, 1]))
            ordinary_segments.append((pa, pproj))
            diag_proj_a_coords.append(pproj)

    if plot_b_to_diagonal and len(matching.b_to_diagonal) > 0:
        projs = point_to_diagonal(dgm_b, indices=matching.b_to_diagonal)
        for local_i, ib in enumerate(matching.b_to_diagonal):
            pb = _point_coords_for_edge(dgm_b, int(ib), inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
            pproj = (float(projs[local_i, 0]), float(projs[local_i, 1]))
            ordinary_segments.append((pb, pproj))
            diag_proj_b_coords.append(pproj)

    if plot_essential:
        for _kind, pairs in matching.essential.items():
            for ia, ib in pairs:
                pa = _point_coords_for_edge(dgm_a, int(ia), inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
                pb = _point_coords_for_edge(dgm_b, int(ib), inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
                ordinary_segments.append((pa, pb))

    if ordinary_segments and use_density_plot and 0.0 < edge_quantile < 1.0:
        # In density mode keep only the longest few percent of ordinary edges.
        # Most matchings of large diagrams are dominated by short noise-to-noise
        # pairs near the diagonal that pile up into a featureless gray hairball.
        seg_arr = np.array(ordinary_segments, dtype=float)
        lengths = np.hypot(
            seg_arr[:, 1, 0] - seg_arr[:, 0, 0],
            seg_arr[:, 1, 1] - seg_arr[:, 0, 1],
        )
        threshold = float(np.quantile(lengths, edge_quantile))
        keep = lengths >= threshold
        ordinary_segments = [s for s, k in zip(ordinary_segments, keep) if k]

    if ordinary_segments:
        ax.add_collection(LineCollection(ordinary_segments, **ordinary_edge_style))

    # Diagonal projection markers (after edges so they sit on top).
    if plot_diagonal_projections:
        if diag_proj_a_coords:
            arr = np.array(diag_proj_a_coords)
            ax.scatter(arr[:, 0], arr[:, 1], **diagonal_projection_a_style)
        if diag_proj_b_coords:
            arr = np.array(diag_proj_b_coords)
            ax.scatter(arr[:, 0], arr[:, 1], **diagonal_projection_b_style)

    # Highlight longest edges for bottleneck.
    if highlight_longest and is_bottleneck:
        longest_segments = []
        for e in matching.longest.finite:
            longest_segments.append((e.point_a, e.point_b))
        for _kind, edges in matching.longest.essential.items():
            for e in edges:
                pa = _point_coords_for_edge(dgm_a, e.idx_a, inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
                pb = _point_coords_for_edge(dgm_b, e.idx_b, inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)
                longest_segments.append((pa, pb))
        if longest_segments:
            ax.add_collection(LineCollection(longest_segments, **longest_edge_style))

    # Axis limits
    x_left = None if "xmin" not in bounds else float(bounds["xmin"])
    x_right = None if "xmax" not in bounds else float(bounds["xmax"])
    y_bottom = None if "ymin" not in bounds else float(bounds["ymin"])
    y_top = None if "ymax" not in bounds else float(bounds["ymax"])
    if x_left is not None or x_right is not None:
        ax.set_xlim(left=x_left, right=x_right)
    if y_bottom is not None or y_top is not None:
        ax.set_ylim(bottom=y_bottom, top=y_top)

    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    if title is not None:
        ax.set_title(title)

    # Legend: only the diagram-point labels are registered, so this is safe.
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    return ax


# ---------------------------------------------------------------------------
# plot_chain
# ---------------------------------------------------------------------------

def _cube_filtration_types():
    """Return the tuple of CubeFiltration_*D types or () if unavailable."""
    try:
        from .. import _oineus
        return (
            _oineus.CubeFiltration_1D,
            _oineus.CubeFiltration_2D,
            _oineus.CubeFiltration_3D,
        )
    except (ImportError, AttributeError):
        return ()


def _is_cubical_filtration(filtration):
    cube_types = _cube_filtration_types()
    return bool(cube_types) and isinstance(filtration, cube_types)


def _resolve_source_kind(filtration, override):
    if override is not None:
        if override not in ("points", "field"):
            raise ValueError(
                f"source_kind must be 'points' or 'field', got {override!r}."
            )
        return override
    # Prefer the FiltrationKind tag set by the constructor (Phase 2);
    # fall back to type-based detection for hand-built filtrations
    # whose kind was left at User.
    kind = getattr(filtration, "kind", None)
    if kind is not None:
        try:
            from .. import _oineus
            FK = _oineus.FiltrationKind
            if kind in (FK.Cubical, FK.Freudenthal):
                return "field"
            if kind in (FK.Vr, FK.Alpha, FK.WeakAlpha, FK.CechDelaunay):
                return "points"
        except (ImportError, AttributeError):
            pass
    if _is_cubical_filtration(filtration):
        return "field"
    return "points"


def _square_corners_cyclic(corners):
    """Reorder 4 axis-aligned-square corners into BL, BR, TR, TL cyclic order
    for polygon rendering. Each input corner is a length-2 sequence (i, j)."""
    arr = np.asarray(corners, dtype=float)
    i_min, j_min = float(arr[:, 0].min()), float(arr[:, 1].min())
    i_max, j_max = float(arr[:, 0].max()), float(arr[:, 1].max())
    return [(i_min, j_min), (i_min, j_max), (i_max, j_max), (i_max, j_min)]


def _cube_3d_face_polys(corners):
    """Return the 6 axis-aligned square faces of a 3-cube as lists of 4
    (i, j, k) corners each. ``corners`` is the 8-corner list."""
    arr = np.asarray(corners, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    i0, j0, k0 = mins
    i1, j1, k1 = maxs
    return [
        [(i0, j0, k0), (i1, j0, k0), (i1, j1, k0), (i0, j1, k0)],  # k=k0
        [(i0, j0, k1), (i1, j0, k1), (i1, j1, k1), (i0, j1, k1)],  # k=k1
        [(i0, j0, k0), (i1, j0, k0), (i1, j0, k1), (i0, j0, k1)],  # j=j0
        [(i0, j1, k0), (i1, j1, k0), (i1, j1, k1), (i0, j1, k1)],  # j=j1
        [(i0, j0, k0), (i0, j1, k0), (i0, j1, k1), (i0, j0, k1)],  # i=i0
        [(i1, j0, k0), (i1, j1, k0), (i1, j1, k1), (i1, j0, k1)],  # i=i1
    ]


def _tet_face_polys(verts):
    """Return the 4 triangular faces of a tetrahedron given its 4 vertex
    coordinates (each a length-3 array)."""
    v = list(verts)
    return [
        [v[0], v[1], v[2]],
        [v[0], v[1], v[3]],
        [v[0], v[2], v[3]],
        [v[1], v[2], v[3]],
    ]


def plot_chain(
    source,
    filtration,
    chain,
    *,
    ax=None,
    source_kind: typing.Optional[str] = None,
    dualize=False,
    edge_style: typing.Optional[dict] = None,
    triangle_style: typing.Optional[dict] = None,
    tetrahedron_style: typing.Optional[dict] = None,
    vertex_style: typing.Optional[dict] = None,
    point_style: typing.Optional[dict] = None,
    title: typing.Optional[str] = None,
    plot_source: bool = True,
    field_cmap: str = "viridis",
):
    """Render a chain of cells over its underlying source.

    ``source`` is one of:

    - a 2D point cloud as ``(N, 2)`` ndarray (simplicial filtration),
    - a 3D point cloud as ``(N, 3)`` ndarray (simplicial filtration),
    - a 2D scalar field as ``(H, W)`` ndarray (cubical or Freudenthal),
    - a 3D scalar field as ``(D, H, W)`` ndarray (cubical or Freudenthal).

    The dispatch is driven by ``source_kind`` (``"points"`` or ``"field"``)
    and, for kind ``"points"``, by ``source.shape[1]`` (2 vs 3). When
    ``source_kind`` is ``None`` we route ``CubeFiltration_*D`` to ``"field"``
    and everything else to ``"points"``.

    The chain may be a list, ndarray, range, or scipy.sparse column / row
    slice (e.g. ``dcmp.v_as_csc()[:, j]``). Each entry is normally a
    *filtration sorted-id*; cells of dim 0/1/2/3 render as vertices /
    edges / triangles-or-squares / tetrahedra-or-voxels respectively.

    The matrices ``v_data``, ``r_data``, ``u_data_t`` of a
    ``Decomposition(fil, dualize=True)`` are indexed in *cohomology matrix
    space* rather than filtration space (matrix index = ``n - 1 -
    filtration_id``). Pass ``dualize=True`` (or ``dualize=dcmp`` and we
    will read ``dcmp.dualize`` for you) so plot_chain translates the
    chain back to filtration ids before looking up cells. The default
    ``False`` matches homology decompositions and indices that have
    already been translated (e.g. those returned by
    ``TopologyOptimizer.increase_birth``).
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plot_chain.")

    kind = _resolve_source_kind(filtration, source_kind)
    chain_ids = _coerce_chain(chain)

    if hasattr(dualize, "dualize"):
        # Decomposition (or duck-type with a .dualize attr) -- read the flag.
        dualize_flag = bool(dualize.dualize)
    else:
        dualize_flag = bool(dualize)
    if dualize_flag and chain_ids.size:
        chain_ids = (filtration.size() - 1) - chain_ids

    vertex_style = _resolve_style(vertex_style, default_chain_vertex_style)
    edge_style = _resolve_style(edge_style, default_chain_edge_style)
    triangle_style = _resolve_style(triangle_style, default_chain_triangle_style)
    tetrahedron_style = _resolve_style(tetrahedron_style, default_chain_tetrahedron_style)
    point_style = _resolve_style(point_style, default_point_cloud_style)

    if kind == "points":
        points = np.asarray(source, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(
                f"For source_kind='points', expected an (N, 2) or (N, 3) "
                f"array; got shape {points.shape}."
            )
        if points.shape[1] == 2:
            return _render_chain_points_2d(
                ax, points, filtration, chain_ids,
                vertex_style=vertex_style,
                edge_style=edge_style,
                triangle_style=triangle_style,
                point_style=point_style,
                plot_source=plot_source,
                title=title,
            )
        return _render_chain_points_3d(
            ax, points, filtration, chain_ids,
            vertex_style=vertex_style,
            edge_style=edge_style,
            triangle_style=triangle_style,
            tetrahedron_style=tetrahedron_style,
            point_style=point_style,
            plot_source=plot_source,
            title=title,
        )

    # kind == "field"
    field = np.asarray(source)
    if field.ndim == 2:
        return _render_chain_field_2d(
            ax, field, filtration, chain_ids,
            vertex_style=vertex_style,
            edge_style=edge_style,
            triangle_style=triangle_style,
            field_cmap=field_cmap,
            plot_source=plot_source,
            title=title,
        )
    if field.ndim == 3:
        return _render_chain_field_3d(
            ax, field, filtration, chain_ids,
            vertex_style=vertex_style,
            edge_style=edge_style,
            triangle_style=triangle_style,
            tetrahedron_style=tetrahedron_style,
            point_style=point_style,
            plot_source=plot_source,
            title=title,
        )
    raise ValueError(
        f"For source_kind='field', expected a 2D (H, W) or 3D (D, H, W) "
        f"array; got shape {field.shape}."
    )


# ---------------------------------------------------------------------------
# plot_chain renderers
# ---------------------------------------------------------------------------

def _render_chain_points_2d(
    ax, points, filtration, chain_ids,
    *, vertex_style, edge_style, triangle_style,
    point_style, plot_source, title,
):
    if ax is None:
        _, ax = plt.subplots()

    if plot_source:
        ax.scatter(points[:, 0], points[:, 1], **point_style)

    vertex_coords = []
    edge_segments = []
    triangle_polys = []
    skipped_high_dim = 0

    for cell_id in chain_ids:
        cell = filtration[int(cell_id)]
        verts = list(cell.vertices)
        if len(verts) == 1:
            vertex_coords.append(points[verts[0]])
        elif len(verts) == 2:
            edge_segments.append((points[verts[0]], points[verts[1]]))
        elif len(verts) == 3:
            triangle_polys.append([points[v] for v in verts])
        else:
            skipped_high_dim += 1

    if skipped_high_dim:
        warnings.warn(
            f"plot_chain: skipped {skipped_high_dim} cells of dim >= 3 "
            f"(2D point-cloud rendering only supports vertices, edges, "
            f"triangles).",
            stacklevel=3,
        )

    if triangle_polys:
        ax.add_collection(PolyCollection(triangle_polys, **triangle_style))
    if edge_segments:
        ax.add_collection(LineCollection(edge_segments, **edge_style))
    if vertex_coords:
        arr = np.asarray(vertex_coords)
        ax.scatter(arr[:, 0], arr[:, 1], **vertex_style)

    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def _ensure_3d_axes(ax):
    if ax is None:
        fig = plt.figure()
        return fig.add_subplot(111, projection="3d")
    if getattr(ax, "name", "") != "3d":
        raise ValueError(
            "3D plot_chain rendering requires a 3D Axes; pass ax=None to "
            "auto-create one or build with projection='3d'."
        )
    return ax


def _render_chain_points_3d(
    ax, points, filtration, chain_ids,
    *, vertex_style, edge_style, triangle_style, tetrahedron_style,
    point_style, plot_source, title,
):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    ax = _ensure_3d_axes(ax)

    if plot_source:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], **point_style)

    vertex_coords = []
    edge_segments = []
    triangle_polys = []
    tetrahedron_face_polys = []
    skipped_high_dim = 0

    for cell_id in chain_ids:
        cell = filtration[int(cell_id)]
        verts = list(cell.vertices)
        if len(verts) == 1:
            vertex_coords.append(points[verts[0]])
        elif len(verts) == 2:
            edge_segments.append((points[verts[0]], points[verts[1]]))
        elif len(verts) == 3:
            triangle_polys.append([points[v] for v in verts])
        elif len(verts) == 4:
            tetrahedron_face_polys.extend(_tet_face_polys([points[v] for v in verts]))
        else:
            skipped_high_dim += 1

    if skipped_high_dim:
        warnings.warn(
            f"plot_chain: skipped {skipped_high_dim} cells of dim >= 4 "
            f"(3D point-cloud rendering only supports cells up to "
            f"tetrahedra).",
            stacklevel=3,
        )

    if tetrahedron_face_polys:
        ax.add_collection3d(Poly3DCollection(tetrahedron_face_polys, **tetrahedron_style))
    if triangle_polys:
        ax.add_collection3d(Poly3DCollection(triangle_polys, **triangle_style))
    if edge_segments:
        ax.add_collection3d(Line3DCollection(edge_segments, **edge_style))
    if vertex_coords:
        arr = np.asarray(vertex_coords)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], **vertex_style)

    if title is not None:
        ax.set_title(title)
    return ax


def _render_chain_field_2d(
    ax, field, filtration, chain_ids,
    *, vertex_style, edge_style, triangle_style,
    field_cmap, plot_source, title,
):
    if ax is None:
        _, ax = plt.subplots()

    H, W = field.shape
    if plot_source:
        # origin='lower' so that array index i maps to plot y = i, j to x = j.
        ax.imshow(field, origin="lower", cmap=field_cmap,
                  extent=(-0.5, W - 0.5, -0.5, H - 0.5), zorder=0)

    cubical = _is_cubical_filtration(filtration)

    vertex_xy = []
    edge_segments = []
    polys = []
    skipped_high_dim = 0

    if cubical:
        for cell_id in chain_ids:
            cell = filtration[int(cell_id)]
            corners = [tuple(c) for c in cell.vertices]  # list of (i, j)
            if len(corners) == 1:
                i, j = corners[0]
                vertex_xy.append((j, i))
            elif len(corners) == 2:
                (i0, j0), (i1, j1) = corners
                edge_segments.append(((j0, i0), (j1, i1)))
            elif len(corners) == 4:
                rect = _square_corners_cyclic(corners)  # [(i, j), ...]
                polys.append([(j, i) for (i, j) in rect])
            else:
                skipped_high_dim += 1
    else:
        # Simplicial Freudenthal: vertex IDs ravel C-order over (H, W).
        for cell_id in chain_ids:
            cell = filtration[int(cell_id)]
            verts = list(cell.vertices)
            if len(verts) == 1:
                i, j = _id_to_grid(verts[0], (H, W))
                vertex_xy.append((j, i))
            elif len(verts) == 2:
                (i0, j0) = _id_to_grid(verts[0], (H, W))
                (i1, j1) = _id_to_grid(verts[1], (H, W))
                edge_segments.append(((j0, i0), (j1, i1)))
            elif len(verts) == 3:
                tri = [_id_to_grid(v, (H, W)) for v in verts]
                polys.append([(j, i) for (i, j) in tri])
            else:
                skipped_high_dim += 1

    if skipped_high_dim:
        warnings.warn(
            f"plot_chain: skipped {skipped_high_dim} cells of dim >= 3 "
            f"(2D field rendering only supports vertices, edges, faces).",
            stacklevel=3,
        )

    if polys:
        ax.add_collection(PolyCollection(polys, **triangle_style))
    if edge_segments:
        ax.add_collection(LineCollection(edge_segments, **edge_style))
    if vertex_xy:
        arr = np.asarray(vertex_xy, dtype=float)
        ax.scatter(arr[:, 0], arr[:, 1], **vertex_style)

    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return ax


def _render_chain_field_3d(
    ax, field, filtration, chain_ids,
    *, vertex_style, edge_style, triangle_style, tetrahedron_style,
    point_style, plot_source, title,
):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    ax = _ensure_3d_axes(ax)
    D, H, W = field.shape

    if plot_source:
        # Render the grid as a sparse scatter with color = field value.
        # Drawing every grid point is overkill at large sizes, but keeps the
        # demo straightforward; users can disable with plot_source=False.
        ii, jj, kk = np.mgrid[0:D, 0:H, 0:W]
        ax.scatter(
            kk.ravel(), jj.ravel(), ii.ravel(),
            c=field.ravel(),
            cmap="viridis",
            s=point_style.get("s", 12.0),
            alpha=point_style.get("alpha", 0.4),
        )

    cubical = _is_cubical_filtration(filtration)

    vertex_xyz = []
    edge_segments = []
    triangle_polys = []
    cube_face_polys = []
    skipped_high_dim = 0

    def _ijk_to_xyz(ijk):
        # Plot convention: x=k, y=j, z=i so the third array dim runs along x.
        i, j, k = ijk
        return (k, j, i)

    if cubical:
        for cell_id in chain_ids:
            cell = filtration[int(cell_id)]
            corners = [tuple(c) for c in cell.vertices]
            if len(corners) == 1:
                vertex_xyz.append(_ijk_to_xyz(corners[0]))
            elif len(corners) == 2:
                edge_segments.append((_ijk_to_xyz(corners[0]), _ijk_to_xyz(corners[1])))
            elif len(corners) == 4:
                # 2-cube (square face). Reorder to cyclic by bounding-box.
                arr = np.asarray(corners, dtype=float)
                axis = int(np.argmin(arr.ptp(axis=0)))  # the constant axis
                others = [d for d in range(3) if d != axis]
                a0, a1 = arr[:, others[0]].min(), arr[:, others[0]].max()
                b0, b1 = arr[:, others[1]].min(), arr[:, others[1]].max()
                c = arr[0, axis]
                base = [None] * 3
                ordered = []
                for (a, b) in [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]:
                    base[axis] = c
                    base[others[0]] = a
                    base[others[1]] = b
                    ordered.append(_ijk_to_xyz(tuple(base)))
                triangle_polys.append(ordered)
            elif len(corners) == 8:
                for face in _cube_3d_face_polys(corners):
                    cube_face_polys.append([_ijk_to_xyz(c) for c in face])
            else:
                skipped_high_dim += 1
    else:
        # Simplicial Freudenthal in 3D.
        for cell_id in chain_ids:
            cell = filtration[int(cell_id)]
            verts = list(cell.vertices)
            coords = [_ijk_to_xyz(_id_to_grid(v, (D, H, W))) for v in verts]
            if len(coords) == 1:
                vertex_xyz.append(coords[0])
            elif len(coords) == 2:
                edge_segments.append((coords[0], coords[1]))
            elif len(coords) == 3:
                triangle_polys.append(coords)
            elif len(coords) == 4:
                cube_face_polys.extend(_tet_face_polys(coords))
            else:
                skipped_high_dim += 1

    if skipped_high_dim:
        warnings.warn(
            f"plot_chain: skipped {skipped_high_dim} cells of dim >= 4 "
            f"(3D field rendering only supports cells up to 3-cubes / tets).",
            stacklevel=3,
        )

    if cube_face_polys:
        ax.add_collection3d(Poly3DCollection(cube_face_polys, **tetrahedron_style))
    if triangle_polys:
        ax.add_collection3d(Poly3DCollection(triangle_polys, **triangle_style))
    if edge_segments:
        ax.add_collection3d(Line3DCollection(edge_segments, **edge_style))
    if vertex_xyz:
        arr = np.asarray(vertex_xyz, dtype=float)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], **vertex_style)

    if title is not None:
        ax.set_title(title)
    return ax
