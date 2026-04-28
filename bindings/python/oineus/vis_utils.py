from __future__ import annotations

import typing

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

try:
    import mpl_scatter_density  # noqa: F401
    _HAS_MPL_SCATTER_DENSITY = True
except Exception:
    _HAS_MPL_SCATTER_DENSITY = False


# ---------------------------------------------------------------------------
# Style-dict infrastructure
#
# Mutable module-level defaults that every plotting helper below consults.
# Callers get a fresh dict from the default_*_style() getters, modify it,
# and pass it back via the corresponding kwarg for a one-shot override.
# For global overrides (once per script / notebook), mutate the module-level
# DEFAULT_*_STYLE dict in place: oineus.DEFAULT_MATCHING_EDGE_STYLE["linewidth"] = 2.
# ---------------------------------------------------------------------------

DEFAULT_POINT_STYLE: dict = {
    "marker": "o",
    "s": 25.0,
    "alpha": 0.9,
    "edgecolors": "none",
}

DEFAULT_DIAGRAM_A_POINT_STYLE: dict = {**DEFAULT_POINT_STYLE, "c": "tab:red"}
DEFAULT_DIAGRAM_B_POINT_STYLE: dict = {**DEFAULT_POINT_STYLE, "c": "tab:blue"}

DEFAULT_MATCHING_EDGE_STYLE: dict = {
    "linewidth": 0.8,
    "linestyle": "-",
    "color": "gray",
    "alpha": 0.5,
}

DEFAULT_LONGEST_EDGE_STYLE: dict = {
    "linewidth": 1.6,
    "linestyle": "-",
    "color": "red",
    "alpha": 0.95,
}

DEFAULT_DIAGONAL_STYLE: dict = {
    "linestyle": "--",
    "color": "gray",
    "alpha": 0.7,
    "linewidth": 1.0,
}

DEFAULT_DIAGONAL_PROJECTION_A_STYLE: dict = {
    "marker": "x",
    "s": 30.0,
    "alpha": 0.8,
    "c": "tab:red",
}

DEFAULT_DIAGONAL_PROJECTION_B_STYLE: dict = {
    "marker": "x",
    "s": 30.0,
    "alpha": 0.8,
    "c": "tab:blue",
}

DEFAULT_INF_LINE_STYLE: dict = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 1.0,
}

DEFAULT_DIAGRAM_GRADIENT_STYLE: dict = {
    "color": "tab:green",
    "alpha": 0.85,
    "width": 0.004,
    "headwidth": 3.5,
    "headlength": 5.0,
    "angles": "xy",
    "scale_units": "xy",
    "scale": 1.0,
}

# Above this point count we switch large diagrams from per-point scatter to
# a density-aggregated raster (via mpl_scatter_density). Tuned so that the
# perceptual oversaturation that starts around 5k is mostly hidden, while
# small diagrams still get the crisp scatter rendering users expect.
DEFAULT_DENSITY_THRESHOLD: int = 20_000

# Style passed to the underlying ScatterDensityArtist. The norm is set by
# the caller (defaults to PowerNorm(0.5) inside _add_density_artist).
DEFAULT_DENSITY_STYLE: dict = {
    "cmap": "viridis",
    "dpi": 72,
    "downres_factor": 4,
}

# Edges with length below this quantile are hidden in matching-density mode.
# 0.99 keeps the top 1% of edges -- typically the ones that carry the
# matching-cost signal.
DEFAULT_MATCHING_EDGE_QUANTILE: float = 0.99

# Number of arrows to draw in gradient-density mode (largest |grad| points).
DEFAULT_GRADIENT_TOP_K_ARROWS: int = 200


def default_point_style() -> dict:
    return dict(DEFAULT_POINT_STYLE)


def default_diagram_a_point_style() -> dict:
    return dict(DEFAULT_DIAGRAM_A_POINT_STYLE)


def default_diagram_b_point_style() -> dict:
    return dict(DEFAULT_DIAGRAM_B_POINT_STYLE)


def default_matching_edge_style() -> dict:
    return dict(DEFAULT_MATCHING_EDGE_STYLE)


def default_longest_edge_style() -> dict:
    return dict(DEFAULT_LONGEST_EDGE_STYLE)


def default_diagonal_style() -> dict:
    return dict(DEFAULT_DIAGONAL_STYLE)


def default_diagonal_projection_a_style() -> dict:
    return dict(DEFAULT_DIAGONAL_PROJECTION_A_STYLE)


def default_diagonal_projection_b_style() -> dict:
    return dict(DEFAULT_DIAGONAL_PROJECTION_B_STYLE)


def default_inf_line_style() -> dict:
    return dict(DEFAULT_INF_LINE_STYLE)


def default_diagram_gradient_style() -> dict:
    return dict(DEFAULT_DIAGRAM_GRADIENT_STYLE)


def default_density_style() -> dict:
    return dict(DEFAULT_DENSITY_STYLE)


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


def _resolve_style(user: typing.Optional[dict], default_getter) -> dict:
    """Merge a fresh copy of the default with user overrides (user wins)."""
    style = default_getter()
    if user:
        style.update(user)
    return style


# ---------------------------------------------------------------------------
# Diagram input coercion
# ---------------------------------------------------------------------------

def _diagram_points_to_array(diagram_points) -> np.ndarray:
    if len(diagram_points) == 0:
        return np.empty((0, 2), dtype=float)
    return np.asarray([(p.birth, p.death) for p in diagram_points], dtype=float)


def _array_diagram(a) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Diagram array must have shape (n, 2).")
    return arr


def _to_dim_diagrams(
    diagrams,
    dims: typing.Optional[typing.Iterable[int]],
    max_dimension: typing.Optional[int],
) -> typing.Dict[int, np.ndarray]:
    if isinstance(diagrams, dict):
        out = {}
        for dim, dgm in diagrams.items():
            if isinstance(dgm, np.ndarray):
                out[int(dim)] = _array_diagram(dgm)
            else:
                out[int(dim)] = _diagram_points_to_array(dgm)
        return out

    if hasattr(diagrams, "in_dimension"):
        if dims is None:
            max_scan_dim = 64 if max_dimension is None else int(max_dimension)
            out = {}
            seen_non_empty = False
            for dim in range(max_scan_dim + 1):
                try:
                    arr = _array_diagram(diagrams.in_dimension(dim, as_numpy=True))
                except (IndexError, RuntimeError):
                    break
                if arr.shape[0] == 0:
                    if seen_non_empty:
                        break
                    continue
                seen_non_empty = True
                out[dim] = arr
            if not out:
                out[0] = np.empty((0, 2), dtype=float)
            return out
        return {int(dim): _array_diagram(diagrams.in_dimension(int(dim), as_numpy=True)) for dim in dims}

    if isinstance(diagrams, np.ndarray):
        return {0: _array_diagram(diagrams)}

    if isinstance(diagrams, list):
        if len(diagrams) == 0:
            return {0: np.empty((0, 2), dtype=float)}

        first = diagrams[0]
        if isinstance(first, np.ndarray):
            return {dim: _array_diagram(dgm) for dim, dgm in enumerate(diagrams)}

        if hasattr(first, "birth") and hasattr(first, "death"):
            return {0: _diagram_points_to_array(diagrams)}

        raise TypeError(
            "Unsupported list input. Expected list[DiagramPoint] or list[np.ndarray]."
        )

    raise TypeError(
        "Unsupported diagrams input. Expected Diagrams, list[DiagramPoint], "
        "numpy.ndarray, list[numpy.ndarray], or dict[int, numpy.ndarray]."
    )


def _resolve_color(color, dim: int, dim_idx: int):
    if color is None:
        return None
    if isinstance(color, dict):
        return color.get(dim, None)
    if isinstance(color, list):
        if len(color) == 0:
            return None
        return color[dim_idx % len(color)]
    return color


def _shift_for_log(values: np.ndarray, use_log: bool) -> float:
    if not use_log or values.size == 0:
        return 0.0
    min_val = float(np.min(values[np.isfinite(values)]))
    if min_val <= 0.0:
        return 1.0 - min_val
    return 0.0


# ---------------------------------------------------------------------------
# Shared inf-line / axis-range helper
# ---------------------------------------------------------------------------

def _compute_plot_limits(finite_xs: np.ndarray, finite_ys: np.ndarray,
                         extra_xs: typing.Sequence[np.ndarray] = (),
                         extra_ys: typing.Sequence[np.ndarray] = (),
                         inf_line_margin: float = 0.05):
    """Compute (x_min, x_max, y_min, y_max, x_span, y_span, inf_x_pos,
    inf_x_neg, inf_y_pos, inf_y_neg) given the finite data and any extra
    finite-coordinate arrays that should be included for positioning.
    """
    xs = [finite_xs] + list(extra_xs)
    ys = [finite_ys] + list(extra_ys)
    xs = [a for a in xs if a.size > 0]
    ys = [a for a in ys if a.size > 0]
    all_x = np.concatenate(xs) if xs else np.empty((0,), dtype=float)
    all_y = np.concatenate(ys) if ys else np.empty((0,), dtype=float)

    if all_y.size:
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    elif all_x.size:
        y_min, y_max = float(np.min(all_x)), float(np.max(all_x))
    else:
        y_min, y_max = -1.0, 1.0

    if all_x.size:
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    else:
        x_min, x_max = y_min, y_max

    y_span = y_max - y_min
    x_span = x_max - x_min
    if y_span <= 0.0:
        y_span = max(abs(y_max), abs(y_min), 1.0)
    if x_span <= 0.0:
        x_span = max(abs(x_max), abs(x_min), 1.0)

    inf_y_pos = y_max + inf_line_margin * y_span
    inf_y_neg = y_min - inf_line_margin * y_span
    inf_x_pos = x_max + inf_line_margin * x_span
    inf_x_neg = x_min - inf_line_margin * x_span

    return (x_min, x_max, y_min, y_max, x_span, y_span,
            inf_x_pos, inf_x_neg, inf_y_pos, inf_y_neg)


# ---------------------------------------------------------------------------
# plot_diagram (refactored from plot_persistence_diagram)
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

def _coerce_diagram_with_grad(diagram, gradient, dims):
    """Normalize ``diagram`` + ``gradient`` to two parallel
    ``dict[int, np.ndarray]``, both ``(n, 2)`` per dim.

    Accepted ``diagram`` forms: ``torch.Tensor``, ``np.ndarray``, native
    ``oineus.Diagrams``, dict-like (incl. ``oineus.diff.PersistenceDiagrams``).
    When ``gradient`` is ``None`` the input must be torch-backed and the
    gradient is pulled from ``.grad``.
    """
    try:
        import torch
        has_torch = True
    except ImportError:  # pragma: no cover - torch is optional
        torch = None
        has_torch = False

    def _torch_to_array(t):
        return t.detach().cpu().numpy().astype(float, copy=False)

    def _grad_for_tensor(t):
        if t.grad is None:
            raise ValueError(
                "Torch diagram tensor has no .grad. Make sure a loss derived "
                "from this tensor was followed by .backward(). For non-leaf "
                "tensors (e.g. those returned by "
                "oineus.diff.persistence_diagram), call dgm.retain_grad() "
                "before .backward()."
            )
        return _torch_to_array(t.grad)

    def _array_2d(a, what):
        arr = np.asarray(a, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{what} must have shape (n, 2); got {arr.shape}.")
        return arr

    def _gradient_array(g):
        if has_torch and isinstance(g, torch.Tensor):
            return _array_2d(_torch_to_array(g), "gradient")
        return _array_2d(g, "gradient")

    if has_torch and isinstance(diagram, torch.Tensor):
        if diagram.ndim != 2 or diagram.shape[1] != 2:
            raise ValueError("torch diagram tensor must have shape (n, 2).")
        pts = _torch_to_array(diagram)
        grad = _grad_for_tensor(diagram) if gradient is None else _gradient_array(gradient)
        return {0: pts}, {0: grad}

    if isinstance(diagram, np.ndarray):
        if gradient is None:
            raise ValueError("When diagram is a numpy array, gradient must be supplied.")
        return {0: _array_2d(diagram, "diagram")}, {0: _gradient_array(gradient)}

    is_dict_like = isinstance(diagram, dict) or hasattr(diagram, "items")
    if is_dict_like:
        diag_items = list(diagram.items())
        if dims is not None:
            wanted = {int(d) for d in dims}
            diag_items = [(d, v) for (d, v) in diag_items if int(d) in wanted]
        grad_dict = gradient if isinstance(gradient, dict) else None
        if gradient is not None and grad_dict is None:
            raise ValueError(
                "When diagram is dict-like, gradient must also be a "
                "dict[int, ndarray | Tensor] or None."
            )
        pts_out: typing.Dict[int, np.ndarray] = {}
        grad_out: typing.Dict[int, np.ndarray] = {}
        for dim, val in diag_items:
            dim_int = int(dim)
            if has_torch and isinstance(val, torch.Tensor):
                pts_out[dim_int] = _torch_to_array(val)
                if grad_dict is not None and dim_int in grad_dict:
                    grad_out[dim_int] = _gradient_array(grad_dict[dim_int])
                elif gradient is None:
                    grad_out[dim_int] = _grad_for_tensor(val)
                else:
                    raise ValueError(
                        f"gradient dict is missing dimension {dim_int}."
                    )
            else:
                pts_out[dim_int] = _array_2d(val, f"diagram[{dim_int}]")
                if grad_dict is None or dim_int not in grad_dict:
                    raise ValueError(
                        f"Non-torch diagram entry at dim {dim_int} requires "
                        "an explicit gradient."
                    )
                grad_out[dim_int] = _gradient_array(grad_dict[dim_int])
        return pts_out, grad_out

    if hasattr(diagram, "in_dimension"):
        if not isinstance(gradient, dict):
            raise ValueError(
                "Native oineus.Diagrams requires gradient as a "
                "dict[int, ndarray | Tensor]."
            )
        scan_dims = sorted(int(k) for k in gradient.keys()) if dims is None else [int(d) for d in dims]
        pts_out = {}
        grad_out = {}
        for dim in scan_dims:
            try:
                arr = diagram.in_dimension(int(dim), as_numpy=True)
            except (IndexError, RuntimeError):
                continue
            if int(dim) not in gradient:
                raise ValueError(f"gradient dict is missing dimension {dim}.")
            pts_out[int(dim)] = _array_2d(arr, f"diagram[{dim}]")
            grad_out[int(dim)] = _gradient_array(gradient[int(dim)])
        return pts_out, grad_out

    raise TypeError(
        f"Unsupported diagram type: {type(diagram).__name__}. Expected "
        "torch.Tensor, numpy.ndarray, native oineus.Diagrams, "
        "oineus.diff.PersistenceDiagrams, or dict[int, ndarray | Tensor]."
    )


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

def _build_diagram_arrays(dgm) -> np.ndarray:
    if isinstance(dgm, np.ndarray):
        return _array_diagram(dgm)
    if hasattr(dgm, "birth") and hasattr(dgm, "death"):
        raise TypeError("Single DiagramPoint cannot be plotted; pass a diagram.")
    if isinstance(dgm, list):
        return _diagram_points_to_array(dgm)
    return _array_diagram(dgm)


def _classify_points(dgm: np.ndarray):
    """Return masks and indices for (finite, pos_inf_death, neg_inf_death,
    pos_inf_birth, neg_inf_birth) points in a (n, 2) diagram."""
    b = dgm[:, 0]
    d = dgm[:, 1]
    finite_mask = np.isfinite(b) & np.isfinite(d)
    pos_inf_d = np.isfinite(b) & np.isposinf(d)
    neg_inf_d = np.isfinite(b) & np.isneginf(d)
    pos_inf_b = np.isposinf(b) & np.isfinite(d)
    neg_inf_b = np.isneginf(b) & np.isfinite(d)
    return finite_mask, pos_inf_d, neg_inf_d, pos_inf_b, neg_inf_b


def _point_coords_for_edge(dgm: np.ndarray, idx: int,
                           inf_x_pos: float, inf_x_neg: float,
                           inf_y_pos: float, inf_y_neg: float) -> typing.Tuple[float, float]:
    """Coordinates to use when drawing an edge endpoint for a diagram point,
    mapping infinities to the capped inf-line coordinates."""
    b = float(dgm[idx, 0])
    d = float(dgm[idx, 1])
    if np.isposinf(d):
        return (b, inf_y_pos)
    if np.isneginf(d):
        return (b, inf_y_neg)
    if np.isposinf(b):
        return (inf_x_pos, d)
    if np.isneginf(b):
        return (inf_x_neg, d)
    return (b, d)


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
    from .matching import (
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

    def _split_near_diagonal(births, deaths):
        if births.size == 0:
            return births, deaths, births, deaths
        near_mask = np.abs(deaths - births) <= near_thr
        return (
            births[near_mask], deaths[near_mask],
            births[~near_mask], deaths[~near_mask],
        )

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
        near_b, near_d, far_b, far_d = _split_near_diagonal(finite[:, 0], finite[:, 1])
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
    ordinary_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

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
