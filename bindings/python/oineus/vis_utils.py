from __future__ import annotations

import typing

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

try:
    import mpl_scatter_density  # noqa: F401
    _HAS_MPL_SCATTER_DENSITY = True
except Exception:
    _HAS_MPL_SCATTER_DENSITY = False


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
    # dict: dim -> ndarray or list[DiagramPoint]
    if isinstance(diagrams, dict):
        out = {}
        for dim, dgm in diagrams.items():
            if isinstance(dgm, np.ndarray):
                out[int(dim)] = _array_diagram(dgm)
            else:
                out[int(dim)] = _diagram_points_to_array(dgm)
        return out

    # Diagrams object from oineus bindings
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

    # Single numpy diagram
    if isinstance(diagrams, np.ndarray):
        return {0: _array_diagram(diagrams)}

    # list[DiagramPoint] or list[np.ndarray]
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


def plot_persistence_diagram(
    diagrams,
    ax=None,
    *,
    marker: str = "o",
    marker_size: float = 16.0,
    color=None,
    alpha: float = 0.9,
    log_x: bool = False,
    log_y: bool = False,
    title: typing.Optional[str] = None,
    suptitle: typing.Optional[str] = None,
    axis_bounds: typing.Optional[typing.Mapping[str, float]] = None,
    dims: typing.Optional[typing.Iterable[int]] = None,
    max_dimension: typing.Optional[int] = None,
    use_density: bool = True,
    density_threshold: int = 50000,
    near_diagonal_fraction: float = 0.03,
    density_cmap: str = "viridis",
    inf_line_margin: float = 0.05,
    inf_line_color: str = "black",
    inf_line_style: str = "--",
    diag_line_color: str = "gray",
    diag_line_style: str = "--",
    diag_line_alpha: float = 0.7,
    scatter_kwargs: typing.Optional[dict] = None,
):
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plot_persistence_diagram.")

    bounds = {} if axis_bounds is None else dict(axis_bounds)
    scatter_kwargs = {} if scatter_kwargs is None else dict(scatter_kwargs)
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
    if use_density_plot and not _HAS_MPL_SCATTER_DENSITY:
        raise ImportError(
            "mpl_scatter_density is required when density rendering is used. "
            "Install mpl_scatter_density, or disable density plotting "
            "(use_density=False / increase density_threshold)."
        )

    if ax is None:
        if use_density_plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        else:
            _, ax = plt.subplots()
    elif use_density_plot and not hasattr(ax, "scatter_density"):
        raise ValueError(
            "For density rendering, axis must use projection='scatter_density'. "
            "Pass ax=None or create the axis with that projection."
        )

    y_values_for_shift = all_finite_deaths
    if any_pos_inf:
        y_values_for_shift = np.concatenate([y_values_for_shift, np.asarray([inf_y_pos])])
    if any_neg_inf:
        y_values_for_shift = np.concatenate([y_values_for_shift, np.asarray([inf_y_neg])])

    x_shift = _shift_for_log(all_births_for_limits, log_x)
    y_shift = _shift_for_log(y_values_for_shift, log_y)

    if use_density_plot:
        ax.scatter_density(near_x + x_shift, near_y + y_shift, cmap=density_cmap)

    for dim_idx, dim in enumerate(dims_sorted):
        births, deaths = finite_by_dim[dim]
        if births.size:
            mask = ~near_mask_by_dim[dim] if use_density_plot else np.ones_like(births, dtype=bool)
            if np.any(mask):
                ax.scatter(
                    births[mask] + x_shift,
                    deaths[mask] + y_shift,
                    s=marker_size,
                    marker=marker,
                    c=_resolve_color(color, dim, dim_idx),
                    alpha=alpha,
                    label=f"H{dim}",
                    **scatter_kwargs,
                )

        label_for_inf = f"H{dim}" if births.size == 0 else None

        pos_inf_births = pos_inf_birth_by_dim[dim]
        if pos_inf_births.size:
            ax.scatter(
                pos_inf_births + x_shift,
                np.full_like(pos_inf_births, inf_y_pos + y_shift),
                s=marker_size,
                marker=marker,
                c=_resolve_color(color, dim, dim_idx),
                alpha=alpha,
                label=label_for_inf,
                **scatter_kwargs,
            )
            label_for_inf = None

        neg_inf_births = neg_inf_birth_by_dim[dim]
        if neg_inf_births.size:
            ax.scatter(
                neg_inf_births + x_shift,
                np.full_like(neg_inf_births, inf_y_neg + y_shift),
                s=marker_size,
                marker=marker,
                c=_resolve_color(color, dim, dim_idx),
                alpha=alpha,
                label=label_for_inf,
                **scatter_kwargs,
            )

    if any_pos_inf:
        ax.axhline(inf_y_pos + y_shift, color=inf_line_color, linestyle=inf_line_style, linewidth=1.0)
    if any_neg_inf:
        ax.axhline(inf_y_neg + y_shift, color=inf_line_color, linestyle=inf_line_style, linewidth=1.0)

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
        ax.plot([lo, hi], [lo, hi], linestyle=diag_line_style, color=diag_line_color, alpha=diag_line_alpha, linewidth=1.0)

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
