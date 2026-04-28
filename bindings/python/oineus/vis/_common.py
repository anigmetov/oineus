"""Backend-agnostic helpers for the visualization layer.

Pure numpy + Python -- no matplotlib (or plotly) imports. All input coercion,
geometry, and style-merging logic lives here so backend modules
(``_matplotlib``, eventually ``_plotly``) can rely on a shared foundation.
"""
from __future__ import annotations

import typing

import numpy as np


# ---------------------------------------------------------------------------
# Style merging
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _shift_for_log(values: np.ndarray, use_log: bool) -> float:
    if not use_log or values.size == 0:
        return 0.0
    min_val = float(np.min(values[np.isfinite(values)]))
    if min_val <= 0.0:
        return 1.0 - min_val
    return 0.0


def _compute_plot_limits(
    finite_xs: np.ndarray,
    finite_ys: np.ndarray,
    extra_xs: typing.Sequence[np.ndarray] = (),
    extra_ys: typing.Sequence[np.ndarray] = (),
    inf_line_margin: float = 0.05,
):
    """Compute (x_min, x_max, y_min, y_max, x_span, y_span, inf_x_pos,
    inf_x_neg, inf_y_pos, inf_y_neg) given the finite data and any extra
    finite-coordinate arrays that should be included for positioning."""
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


def _classify_points(dgm: np.ndarray):
    """Return masks for (finite, pos_inf_death, neg_inf_death,
    pos_inf_birth, neg_inf_birth) points in a (n, 2) diagram."""
    b = dgm[:, 0]
    d = dgm[:, 1]
    finite_mask = np.isfinite(b) & np.isfinite(d)
    pos_inf_d = np.isfinite(b) & np.isposinf(d)
    neg_inf_d = np.isfinite(b) & np.isneginf(d)
    pos_inf_b = np.isposinf(b) & np.isfinite(d)
    neg_inf_b = np.isneginf(b) & np.isfinite(d)
    return finite_mask, pos_inf_d, neg_inf_d, pos_inf_b, neg_inf_b


def _build_diagram_arrays(dgm) -> np.ndarray:
    if isinstance(dgm, np.ndarray):
        return _array_diagram(dgm)
    if hasattr(dgm, "birth") and hasattr(dgm, "death"):
        raise TypeError("Single DiagramPoint cannot be plotted; pass a diagram.")
    if isinstance(dgm, list):
        return _diagram_points_to_array(dgm)
    return _array_diagram(dgm)


def _point_coords_for_edge(
    dgm: np.ndarray,
    idx: int,
    inf_x_pos: float,
    inf_x_neg: float,
    inf_y_pos: float,
    inf_y_neg: float,
) -> typing.Tuple[float, float]:
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


def _split_near_diagonal(births, deaths, near_thr):
    """Partition points by distance to the diagonal: returns
    (near_b, near_d, far_b, far_d). ``near_thr`` is a coordinate-space
    threshold, i.e. ``|death - birth| <= near_thr`` defines the bulk."""
    if births.size == 0:
        return births, deaths, births, deaths
    near_mask = np.abs(deaths - births) <= near_thr
    return (
        births[near_mask], deaths[near_mask],
        births[~near_mask], deaths[~near_mask],
    )


# ---------------------------------------------------------------------------
# Gradient-input coercion (used by plot_diagram_gradient)
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


# ---------------------------------------------------------------------------
# Chain coercion (plot_chain)
# ---------------------------------------------------------------------------

def _id_to_grid(vertex_id: int, shape: typing.Sequence[int]) -> typing.Tuple[int, ...]:
    """Convert a C-order ravel index back to grid coordinates.

    Mirrors oineus's Freudenthal vertex-id convention: for shape ``(H, W)``
    the id is ``i*W + j`` so this returns ``(i, j)``. For shape ``(D, H, W)``
    the id is ``i*H*W + j*W + k`` returning ``(i, j, k)``. C-order matches
    the strides computed in ``include/oineus/grid_domain.h``.
    """
    coords = []
    rem = int(vertex_id)
    for s in reversed(shape):
        coords.append(rem % int(s))
        rem //= int(s)
    return tuple(reversed(coords))


def _coerce_chain(chain) -> np.ndarray:
    """Coerce a chain into a flat numpy array of int sorted-ids.

    Accepts: list[int], np.ndarray, range, tuple, scipy.sparse vector slice
    (1xN or Nx1), or any object with an ``indices`` attribute (e.g. CSR/CSC
    column slice). Empty input returns an empty int64 array.
    """
    # scipy.sparse vectors / matrices
    indices_attr = getattr(chain, "indices", None)
    nnz_attr = getattr(chain, "nnz", None)
    if indices_attr is not None and nnz_attr is not None:
        return np.asarray(indices_attr, dtype=np.int64)

    if hasattr(chain, "nonzero") and not isinstance(chain, np.ndarray):
        nz = chain.nonzero()
        if len(nz) == 2:
            shape = getattr(chain, "shape", None)
            if shape is not None and shape[0] == 1:
                return np.asarray(nz[1], dtype=np.int64)
            return np.asarray(nz[0], dtype=np.int64)
        return np.asarray(nz[0], dtype=np.int64)

    arr = np.asarray(list(chain), dtype=np.int64)
    return arr.ravel()
