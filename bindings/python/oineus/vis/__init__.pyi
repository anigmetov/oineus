"""Type stubs for oineus.vis (Matplotlib persistence diagram & chain plotting)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

OKABE_ITO_BLUE: str
OKABE_ITO_VERMILLION: str

DEFAULT_POINT_STYLE: dict[str, Any]
DEFAULT_DIAGRAM_A_POINT_STYLE: dict[str, Any]
DEFAULT_DIAGRAM_B_POINT_STYLE: dict[str, Any]
DEFAULT_MATCHING_EDGE_STYLE: dict[str, Any]
DEFAULT_LONGEST_EDGE_STYLE: dict[str, Any]
DEFAULT_DIAGONAL_STYLE: dict[str, Any]
DEFAULT_DIAGONAL_PROJECTION_A_STYLE: dict[str, Any]
DEFAULT_DIAGONAL_PROJECTION_B_STYLE: dict[str, Any]
DEFAULT_INF_LINE_STYLE: dict[str, Any]
DEFAULT_INF_POINT_STYLE: dict[str, Any]
DEFAULT_DIAGRAM_GRADIENT_STYLE: dict[str, Any]
DEFAULT_DENSITY_STYLE: dict[str, Any]
DEFAULT_DENSITY_THRESHOLD: int
DEFAULT_GRID_STYLE: dict[str, Any]
DEFAULT_MATCHING_EDGE_QUANTILE: float
DEFAULT_GRADIENT_TOP_K_ARROWS: int
DEFAULT_CHAIN_VERTEX_STYLE: dict[str, Any]
DEFAULT_CHAIN_EDGE_STYLE: dict[str, Any]
DEFAULT_CHAIN_TRIANGLE_STYLE: dict[str, Any]
DEFAULT_CHAIN_TETRAHEDRON_STYLE: dict[str, Any]
DEFAULT_POINT_CLOUD_STYLE: dict[str, Any]

def default_point_style() -> dict[str, Any]: ...
def default_diagram_a_point_style() -> dict[str, Any]: ...
def default_diagram_b_point_style() -> dict[str, Any]: ...
def default_matching_edge_style() -> dict[str, Any]: ...
def default_longest_edge_style() -> dict[str, Any]: ...
def default_diagonal_style() -> dict[str, Any]: ...
def default_diagonal_projection_a_style() -> dict[str, Any]: ...
def default_diagonal_projection_b_style() -> dict[str, Any]: ...
def default_inf_line_style() -> dict[str, Any]: ...
def default_inf_point_style() -> dict[str, Any]: ...
def default_diagram_gradient_style() -> dict[str, Any]: ...
def default_density_style() -> dict[str, Any]: ...
def default_grid_style() -> dict[str, Any]: ...
def default_chain_vertex_style() -> dict[str, Any]: ...
def default_chain_edge_style() -> dict[str, Any]: ...
def default_chain_triangle_style() -> dict[str, Any]: ...
def default_chain_tetrahedron_style() -> dict[str, Any]: ...
def default_point_cloud_style() -> dict[str, Any]: ...

def plot_diagram(
    diagrams: Any,
    ax: Any = None,
    *,
    color: Any = None,
    cmap: Any = None,
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
    suptitle: str | None = None,
    axis_bounds: Mapping[str, float] | None = None,
    dims: Iterable[int] | None = None,
    max_dimension: int | None = None,
    scatter_only: bool = True,
    density_threshold: int = ...,
    near_diagonal_fraction: float = 0.03,
    density_style: dict[str, Any] | None = None,
    inf_line_margin: float = 0.05,
    point_style: dict[str, Any] | None = None,
    inf_point_style: dict[str, Any] | None = None,
    diagonal_style: dict[str, Any] | None = None,
    inf_line_style: dict[str, Any] | None = None,
    grid_style: dict[str, Any] | bool | None = None,
    dim_label_fmt: str = "H{dim}",
    use_density: Any = ...,
) -> Any: ...

def plot_diagram_gradient(
    diagram: Any,
    gradient: Any = None,
    *,
    ax: Any = None,
    dims: Iterable[int] | None = None,
    descent: bool = False,
    plot_points: bool = True,
    scatter_only: bool = True,
    density_threshold: int = ...,
    min_persistence: float | None = None,
    top_k_arrows: int | None = None,
    arrow_overlay_threshold: int = 1000,
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
    axis_bounds: Mapping[str, float] | None = None,
    inf_line_margin: float = 0.05,
    diagram_color: Any = None,
    grad_color: Any = None,
    cmap: Any = None,
    quiver_style: dict[str, Any] | None = None,
    point_style: dict[str, Any] | None = None,
    inf_point_style: dict[str, Any] | None = None,
    diagonal_style: dict[str, Any] | None = None,
    inf_line_style: dict[str, Any] | None = None,
    density_style: dict[str, Any] | None = None,
    dim_label_fmt: str = "H{dim}",
    use_density: Any = ...,
) -> Any: ...

def plot_matching(
    dgm_a: Any,
    dgm_b: Any,
    matching: Any,
    ax: Any = None,
    *,
    plot_finite_to_finite: bool | None = None,
    plot_a_to_diagonal: bool | None = None,
    plot_b_to_diagonal: bool | None = None,
    plot_essential: bool | None = None,
    highlight_longest: bool | None = None,
    plot_points: bool = True,
    plot_diagonal_projections: bool = False,
    plot_diagonal: bool = True,
    scatter_only: bool = True,
    density_threshold: int = ...,
    near_diagonal_fraction: float = 0.03,
    edge_quantile: float = ...,
    min_persistence: float | None = None,
    top_k_pairs: int | None = None,
    pair_filter: str = "either",
    pair_overlay_threshold: int = 1000,
    color_dgm_a: Any = None,
    color_dgm_b: Any = None,
    match_color: Any = None,
    cmap_a: Any = None,
    cmap_b: Any = None,
    density_style: dict[str, Any] | None = None,
    dgm_a_point_style: dict[str, Any] | None = None,
    dgm_b_point_style: dict[str, Any] | None = None,
    ordinary_edge_style: dict[str, Any] | None = None,
    longest_edge_style: dict[str, Any] | None = None,
    diagonal_style: dict[str, Any] | None = None,
    diagonal_projection_a_style: dict[str, Any] | None = None,
    diagonal_projection_b_style: dict[str, Any] | None = None,
    inf_line_style: dict[str, Any] | None = None,
    dgm_a_label: str = "Diagram A",
    dgm_b_label: str = "Diagram B",
    title: str | None = None,
    axis_bounds: Mapping[str, float] | None = None,
    inf_line_margin: float = 0.05,
    use_density: Any = ...,
) -> Any: ...

def plot_chain(
    source: Any,
    filtration: Any,
    chain: Any,
    *,
    ax: Any = None,
    source_kind: str | None = None,
    dualize: Any = False,
    chain_color: Any = None,
    point_cloud_color: Any = None,
    edge_style: dict[str, Any] | None = None,
    triangle_style: dict[str, Any] | None = None,
    tetrahedron_style: dict[str, Any] | None = None,
    vertex_style: dict[str, Any] | None = None,
    point_style: dict[str, Any] | None = None,
    title: str | None = None,
    plot_source: bool = True,
    field_cmap: str = "viridis",
) -> Any: ...

__all__: list[str]
