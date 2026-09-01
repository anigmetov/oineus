from __future__ import absolute_import

__version__ = "0.9.36"

import copy
import typing
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from . import _oineus

from ._oineus import ConflictStrategy, DenoiseStrategy, VREdge, FiltrationKind
from ._oineus import DiagramPlaneDomain, FrechetMeanInit
from ._oineus import CombinatorialProdSimplex, CombinatorialSimplex
# Simplex / ProdSimplex are Real-templated valued cells (a float32 build registers distinct
# classes under _f32); they are re-exposed below as cross-backend markers so isinstance works on
# float32 cells. CombinatorialSimplex / CombinatorialProdSimplex are value-less (Real-independent),
# so the single shared class is fine.
from ._oineus import Decomposition, IndexDiagramPoint
# Diagrams / DiagramPoint are Real-templated (a float32 build registers distinct classes under
# _f32), so the bare top-module class would fail isinstance on a float32 diagram. They are
# re-exposed below as cross-backend markers, mirroring Filtration / ProdFiltration.
# IndexDiagramPoint is Real-independent (indices are ints), so the single shared class is fine.
# reduce is the fused one-shot build+reduce free function; it is Real-templated (registered per
# backend), so it is re-exposed below as a dtype-routing facade (like the other routed helpers).
from ._oineus import DecompositionManipStats
from ._oineus import ReductionParams, ReductionParamsAdvanced, ReductionTimings, UComputeTimings, KICRParams
from ._oineus import ColumnRepr
# KerImCokReduced(+Prod), IndicesValues(+Prod) and the concrete per-cell-type TopologyOptimizer
# classes (Prod / Cube_ND) are Real-templated (and several are per-cell-type too); they are
# re-exposed below as cross-backend markers so isinstance works on float32 / packed / slim results
# and the direct constructors route by dtype. KICRParams / ReductionParams / ColumnRepr are
# Real-independent (shared), so the single class is fine as a direct import.
from ._oineus import get_boundary_matrix
# compute_relative_diagrams / get_denoise_target / get_induced_matching / get_nth_persistence /
# get_permutation(_dtv) are dtype- and cell-type-routed Python wrappers defined below: the C++
# overloads live per-Real (top module for float64, _f32 for float32) and per cell type, so the
# bare top-module symbol would reject the now-default packed/slim and any float32 filtration.
from ._oineus import bottleneck_distance as _bottleneck_distance_cpp
from ._oineus import wasserstein_distance as _wasserstein_distance_cpp
from ._oineus import init_frechet_mean_first_diagram as _init_frechet_mean_first_diagram_cpp
from ._oineus import init_frechet_mean_random_diagram as _init_frechet_mean_random_diagram_cpp
from ._oineus import init_frechet_mean_medoid_diagram as _init_frechet_mean_medoid_diagram_cpp
from ._oineus import init_frechet_mean_diagonal_grid as _init_frechet_mean_diagonal_grid_cpp
from ._oineus import frechet_mean as _frechet_mean_cpp
# Cube_ND and Grid_ND are Real-templated (re-exposed below as cross-backend markers; Grid carries
# the Real-typed value array, so its construction routes by dtype). The GridDomain / CombinatorialCube
# types are Real-independent, kept as direct imports.
from ._oineus import GridDomain_1D, CombinatorialCube_1D
from ._oineus import GridDomain_2D, CombinatorialCube_2D
from ._oineus import GridDomain_3D, CombinatorialCube_3D
from ._oineus import GridDomain_4D, CombinatorialCube_4D

from ._dtype import (REAL_DTYPE, DEFAULT_REAL_DTYPE, REAL_MODULES, as_real_numpy,
                     detect_real_dtype, real_module_for, module_of_oineus_obj)


def _merge_over_reals(per_module):
    """Merge a per-submodule dict over every compiled Real (float64 = the top
    module, float32 = _f32 when present). Class names are identical across
    submodules, but a float32 _Filtration is a distinct Python class from the
    float64 one, so the merged map dispatches both dtypes from one lookup."""
    merged = {}
    for sub in REAL_MODULES.values():
        merged.update(per_module(sub))
    return merged


# Maps the fat cell-with-value type a user hands to Filtration(...) to the concrete C++
# filtration class that consumes it. The per-encoding internal filtrations are distinct C++
# types (cube vs simplex vs product), but the user sees one Filtration.
_FIL_CLASS_BY_CELL_TYPE = _merge_over_reals(lambda s: {
    s.Simplex:      s._Filtration,        # fat Simplex (VR / alpha / user-built)
    s.ProdSimplex:  s._ProdFiltration,    # product cells (mapping cylinders)
    s.Cube_1D:      s._CubeFiltration_1D,  # fat cubes (hand-built cubical complexes)
    s.Cube_2D:      s._CubeFiltration_2D,
    s.Cube_3D:      s._CubeFiltration_3D,
    s.Cube_4D:      s._CubeFiltration_4D,
})

# Every concrete filtration C++ type (both dtypes), for isinstance(x, oineus.Filtration).
# Includes the factory-produced slim Freudenthal / bit-packed ones, which a user never
# constructs by hand but should still recognize as filtrations.
_ALL_FILTRATION_TYPES = tuple(
    f for s in REAL_MODULES.values() for f in (
        s._Filtration, s._ProdFiltration,
        s._CubeFiltration_1D, s._CubeFiltration_2D, s._CubeFiltration_3D, s._CubeFiltration_4D,
        s._FreudenthalFiltration_1D, s._FreudenthalFiltration_2D, s._FreudenthalFiltration_3D, s._FreudenthalFiltration_4D,
        s._PackedSimplexFiltration_64, s._PackedSimplexFiltration_128,
    ))


class _FiltrationMeta(type):
    # isinstance(x, oineus.Filtration) is True for any concrete filtration the library builds,
    # even though Filtration() returns the concrete C++ object (not a _FiltrationMeta instance).
    def __instancecheck__(cls, obj):
        return isinstance(obj, _ALL_FILTRATION_TYPES)

    def __subclasscheck__(cls, sub):
        # `sub is cls` keeps the relation reflexive: issubclass(Filtration, Filtration)
        # must be True even though the facade is not itself a concrete C++ type.
        return sub is cls or issubclass(sub, _ALL_FILTRATION_TYPES)


class Filtration(metaclass=_FiltrationMeta):
    """A filtration: an ordered list of cells, each with a filtration value.

    Construct one from a list of fat cells with values, dispatching on the cell type::

        oineus.Filtration([oineus.Simplex([0], 0.0), oineus.Simplex([0, 1], 1.0), ...])  # simplicial
        oineus.Filtration([oineus.Cube_2D(...), ...])                                     # cubical
        oineus.Filtration([oineus.ProdSimplex(...), ...])                                 # product cells

    For the common constructions use the factory functions instead, which build the cells for
    you (and pick an efficient internal cell encoding): vr_filtration / alpha_filtration for
    point clouds, freudenthal_filtration / cube_filtration for functions on grids.

    isinstance(x, oineus.Filtration) is True for any filtration the library produces, including
    the factory-built ones whose concrete C++ type is an internal detail.
    """

    def __new__(cls, cells, *args, **kwargs):
        if not hasattr(cells, "__getitem__"):
            # accept generators / iterators, not just sequences: materialize once so
            # we can both peek cells[0] to dispatch and hand the full list to C++
            cells = list(cells)
        try:
            first = cells[0]
        except IndexError:
            # empty list -> the universal fat Simplex filtration (historical default; there is
            # no cell to dispatch on)
            return _oineus._Filtration(cells, *args, **kwargs)
        except TypeError:
            raise ValueError(
                "Filtration(cells): cells must be a list of fat cells with values "
                "(Simplex / Cube_1D/2D/3D / ProdSimplex). For point clouds use vr_filtration or "
                "alpha_filtration; for functions on grids use freudenthal_filtration or "
                "cube_filtration.")
        fil_cls = _FIL_CLASS_BY_CELL_TYPE.get(type(first))
        if fil_cls is None:
            if isinstance(first, tuple):
                # (vertices, value) pairs -> the universal simplicial constructor, which builds
                # the Simplex cells itself (used by the diode alpha / Cech-Delaunay paths)
                return _oineus._Filtration(cells, *args, **kwargs)
            raise TypeError(
                f"Filtration(cells): unsupported cell type {type(first).__name__}; expected one "
                f"of {[t.__name__ for t in _FIL_CLASS_BY_CELL_TYPE]} or (vertices, value) tuples.")
        return fil_cls(cells, *args, **kwargs)


# Product-cell filtrations: those whose cells are oineus.ProdSimplex (ProductCell), i.e. the
# output of mapping_cylinder / multiply_filtration and of Filtration([ProdSimplex, ...]). Both
# Real backends register the concrete C++ type under the private name _ProdFiltration.
_PROD_FILTRATION_TYPES = tuple(s._ProdFiltration for s in REAL_MODULES.values())


class _ProdFiltrationMeta(type):
    # isinstance(x, oineus.ProdFiltration) is True for any product-cell filtration, in either
    # Real backend. ProdFiltration is a marker only -- it never instantiates (see __new__).
    def __instancecheck__(cls, obj):
        return isinstance(obj, _PROD_FILTRATION_TYPES)

    def __subclasscheck__(cls, sub):
        # `sub is cls` keeps the relation reflexive (see _FiltrationMeta).
        return sub is cls or issubclass(sub, _PROD_FILTRATION_TYPES)


class ProdFiltration(metaclass=_ProdFiltrationMeta):
    """Marker for product-cell filtrations -- those whose cells are oineus.ProdSimplex
    (ProductCell), e.g. the result of mapping_cylinder or multiply_filtration.

    Use it for membership tests only::

        isinstance(fil, oineus.ProdFiltration)   # True for any product-cell filtration

    It is a cell-type marker, distinct from the filtration's FiltrationKind (which records how
    the filtration was built). It is NOT a constructor; build a product filtration through the
    unified facade::

        oineus.Filtration([oineus.ProdSimplex([0], [0], 0.0), ...])
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "oineus.ProdFiltration is an isinstance marker, not a constructor. Build a product "
            "filtration with oineus.Filtration([oineus.ProdSimplex(...), ...]).")


# Several leaf C++ types are Real-templated: a float32 build registers a distinct class with the
# same name under _oineus._f32, so the bare top-module class fails isinstance on a float32 object.
# The marker below spans both backends for isinstance/issubclass, while construction returns the
# float64 concrete class (these leaf types -- diagram points, valued cells -- carry no input array
# to route construction on). Mirrors how Filtration / ProdFiltration span the backends; this is
# the leaf-type analogue of the _ALL_FILTRATION_TYPES marker above.
def _real_templated_marker(class_name, doc, *, isinstance_names=None, route=None, route_kw=None):
    """Cross-backend marker for a Real-templated C++ type registered under `class_name` in every
    Real backend (float64 on the top module, float32 under _f32).

    isinstance / issubclass span the concrete types across both backends. By default the marker
    covers just `class_name`; pass `isinstance_names` (a name -> bool predicate) to union a whole
    family of per-encoding concrete types -- needed for results whose concrete class depends on
    the filtration's cell encoding, not just its dtype (e.g. the optimizer returns an
    IndicesValuesFreudenthal_2D, the KICR an _KerImCokReduced_Cube_2D).

    Construction builds `class_name` from the float64 top module by default. Pass `route`
    (a callable mapping the first constructor argument to its backend (sub)module) to route
    construction by the dtype-bearing argument: real_module_for for a numpy array (Grid_ND),
    module_of_oineus_obj for an oineus filtration (KerImCokReduced, the concrete optimizers).
    `route_kw` is the name of that argument so routing also works when it is passed by keyword
    (e.g. Grid_2D(data=arr)); route(None) falls back to float64, so a missing argument is safe.
    Leaf types with no dtype-bearing argument (DiagramPoint, Simplex) keep the float64 default.
    """
    if isinstance_names is None:
        concrete = tuple(getattr(s, class_name) for s in REAL_MODULES.values())
    else:
        concrete = tuple(getattr(s, n) for s in REAL_MODULES.values() for n in dir(s)
                         if isinstance_names(n) and isinstance(getattr(s, n, None), type))

    class _Meta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, concrete)

        def __subclasscheck__(cls, sub):
            return issubclass(sub, concrete)

    def __new__(cls, *args, **kwargs):
        if route is None:
            mod = _oineus
        else:
            mod = route(args[0] if args else kwargs.get(route_kw))
        return getattr(mod, class_name)(*args, **kwargs)

    return _Meta(class_name, (), {"__doc__": doc, "__new__": __new__})


# Persistence diagrams and points (birth/death are stored in the Real dtype).
Diagrams = _real_templated_marker("Diagrams",
    "Persistence diagrams indexed by homology dimension.\n\n"
    "isinstance(x, oineus.Diagrams) is True for a diagram from either Real backend (float64 or\n"
    "float32). Extract one dimension with dgm.in_dimension(d) (NumPy (n, 2) array) or\n"
    "dgm.in_dimension(d, as_numpy=False) (list of DiagramPoint). Constructing\n"
    "oineus.Diagrams(max_dim) returns a float64 diagram container.")
DiagramPoint = _real_templated_marker("DiagramPoint",
    "A single persistence-diagram point with birth and death attributes.\n\n"
    "isinstance(x, oineus.DiagramPoint) is True for a point from either Real backend.\n"
    "Constructing oineus.DiagramPoint(birth, death) returns a float64 point.")

# Valued cells -- a cell together with its filtration value (the value is the Real dtype, so these
# are Real-templated, unlike the value-less CombinatorialSimplex / CombinatorialCube). fil[i]
# returns one of these; the marker makes isinstance(fil32[i], oineus.Simplex) True and lets
# multiply_filtration / mapping_cylinder accept a valued cell from a float32 filtration.
Simplex = _real_templated_marker("Simplex",
    "A simplex with a filtration value. isinstance spans both Real backends; constructing\n"
    "oineus.Simplex(vertices, value) (optionally with an id) returns a float64 simplex.")
ProdSimplex = _real_templated_marker("ProdSimplex",
    "A product cell (ProductCell of two simplices) with a filtration value. isinstance spans\n"
    "both Real backends; construction returns a float64 product cell.")
Cube_1D = _real_templated_marker("Cube_1D",
    "A 1D cube with a filtration value. isinstance spans both Real backends; construction\n"
    "returns a float64 cube.")
Cube_2D = _real_templated_marker("Cube_2D",
    "A 2D cube with a filtration value. isinstance spans both Real backends; construction\n"
    "returns a float64 cube.")
Cube_3D = _real_templated_marker("Cube_3D",
    "A 3D cube with a filtration value. isinstance spans both Real backends; construction\n"
    "returns a float64 cube.")
Cube_4D = _real_templated_marker("Cube_4D",
    "A 4D cube with a filtration value. isinstance spans both Real backends; construction\n"
    "returns a float64 cube.")

# Grids -- Real-templated (the lower-star value array is the Real dtype). Construction routes by
# the data array's dtype, so oineus.Grid_2D(float32_array) builds a genuine float32 grid (whose
# freudenthal_filtration / cube_filtration is then float32), mirroring the freudenthal_filtration
# facade -- a bare float64 Grid_2D would silently widen the data to float64.
Grid_1D = _real_templated_marker("Grid_1D",
    "A 1D regular grid of scalar values. isinstance spans both Real backends; constructing\n"
    "oineus.Grid_1D(data) routes to the backend matching data's dtype (float32 data -> a\n"
    "float32 grid).", route=real_module_for, route_kw="data")
Grid_2D = _real_templated_marker("Grid_2D",
    "A 2D regular grid of scalar values. isinstance spans both Real backends; construction\n"
    "routes by the data array's dtype.", route=real_module_for, route_kw="data")
Grid_3D = _real_templated_marker("Grid_3D",
    "A 3D regular grid of scalar values. isinstance spans both Real backends; construction\n"
    "routes by the data array's dtype.", route=real_module_for, route_kw="data")
Grid_4D = _real_templated_marker("Grid_4D",
    "A 4D regular grid of scalar values. isinstance spans both Real backends; construction\n"
    "routes by the data array's dtype.", route=real_module_for, route_kw="data")

# Kernel / image / cokernel reductions. Real-templated AND per-cell-type: the concrete class
# depends on the filtration encoding, so isinstance must span EVERY KerImCokReduced encoding
# across both backends (KerImCokReduced / KerImCokReducedProd / the internal _KerImCokReduced_*),
# so isinstance(compute_kernel_image_cokernel_reduction(...), oineus.KerImCokReduced) holds for
# packed/slim/cube results too. The direct ctor (simplex K/L) routes by the filtration's dtype.
KerImCokReduced = _real_templated_marker("KerImCokReduced",
    "Kernel / image / cokernel persistence of a pair (K, L). isinstance(x, oineus.KerImCokReduced)\n"
    "is True for the result of compute_kernel_image_cokernel_reduction on any cell encoding, in\n"
    "either Real backend. The direct constructor KerImCokReduced(K, L, params) (simplicial K, L)\n"
    "routes by the filtration's dtype.",
    isinstance_names=lambda n: "KerImCokReduced" in n, route=module_of_oineus_obj, route_kw="K")
KerImCokReducedProd = _real_templated_marker("KerImCokReducedProd",
    "Kernel / image / cokernel persistence over product cells (mapping cylinders). isinstance\n"
    "spans both Real backends; the direct constructor routes by the filtration's dtype.",
    route=module_of_oineus_obj, route_kw="K")

# Optimizer targets (IndicesValues). Real-templated AND per-cell-type, and return-only (no public
# constructor). isinstance must span every IndicesValues encoding across both backends so that the
# result of an optimizer over a packed/slim/cube filtration is recognized as an oineus.IndicesValues.
IndicesValues = _real_templated_marker("IndicesValues",
    "(simplex indices, target values) returned by TopologyOptimizer methods. isinstance(x,\n"
    "oineus.IndicesValues) is True for an optimizer result over any cell encoding, in either Real\n"
    "backend. It is a result type, not user-constructed.",
    isinstance_names=lambda n: n.startswith("IndicesValues"))
IndicesValuesProd = _real_templated_marker("IndicesValuesProd",
    "Optimizer targets for a product-cell filtration. isinstance spans both Real backends; it is\n"
    "a result type, not user-constructed.")

# Concrete per-cell-type optimizers. The generic oineus.TopologyOptimizer facade already
# dispatches by filtration type; these concrete names route construction by the filtration's dtype
# and span both Real backends for isinstance.
TopologyOptimizerProd = _real_templated_marker("TopologyOptimizerProd",
    "TopologyOptimizer for a product-cell filtration. Prefer the generic oineus.TopologyOptimizer\n"
    "facade; this direct constructor routes by the filtration's dtype.", route=module_of_oineus_obj, route_kw="fil")
TopologyOptimizerCube_1D = _real_templated_marker("TopologyOptimizerCube_1D",
    "TopologyOptimizer for a 1D cubical filtration; direct ctor routes by the filtration's dtype.",
    route=module_of_oineus_obj, route_kw="fil")
TopologyOptimizerCube_2D = _real_templated_marker("TopologyOptimizerCube_2D",
    "TopologyOptimizer for a 2D cubical filtration; direct ctor routes by the filtration's dtype.",
    route=module_of_oineus_obj, route_kw="fil")
TopologyOptimizerCube_3D = _real_templated_marker("TopologyOptimizerCube_3D",
    "TopologyOptimizer for a 3D cubical filtration; direct ctor routes by the filtration's dtype.",
    route=module_of_oineus_obj, route_kw="fil")
TopologyOptimizerCube_4D = _real_templated_marker("TopologyOptimizerCube_4D",
    "TopologyOptimizer for a 4D cubical filtration; direct ctor routes by the filtration's dtype.",
    route=module_of_oineus_obj, route_kw="fil")


def _apply_reduction_kwargs(params, kwargs):
    """Return ReductionParams combining params with keyword overrides.

    With no kwargs, params is returned as-is (or a default instance if None).
    Otherwise the overrides are set on a copy, so the caller's params object
    is never mutated. Fields of params.advanced (chunk_size, col_repr,
    dims_to_restore_elz) are accepted flat and routed there. An unknown
    field name raises TypeError.
    """
    if not kwargs:
        return ReductionParams() if params is None else params
    params = ReductionParams() if params is None else copy.copy(params)
    for name, value in kwargs.items():
        try:
            setattr(params, name, value)
            continue
        except AttributeError:
            pass
        try:
            setattr(params.advanced, name, value)
        except AttributeError:
            raise TypeError(f"reduce() got an unexpected keyword argument {name!r}"
                            " (not a ReductionParams or ReductionParams.advanced field)") from None
    return params


def reduce(filtration, params=None, dualize=False, **kwargs):
    """Reduce a filtration in one fused build+reduce step, returning a Decomposition.

    Routes to the backend (float64 / float32) matching the filtration's dtype, so it accepts the
    now-default packed/slim and float32 filtrations. Equivalent to
    ``d = oineus.Decomposition(filtration); d.reduce(params)`` but uses the fused fast path.

    ReductionParams fields can be given directly as keyword arguments, e.g.
    ``oineus.reduce(fil, n_threads=8, compute_v=True)``; they are set on a copy
    of params (or of a default ReductionParams), so the caller's params object
    is never mutated. An unknown field name raises TypeError.
    """
    params = _apply_reduction_kwargs(params, kwargs)
    return module_of_oineus_obj(filtration).reduce(filtration, params, dualize)


# pythonic kwargs layer over the C++ Decomposition.reduce(params) method:
# dcmp.reduce(n_threads=1, compute_u=True) works like the free reduce above
_decomposition_reduce_cpp = Decomposition.reduce


def _decomposition_reduce(self, params=None, **kwargs):
    """Reduce this decomposition with the given ReductionParams.

    ReductionParams fields can be given directly as keyword arguments, e.g.
    ``dcmp.reduce(n_threads=1, compute_u=True)``; they are set on a copy of
    params (or of a default ReductionParams), so the caller's params object is
    never mutated. An unknown field name raises TypeError.
    """
    return _decomposition_reduce_cpp(self, _apply_reduction_kwargs(params, kwargs))


Decomposition.reduce = _decomposition_reduce


# Visualization helpers require matplotlib, an optional extra
# (`pip install oineus[vis]`). When it is absent, the plot_* helpers and
# style constants are simply unavailable; the rest of oineus works normally.
try:
    from .vis import (
        plot_diagram,
        plot_diagram_gradient,
        plot_matching,
        plot_chain,
        default_point_style,
        default_diagram_a_point_style,
        default_diagram_b_point_style,
        default_matching_edge_style,
        default_longest_edge_style,
        default_diagonal_style,
        default_diagonal_projection_a_style,
        default_diagonal_projection_b_style,
        default_inf_line_style,
        default_inf_point_style,
        default_diagram_gradient_style,
        default_density_style,
        default_grid_style,
        default_chain_vertex_style,
        default_chain_edge_style,
        default_chain_triangle_style,
        default_chain_tetrahedron_style,
        default_point_cloud_style,
        DEFAULT_POINT_STYLE,
        DEFAULT_DIAGRAM_A_POINT_STYLE,
        DEFAULT_DIAGRAM_B_POINT_STYLE,
        DEFAULT_MATCHING_EDGE_STYLE,
        DEFAULT_LONGEST_EDGE_STYLE,
        DEFAULT_DIAGONAL_STYLE,
        DEFAULT_DIAGONAL_PROJECTION_A_STYLE,
        DEFAULT_DIAGONAL_PROJECTION_B_STYLE,
        DEFAULT_INF_LINE_STYLE,
        DEFAULT_INF_POINT_STYLE,
        DEFAULT_DIAGRAM_GRADIENT_STYLE,
        DEFAULT_DENSITY_STYLE,
        DEFAULT_DENSITY_THRESHOLD,
        DEFAULT_GRID_STYLE,
        DEFAULT_MATCHING_EDGE_QUANTILE,
        DEFAULT_GRADIENT_TOP_K_ARROWS,
        DEFAULT_CHAIN_VERTEX_STYLE,
        DEFAULT_CHAIN_EDGE_STYLE,
        DEFAULT_CHAIN_TRIANGLE_STYLE,
        DEFAULT_CHAIN_TETRAHEDRON_STYLE,
        DEFAULT_POINT_CLOUD_STYLE,
        OKABE_ITO_BLUE,
        OKABE_ITO_VERMILLION,
    )
    # Keep vis_utils as a backward-compat alias.
    from . import vis_utils  # noqa: F401
except ImportError:
    pass
from .matching import (
    DiagramMatching,
    BottleneckMatching,
    InfKind,
    EssentialMatches,
    EssentialLongestEdges,
    LongestEdges,
    FiniteLongestEdge,
    EssentialLongestEdge,
    point_to_diagonal,
    wasserstein_matching,
    bottleneck_matching,
)
from .sliced_wasserstein import (
    sliced_wasserstein_distance,
    sliced_wasserstein_distance_diag_corrected,
)
from .mixup import (
    MixupBarcodes,
    mixup_barcodes,
    mixup_barcodes_of_filtrations,
)
from .function_delaunay import function_delaunay_bifiltration
# from ._oineus import Z2_Column, Z2_Matrix

try:
    import diode
    _HAS_DIODE = True
except:
    _HAS_DIODE = False

# Newer diode builds add the structured-array exporters (combinatorics/values as
# NumPy arrays instead of one Python tuple per simplex). Probed once at import;
# when False, the code falls back to the list-of-(vertices, value) API.
_HAS_DIODE_ARRAYS = _HAS_DIODE and hasattr(diode, "fill_delaunay_arrays") \
    and hasattr(diode, "fill_alpha_shapes_arrays")
_HAS_DIODE_PERIODIC_LIFTS = _HAS_DIODE \
    and hasattr(diode, "fill_periodic_delaunay_lifts_arrays")


# Maps each filtration cell encoding to its C++ TopologyOptimizer instantiation. The
# reduction core is cell-agnostic, but the optimizer is templated on the cell type, so
# there is one bound class per encoding (universal Simplex, product, slim cube, slim
# Freudenthal, bit-packed VR/alpha). Single source of truth: oineus.diff reuses it.
_OPT_CLASS_BY_FIL_TYPE = _merge_over_reals(lambda s: {
    s._Filtration:               s.TopologyOptimizer,
    s._ProdFiltration:           s.TopologyOptimizerProd,
    s._CubeFiltration_1D:        s.TopologyOptimizerCube_1D,
    s._CubeFiltration_2D:        s.TopologyOptimizerCube_2D,
    s._CubeFiltration_3D:        s.TopologyOptimizerCube_3D,
    s._CubeFiltration_4D:        s.TopologyOptimizerCube_4D,
    s._FreudenthalFiltration_1D: s.TopologyOptimizerFreudenthal_1D,
    s._FreudenthalFiltration_2D: s.TopologyOptimizerFreudenthal_2D,
    s._FreudenthalFiltration_3D: s.TopologyOptimizerFreudenthal_3D,
    s._FreudenthalFiltration_4D: s.TopologyOptimizerFreudenthal_4D,
    s._PackedSimplexFiltration_64:  s.TopologyOptimizerPacked_64,
    s._PackedSimplexFiltration_128: s.TopologyOptimizerPacked_128,
})


# Maps each filtration cell encoding to its C++ KerImCokReduced (kernel/image/cokernel)
# instantiation. kernel.h is cell-agnostic, so KICR is wired for every encoding; the two
# fat classes keep their public names, the rest are hidden underscore names. Keyed by the
# filtration type (type(K)) -- NOT by K[0], whose materialized fat cell would misdispatch
# slim/packed filtrations into the fat ctor.
_KICR_CLASS_BY_FIL_TYPE = _merge_over_reals(lambda s: {
    s._Filtration:               s.KerImCokReduced,
    s._ProdFiltration:           s.KerImCokReducedProd,
    s._CubeFiltration_1D:        s._KerImCokReduced_Cube_1D,
    s._CubeFiltration_2D:        s._KerImCokReduced_Cube_2D,
    s._CubeFiltration_3D:        s._KerImCokReduced_Cube_3D,
    s._CubeFiltration_4D:        s._KerImCokReduced_Cube_4D,
    s._FreudenthalFiltration_1D: s._KerImCokReduced_Fr_1D,
    s._FreudenthalFiltration_2D: s._KerImCokReduced_Fr_2D,
    s._FreudenthalFiltration_3D: s._KerImCokReduced_Fr_3D,
    s._FreudenthalFiltration_4D: s._KerImCokReduced_Fr_4D,
    s._PackedSimplexFiltration_64:  s._KerImCokReduced_Packed_64,
    s._PackedSimplexFiltration_128: s._KerImCokReduced_Packed_128,
})


class TopologyOptimizer:
    """Topology optimizer for a filtration of any cell encoding.

    Dispatches on the filtration's cell type -- universal Simplex (VR / alpha / user),
    product, slim cube, slim Freudenthal, or bit-packed VR/alpha -- and returns the
    matching C++ optimizer instance directly, so its full native API (reduce_all,
    compute_diagram, simplify, match, singletons, combine_loss, ...) is available
    unchanged. Constructor keywords (with_crit_sets, dims_to_restore_elz, n_threads,
    u_strategy) are forwarded verbatim.

    oineus.diff.TopologyOptimizer is the differentiable-pipeline wrapper built on the
    same dispatch; use this bare class for direct (non-autograd) topology optimization.
    """

    def __new__(cls, fil, *args, **kwargs):
        opt_cls = _OPT_CLASS_BY_FIL_TYPE.get(type(fil))
        if opt_cls is None:
            raise TypeError(
                f"TopologyOptimizer: unsupported filtration type "
                f"{type(fil).__name__}; expected one of "
                f"{[t.__name__ for t in _OPT_CLASS_BY_FIL_TYPE]}"
            )
        return opt_cls(fil, *args, **kwargs)


def _check_numpy_diagram_shape(dgm):
    """If ``dgm`` is a numpy array, assert shape (n, 2). Otherwise pass through."""
    if isinstance(dgm, np.ndarray) and (dgm.ndim != 2 or dgm.shape[1] != 2):
        raise ValueError("Expected NumPy array with shape (n_points, 2)")
    return dgm


def _normalize_frechet_weights(n_diagrams: int, weights):
    if n_diagrams == 0:
        return np.empty((0,), dtype=REAL_DTYPE)

    if weights is None:
        return np.full(n_diagrams, 1.0 / n_diagrams, dtype=REAL_DTYPE)

    arr = np.asarray(weights)
    if arr.ndim != 1:
        raise ValueError("weights must be a 1D array")
    if arr.shape[0] != n_diagrams:
        raise ValueError("weights must have same length as diagrams")
    if np.any(arr < 0.0):
        raise ValueError("weights must be nonnegative")

    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")

    return arr / total


def _diagram_persistences(dgm: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return np.empty((0,), dtype=dgm.dtype)

    pers = np.empty(dgm.shape[0], dtype=dgm.dtype)
    finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    pers[finite_mask] = np.abs(dgm[finite_mask, 1] - dgm[finite_mask, 0])
    pers[~finite_mask] = np.inf
    return pers


def _threshold_diagram_by_persistence(dgm: np.ndarray, min_persistence: float, *, include_infinite_points: bool = True):
    if dgm.size == 0:
        return dgm.reshape((0, 2))

    pers = _diagram_persistences(dgm)
    finite_mask = np.isfinite(pers)
    keep_mask = np.zeros(dgm.shape[0], dtype=bool)
    keep_mask[finite_mask] = pers[finite_mask] >= min_persistence
    if include_infinite_points:
        keep_mask |= ~finite_mask
    return np.ascontiguousarray(dgm[keep_mask])


def _newly_active_diagram_points(dgm: np.ndarray, previous_threshold: float, current_threshold: float):
    if dgm.size == 0:
        return dgm.reshape((0, 2))

    pers = _diagram_persistences(dgm)
    finite_mask = np.isfinite(pers)
    keep_mask = finite_mask & (pers >= current_threshold) & (pers < previous_threshold)
    return np.ascontiguousarray(dgm[keep_mask])


def _diagrams_to_numpy_list(diagrams):
    """Convert each diagram in the sequence to a Real-dtype ``(n, 2)`` numpy array.

    Used by helpers that manipulate diagram points in Python (thresholding,
    persistence scheduling, pairwise distances, ...). For direct pass-through
    to the C++ Hera bindings, prefer :func:`as_real_numpy` instead — nanobind
    picks the correct overload based on input type.
    """
    result = []
    for dgm in diagrams:
        if isinstance(dgm, np.ndarray):
            result.append(as_real_numpy(_check_numpy_diagram_shape(dgm)))
        elif hasattr(dgm, "in_dimension"):  # oineus.Diagrams
            if len(dgm) != 1:
                raise ValueError(
                    "Cannot convert multi-dimensional oineus.Diagrams: specify dim=... "
                    "or extract .in_dimension(d) before calling"
                )
            result.append(dgm.in_dimension(0, as_numpy=True))
        else:  # list of DiagramPoint or similar
            if len(dgm) == 0:
                result.append(np.empty((0, 2), dtype=REAL_DTYPE))
            else:
                result.append(
                    np.array([[p[0], p[1]] for p in dgm], dtype=REAL_DTYPE).reshape((-1, 2))
                )
    return result


def _resolve_multistart_seed(diagrams,
                             seed,
                             *,
                             weights,
                             domain,
                             random_noise_scale,
                             random_seed,
                             grid_n_x_bins,
                             grid_n_y_bins,
                             wasserstein_delta,
                             internal_p,
                             n_threads: int = 1):
    if not isinstance(seed, str):
        return as_real_numpy(_check_numpy_diagram_shape(seed))

    if seed == "first":
        return init_frechet_mean_first_diagram(diagrams)
    if seed == "medoid":
        return init_frechet_mean_medoid_diagram(diagrams, weights=weights, n_threads=n_threads)
    if seed == "grid":
        return init_frechet_mean_diagonal_grid(
            diagrams,
            weights=weights,
            domain=domain,
            grid_n_x_bins=grid_n_x_bins,
            grid_n_y_bins=grid_n_y_bins,
        )
    if seed == "random":
        return init_frechet_mean_random_diagram(
            diagrams,
            domain=domain,
            random_noise_scale=random_noise_scale,
            random_seed=random_seed,
        )

    # Need numpy arrays for .copy() in the "farthest_from_medoid"/"second_medoid" paths.
    diagrams = _diagrams_to_numpy_list(diagrams)
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    n = len(diagrams)
    d2 = np.zeros((n, n), dtype=REAL_DTYPE)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def pair_dist(ij):
        i, j = ij
        return wasserstein_distance(
            diagrams[i], diagrams[j],
            q=2.0, delta=wasserstein_delta, internal_p=internal_p,
        )

    if n_threads <= 1 or len(pairs) <= 1:
        dists = [pair_dist(p) for p in pairs]
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            dists = list(executor.map(pair_dist, pairs))
    for (i, j), dist in zip(pairs, dists):
        d2[i, j] = dist * dist
        d2[j, i] = d2[i, j]

    medoid_idx = int(np.argmin(d2 @ normalized_weights))

    if seed == "farthest_from_medoid":
        return diagrams[int(np.argmax(d2[:, medoid_idx]))].copy()

    if seed == "second_medoid":
        provisional = int(np.argmax(d2[:, medoid_idx]))
        c2 = np.where(d2[:, provisional] < d2[:, medoid_idx])[0]
        if c2.size == 0:
            return diagrams[provisional].copy()
        restricted_scores = np.array([
            np.sum(normalized_weights[c2] * d2[idx, c2])
            for idx in c2
        ])
        return diagrams[int(c2[int(np.argmin(restricted_scores))])].copy()

    raise ValueError(f"Unknown Fréchet-mean seed '{seed}'")


def _prepare_distance_args(dgm_1, dgm_2):
    """Coerce two single-dimension diagram inputs for the C++ distance entry.

    Both inputs must be a numpy ``(n, 2)`` array or a ``list[DiagramPoint]``.
    A multi-dimensional ``oineus.Diagrams`` is rejected — the caller must
    extract a single dimension first via ``dgm.in_dimension(d)``.
    """
    def coerce(dgm):
        if isinstance(dgm, Diagrams):
            raise TypeError(
                "Pass a single-dimension diagram: use `dgm.in_dimension(d)` "
                "instead of passing the multi-dimensional Diagrams object.")
        return as_real_numpy(_check_numpy_diagram_shape(dgm))
    return (coerce(dgm_1), coerce(dgm_2))


def bottleneck_distance(dgm_1, dgm_2, delta: float=0.01):
    """Compute the bottleneck distance between two persistence diagrams.

    Args:
        dgm_1: Single-dimension persistence diagram: a NumPy array of shape
            ``(n_points, 2)`` or a ``list[DiagramPoint]``. To pass an Oineus
            ``Diagrams`` object, extract the dimension first via
            ``dgm.in_dimension(d)``.
        dgm_2: Same forms as ``dgm_1``.
        delta: Relative error requested from Hera. Set `delta=0.0` to request
            the exact bottleneck distance.

    Returns:
        The bottleneck distance as a Python float.
    """
    return _bottleneck_distance_cpp(*_prepare_distance_args(dgm_1, dgm_2), delta=delta)


def _diagram_arrays_equal_for_zero_check(dgm_1, dgm_2):
    if isinstance(dgm_1, np.ndarray):
        arr_1 = dgm_1
    else:
        arr_1 = np.array([[p[0], p[1]] for p in dgm_1], dtype=REAL_DTYPE).reshape((-1, 2))

    if isinstance(dgm_2, np.ndarray):
        arr_2 = dgm_2
    else:
        arr_2 = np.array([[p[0], p[1]] for p in dgm_2], dtype=REAL_DTYPE).reshape((-1, 2))

    if arr_1.shape != arr_2.shape:
        return False

    if arr_1.size == 0:
        return True

    sort_idx_1 = np.lexsort((arr_1[:, 1], arr_1[:, 0]))
    sort_idx_2 = np.lexsort((arr_2[:, 1], arr_2[:, 0]))
    arr_1 = arr_1[sort_idx_1]
    arr_2 = arr_2[sort_idx_2]

    finite_mask = np.isfinite(arr_1) & np.isfinite(arr_2)
    matching_inf_mask = np.isinf(arr_1) & np.isinf(arr_2) & (np.signbit(arr_1) == np.signbit(arr_2))

    if not np.all(finite_mask | matching_inf_mask):
        return False

    diff = np.zeros_like(arr_1)
    diff[finite_mask] = np.abs(arr_1[finite_mask] - arr_2[finite_mask])

    return np.all(diff < np.finfo(arr_1.dtype).eps)


def wasserstein_distance(dgm_1, dgm_2, q: float=1.0, delta: float=0.01, internal_p: float=np.inf,
                         wasserstein_q: typing.Optional[float]=None,
                         check_for_zero: bool=True):
    """Compute the q-Wasserstein distance between two persistence diagrams.

    Args:
        dgm_1: Single-dimension persistence diagram: a NumPy array of shape
            ``(n_points, 2)`` or a ``list[DiagramPoint]``. To pass an Oineus
            ``Diagrams`` object, extract the dimension first via
            ``dgm.in_dimension(d)``.
        dgm_2: Same forms as ``dgm_1``.
        q: Wasserstein exponent.
        delta: Relative error requested from Hera.
        internal_p: Ground-metric norm in the plane. Use `np.inf` for the
            `L_infinity` norm.
        wasserstein_q: Alias for `q`, kept for API compatibility.
        check_for_zero: If `True`, skip Hera when the two inputs are numpy
            arrays of equal points.

    Returns:
        The Wasserstein distance as a Python float.
    """
    if wasserstein_q is not None:
        q = wasserstein_q
    if np.isinf(internal_p):
        internal_p = -1.0

    prepared = _prepare_distance_args(dgm_1, dgm_2)

    if check_for_zero and _diagram_arrays_equal_for_zero_check(*prepared):
        return 0.0

    return _wasserstein_distance_cpp(*prepared, q=q, delta=delta, internal_p=internal_p)


def init_frechet_mean_first_diagram(diagrams):
    return _init_frechet_mean_first_diagram_cpp([as_real_numpy(d) for d in diagrams])


def init_frechet_mean_random_diagram(diagrams,
                                     *,
                                     domain=DiagramPlaneDomain.AboveDiagonal,
                                     random_noise_scale: float = 1.0,
                                     random_seed: int = 42):
    return _init_frechet_mean_random_diagram_cpp(
        [as_real_numpy(d) for d in diagrams],
        domain=domain,
        random_noise_scale=random_noise_scale,
        random_seed=random_seed,
    )


def init_frechet_mean_medoid_diagram(diagrams, *, weights=None, n_threads: int = 1):
    return _init_frechet_mean_medoid_diagram_cpp(
        [as_real_numpy(d) for d in diagrams], weights=weights, n_threads=n_threads
    )


def init_frechet_mean_diagonal_grid(diagrams,
                                    *,
                                    weights=None,
                                    domain=DiagramPlaneDomain.AboveDiagonal,
                                    grid_n_x_bins: int = 16,
                                    grid_n_y_bins: int = 16):
    return _init_frechet_mean_diagonal_grid_cpp(
        [as_real_numpy(d) for d in diagrams],
        weights=weights,
        domain=domain,
        grid_n_x_bins=grid_n_x_bins,
        grid_n_y_bins=grid_n_y_bins,
    )


def frechet_mean_objective(diagrams,
                           barycenter,
                           *,
                           weights=None,
                           wasserstein_delta: float = 0.01,
                           internal_p: float = np.inf,
                           n_threads: int = 1):
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)

    def term(i_diagram):
        i, diagram = i_diagram
        return normalized_weights[i] * wasserstein_distance(
            barycenter,
            diagram,
            q=2.0,
            delta=wasserstein_delta,
            internal_p=internal_p,
        ) ** 2

    if n_threads <= 1 or len(diagrams) <= 1:
        return float(sum(term((i, d)) for i, d in enumerate(diagrams)))

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        terms = list(executor.map(term, enumerate(diagrams)))
    return float(sum(terms))


def make_frechet_mean_persistence_schedule(diagrams,
                                           *,
                                           initial_threshold_fraction: float = 0.5,
                                           max_active_growth: float = 0.10,
                                           min_persistence: float = 0.0):
    diagrams = _diagrams_to_numpy_list(diagrams)
    if initial_threshold_fraction <= 0.0:
        raise ValueError("initial_threshold_fraction must be positive")
    if max_active_growth < 0.0:
        raise ValueError("max_active_growth must be nonnegative")

    finite_persistences = []
    for dgm in diagrams:
        pers = _diagram_persistences(dgm)
        finite_persistences.extend(pers[np.isfinite(pers)].tolist())

    if not finite_persistences:
        return [float(min_persistence)]

    values = np.array(sorted(set(float(p) for p in finite_persistences if p >= min_persistence), reverse=True))
    if values.size == 0:
        return [float(min_persistence)]

    start_target = max(float(min_persistence), float(initial_threshold_fraction) * float(values[0]))
    start_idx = int(np.where(values >= start_target)[0][-1])
    schedule = [float(values[start_idx])]
    current_idx = start_idx

    counts = np.array([
        sum(int(np.count_nonzero(np.isfinite(_diagram_persistences(dgm)) & (_diagram_persistences(dgm) >= thr))) for dgm in diagrams)
        for thr in values
    ], dtype=np.int64)

    while current_idx + 1 < values.size:
        current_count = max(int(counts[current_idx]), 1)
        max_count = int(np.floor((1.0 + max_active_growth) * current_count))
        next_idx = current_idx + 1
        valid = np.where(counts[current_idx + 1:] <= max_count)[0]
        if valid.size > 0:
            next_idx = current_idx + 1 + int(valid[-1])
        schedule.append(float(values[next_idx]))
        current_idx = next_idx

    if schedule[-1] > float(min_persistence):
        schedule.append(float(min_persistence))

    deduped = []
    for threshold in schedule:
        if not deduped or threshold < deduped[-1]:
            deduped.append(threshold)
    return deduped


def frechet_mean_newborn_points_from_newly_active(newly_active_diagrams, *, weights=None):
    newly_active_diagrams = _diagrams_to_numpy_list(newly_active_diagrams)
    normalized_weights = _normalize_frechet_weights(len(newly_active_diagrams), weights)

    new_points = []
    for diagram_weight, dgm in zip(normalized_weights, newly_active_diagrams):
        if dgm.size == 0:
            continue
        finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
        finite_points = dgm[finite_mask]
        if finite_points.size == 0:
            continue
        midpoints = 0.5 * (finite_points[:, 0] + finite_points[:, 1])
        births = diagram_weight * finite_points[:, 0] + (1.0 - diagram_weight) * midpoints
        deaths = diagram_weight * finite_points[:, 1] + (1.0 - diagram_weight) * midpoints
        new_points.append(np.column_stack((births, deaths)))

    if not new_points:
        return np.empty((0, 2), dtype=REAL_DTYPE)

    return np.ascontiguousarray(np.vstack(new_points))


def frechet_mean_multistart(diagrams,
                            *,
                            weights=None,
                            starts=("medoid", "second_medoid", "farthest_from_medoid"),
                            return_details: bool = False,
                            n_threads: int = 1,
                            **kwargs):
    diagrams = [as_real_numpy(_check_numpy_diagram_shape(d)) for d in diagrams]
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    if not starts:
        raise ValueError("starts must be non-empty")

    results = []
    for start_idx, start in enumerate(starts):
        if isinstance(start, dict):
            local_kwargs = dict(kwargs)
            local_kwargs.update(start)
            local_kwargs.pop("init_strategy", None)
            local_kwargs.pop("custom_initial_barycenter", None)
            seed = _resolve_multistart_seed(
                diagrams,
                local_kwargs.pop("seed", "medoid"),
                weights=normalized_weights,
                domain=local_kwargs.get("domain", DiagramPlaneDomain.AboveDiagonal),
                random_noise_scale=local_kwargs.get("random_noise_scale", 1.0),
                random_seed=local_kwargs.get("random_seed", 42 + start_idx),
                grid_n_x_bins=local_kwargs.get("grid_n_x_bins", 16),
                grid_n_y_bins=local_kwargs.get("grid_n_y_bins", 16),
                wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
                internal_p=local_kwargs.get("internal_p", np.inf),
                n_threads=n_threads,
            )
        else:
            local_kwargs = dict(kwargs)
            local_kwargs.pop("init_strategy", None)
            local_kwargs.pop("custom_initial_barycenter", None)
            seed = _resolve_multistart_seed(
                diagrams,
                start,
                weights=normalized_weights,
                domain=local_kwargs.get("domain", DiagramPlaneDomain.AboveDiagonal),
                random_noise_scale=local_kwargs.get("random_noise_scale", 1.0),
                random_seed=local_kwargs.get("random_seed", 42 + start_idx),
                grid_n_x_bins=local_kwargs.get("grid_n_x_bins", 16),
                grid_n_y_bins=local_kwargs.get("grid_n_y_bins", 16),
                wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
                internal_p=local_kwargs.get("internal_p", np.inf),
                n_threads=n_threads,
            )

        barycenter = frechet_mean(
            diagrams,
            weights=normalized_weights,
            init_strategy=FrechetMeanInit.Custom,
            custom_initial_barycenter=seed,
            n_threads=n_threads,
            **local_kwargs,
        )
        objective = frechet_mean_objective(
            diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
            n_threads=n_threads,
        )
        results.append({"start": start, "barycenter": barycenter, "objective": objective})

    best = min(results, key=lambda item: item["objective"])
    if return_details:
        return best["barycenter"], {"objective": best["objective"], "runs": results}
    return best["barycenter"]


def progressive_frechet_mean(diagrams,
                             *,
                             weights=None,
                             thresholds=None,
                             initial_threshold_fraction: float = 0.5,
                             max_active_growth: float = 0.10,
                             min_persistence: float = 0.0,
                             initial_seed="medoid",
                             support_update_predicate=None,
                             support_update_fn=None,
                             return_details: bool = False,
                             n_threads: int = 1,
                             **kwargs):
    diagrams = _diagrams_to_numpy_list(diagrams)
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    ignore_infinite_points = bool(kwargs.get("ignore_infinite_points", False))

    if thresholds is None:
        thresholds = make_frechet_mean_persistence_schedule(
            diagrams,
            initial_threshold_fraction=initial_threshold_fraction,
            max_active_growth=max_active_growth,
            min_persistence=min_persistence,
        )
    else:
        thresholds = [float(t) for t in thresholds]
        if not thresholds:
            raise ValueError("thresholds must be non-empty")

    local_kwargs = dict(kwargs)
    local_kwargs.pop("init_strategy", None)
    local_kwargs.pop("custom_initial_barycenter", None)

    barycenter = None
    history = []
    previous_threshold = np.inf

    for stage_idx, threshold in enumerate(thresholds):
        active_diagrams = [
            _threshold_diagram_by_persistence(
                dgm,
                threshold,
                include_infinite_points=not ignore_infinite_points,
            )
            for dgm in diagrams
        ]

        if barycenter is None:
            seed = _resolve_multistart_seed(
                active_diagrams,
                initial_seed,
                weights=normalized_weights,
                domain=local_kwargs.get("domain", DiagramPlaneDomain.AboveDiagonal),
                random_noise_scale=local_kwargs.get("random_noise_scale", 1.0),
                random_seed=local_kwargs.get("random_seed", 42),
                grid_n_x_bins=local_kwargs.get("grid_n_x_bins", 16),
                grid_n_y_bins=local_kwargs.get("grid_n_y_bins", 16),
                wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
                internal_p=local_kwargs.get("internal_p", np.inf),
                n_threads=n_threads,
            )
        else:
            seed = barycenter
            if support_update_predicate is not None and support_update_fn is not None:
                newly_active_diagrams = [
                    _newly_active_diagram_points(dgm, previous_threshold, threshold)
                    for dgm in diagrams
                ]
                should_update = bool(support_update_predicate(
                    stage_index=stage_idx,
                    threshold=threshold,
                    previous_threshold=previous_threshold,
                    current_barycenter=seed,
                    active_diagrams=active_diagrams,
                    newly_active_diagrams=newly_active_diagrams,
                    weights=normalized_weights,
                ))
                if should_update:
                    new_points = support_update_fn(
                        stage_index=stage_idx,
                        threshold=threshold,
                        previous_threshold=previous_threshold,
                        current_barycenter=seed,
                        active_diagrams=active_diagrams,
                        newly_active_diagrams=newly_active_diagrams,
                        weights=normalized_weights,
                    )
                    if new_points is not None:
                        new_points = as_real_numpy(_check_numpy_diagram_shape(new_points))
                        if new_points.size != 0:
                            seed = np.ascontiguousarray(np.vstack([seed, new_points]))

        barycenter = frechet_mean(
            active_diagrams,
            weights=normalized_weights,
            init_strategy=FrechetMeanInit.Custom,
            custom_initial_barycenter=seed,
            n_threads=n_threads,
            **local_kwargs,
        )

        objective = frechet_mean_objective(
            active_diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
            n_threads=n_threads,
        )
        history.append({
            "stage_index": stage_idx,
            "threshold": threshold,
            "n_active_points": int(sum(dgm.shape[0] for dgm in active_diagrams)),
            "barycenter": barycenter,
            "objective": objective,
        })
        previous_threshold = threshold

    if return_details:
        return barycenter, {"thresholds": thresholds, "history": history}
    return barycenter


def progressive_frechet_mean_multistart(diagrams,
                                        *,
                                        weights=None,
                                        starts=("medoid", "second_medoid", "farthest_from_medoid"),
                                        return_details: bool = False,
                                        n_threads: int = 1,
                                        **kwargs):
    diagrams = [as_real_numpy(_check_numpy_diagram_shape(d)) for d in diagrams]
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    if not starts:
        raise ValueError("starts must be non-empty")

    results = []
    for start_idx, start in enumerate(starts):
        local_kwargs = dict(kwargs)
        initial_seed = local_kwargs.pop("initial_seed", start)

        if isinstance(start, dict):
            local_kwargs.update(start)
            initial_seed = local_kwargs.pop("initial_seed", local_kwargs.pop("seed", "medoid"))

        barycenter, details = progressive_frechet_mean(
            diagrams,
            weights=normalized_weights,
            initial_seed=initial_seed,
            return_details=True,
            n_threads=n_threads,
            **local_kwargs,
        )
        objective = frechet_mean_objective(
            diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
            n_threads=n_threads,
        )
        results.append({
            "start": start,
            "barycenter": barycenter,
            "objective": objective,
            "progressive_details": details,
        })

    best = min(results, key=lambda item: item["objective"])
    if return_details:
        return best["barycenter"], {
            "objective": best["objective"],
            "runs": results,
            "thresholds": best["progressive_details"]["thresholds"],
            "history": best["progressive_details"]["history"],
        }
    return best["barycenter"]


def frechet_mean(diagrams,
                 *,
                 weights=None,
                 max_iter: int = 100,
                 tol: float = 1e-7,
                 wasserstein_delta: float = 0.01,
                 internal_p: float = np.inf,
                 init_strategy=FrechetMeanInit.Grid,
                 domain=DiagramPlaneDomain.AboveDiagonal,
                 ignore_infinite_points: bool = False,
                 random_noise_scale: float = 1.0,
                 random_seed: int = 42,
                 grid_n_x_bins: int = 16,
                 grid_n_y_bins: int = 16,
                 custom_initial_barycenter=None,
                 n_threads: int = 1):
    diagrams = [as_real_numpy(_check_numpy_diagram_shape(d)) for d in diagrams]
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if weights.shape[0] != len(diagrams):
            raise ValueError("weights must have same length as diagrams")
    custom_initial_barycenter = (
        None if custom_initial_barycenter is None
        else as_real_numpy(_check_numpy_diagram_shape(custom_initial_barycenter))
    )

    if np.isinf(internal_p):
        internal_p = -1.0

    return _frechet_mean_cpp(diagrams,
                             weights=weights,
                             max_iter=max_iter,
                             tol=tol,
                             wasserstein_delta=wasserstein_delta,
                             internal_p=internal_p,
                             init_strategy=init_strategy,
                             domain=domain,
                             ignore_infinite_points=ignore_infinite_points,
                             random_noise_scale=random_noise_scale,
                             random_seed=random_seed,
                             grid_n_x_bins=grid_n_x_bins,
                             grid_n_y_bins=grid_n_y_bins,
                             custom_initial_barycenter=custom_initial_barycenter,
                             n_threads=n_threads)


def to_scipy_matrix(sparse_cols, shape=None):
    import scipy.sparse  # local: keep scipy off the `import oineus` path
    if shape is None:
        shape = (len(sparse_cols), len(sparse_cols))
    row_ind = [j for i in range(len(sparse_cols)) for j in sparse_cols[i]]
    col_ind = [i for i in range(len(sparse_cols)) for _ in sparse_cols[i]]
    assert (len(row_ind) == len(col_ind))
    data = [1 for _ in range(len(row_ind))]
    return scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)


def max_distance(data: np.ndarray, from_pwdists: bool=False):
    """Enclosing radius of a point cloud, for use as a Vietoris-Rips cutoff.

    Returns min_i max_j ||x_i - x_j|| (scaled by 1.00001 so it sits strictly
    above the true value after rounding) -- the smallest radius from which some
    single point sees every other. This is the standard Vietoris-Rips threshold:
    beyond it the complex is a cone and carries no more topology, as in Ripser.
    It is NOT the diameter max_i max_j ||x_i - x_j||; for three collinear points
    at 0, 1, 2 it returns 1, not 2. Pass it to vr_filtration as max_diameter.

    Args:
        data: an (n, d) array of n points, or -- when from_pwdists is True -- an
            (n, n) matrix of pairwise distances.
        from_pwdists: if True, read the enclosing radius directly off a
            pairwise-distance matrix instead of a point cloud.

    Raises:
        ValueError: if data is not a 2D array with at least two rows, contains
            non-finite values, or has a coordinate spread that overflows float64
            (rescale first).
    """
    if from_pwdists:
        return 1.00001 * np.min(np.max(data, axis=1))
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("max_distance: data must be a 2D array with at least 2 rows")
    x = np.asarray(data, dtype=np.float64)
    # A non-finite coordinate makes the radius meaningless; reject it rather than
    # let it propagate to inf/nan downstream.
    if not np.all(np.isfinite(x)):
        raise ValueError("max_distance: data contains non-finite values (nan/inf)")
    x = np.ascontiguousarray(x)
    n, d = x.shape
    # Direct pairwise differences, NOT the ||x||^2 + ||y||^2 - 2<x,y> Gram identity:
    # the Gram form overflows and emits spurious "matmul" RuntimeWarnings on large
    # or extreme-magnitude data, and loses a constant cloud to catastrophic
    # cancellation (its true radius is 0). Differences are exact for identical
    # points, so a constant cloud correctly gives 0. Chunk the rows so the
    # (chunk, n, d) block stays bounded (~32 MB) instead of the full (n, n, d)
    # temporary a single broadcast would build.
    chunk = max(1, (2 ** 22) // (n * max(1, d)))
    min_of_max = np.inf
    for beg in range(0, n, chunk):
        diffs = x[beg:beg + chunk, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.einsum('ijk,ijk->ij', diffs, diffs))
        min_of_max = min(min_of_max, dists.max(axis=1).min())
    result = 1.00001 * min_of_max
    # a constant cloud gives 0 (finite); only genuinely extreme coordinate spread
    # (> ~1.3e154 in a dimension) overflows diffs**2 to inf -- fail loud rather than
    # hand back an infinite max_diameter
    if not np.isfinite(result):
        raise ValueError("max_distance: coordinate spread overflows float64; rescale the data")
    return result


def freudenthal_filtration(data: np.ndarray,
                           negate: bool=False,
                           wrap: bool=False,
                           max_dim: int = 3,
                           with_critical_vertices: bool=False,
                           *,
                           slim: bool=True,
                           n_threads: int=1):
    # route to the float32 or float64 backend by the input dtype (float32 numpy/torch
    # arrays build a genuine float32 filtration; everything else defaults to float64)
    dt = detect_real_dtype(data)
    sub = REAL_MODULES[dt]
    data = as_real_numpy(data, dtype=dt)
    max_dim = min(max_dim, data.ndim)
    # slim (the default) returns the compact (anchor,type) _FreudenthalFiltration_ND (one shared
    # FrGeometry, fat simplices materialized on access) for D=1,2,3,4 on non-wrap grids; it reduces,
    # produces diagrams, optimizes (oineus.TopologyOptimizer dispatches it) and supports KICR
    # identically to the fat path but with a far smaller boundary-build footprint. wrap grids and
    # D>=5 always fall back to the fat universal Filtration (FrGeometry rejects wrap; the slim
    # builder is bound only for D=1,2,3,4). Pass slim=False to force the fat path.
    use_slim = slim and (not wrap) and (1 <= data.ndim <= 4)
    if use_slim:
        grid_cls = {1: sub.Grid_1D, 2: sub.Grid_2D, 3: sub.Grid_3D, 4: sub.Grid_4D}[data.ndim]
        grid = grid_cls(data, wrap=wrap, values_on="vertices")
        if with_critical_vertices:
            fil, vertices = grid.freudenthal_filtration_and_critical_vertices_slim(max_dim=max_dim, negate=negate, n_threads=n_threads)
            vertices = np.array(vertices, dtype=np.int64)
            return fil, vertices
        return grid.freudenthal_filtration_slim(max_dim=max_dim, negate=negate, n_threads=n_threads)
    if with_critical_vertices:
        fil, vertices = sub.get_freudenthal_filtration_and_crit_vertices(data=data, negate=negate, wrap=wrap, max_dim=max_dim, n_threads=n_threads)
        vertices = np.array(vertices, dtype=np.int64)
        return fil, vertices
    return sub.get_freudenthal_filtration(data=data, negate=negate, wrap=wrap, max_dim=max_dim, n_threads=n_threads)


def _vr_packed_word_suffix(n_points, max_dim):
    # Smallest packed word that holds a (max_dim)-simplex over n_points vertices:
    # bits = ceil(log2(n_points)) per field (== C++ packed_vertex_bits), (max_dim+1)
    # fields. Returns "64", "128", or None (too wide -> fat fallback).
    bits = _packed_bits(n_points)
    width = (int(max_dim) + 1) * bits
    if width <= 64:
        return "64"
    if width <= 128:
        return "128"
    return None


def _packed_bits(n_points):
    # bits per packed vertex field == C++ oin::packed_vertex_bits(n_points);
    # passed to the packed array builders so they skip a full max-id scan.
    return max(1, (int(n_points) - 1).bit_length())


def vr_filtration(data: np.ndarray,
                  from_pwdists: bool = False,
                  max_dim: int = -1,
                  max_diameter: float = -1.0,
                  with_critical_edges: bool = False,
                  *,
                  packed: bool = True,
                  n_threads: int = 1):
    """Build a Vietoris-Rips filtration from points or pairwise distances.

    Construction uses the in-order generation (VRE) algorithm of
    Vejdemo-Johansson, Matuszewski & Bauer ("In-order generation of
    Vietoris-Rips Complexes", arXiv:2411.05495).

    Parameters
    ----------
    data : np.ndarray
        (n, d) array of points, or (n, n) pairwise distance matrix.
    from_pwdists : bool
        Treat ``data`` as a pairwise distance matrix.
    max_dim : int
        Largest simplex dimension to enumerate. Default: data dimensionality.
    max_diameter : float
        Truncation threshold; only simplices with diameter <= this value are
        kept. Default: enclosing radius of the point cloud.
    with_critical_edges : bool
        Also return an array (one per simplex) of an edge whose length equals
        the simplex's filtration value.
    packed : bool
        Use the compact bit-packed cell encoding (the default) when the vertex
        ids fit a 64- or 128-bit word; falls back to the fat encoding otherwise.
        Pass packed=False to force the fat universal Simplex filtration.
    n_threads : int
        Threads used for the (parallel) sort inside the Filtration ctor.
        Enumeration itself is single-threaded.
    """
    # route to the float32 or float64 backend by the input dtype, then coerce the
    # point/distance array to that Real so the matching get_vr_* builder accepts it
    dt = detect_real_dtype(data)
    sub = REAL_MODULES[dt]
    data = as_real_numpy(data, dtype=dt)

    if data.ndim != 2:
        raise ValueError("data must be a 2D array")

    if from_pwdists and data.shape[0] != data.shape[1]:
        raise ValueError("from_pwdists=True requires a square pairwise-distance matrix")

    if max_diameter < 0:
        max_diameter = max_distance(data, from_pwdists)

    if max_dim < 0:
        if from_pwdists:
            raise RuntimeError("vr_filtration: if input is pairwise distance matrix, max_dim must be specified")
        else:
            max_dim = data.shape[1]

    # packed (the default) returns a bit-packed _PackedSimplexFiltration_64/128 (compact cells,
    # one shared PackedGeom) when the vertex ids fit a 64- or 128-bit word; if they do not fit
    # (very large/high-dim complex) it transparently falls back to the fat path. It reduces,
    # produces diagrams, optimizes (oineus.TopologyOptimizer dispatches it), supports KICR and
    # the uid-keyed accessors (via the combinatorial-uid translation) identically to fat but with
    # a smaller footprint. Pass packed=False to force the fat universal Simplex filtration.
    suffix = _vr_packed_word_suffix(data.shape[0], max_dim) if packed else None

    if from_pwdists:
        if suffix is not None:
            base = "get_vr_filtration_and_critical_edges_packed" if with_critical_edges else "get_vr_filtration_packed"
            func = getattr(sub, base + suffix + "_from_pwdists")
        elif with_critical_edges:
            func = sub.get_vr_filtration_and_critical_edges_from_pwdists
        else:
            func = sub.get_vr_filtration_from_pwdists
    else:
        if suffix is not None:
            base = "get_vr_filtration_and_critical_edges_packed" if with_critical_edges else "get_vr_filtration_packed"
            func = getattr(sub, base + suffix)
        elif with_critical_edges:
            func = sub.get_vr_filtration_and_critical_edges
        else:
            func = sub.get_vr_filtration

    result = func(data, max_dim=max_dim, max_diameter=max_diameter, n_threads=n_threads)
    if with_critical_edges:
        # convert list of VREdges to numpy array
        edges = [ [ e.x, e.y] for e in result[1] ]
        edges = np.array(edges, dtype=np.int64)
        return result[0], edges
    else:
        return result


def is_reduced(a):
    """Check whether a Z_2 boundary matrix is reduced.

    A column is treated as nonzero in row ``i`` iff ``a[i, col] % 2 == 1``,
    so any integer dtype with mod-2 semantics works (binary 0/1 matrices,
    or unreduced count matrices). Returns ``True`` iff every nonzero
    column has a distinct lowest-1 row index, which is the definition of
    a reduced matrix in the standard persistence reduction.

    Args:
        a: 2D array-like with ``.shape[1]`` columns and integer entries.

    Returns:
        bool: True if the matrix is reduced.
    """
    lowest_ones = []
    for col_idx in range(a.shape[1]):
        if np.any(a[:, col_idx] % 2 == 1):
            lowest_ones.append(np.max(np.where(a[:, col_idx] % 2 == 1)))
    return len(lowest_ones) == len(set(lowest_ones))

_SLIM_SIMPLEX_FIL_TYPES = tuple(
    f for s in REAL_MODULES.values() for f in (
        s._FreudenthalFiltration_1D, s._FreudenthalFiltration_2D,
        s._FreudenthalFiltration_3D, s._FreudenthalFiltration_4D,
        s._PackedSimplexFiltration_64, s._PackedSimplexFiltration_128,
    ))


def _to_fat_simplex_filtration(fil):
    """Materialize a slim/packed simplicial filtration into a fat _Filtration.

    mapping_cylinder / multiply_filtration build fat ProductCell<Simplex, Simplex>
    filtrations, so they need fat Simplex factors. A slim Freudenthal / bit-packed
    filtration materializes fat Simplex cells on access, so rebuild a fat _Filtration
    from them. Fat Simplex / product filtrations pass through unchanged; cube (and
    anything else) is returned as-is so the C++ overload rejects it as before.
    """
    if isinstance(fil, _SLIM_SIMPLEX_FIL_TYPES):
        # fil.cells() materializes fat Simplex cells in fil's own (float32/float64)
        # submodule, so build the fat _Filtration from the matching backend
        return module_of_oineus_obj(fil)._Filtration(fil.cells(), fil.negate)
    return fil


def mapping_cylinder(fil_domain, fil_codomain, v_domain, v_codomain,
                     v_domain_value=None, v_codomain_value=None, with_indices=False):
    """Build the mapping cylinder of the inclusion fil_domain -> fil_codomain.

    The auxiliary vertex values default to filtration-order -inf
    (``fil_domain.neg_infinity()`` / ``fil_codomain.neg_infinity()``), which
    keeps the cylinder's persistent homology equivalent to the inclusion's.
    Pass explicit values only if you intentionally want the auxiliary vertices
    to enter at a finite point in the filtration.

    Slim Freudenthal / bit-packed inputs are materialized to the fat encoding
    first (the cylinder is built over fat product cells); the cells keep their
    combinatorial uids, so uid-keyed lookups against the original filtrations
    still resolve.
    """
    fil_domain = _to_fat_simplex_filtration(fil_domain)
    fil_codomain = _to_fat_simplex_filtration(fil_codomain)
    # Accept either valued (oin.Simplex / SimplexValue) or bare (CombinatorialSimplex)
    # auxiliary vertices. Strip the value -- we use the explicit *_value args
    # below, so any value baked into the simplex would only confuse readers.
    # Simplex / ProdSimplex are the cross-backend markers, so a valued cell from a float32
    # filtration is recognized and stripped too (combinatorial_cell is shared / Real-independent).
    if isinstance(v_codomain, Simplex) or isinstance(v_codomain, ProdSimplex):
        v_codomain = v_codomain.combinatorial_cell
    if isinstance(v_domain, Simplex) or isinstance(v_domain, ProdSimplex):
        v_domain = v_domain.combinatorial_cell
    if v_domain_value is None:
        v_domain_value = fil_domain.neg_infinity()
    if v_codomain_value is None:
        v_codomain_value = fil_codomain.neg_infinity()
    # route to the backend matching the (fattened) filtrations' dtype
    sub = module_of_oineus_obj(fil_domain)
    if with_indices:
        return sub._mapping_cylinder_with_indices(fil_domain, fil_codomain, v_domain, v_codomain,
                                                  v_domain_value, v_codomain_value)
    else:
        return sub._mapping_cylinder(fil_domain, fil_codomain, v_domain, v_codomain,
                                     v_domain_value, v_codomain_value)

def multiply_filtration(fil, sigma, sigma_value=None):
    """Multiply every cell in fil by the auxiliary simplex sigma.

    Each product cell receives value ``fil.fil_max(cell.value, sigma_value)``.
    sigma_value defaults to ``fil.neg_infinity()`` so each product cell
    inherits its primary factor's value unchanged.

    A slim Freudenthal / bit-packed fil is materialized to the fat encoding first
    (the product cells are fat).
    """
    fil = _to_fat_simplex_filtration(fil)
    # Simplex is the cross-backend marker, so a valued cell from a float32 filtration is
    # recognized and stripped to its (shared) combinatorial cell too.
    if isinstance(sigma, Simplex):
        sigma = sigma.combinatorial_cell
    if sigma_value is None:
        sigma_value = fil.neg_infinity()
    return module_of_oineus_obj(fil)._multiply_filtration(fil, sigma, sigma_value)

def min_filtration(fil_1, fil_2, with_indices=False):
    # route to the backend matching the input filtrations' dtype
    sub = module_of_oineus_obj(fil_1)
    if with_indices:
        return sub._min_filtration_with_indices(fil_1, fil_2)
    else:
        return sub._min_filtration(fil_1, fil_2)


# The five helpers below take a filtration argument and were historically bound only for the
# fat float64 Simplex. The C++ overloads are now folded over every cell type and registered in
# both Real backends; these wrappers route each call to the (sub)module matching the
# filtration's dtype, so they work on the now-default packed VR / slim Freudenthal / cube
# filtrations and on float32 filtrations -- mirroring min_filtration / mapping_cylinder.

def get_nth_persistence(fil, rv_matrix, dim, n):
    """The n-th largest persistence value in the given homology dimension."""
    return module_of_oineus_obj(fil).get_nth_persistence(fil, rv_matrix, dim, n)


def get_denoise_target(dim, fil, rv_matrix, eps, strategy):
    """Target values for topological denoising (DiagramToValues)."""
    return module_of_oineus_obj(fil).get_denoise_target(dim, fil, rv_matrix, eps, strategy)


def get_permutation(target_matching, fil):
    """Permutation realizing the requested simplex-to-value targets (warm starts)."""
    return module_of_oineus_obj(fil).get_permutation(target_matching, fil)


def get_permutation_dtv(diagram_to_values, fil):
    """Permutation realizing the requested diagram-to-value targets (warm starts)."""
    return module_of_oineus_obj(fil).get_permutation_dtv(diagram_to_values, fil)


def compute_relative_diagrams(fil, rel, include_inf_points=True):
    """Relative persistence diagrams of the pair (fil, rel). fil and rel must share
    the same cell type and Real dtype."""
    return module_of_oineus_obj(fil).compute_relative_diagrams(
        fil, rel, include_inf_points=include_inf_points)


def get_induced_matching(included_filtration, containing_filtration, dim=-1, n_threads=1):
    """Induced matching between the diagrams of two filtrations on the same complex.

    dim < 0 (the default) matches across all homology dimensions."""
    sub = module_of_oineus_obj(included_filtration)
    # dim_type is unsigned in C++; -1 ("all dims") is its SIZE_MAX default, which a Python -1
    # cannot convert to -- so omit the argument and let the C++ default apply.
    if dim is None or dim < 0:
        return sub.get_induced_matching(included_filtration, containing_filtration, n_threads=n_threads)
    return sub.get_induced_matching(
        included_filtration, containing_filtration, dim=dim, n_threads=n_threads)

def remove_simplices(fil, dcmp, seeds, *, close_star=True, stats=None, n_threads=1):
    """SiRUP: remove a coface-closed set of cells from a reduced decomposition.

    Updates ``dcmp`` in place to the reduced R = D V decomposition of the
    filtration with the requested cells removed, updating both the barcode and
    the representative cycles, instead of recomputing from scratch (Giunti and
    Lazovskis, "Pruning vineyards: updating barcodes and representative cycles
    by removing simplices").

    Parameters
    ----------
    fil
        The filtration ``dcmp`` was reduced from.
    dcmp
        A reduced Decomposition with V (reduce with compute_v = True), from the
        classic ``oin.Decomposition(fil); dcmp.reduce(params)`` path. Homology
        only. Mutated in place.
    seeds
        sorted_ids of the cells to remove. By default their coface up-closure
        (union of stars) is taken so that the result is a valid filtration; set
        ``close_star=False`` if ``seeds`` is already coface-closed.
    close_star
        Whether to expand ``seeds`` to ``fil.star_closure(seeds)`` first.
    stats
        Optional DecompositionManipStats collecting column-op counts and timings.
    n_threads
        Threads for the internal row-index / closure build.

    Returns
    -------
    A new filtration on the surviving cells, in the same order as the updated
    decomposition's columns, so ``dcmp.diagram(new_fil)`` gives the updated
    diagram.
    """
    seeds = [int(s) for s in seeds]
    cells = fil.star_closure(seeds, n_threads) if close_star else seeds
    dcmp.remove_simplices(cells, stats, n_threads)
    return fil.without_cells(cells)

def compute_diagrams_ls(data: np.ndarray, negate: bool=False, wrap: bool=False,
                        max_dim: typing.Optional[int]=None, params: typing.Optional[ReductionParams]=None,
                        include_inf_points: bool=True, dualize: bool=False):
    if max_dim is None:
        max_dim = data.ndim - 1
    if params is None:
        params = _oineus.ReductionParams()
    # max_dim is maximal dimension of the _diagram_, we need simplices one dimension higher, hence +1
    fil = freudenthal_filtration(data=data, negate=negate, wrap=wrap, max_dim=max_dim + 1, n_threads=params.n_threads)
    dcmp = _oineus.Decomposition(fil, dualize)
    dcmp.reduce(params)
    return dcmp.diagram(fil=fil, include_inf_points=include_inf_points, n_threads=params.n_threads)


def compute_diagrams_vr(data: np.ndarray, from_pwdists: bool=False, max_dim: int=-1, max_diameter: float = -1.0, params: typing.Optional[ReductionParams]=None, include_inf_points: bool=True, dualize: bool=True):
    if params is None:
        params = _oineus.ReductionParams()
    # max_dim is maximal dimension of the _diagram_, we need simplices one dimension higher, hence +1
    fil = vr_filtration(data, from_pwdists, max_dim=max_dim, max_diameter=max_diameter, with_critical_edges=False, n_threads=params.n_threads)
    dcmp = _oineus.Decomposition(fil, dualize)
    dcmp.reduce(params)
    return dcmp.diagram(fil=fil, include_inf_points=include_inf_points, n_threads=params.n_threads)


def alpha_filtration(points: np.ndarray,
                     weights: typing.Optional[np.ndarray]=None,
                     exact: bool=False,
                     periodic: bool=False,
                     compute_bounding_box: bool=True,
                     bbox_min=None,
                     bbox_max=None,
                     *,
                     packed: bool=True,
                     n_threads: int=1):
    """Build an alpha-shape filtration from a 2D or 3D point cloud.

    Combinatorics come from diode (CGAL Delaunay); filtration values are the
    alpha values returned by diode. For one-shot diagrams use
    :func:`compute_diagrams_alpha`; the differentiable Cech-Delaunay path
    reuses the same combinatorics with autograd-attached values.

    Args:
        points: NumPy array of shape (n, 2) or (n, 3).
        weights: Optional 1D array of length n. If provided, computes
            weighted (regular) alpha-shapes; currently 3D only.
        exact: Use CGAL's exact kernel. Slower but robust against numerical
            pathologies.
        periodic: Use periodic alpha-shapes on a torus.
        compute_bounding_box: If True, use the bounding box of the data.
        bbox_min, bbox_max: Bounding box if compute_bounding_box=False.
        packed: Use the compact bit-packed cell encoding (the default) on the
            fast unweighted/non-periodic array path when the vertex ids fit a
            64/128-bit word. Pass packed=False to force the fat encoding.
        n_threads: Threads used inside the Filtration constructor.

    Returns:
        A Filtration over alpha-shape simplices.
    """
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n_points, dim)")
    if points.shape[1] not in (2, 3):
        raise ValueError("Alpha-shapes only support 2D and 3D point clouds")
    if not _HAS_DIODE:
        raise ImportError(
            "Alpha-shape construction requires the `diode` package "
            "(https://github.com/mrzv/diode). Install it via "
            "`pip install diode` or build from source."
        )

    # Diode wants lists, we accept NumPy array for convenience
    if isinstance(bbox_min, np.ndarray):
        bbox_min = [ float(x) for x in bbox_min ]
    if isinstance(bbox_max, np.ndarray):
        bbox_max = [ float(x) for x in bbox_max ]

    if compute_bounding_box:
        bbox_min = [ float(np.min(points[:, d])) for d in range(points.shape[1]) ]
        bbox_max = [ float(np.max(points[:, d])) for d in range(points.shape[1]) ]
    else:
        if bbox_max is None:
            bbox_max = [ 1.0 for d in range(points.shape[1]) ]
        if  bbox_min is None:
            bbox_min = [ 0.0 for d in range(points.shape[1]) ]

    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if weights.shape[0] != points.shape[0]:
            raise ValueError("weights must have same length as points")
        if points.shape[1] != 3:
            raise ValueError("Weighted alpha-shapes require 3D points")

        weighted_points = np.column_stack((points, weights))

        if periodic:
            if not hasattr(diode, "fill_weighted_periodic_alpha_shapes"):
                raise RuntimeError("diode.fill_weighted_periodic_alpha_shapes is not available in this diode build")
            fil_diode = diode.fill_weighted_periodic_alpha_shapes(weighted_points, exact, bbox_min, bbox_max)
        else:
            fil_diode = diode.fill_weighted_alpha_shapes(weighted_points, exact=exact)
    else:
        if periodic:
            fil_diode = diode.fill_periodic_alpha_shapes(points, exact, bbox_min, bbox_max)
        elif _HAS_DIODE_ARRAYS:
            # Fast array path: simplices and alpha values come back as NumPy
            # arrays, avoiding one Python tuple per simplex. Same combinatorics
            # and values as fill_alpha_shapes.
            verts_by_dim, vals_by_dim = diode.fill_alpha_shapes_arrays(points, exact=exact)
            # Route the filtration to the backend matching the points' dtype (float32 points ->
            # a genuine float32 alpha filtration, like vr_filtration / freudenthal_filtration).
            # CGAL/diode compute the alpha values in double, so narrow them to the target Real.
            dt = detect_real_dtype(points)
            sub = REAL_MODULES[dt]
            if dt != DEFAULT_REAL_DTYPE:
                vals_by_dim = [np.ascontiguousarray(v, dtype=dt) for v in vals_by_dim]
            # packed (the default) returns a bit-packed PackedSimplexFiltration when the vertex
            # ids fit a 64/128-bit word (only this fast array path supports it; the
            # weighted/periodic/list fallbacks below stay fat). Reduces and produces
            # diagrams identically to fat, with a smaller footprint.
            suffix = _vr_packed_word_suffix(points.shape[0], points.shape[1]) if packed else None
            if suffix is not None:
                # bits passed directly (skips the C++ max-id scan); diode rows are
                # not vertex-sorted, so assume_sorted stays False.
                fil = getattr(sub, "_filtration_from_arrays_packed" + suffix)(
                    verts_by_dim, vals_by_dim, n_threads=n_threads, bits=_packed_bits(points.shape[0]))
            else:
                fil = sub._filtration_from_arrays(verts_by_dim, vals_by_dim, n_threads=n_threads)
            fil.kind = _oineus.FiltrationKind.Alpha
            return fil
        else:
            fil_diode = diode.fill_alpha_shapes(points, exact=exact)

    # Route the weighted / periodic / non-array-exporter fallbacks to the backend matching the
    # points' dtype, like the fast array path above (and vr_filtration / freudenthal_filtration).
    # diode computes alpha values in double; the float32 _Filtration ctor narrows them, same as
    # the array path's explicit narrowing.
    sub = REAL_MODULES[detect_real_dtype(points)]
    fil = sub._Filtration(
        fil_diode,
        duplicates_possible=periodic,
        n_threads=n_threads,
    )
    fil.kind = _oineus.FiltrationKind.Alpha
    return fil


def _delaunay_combinatorics(points: np.ndarray, exact: bool=False, packed: bool=False, n_threads: int=1):
    """Build the Delaunay complex as a Filtration, for its combinatorics only.

    For callers that recompute and set their own values (the differentiable
    Cech-Delaunay and weak-alpha filtrations), so the filtration values here are
    not meaningful and must be overwritten via set_values. The fast path
    (diode's fill_delaunay_arrays) leaves them at 0; the fallback
    (alpha_filtration, used when the array exporters are absent) carries alpha
    values instead. Either way the simplex set is identical for full-dimensional
    input.

    Args:
        points: NumPy array of shape (n, 2) or (n, 3).
        exact: Use CGAL's exact kernel.
        packed: Use the compact bit-packed cell encoding when the vertex ids fit
            a 64/128-bit word (only the fast diode-array path supports it; the
            alpha_filtration fallback honors packed too).
        n_threads: Threads used inside the Filtration constructor.

    Returns:
        A Filtration over the Delaunay simplices; its values are not meaningful
        and are expected to be overwritten by the caller.
    """
    if _HAS_DIODE_ARRAYS:
        verts_by_dim = diode.fill_delaunay_arrays(points, exact=exact)
        # route to the backend matching the points' dtype so a float32 point cloud yields a
        # float32 Delaunay filtration (the differentiable cech/weak paths then set float32
        # values into it via real_buffer_for). Only the vertex arrays are used here; the
        # caller overwrites the values, so no value-array dtype handling is needed.
        sub = real_module_for(points)
        suffix = _vr_packed_word_suffix(points.shape[0], points.shape[1]) if packed else None
        if suffix is not None:
            return getattr(sub, "_filtration_from_arrays_packed" + suffix)(
                verts_by_dim, None, n_threads=n_threads, bits=_packed_bits(points.shape[0]))
        return sub._filtration_from_arrays(verts_by_dim, None, n_threads=n_threads)
    return alpha_filtration(points, exact=exact, packed=packed, n_threads=n_threads)


def _triangle_meb_np(p0, p1, p2, eps=0.0):
    """Numpy mirror of oineus.diff.cech_delaunay.triangle_meb.

    Returns (centers, radii_sq) of the minimum enclosing balls of n
    triangles given as (n, d) arrays, d in {2, 3}. Vertices of a triangle
    must be pairwise distinct points.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    a = p1 - p0
    b = p2 - p0
    c = p2 - p1

    a_sq = np.sum(a * a, axis=1)
    b_sq = np.sum(b * b, axis=1)
    c_sq = np.sum(c * c, axis=1)

    d = p0.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        if d == 2:
            cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            area_2_sq = cross ** 2
        else:
            cross = np.cross(a, b)
            area_2_sq = np.sum(cross * cross, axis=1)

        circum_radii_sq = (a_sq * b_sq * c_sq + eps) / (4 * area_2_sq + eps)

        if d == 3:
            cross_ab_sq = area_2_sq[:, None]
            b_cross_axb = np.cross(b, cross)
            axb_cross_a = np.cross(cross, a)
            circum_centers = p0 + (a_sq[:, None] * b_cross_axb + b_sq[:, None] * axb_cross_a) / (2 * cross_ab_sq + eps)
        else:
            D = 2 * cross[:, None]
            ux = (b[:, 1:2] * a_sq[:, None] - a[:, 1:2] * b_sq[:, None]) / (D + eps)
            uy = (a[:, 0:1] * b_sq[:, None] - b[:, 0:1] * a_sq[:, None]) / (D + eps)
            circum_centers = p0 + np.concatenate([ux, uy], axis=1)

    abc_sq = np.stack((a_sq, b_sq, c_sq), axis=0)
    sort_idx = np.argsort(abc_sq, axis=0)
    s_abc_sq = np.take_along_axis(abc_sq, sort_idx, axis=0)
    # degenerate (collinear) triangles are always obtuse, so they take the
    # exact diametral-ball branch and never see the circumsphere formula
    obtuse_mask = s_abc_sq[2, :] > s_abc_sq[0, :] + s_abc_sq[1, :]
    longest_edge_idx = sort_idx[2, :]

    midpoints = np.stack(((p0 + p1) / 2, (p0 + p2) / 2, (p1 + p2) / 2), axis=0)
    diametral_centers = np.take_along_axis(midpoints, longest_edge_idx[None, :, None], axis=0)[0]

    centers = np.where(obtuse_mask[:, None], diametral_centers, circum_centers)
    radii_sq = np.where(obtuse_mask, s_abc_sq[2, :] / 4, circum_radii_sq)
    return centers, radii_sq


def _triangle_meb_sq_np(p0, p1, p2, eps=0.0):
    """Squared MEB radii of n triangles; numpy, see _triangle_meb_np."""
    return _triangle_meb_np(p0, p1, p2, eps)[1]


def _tetrahedron_meb_sq_np(p0, p1, p2, p3, eps=0.0, flat_tol=1e-9):
    """Numpy mirror of oineus.diff.cech_delaunay.tetrahedron_meb (radii only).

    Handles arbitrary tetrahedra given as (n, 3) arrays, including
    (near-)flat and coplanar ones: the ill-conditioned circumsphere
    candidate is discarded for flat tets, whose MEB is always attained on
    a face. Vertices of a tetrahedron must be pairwise distinct points.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    a = p1 - p0
    b = p2 - p0
    c = p3 - p0

    a_sq = np.sum(a * a, axis=1)
    b_sq = np.sum(b * b, axis=1)
    c_sq = np.sum(c * c, axis=1)

    cross_bc = np.cross(b, c)
    cross_ca = np.cross(c, a)
    cross_ab = np.cross(a, b)

    volume_6 = np.sum(a * cross_bc, axis=1)

    numerator_vec = a_sq[:, None] * cross_bc + b_sq[:, None] * cross_ca + c_sq[:, None] * cross_ab
    denom = 2 * volume_6 + np.copysign(np.full_like(volume_6, eps), volume_6)
    with np.errstate(divide="ignore", invalid="ignore"):
        circum_disp = numerator_vec / denom[:, None]
        circum_radii_sq = np.sum(circum_disp * circum_disp, axis=1)

    # same flat-tet mask as the torch version: for flat 4-point sets the MEB
    # is attained on a face, so dropping the circumsphere candidate is exact
    e12_sq = np.sum((p2 - p1) ** 2, axis=1)
    e13_sq = np.sum((p3 - p1) ** 2, axis=1)
    e23_sq = np.sum((p3 - p2) ** 2, axis=1)
    scale_sq = np.max(np.stack([a_sq, b_sq, c_sq, e12_sq, e13_sq, e23_sq]), axis=0)
    flat_mask = np.abs(volume_6) <= flat_tol * scale_sq ** 1.5

    face_centers_0, face_radii_sq_0 = _triangle_meb_np(p1, p2, p3, eps)
    face_centers_1, face_radii_sq_1 = _triangle_meb_np(p0, p2, p3, eps)
    face_centers_2, face_radii_sq_2 = _triangle_meb_np(p0, p1, p3, eps)
    face_centers_3, face_radii_sq_3 = _triangle_meb_np(p0, p1, p2, eps)

    dist_sq_0 = np.sum((p0 - face_centers_0) ** 2, axis=1)
    dist_sq_1 = np.sum((p1 - face_centers_1) ** 2, axis=1)
    dist_sq_2 = np.sum((p2 - face_centers_2) ** 2, axis=1)
    dist_sq_3 = np.sum((p3 - face_centers_3) ** 2, axis=1)

    rel_slack = 4 * np.finfo(np.float64).eps
    contains_0 = dist_sq_0 <= face_radii_sq_0 * (1 + rel_slack) + eps
    contains_1 = dist_sq_1 <= face_radii_sq_1 * (1 + rel_slack) + eps
    contains_2 = dist_sq_2 <= face_radii_sq_2 * (1 + rel_slack) + eps
    contains_3 = dist_sq_3 <= face_radii_sq_3 * (1 + rel_slack) + eps

    inf = np.float64(np.inf)
    all_radii_sq = np.stack([
        np.where(flat_mask, inf, circum_radii_sq),
        np.where(contains_0, face_radii_sq_0, inf),
        np.where(contains_1, face_radii_sq_1, inf),
        np.where(contains_2, face_radii_sq_2, inf),
        np.where(contains_3, face_radii_sq_3, inf),
    ], axis=0)
    # NaNs can only appear in the circumsphere row of flat tets (0/0 with
    # eps=0); the flat mask has already replaced those with inf
    min_radii_sq = np.min(all_radii_sq, axis=0)

    # Belt-and-suspenders for flat tets whose on-boundary vertex rounds
    # outside every face ball by more than rel_slack (all candidates inf):
    # grow the best face ball just enough to contain its opposite vertex.
    # Exact up to ulps in this tie case; an inf here would silently drop a
    # valid simplex from a Cech complex.
    no_candidate = np.isinf(min_radii_sq)
    if np.any(no_candidate):
        grown = np.min(np.stack([
            np.maximum(face_radii_sq_0, dist_sq_0),
            np.maximum(face_radii_sq_1, dist_sq_1),
            np.maximum(face_radii_sq_2, dist_sq_2),
            np.maximum(face_radii_sq_3, dist_sq_3),
        ], axis=0), axis=0)
        min_radii_sq = np.where(no_candidate, grown, min_radii_sq)
    return min_radii_sq


def _pack_vertex_rows(rows, n_points):
    # canonical int64 key per row-sorted vertex row (base-n_points digits).
    # Overflow would alias keys and silently corrupt membership tests, so
    # fail loud instead (unreachable for any enumerable complex size).
    if len(rows) and n_points ** rows.shape[1] >= 2 ** 63:
        raise ValueError("_pack_vertex_rows: n_points too large for int64 row keys")
    key = np.zeros(len(rows), dtype=np.int64)
    for col in range(rows.shape[1]):
        key = key * n_points + rows[:, col]
    return key


def cech_filtration(points: np.ndarray,
                    max_dim: int = -1,
                    max_radius: float = -1.0,
                    *,
                    vertex_ids: typing.Optional[np.ndarray] = None,
                    eps: float = 0.0,
                    n_threads: int = 1):
    """Build a full Cech filtration of a point cloud (non-differentiable).

    Enumerates ALL simplices on the input points up to max_dim whose minimum
    enclosing ball (MEB) radius is at most max_radius; the filtration value
    of a simplex is its SQUARED MEB radius (the same convention as
    alpha_filtration and oineus.diff.cech_delaunay_filtration, so diagrams
    are directly comparable).

    With the default max_radius (the enclosing radius of the cloud) the
    complex contains the complete max_dim-skeleton of the simplex on all n
    vertices, so diagrams in dimensions 0 .. max_dim-1 are complete; the
    dimension-max_dim diagram is skeleton-truncated and unreliable. Mind the
    combinatorial cost: at the default radius the number of q-simplices is
    C(n, q+1); pass an explicit smaller max_radius beyond a few hundred
    points at max_dim=3.

    Args:
        points: (n, d) array, d in {2, 3}, pairwise-distinct points.
        max_dim: Largest simplex dimension; default d. Must be <= d (in
            2D, 3-simplices would need 4-coplanar-point MEBs; if needed,
            z-pad the points to 3D instead).
        max_radius: Unsquared radius threshold; a simplex is kept iff
            meb_radius_sq <= max_radius**2. Default: enclosing radius
            (oineus.max_distance).
        vertex_ids: Optional (n,) integer array relabeling vertex i to
            vertex_ids[i]. With ids drawn from a larger cloud this makes the
            result a genuine subcomplex (equal uids and values) of the full
            cloud's cech_filtration -- pass the SAME explicit max_radius to
            both calls, since the defaults differ.
        eps: Numerical-stability epsilon of the MEB formulas.
        n_threads: Threads for the Filtration constructor sort.

    Returns:
        Filtration with kind=FiltrationKind.Cech and squared-MEB values.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("cech_filtration: points must be a 2D array with at least 2 rows")
    n, d = points.shape
    if d not in (2, 3):
        raise ValueError(f"cech_filtration: ambient dimension must be 2 or 3, got {d}")
    if max_dim < 0:
        max_dim = d
    if max_dim > d:
        raise ValueError(f"cech_filtration: max_dim={max_dim} > ambient dimension {d} is not supported "
                         "(MEBs of degenerate simplices; z-pad the points to 3D if you need 2D tetrahedra)")
    if max_radius < 0:
        max_radius = max_distance(points)

    dt = detect_real_dtype(points)
    sub = REAL_MODULES[dt]
    pts64 = np.ascontiguousarray(points, dtype=np.float64)

    # combinatorics via the VR enumerator: meb_radius <= R implies
    # diameter <= 2R (tight for edges and diametral simplices; Jung's
    # theorem bounds the other direction and cannot shrink this), so the
    # VR complex at diameter 2R is a superset of the Cech complex at R
    vr_fil = vr_filtration(pts64, max_dim=max_dim, max_diameter=2 * max_radius,
                           packed=True, n_threads=n_threads)

    verts_by_dim = []
    vals_by_dim = []
    max_radius_sq = max_radius * max_radius
    kept_keys_prev = None
    for q in range(max_dim + 1):
        # a dim can be empty (or absent) when max_radius truncates hard; all
        # higher dims are then empty too (VR structure + closure pruning)
        if q > vr_fil.max_dim or vr_fil.size_in_dimension(q) == 0:
            break
        verts = np.ascontiguousarray(vr_fil.get_simplices_as_arr(q), dtype=np.int64)
        verts = np.sort(verts, axis=1)
        if q == 0:
            vals = np.zeros(len(verts), dtype=np.float64)
        elif q == 1:
            vals = 0.25 * np.sum((pts64[verts[:, 0]] - pts64[verts[:, 1]]) ** 2, axis=1)
        elif q == 2:
            vals = _triangle_meb_sq_np(pts64[verts[:, 0]], pts64[verts[:, 1]], pts64[verts[:, 2]], eps)
        else:
            vals = _tetrahedron_meb_sq_np(pts64[verts[:, 0]], pts64[verts[:, 1]],
                                          pts64[verts[:, 2]], pts64[verts[:, 3]], eps)
        keep = vals <= max_radius_sq
        # closure repair: MEB radius is monotone under faces, so this can
        # only fire on last-ulp non-monotonicity exactly at the threshold;
        # without it a dropped facet of a kept simplex would break the
        # boundary lookups during reduction
        if q >= 2:
            for i in range(q + 1):
                facet_keys = _pack_vertex_rows(np.delete(verts, i, axis=1), n)
                keep &= np.isin(facet_keys, kept_keys_prev)
        verts, vals = verts[keep], vals[keep]
        if len(verts) == 0:
            break
        kept_keys_prev = _pack_vertex_rows(verts, n)
        verts_by_dim.append(verts)
        vals_by_dim.append(np.ascontiguousarray(vals, dtype=dt))

    if vertex_ids is not None:
        vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
        if vertex_ids.shape != (n,):
            raise ValueError("cech_filtration: vertex_ids must have shape (n_points,)")
        if np.unique(vertex_ids).size != n:
            raise ValueError("cech_filtration: vertex_ids must be pairwise distinct")
        verts_by_dim = [np.ascontiguousarray(vertex_ids[verts]) for verts in verts_by_dim]

    fil = sub._filtration_from_arrays(verts_by_dim, vals_by_dim, n_threads=n_threads)
    fil.kind = _oineus.FiltrationKind.Cech
    return fil
def _periodic_delaunay_combinatorics(points: np.ndarray,
                                     bbox_min,
                                     bbox_max,
                                     exact: bool=False,
                                     packed: bool=False,
                                     n_threads: int=1):
    """Build periodic Delaunay combinatorics and aligned coherent lifts.

    The temporary values are strictly increasing in diode row order, with all
    lower-dimensional rows before higher-dimensional rows. This makes the
    filtration retain the array order until the caller replaces the values, so
    the returned offset rows stay aligned without Python per-simplex maps.

    Returns:
        A tuple ``(filtration, vertices_by_dim, offsets_by_dim)``.
    """
    if not _HAS_DIODE_PERIODIC_LIFTS:
        raise RuntimeError(
            "Differentiable periodic Cech-Delaunay requires a diode build with "
            "fill_periodic_delaunay_lifts_arrays"
        )

    vertices_by_dim, offsets_by_dim = diode.fill_periodic_delaunay_lifts_arrays(
        points,
        exact=exact,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
    if len(vertices_by_dim) != len(offsets_by_dim):
        raise RuntimeError("diode returned misaligned periodic vertex and offset dimensions")

    placeholder_values = []
    next_value = 0
    for dim, (vertices, offsets) in enumerate(zip(vertices_by_dim, offsets_by_dim)):
        expected_width = dim + 1
        expected_offset_shape = (vertices.shape[0], expected_width, points.shape[1])
        if vertices.ndim != 2 or vertices.shape[1] != expected_width:
            raise RuntimeError(f"diode returned invalid periodic vertex array in dimension {dim}")
        if offsets.shape != expected_offset_shape:
            raise RuntimeError(f"diode returned invalid periodic offset array in dimension {dim}")
        count = vertices.shape[0]
        placeholder_values.append(
            np.arange(next_value, next_value + count, dtype=points.dtype)
        )
        next_value += count

    sub = real_module_for(points)
    suffix = _vr_packed_word_suffix(points.shape[0], points.shape[1]) if packed else None
    if suffix is not None:
        filtration = getattr(sub, "_filtration_from_arrays_packed" + suffix)(
            vertices_by_dim,
            placeholder_values,
            n_threads=n_threads,
            bits=_packed_bits(points.shape[0]),
        )
    else:
        filtration = sub._filtration_from_arrays(
            vertices_by_dim, placeholder_values, n_threads=n_threads
        )

    getters = ("get_vertices", "get_edges", "get_triangles", "get_tetrahedra")
    for dim, vertices in enumerate(vertices_by_dim):
        emitted = getattr(filtration, getters[dim])()
        if not np.array_equal(emitted, vertices):
            raise RuntimeError(
                "Periodic Delaunay row order changed while building the filtration; "
                "coherent offsets cannot be aligned safely"
            )

    return filtration, vertices_by_dim, offsets_by_dim


def compute_diagrams_alpha(points: np.ndarray,
                           weights: typing.Optional[np.ndarray]=None,
                           params: typing.Optional[ReductionParams]=None,
                           include_inf_points: bool=True,
                           dualize: bool=False,
                           exact: bool=False,
                           periodic: bool=False,
                           compute_bounding_box: bool=True,
                           bbox_min: typing.Optional[typing.Union[np.ndarray, typing.List[float]]]=None,
                           bbox_max: typing.Optional[typing.Union[np.ndarray, typing.List[float]]]=None,
                           ):
    """Compute alpha-shape persistence diagrams.

    Args:
        points: NumPy array of shape (n, 2) or (n, 3).
        weights: Optional 1D array of length n. If provided, computes weighted
            alpha-shapes (currently 3D only).
        params: Reduction parameters. Defaults to ReductionParams().
        include_inf_points: Include points at infinity in output diagrams.
        dualize: If True, compute cohomology; otherwise homology.
        exact: Passed to diode. If True, uses exact CGAL kernel.
        periodic: If True, uses periodic alpha-shapes. Duplicate simplices
            reported by diode are deduplicated before building the filtration.
        compute_bounding_box: If True, use the bounding box of the data for periodic
            otherwise diode defaults (unit box) will be used.
        bbox_min: lexicographically smallest point of the bounding box.
             NumPy array or list of floats. Ignored, if compute_bounding_box is True.
             Origin will be used, if compute_bounding_box is False and bbox_min is None.
        bbox_max: lexicographically largest point of the bounding box.
             NumPy array or list of floats. Ignored, if compute_bounding_box is True

    Returns:
        Diagrams object indexed by homology dimension.
    """
    if params is None:
        params = _oineus.ReductionParams()
    fil = alpha_filtration(
        points,
        weights=weights,
        exact=exact,
        periodic=periodic,
        compute_bounding_box=compute_bounding_box,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        n_threads=params.n_threads,
    )
    dcmp = _oineus.Decomposition(fil, dualize)
    dcmp.reduce(params)
    return dcmp.diagram(fil=fil, include_inf_points=include_inf_points, n_threads=params.n_threads)


def list_to_filtration(data: typing.List[typing.Tuple[int, typing.List[int], float]]):
    simplices = [ Simplex(id, vertices, val) for id, vertices, val in data ]
    return Filtration(simplices)


def compute_kernel_image_cokernel_reduction(K, L, params=None, reduction_params=None):
    # simplicial filtrations can be supplied as lists,
    # convert to Oineus filtrations if necessary
    if isinstance(K, list):
        K = list_to_filtration(K)
    if isinstance(L, list):
        L = list_to_filtration(L)

    # KICR is templated on the cell type in C++; each encoding has its own bound class.
    # Dispatch on the FILTRATION type, not K[0]: a slim/packed filtration materializes a
    # fat Simplex on K[0], which would misdispatch it into the fat ctor. type(K) is the
    # stable C++ class for every encoding (and is correct for empty filtrations too).
    KICR_Class = _KICR_CLASS_BY_FIL_TYPE.get(type(K))
    if KICR_Class is None:
        raise TypeError(
            f"compute_kernel_image_cokernel_reduction: unsupported filtration type "
            f"{type(K).__name__}; expected one of "
            f"{[t.__name__ for t in _KICR_CLASS_BY_FIL_TYPE]}")

    if params is None:
        # compute all by default
        params = _oineus.KICRParams(kernel=True, image=True, cokernel=True)
    elif not isinstance(params, _oineus.KICRParams):
        raise TypeError("params must be a KICRParams instance")

    if reduction_params is not None:
        if not isinstance(reduction_params, _oineus.ReductionParams):
            raise TypeError("reduction_params must be a ReductionParams instance")
        params.params_f = reduction_params
        params.params_g = reduction_params
        params.params_ker = reduction_params
        params.params_im = reduction_params
        params.params_cok = reduction_params

    return KICR_Class(K, L, params)


def compute_ker_cok_reduction_cyl(fil_2, fil_3):
    fil_min = min_filtration(fil_2, fil_3)

    id_domain = fil_3.size() + fil_min.size() + 1
    id_codomain = id_domain + 1

    # id_domain: id of vertex at the top of the cylinder,
    # i.e., we multiply fil_3 with id_domain
    # id_codomain: id of vertex at the bottom of the cylinder
    # i.e, we multiply fil_min with id_codomain

    # The wrapper functions below default the auxiliary vertex values to
    # fil.neg_infinity(), so the value carried on these Simplex objects is
    # discarded. We still need a Simplex/CombinatorialSimplex to specify
    # the vertex labels.
    v0 = _oineus.CombinatorialSimplex(id_domain, [id_domain])
    v1 = _oineus.CombinatorialSimplex(id_codomain, [id_codomain])

    fil_cyl = mapping_cylinder(fil_3, fil_min, v0, v1)

    # to get a subcomplex, we multiply each fil_3 with id_domain
    fil_3_prod = multiply_filtration(fil_3, v0)

    params = _oineus.KICRParams()
    params.kernel = params.cokernel = True
    params.image = False

    # route the product KICR to the backend matching the cylinder's dtype
    kicr_reduction = module_of_oineus_obj(fil_cyl).KerImCokReducedProd(fil_cyl, fil_3_prod, params)

    return kicr_reduction


def cube_filtration(data: np.ndarray,
                    negate: bool=False,
                    wrap: bool=False,
                    max_dim: typing.Optional[int]=None,
                    values_on: str="vertices",
                    n_threads: int=1,):
    if wrap:
        raise RuntimeError("cube_filtration: wrap=True is not implemented yet")
    # route to the float32 or float64 backend by the input dtype
    dt = detect_real_dtype(data)
    sub = REAL_MODULES[dt]
    data = as_real_numpy(data, dtype=dt)
    if max_dim is None:
        max_dim = data.ndim
    dim = data.ndim
    if dim == 1:
        grid = sub.Grid_1D(data, wrap=wrap, values_on=values_on)
    elif dim == 2:
        grid = sub.Grid_2D(data, wrap=wrap, values_on=values_on)
    elif dim == 3:
        grid = sub.Grid_3D(data, wrap=wrap, values_on=values_on)
    elif dim == 4:
        grid = sub.Grid_4D(data, wrap=wrap, values_on=values_on)
    else:
        raise RuntimeError(f"cube_filtration: dim={data.ndim} not supported, recompile from sources")
    fil = grid.cube_filtration(max_dim=max_dim, n_threads=n_threads, negate=negate)
    return fil


_PUBLIC_API_NAMES = [
    "ConflictStrategy",
    "DenoiseStrategy",
    "VREdge",
    "FiltrationKind",
    "DiagramPlaneDomain",
    "FrechetMeanInit",
    "CombinatorialProdSimplex",
    "CombinatorialSimplex",
    "Simplex",
    "ProdSimplex",
    "Filtration",
    "ProdFiltration",
    "Decomposition",
    "IndexDiagramPoint",
    "DiagramPoint",
    "Diagrams",
    "reduce",
    "DecompositionManipStats",
    "ReductionParams",
    "ReductionParamsAdvanced",
    "ReductionTimings",
    "UComputeTimings",
    "KICRParams",
    "KerImCokReduced",
    "KerImCokReducedProd",
    "ColumnRepr",
    "IndicesValues",
    "IndicesValuesProd",
    "TopologyOptimizer",
    "TopologyOptimizerProd",
    "TopologyOptimizerCube_1D",
    "TopologyOptimizerCube_2D",
    "TopologyOptimizerCube_3D",
    "TopologyOptimizerCube_4D",
    "compute_relative_diagrams",
    "get_boundary_matrix",
    "get_denoise_target",
    "get_induced_matching",
    "get_nth_persistence",
    "get_permutation_dtv",
    "get_permutation",
    "GridDomain_1D",
    "Grid_1D",
    "CombinatorialCube_1D",
    "Cube_1D",
    "GridDomain_2D",
    "Grid_2D",
    "CombinatorialCube_2D",
    "Cube_2D",
    "GridDomain_3D",
    "Grid_3D",
    "CombinatorialCube_3D",
    "Cube_3D",
    "GridDomain_4D",
    "Grid_4D",
    "CombinatorialCube_4D",
    "Cube_4D",
    "DiagramMatching",
    "BottleneckMatching",
    "InfKind",
    "EssentialMatches",
    "EssentialLongestEdges",
    "LongestEdges",
    "FiniteLongestEdge",
    "EssentialLongestEdge",
    "point_to_diagonal",
    "wasserstein_matching",
    "bottleneck_matching",
    "sliced_wasserstein_distance",
    "sliced_wasserstein_distance_diag_corrected",
    "REAL_DTYPE",
    "as_real_numpy",
    "bottleneck_distance",
    "wasserstein_distance",
    "init_frechet_mean_first_diagram",
    "init_frechet_mean_random_diagram",
    "init_frechet_mean_medoid_diagram",
    "init_frechet_mean_diagonal_grid",
    "frechet_mean_objective",
    "make_frechet_mean_persistence_schedule",
    "frechet_mean_newborn_points_from_newly_active",
    "frechet_mean_multistart",
    "progressive_frechet_mean",
    "progressive_frechet_mean_multistart",
    "frechet_mean",
    "to_scipy_matrix",
    "max_distance",
    "freudenthal_filtration",
    "vr_filtration",
    "is_reduced",
    "mapping_cylinder",
    "multiply_filtration",
    "min_filtration",
    "compute_diagrams_ls",
    "compute_diagrams_vr",
    "alpha_filtration",
    "cech_filtration",
    "compute_diagrams_alpha",
    "list_to_filtration",
    "compute_kernel_image_cokernel_reduction",
    "compute_ker_cok_reduction_cyl",
    "MixupBarcodes",
    "mixup_barcodes",
    "mixup_barcodes_of_filtrations",
    "function_delaunay_bifiltration",
    "cube_filtration",
    "plot_diagram",
    "plot_diagram_gradient",
    "plot_matching",
    "plot_chain",
    "default_point_style",
    "default_diagram_a_point_style",
    "default_diagram_b_point_style",
    "default_matching_edge_style",
    "default_longest_edge_style",
    "default_diagonal_style",
    "default_diagonal_projection_a_style",
    "default_diagonal_projection_b_style",
    "default_inf_line_style",
    "default_inf_point_style",
    "default_diagram_gradient_style",
    "default_density_style",
    "default_grid_style",
    "default_chain_vertex_style",
    "default_chain_edge_style",
    "default_chain_triangle_style",
    "default_chain_tetrahedron_style",
    "default_point_cloud_style",
    "DEFAULT_POINT_STYLE",
    "DEFAULT_DIAGRAM_A_POINT_STYLE",
    "DEFAULT_DIAGRAM_B_POINT_STYLE",
    "DEFAULT_MATCHING_EDGE_STYLE",
    "DEFAULT_LONGEST_EDGE_STYLE",
    "DEFAULT_DIAGONAL_STYLE",
    "DEFAULT_DIAGONAL_PROJECTION_A_STYLE",
    "DEFAULT_DIAGONAL_PROJECTION_B_STYLE",
    "DEFAULT_INF_LINE_STYLE",
    "DEFAULT_INF_POINT_STYLE",
    "DEFAULT_DIAGRAM_GRADIENT_STYLE",
    "DEFAULT_DENSITY_STYLE",
    "DEFAULT_DENSITY_THRESHOLD",
    "DEFAULT_GRID_STYLE",
    "DEFAULT_MATCHING_EDGE_QUANTILE",
    "DEFAULT_GRADIENT_TOP_K_ARROWS",
    "DEFAULT_CHAIN_VERTEX_STYLE",
    "DEFAULT_CHAIN_EDGE_STYLE",
    "DEFAULT_CHAIN_TRIANGLE_STYLE",
    "DEFAULT_CHAIN_TETRAHEDRON_STYLE",
    "DEFAULT_POINT_CLOUD_STYLE",
    "OKABE_ITO_BLUE",
    "OKABE_ITO_VERMILLION",
]

__all__ = [name for name in _PUBLIC_API_NAMES if name in globals()]
