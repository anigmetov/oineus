"""Type stubs for the top-level oineus package."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
import enum
from typing import Annotated, Any, Literal, overload

import numpy
from numpy.typing import NDArray
import scipy.sparse

class ConflictStrategy(enum.Enum):
    """
    Rule for resolving overlapping critical-set prescriptions on the same cell.
    """
    Max = 0
    'choose maximal displacement'
    Avg = 1
    'average gradients'
    Sum = 2
    'sum gradients'
    FixCritAvg = 3
    'use matching on critical, average gradients on other cells'

    def as_str(self) -> str:
        ...

class DenoiseStrategy(enum.Enum):
    """
    Target placement on the diagonal when simplifying low-persistence pairs.
    """
    BirthBirth = 0
    '(b, d) maps to (b, b)'
    DeathDeath = 1
    '(b, d) maps to (d, d)'
    Midway = 2
    '((b, d) maps to ((b+d)/2, (b+d)/2)'

    def as_str(self) -> str:
        ...

class UStrategy(enum.Enum):
    """
    U-computation strategy used by the crit-sets backward in oineus.diff. Auto resolves to RowPartial; LegacyInBand is the in-band serial reduction kept as a control.
    """
    Auto = 0
    RowPartial = 1
    LegacyInBand = 2

class VREdge:
    """
    Critical Vietoris-Rips edge (x, y) whose length determines a simplex's filtration value.
    """

    def __init__(self, x: int=0, y: int=0) -> None:
        ...

    @property
    def x(self) -> int:
        """First endpoint vertex index."""

    @x.setter
    def x(self, arg: int, /) -> None:
        ...

    @property
    def y(self) -> int:
        """Second endpoint vertex index."""

    @y.setter
    def y(self, arg: int, /) -> None:
        ...

    def __getitem__(self, arg: int, /) -> int:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: VREdge, /) -> bool:
        ...

    def __ne__(self, arg: VREdge, /) -> bool:
        ...

    def __getstate__(self) -> tuple[int, int]:
        ...

    def __setstate__(self, arg: tuple[int, int], /) -> None:
        ...

class FiltrationKind(enum.Enum):
    """
    Hint set by built-in filtration constructors. Lets oineus.diff pick reduction defaults and oineus.vis pick rendering (point cloud vs scalar field) automatically. User is the safe default for hand-built filtrations.
    """
    User = 0
    Vr = 1
    Alpha = 2
    WeakAlpha = 3
    CechDelaunay = 4
    Cech = 9
    Freudenthal = 5
    Cubical = 6
    MinFil = 7
    MappingCylinder = 8

    def as_str(self) -> str:
        ...

class DiagramPlaneDomain(enum.Enum):
    """
    Half-plane domain constraint for Frechet-mean initializers and barycenters.
    """
    AboveDiagonal = 0
    BelowDiagonal = 1
    Mixed = 2

    def as_str(self) -> str:
        ...

class FrechetMeanInit(enum.Enum):
    """
    Initialization strategy for Frechet-mean (W2 barycenter) Lloyd iterations.
    """
    Custom = 0
    FirstDiagram = 1
    MedoidDiagram = 2
    RandomDiagram = 3
    Grid = 4

    def as_str(self) -> str:
        ...

class ColumnRepr(enum.Enum):
    """Working-column data structure used during reduction"""
    Set = 0
    Any
    Heap = 1
    'lazy max-heap (PHAT A-Heap)'
    Full = 2
    'dense bitset + max-heap (PHAT A-Full)'
    BitTree = 3
    'hierarchical 64-ary bitset (PHAT A-Bit-Tree, default)'

class ReductionTimings:
    """
    Per-phase wall-clock breakdown (seconds) of the last reduce() call, available as Decomposition.timings. Some fields are 0 when a path skips that phase: the serial path reduces in place, so it has no prepare / copy_back / copy_pivots. reduction_total is the path-comparable total.
    """

    def __init__(self) -> None:
        ...

    @property
    def prepare(self) -> float:
        """build the working atomic-pointer matrix (parallel only)"""

    @prepare.setter
    def prepare(self, arg: float, /) -> None:
        ...

    @property
    def reduce(self) -> float:
        """the reduction itself (serial loop or parallel threads)"""

    @reduce.setter
    def reduce(self, arg: float, /) -> None:
        ...

    @property
    def bauer(self) -> float:
        """
        Bauer-trick fill of cleared V columns (only when V materialized under clearing)
        """

    @bauer.setter
    def bauer(self, arg: float, /) -> None:
        ...

    @property
    def restore_elz(self) -> float:
        """ELZ-restore phase (only if dims_to_restore_elz set)"""

    @restore_elz.setter
    def restore_elz(self, arg: float, /) -> None:
        ...

    @property
    def copy_back(self) -> float:
        """move working matrix back into r_data/v_data (parallel only)"""

    @copy_back.setter
    def copy_back(self, arg: float, /) -> None:
        ...

    @property
    def copy_pivots(self) -> float:
        """copy pivots into the at-rest pivot array (parallel only)"""

    @copy_pivots.setter
    def copy_pivots(self, arg: float, /) -> None:
        ...

    @property
    def compute_u(self) -> float:
        """
        full post-reduction parallel VTUT solve (serial in-band U is included in reduce)
        """

    @compute_u.setter
    def compute_u(self, arg: float, /) -> None:
        ...

    @property
    def reduction_total(self) -> float:
        """
        Total reduction wall-clock across every phase -- comparable across serial and parallel paths.
        """

    @property
    def total(self) -> float:
        """Synonym for reduction_total."""

    def reset(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> tuple[float, float, float, float, float, float, float]:
        ...

    def __setstate__(self, arg: tuple[float, float, float, float, float, float, float], /) -> None:
        ...

class UComputeTimings:
    """
    Per-phase wall-clock breakdown (seconds) of the last compute_u_* call. Only the fields the chosen strategy uses are nonzero: row-form (compute_full_u_rows / compute_partial_u_rows) fills transpose_v + row_solve; column-form (compute_u_from_v / compute_u_from_v_1) fills col_solve + col_to_row. total is the strategy-comparable U-compute time.
    """

    def __init__(self) -> None:
        ...

    @property
    def transpose_v(self) -> float:
        """row-form Stage A: build V^T (parallel col->row transpose)"""

    @transpose_v.setter
    def transpose_v(self, arg: float, /) -> None:
        ...

    @property
    def row_solve(self) -> float:
        """row-form Stage B: parallel per-row forward substitution"""

    @row_solve.setter
    def row_solve(self, arg: float, /) -> None:
        ...

    @property
    def col_solve(self) -> float:
        """column-form: solve each U column in parallel"""

    @col_solve.setter
    def col_solve(self, arg: float, /) -> None:
        ...

    @property
    def col_to_row(self) -> float:
        """column-form: transpose column-form U into row form"""

    @col_to_row.setter
    def col_to_row(self, arg: float, /) -> None:
        ...

    @property
    def total(self) -> float:
        """
        Total U-compute wall-clock across every phase -- comparable across strategies.
        """

    def reset(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> tuple[float, float, float, float]:
        ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None:
        ...

class ReductionParamsAdvanced:
    """
    Rarely-tuned reduction knobs, nested as ReductionParams.advanced. The reduce() kwargs layer still accepts them flat: dcmp.reduce(chunk_size=256) routes here.
    """

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, chunk_size: int=128, col_repr: ColumnRepr=ColumnRepr.BitTree, dims_to_restore_elz: Sequence[int]=[]) -> None:
        ...

    @property
    def chunk_size(self) -> int:
        """number of columns each parallel worker claims at a time"""

    @chunk_size.setter
    def chunk_size(self, arg: int, /) -> None:
        ...

    @property
    def col_repr(self) -> ColumnRepr:
        """working-column data structure used during reduction (ColumnRepr)"""

    @col_repr.setter
    def col_repr(self, arg: ColumnRepr, /) -> None:
        ...

    @property
    def dims_to_restore_elz(self) -> list[int]:
        """
        dimensions whose V columns are normalized to ELZ form after a clearing reduce
        """

    @dims_to_restore_elz.setter
    def dims_to_restore_elz(self, arg: Sequence[int], /) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: ReductionParamsAdvanced, /) -> bool:
        ...

    def __ne__(self, arg: ReductionParamsAdvanced, /) -> bool:
        ...

    def __getstate__(self) -> tuple[int, int, list[int]]:
        ...

    def __setstate__(self, arg: tuple[int, int, Sequence[int]], /) -> None:
        ...

class ReductionParams:
    """
    Input recipe for a persistence reduction: basic knobs (n_threads, use_clearing, compute_v, compute_u, use_apparent_pairs, sanity_check, verbose) plus rarely-tuned ones under .advanced. Pure input: timings live on the decomposition (dcmp.timings), so a recipe stays equal to itself after a run and can be shared between reductions.
    """

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, n_threads: int=1, chunk_size: int=128, use_clearing: bool=True, compute_v: bool=False, compute_u: bool=False, dims_to_restore_elz: Sequence[int]=[], col_repr: ColumnRepr=ColumnRepr.BitTree, verbose: bool=False, use_apparent_pairs: bool | None=None) -> None:
        ...

    @property
    def n_threads(self) -> int:
        """
        Number of worker threads for reduction (1 runs the serial in-place path).
        """

    @n_threads.setter
    def n_threads(self, arg: int, /) -> None:
        ...

    @property
    def use_clearing(self) -> bool:
        """
        Enable Chen-Kerber clearing optimization (skips positive columns killed in dimension d+1).
        """

    @use_clearing.setter
    def use_clearing(self, arg: bool, /) -> None:
        ...

    @property
    def compute_v(self) -> bool:
        """Compute and retain the reduction matrix V."""

    @compute_v.setter
    def compute_v(self, arg: bool, /) -> None:
        ...

    @property
    def compute_u(self) -> bool:
        """
        Compute a complete U. With n_threads > 1 this also computes and retains the matching V.
        """

    @compute_u.setter
    def compute_u(self, arg: bool, /) -> None:
        ...

    @property
    def use_apparent_pairs(self) -> object:
        """
        Apparent-pairs optimization, tri-state: None (default) = auto, on only where measured as a pure win (parallel diagram-only homology of a complete cubical grid); True = force on wherever supported; False = off.
        """

    @use_apparent_pairs.setter
    def use_apparent_pairs(self, value: bool | None) -> None:
        ...

    @property
    def sanity_check(self) -> bool:
        """
        Verify factorization invariants (R = D V, reducedness) after reduction.
        """

    @sanity_check.setter
    def sanity_check(self, arg: bool, /) -> None:
        ...

    @property
    def advanced(self) -> ReductionParamsAdvanced:
        """
        Nested ReductionParamsAdvanced tuning options (chunk_size, col_repr, dims_to_restore_elz).
        """

    @advanced.setter
    def advanced(self, arg: ReductionParamsAdvanced, /) -> None:
        ...

    @property
    def clearing_opt(self) -> bool:
        """Deprecated alias for use_clearing."""

    @clearing_opt.setter
    def clearing_opt(self, arg: bool, /) -> None:
        ...

    @property
    def apparent_opt(self) -> object:
        """Deprecated alias for use_apparent_pairs."""

    @apparent_opt.setter
    def apparent_opt(self, value: bool | None) -> None:
        ...

    @property
    def do_sanity_check(self) -> bool:
        """Deprecated alias for sanity_check."""

    @do_sanity_check.setter
    def do_sanity_check(self, arg: bool, /) -> None:
        ...

    @property
    def elapsed(self) -> float:
        ...

    @property
    def timings(self) -> ReductionTimings:
        ...

    @property
    def verbose(self) -> bool:
        ...

    @verbose.setter
    def verbose(self, arg: bool, /) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: ReductionParams, /) -> bool:
        ...

    def __ne__(self, arg: ReductionParams, /) -> bool:
        ...

    def __getstate__(self) -> tuple[int, bool, bool, bool, bool, bool, int, ReductionParamsAdvanced]:
        ...

    def __setstate__(self, arg: tuple[int, bool, bool, bool, bool, bool, int, ReductionParamsAdvanced], /) -> None:
        ...

class KICRParams:
    """
    Configuration parameters for kernel, image, and cokernel persistence reduction.
    """

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, codomain: bool=False, kernel: bool=True, image: bool=True, cokernel: bool=True, include_zero_persistence: bool=False, verbose: bool=False, sanity_check: bool=False, n_threads: int=1, params_f: ReductionParams=..., params_g: ReductionParams=..., params_ker: ReductionParams=..., params_im: ReductionParams=..., params_cok: ReductionParams=...) -> None:
        ...

    @property
    def codomain(self) -> bool:
        """Also extract persistence diagrams of the codomain K."""

    @codomain.setter
    def codomain(self, arg: bool, /) -> None:
        ...

    @property
    def kernel(self) -> bool:
        """Compute kernel persistence diagrams."""

    @kernel.setter
    def kernel(self, arg: bool, /) -> None:
        ...

    @property
    def image(self) -> bool:
        """Compute image persistence diagrams."""

    @image.setter
    def image(self, arg: bool, /) -> None:
        ...

    @property
    def cokernel(self) -> bool:
        """Compute cokernel persistence diagrams."""

    @cokernel.setter
    def cokernel(self, arg: bool, /) -> None:
        ...

    @property
    def include_zero_persistence(self) -> bool:
        """Keep zero-persistence bars (birth == death) in extracted diagrams."""

    @include_zero_persistence.setter
    def include_zero_persistence(self, arg: bool, /) -> None:
        ...

    @property
    def verbose(self) -> bool:
        """Print per-stage progress and timing diagnostics."""

    @verbose.setter
    def verbose(self, arg: bool, /) -> None:
        ...

    @property
    def sanity_check(self) -> bool:
        """Run internal matrix consistency checks."""

    @sanity_check.setter
    def sanity_check(self, arg: bool, /) -> None:
        ...

    @property
    def n_threads(self) -> int:
        """Default thread count propagated to sub-reductions."""

    @n_threads.setter
    def n_threads(self, arg: int, /) -> None:
        ...

    @property
    def params_f(self) -> ReductionParams:
        """ReductionParams override for codomain K."""

    @params_f.setter
    def params_f(self, arg: ReductionParams, /) -> None:
        ...

    @property
    def params_g(self) -> ReductionParams:
        """ReductionParams override for domain L."""

    @params_g.setter
    def params_g(self, arg: ReductionParams, /) -> None:
        ...

    @property
    def params_ker(self) -> ReductionParams:
        """ReductionParams override for the kernel matrix."""

    @params_ker.setter
    def params_ker(self, arg: ReductionParams, /) -> None:
        ...

    @property
    def params_im(self) -> ReductionParams:
        """ReductionParams override for the image matrix."""

    @params_im.setter
    def params_im(self, arg: ReductionParams, /) -> None:
        ...

    @property
    def params_cok(self) -> ReductionParams:
        """ReductionParams override for the cokernel matrix."""

    @params_cok.setter
    def params_cok(self, arg: ReductionParams, /) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: KICRParams, /) -> bool:
        ...

    def __ne__(self, arg: KICRParams, /) -> bool:
        ...

    def __getstate__(self) -> tuple[bool, bool, bool, bool, bool, bool, bool, int, ReductionParams, ReductionParams, ReductionParams, ReductionParams, ReductionParams]:
        ...

    def __setstate__(self, arg: tuple[bool, bool, bool, bool, bool, bool, bool, int, ReductionParams, ReductionParams, ReductionParams, ReductionParams, ReductionParams], /) -> None:
        ...

class DecompositionManipStats:
    """
    Profiling record for decomposition-manipulation methods: per-phase wall-clock times plus column-operation counts (the headline metric of the vineyards / move-schedule / warm-start papers) and fill-in (nnz) before/after.
    """

    def __init__(self) -> None:
        ...

    @property
    def elapsed_total(self) -> float:
        ...

    @elapsed_total.setter
    def elapsed_total(self, arg: float, /) -> None:
        ...

    @property
    def elapsed_schedule_build(self) -> float:
        ...

    @elapsed_schedule_build.setter
    def elapsed_schedule_build(self, arg: float, /) -> None:
        ...

    @property
    def elapsed_transpose(self) -> float:
        ...

    @elapsed_transpose.setter
    def elapsed_transpose(self, arg: float, /) -> None:
        ...

    @property
    def elapsed_move(self) -> float:
        ...

    @elapsed_move.setter
    def elapsed_move(self, arg: float, /) -> None:
        ...

    @property
    def elapsed_permute(self) -> float:
        ...

    @elapsed_permute.setter
    def elapsed_permute(self, arg: float, /) -> None:
        ...

    @property
    def elapsed_rereduce(self) -> float:
        ...

    @elapsed_rereduce.setter
    def elapsed_rereduce(self, arg: float, /) -> None:
        ...

    @property
    def elapsed_resize(self) -> float:
        ...

    @elapsed_resize.setter
    def elapsed_resize(self, arg: float, /) -> None:
        ...

    @property
    def n_transpositions(self) -> int:
        ...

    @n_transpositions.setter
    def n_transpositions(self, arg: int, /) -> None:
        ...

    @property
    def n_moves(self) -> int:
        ...

    @n_moves.setter
    def n_moves(self, arg: int, /) -> None:
        ...

    @property
    def n_column_additions_r(self) -> int:
        ...

    @n_column_additions_r.setter
    def n_column_additions_r(self, arg: int, /) -> None:
        ...

    @property
    def n_column_additions_v(self) -> int:
        ...

    @n_column_additions_v.setter
    def n_column_additions_v(self, arg: int, /) -> None:
        ...

    @property
    def n_queries(self) -> int:
        ...

    @n_queries.setter
    def n_queries(self, arg: int, /) -> None:
        ...

    @property
    def n_columns_scanned(self) -> int:
        ...

    @n_columns_scanned.setter
    def n_columns_scanned(self, arg: int, /) -> None:
        ...

    @property
    def nnz_r_before(self) -> int:
        ...

    @nnz_r_before.setter
    def nnz_r_before(self, arg: int, /) -> None:
        ...

    @property
    def nnz_r_after(self) -> int:
        ...

    @nnz_r_after.setter
    def nnz_r_after(self, arg: int, /) -> None:
        ...

    @property
    def nnz_v_before(self) -> int:
        ...

    @nnz_v_before.setter
    def nnz_v_before(self, arg: int, /) -> None:
        ...

    @property
    def nnz_v_after(self) -> int:
        ...

    @nnz_v_after.setter
    def nnz_v_after(self, arg: int, /) -> None:
        ...

    def n_column_additions(self) -> int:
        """
        Total column additions across R and V (n_column_additions_r + n_column_additions_v).
        """

    def reset(self) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> tuple[float, float, float, float, float, float, float, int, int, int, int, int, int, int, int, int, int]:
        ...

    def __setstate__(self, arg: tuple[float, float, float, float, float, float, float, int, int, int, int, int, int, int, int, int, int], /) -> None:
        ...

class CombinatorialSimplex:
    """
    Combinatorial simplex (sorted list of distinct non-negative vertex ids, without a filtration value).
    """

    @overload
    def __init__(self, vertices: Sequence[int]) -> None:
        ...

    @overload
    def __init__(self, id: int, vertices: Sequence[int]) -> None:
        ...

    def __iter__(self) -> Iterator[int]:
        ...

    def __getitem__(self, i: int) -> int:
        """Return the i-th vertex id in sorted order."""

    @property
    def id(self) -> int:
        """User-assigned cell id."""

    @id.setter
    def id(self, arg: int, /) -> None:
        ...

    @property
    def vertices(self) -> list[int]:
        """Sorted list of vertex indices."""

    @property
    def uid(self) -> int:
        """Deterministic combinatorial uid uniquely identifying the vertex set."""

    @property
    def dim(self) -> int:
        """Topological dimension (len(vertices) - 1)."""

    def boundary(self) -> list[int]:
        """Return codimension-1 boundary face uids."""

    def join(self, new_vertex: int, new_id: int=-1) -> CombinatorialSimplex:
        """Return a new CombinatorialSimplex formed by adjoining new_vertex."""

    def __eq__(self, arg: CombinatorialSimplex, /) -> bool:
        ...

    def __ne__(self, arg: CombinatorialSimplex, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[int, int, list[int]]:
        ...

    def __setstate__(self, arg: tuple[int, int, Sequence[int]], /) -> None:
        ...

    def __repr__(self) -> str:
        ...

class CombinatorialProdSimplex:
    """
    Combinatorial Cartesian product sigma x tau of two simplices (without a filtration value).
    """

    def __init__(self, vertices_1: Sequence[int], vertices_2: Sequence[int]) -> None:
        ...

    @property
    def id(self) -> int:
        """User-assigned cell id."""

    @id.setter
    def id(self, arg: int, /) -> None:
        ...

    @property
    def factor_1(self) -> CombinatorialSimplex:
        """First simplex factor."""

    @property
    def factor_2(self) -> CombinatorialSimplex:
        """Second simplex factor."""

    @property
    def uid(self) -> tuple[int, int]:
        """Combinatorial uid of the product cell."""

    @property
    def dim(self) -> int:
        """Topological dimension (factor_1.dim + factor_2.dim)."""

    def boundary(self) -> list[tuple[int, int]]:
        """Return codimension-1 boundary face uids."""

    def __eq__(self, arg: CombinatorialProdSimplex, /) -> bool:
        ...

    def __ne__(self, arg: CombinatorialProdSimplex, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[int, CombinatorialSimplex, CombinatorialSimplex]:
        ...

    def __setstate__(self, arg: tuple[int, CombinatorialSimplex, CombinatorialSimplex], /) -> None:
        ...

    def __repr__(self) -> str:
        ...

class Simplex:

    @overload
    def __init__(self, vertices: Sequence[int], value: float=0.0) -> None:
        ...

    @overload
    def __init__(self, id: int, vertices: Sequence[int], value: float) -> None:
        ...

    def __iter__(self) -> Iterator[int]:
        ...

    def __getitem__(self, arg: int, /) -> int:
        ...

    @property
    def id(self) -> int:
        """User-assigned cell id."""

    @id.setter
    def id(self, arg: int, /) -> None:
        ...

    @property
    def sorted_id(self) -> int:
        """Position of this simplex in sorted filtration order."""

    @sorted_id.setter
    def sorted_id(self, arg: int, /) -> None:
        ...

    @property
    def vertices(self) -> list[int]:
        """Sorted list of vertex indices."""

    @property
    def uid(self) -> int:
        """Deterministic combinatorial uid of the underlying simplex."""

    @property
    def value(self) -> float:
        """Filtration value assigned to this simplex."""

    @value.setter
    def value(self, arg: float, /) -> None:
        ...

    @property
    def dim(self) -> int:
        """Topological dimension (len(vertices) - 1)."""

    def boundary(self) -> list[int]:
        """Return codimension-1 boundary face uids."""

    @property
    def combinatorial_simplex(self) -> CombinatorialSimplex:
        """Underlying value-free CombinatorialSimplex."""

    @property
    def combinatorial_cell(self) -> CombinatorialSimplex:
        """Underlying value-free CombinatorialSimplex."""

    def join(self, new_vertex: int, value: float, new_id: int=-1) -> Simplex:
        """
        Return a new valued Simplex formed by adjoining new_vertex with filtration value.
        """

    def __eq__(self, arg: Simplex, /) -> bool:
        ...

    def __ne__(self, arg: Simplex, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[float, int, CombinatorialSimplex]:
        ...

    def __setstate__(self, arg: tuple[float, int, CombinatorialSimplex], /) -> None:
        ...

    def __repr__(self) -> str:
        ...

class ProdSimplex:

    @overload
    def __init__(self, vertices_1: Sequence[int], vertices_2: Sequence[int], value: float) -> None:
        ...

    @overload
    def __init__(self, cell_1: Simplex, cell_2: Simplex, value: float) -> None:
        ...

    @overload
    def __init__(self, cell_1: CombinatorialSimplex, cell_2: CombinatorialSimplex, value: float) -> None:
        ...

    @property
    def id(self) -> int:
        """User-assigned cell id."""

    @id.setter
    def id(self, arg: int, /) -> None:
        ...

    @property
    def sorted_id(self) -> int:
        """Position of this product cell in sorted filtration order."""

    @sorted_id.setter
    def sorted_id(self, arg: int, /) -> None:
        ...

    @property
    def factor_1(self) -> CombinatorialSimplex:
        """First CombinatorialSimplex factor."""

    @property
    def factor_2(self) -> CombinatorialSimplex:
        """Second CombinatorialSimplex factor."""

    @property
    def cell_1(self) -> CombinatorialSimplex:
        """Alias for factor_1."""

    @property
    def cell_2(self) -> CombinatorialSimplex:
        """Alias for factor_2."""

    @property
    def uid(self) -> tuple[int, int]:
        """Combinatorial uid of the product cell."""

    @property
    def value(self) -> float:
        """Filtration value assigned to this product cell."""

    @value.setter
    def value(self, arg: float, /) -> None:
        ...

    @property
    def dim(self) -> int:
        """Topological dimension (factor_1.dim + factor_2.dim)."""

    def boundary(self) -> list[tuple[int, int]]:
        """Return codimension-1 boundary face uids."""

    @property
    def combinatorial_cell(self) -> CombinatorialProdSimplex:
        """Underlying value-free CombinatorialProdSimplex."""

    def __eq__(self, arg: ProdSimplex, /) -> bool:
        ...

    def __ne__(self, arg: ProdSimplex, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[float, int, CombinatorialProdSimplex]:
        ...

    def __setstate__(self, arg: tuple[float, int, CombinatorialProdSimplex], /) -> None:
        ...

    def __repr__(self) -> str:
        ...

class GridDomain_1D:
    """1D grid domain"""

    def __init__(self, *args) -> None:
        ...

    @property
    def shape(self) -> list[int]:
        ...

    def __eq__(self, arg: GridDomain_1D, /) -> bool:
        ...

    def __ne__(self, arg: GridDomain_1D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[list[int], list[int], bool, bool]:
        ...

    def __setstate__(self, arg: tuple[Sequence[int], Sequence[int], bool, bool], /) -> None:
        ...

class GridDomain_2D:
    """2D grid domain"""

    def __init__(self, *args) -> None:
        ...

    @property
    def shape(self) -> list[int]:
        ...

    def __eq__(self, arg: GridDomain_2D, /) -> bool:
        ...

    def __ne__(self, arg: GridDomain_2D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[list[int], list[int], bool, bool]:
        ...

    def __setstate__(self, arg: tuple[Sequence[int], Sequence[int], bool, bool], /) -> None:
        ...

class GridDomain_3D:
    """3D grid domain"""

    def __init__(self, *args) -> None:
        ...

    @property
    def shape(self) -> list[int]:
        ...

    def __eq__(self, arg: GridDomain_3D, /) -> bool:
        ...

    def __ne__(self, arg: GridDomain_3D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[list[int], list[int], bool, bool]:
        ...

    def __setstate__(self, arg: tuple[Sequence[int], Sequence[int], bool, bool], /) -> None:
        ...

class GridDomain_4D:
    """4D grid domain"""

    def __init__(self, *args) -> None:
        ...

    @property
    def shape(self) -> list[int]:
        ...

    def __eq__(self, arg: GridDomain_4D, /) -> bool:
        ...

    def __ne__(self, arg: GridDomain_4D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[list[int], list[int], bool, bool]:
        ...

    def __setstate__(self, arg: tuple[Sequence[int], Sequence[int], bool, bool], /) -> None:
        ...

class CombinatorialCube_1D:

    @overload
    def __init__(self, domain: GridDomain_1D, x: int) -> None:
        ...

    @overload
    def __init__(self, anchor_vertex: Sequence[int], spanning_dims: Sequence[int], domain: GridDomain_1D) -> None:
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def uid(self) -> int:
        """Get UID of a cube"""

    @property
    def vertices(self) -> list[list[int]]:
        """Get all vertices of a cube"""

    @property
    def anchor_vertex(self) -> list[int]:
        """Get anchor vertex of a cube"""

    @property
    def id(self) -> int:
        """User ID of a cube"""

    @id.setter
    def id(self, arg: int, /) -> None:
        ...

    @property
    def domain(self) -> GridDomain_1D:
        ...

    def boundary(self) -> list[CombinatorialCube_1D]:
        """boundary of a cube"""

    def coboundary(self) -> list[CombinatorialCube_1D]:
        """coboundary of a cube"""

    def top_cofaces(self) -> list[CombinatorialCube_1D]:
        """top cofaces of a cube"""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, arg: CombinatorialCube_1D, /) -> bool:
        ...

    def __ne__(self, arg: CombinatorialCube_1D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[int, int, GridDomain_1D]:
        ...

    def __setstate__(self, arg: tuple[int, int, GridDomain_1D], /) -> None:
        ...

class CombinatorialCube_2D:

    @overload
    def __init__(self, domain: GridDomain_2D, x: int) -> None:
        ...

    @overload
    def __init__(self, anchor_vertex: Sequence[int], spanning_dims: Sequence[int], domain: GridDomain_2D) -> None:
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def uid(self) -> int:
        """Get UID of a cube"""

    @property
    def vertices(self) -> list[list[int]]:
        """Get all vertices of a cube"""

    @property
    def anchor_vertex(self) -> list[int]:
        """Get anchor vertex of a cube"""

    @property
    def id(self) -> int:
        """User ID of a cube"""

    @id.setter
    def id(self, arg: int, /) -> None:
        ...

    @property
    def domain(self) -> GridDomain_2D:
        ...

    def boundary(self) -> list[CombinatorialCube_2D]:
        """boundary of a cube"""

    def coboundary(self) -> list[CombinatorialCube_2D]:
        """coboundary of a cube"""

    def top_cofaces(self) -> list[CombinatorialCube_2D]:
        """top cofaces of a cube"""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, arg: CombinatorialCube_2D, /) -> bool:
        ...

    def __ne__(self, arg: CombinatorialCube_2D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[int, int, GridDomain_2D]:
        ...

    def __setstate__(self, arg: tuple[int, int, GridDomain_2D], /) -> None:
        ...

class CombinatorialCube_3D:

    @overload
    def __init__(self, domain: GridDomain_3D, x: int) -> None:
        ...

    @overload
    def __init__(self, anchor_vertex: Sequence[int], spanning_dims: Sequence[int], domain: GridDomain_3D) -> None:
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def uid(self) -> int:
        """Get UID of a cube"""

    @property
    def vertices(self) -> list[list[int]]:
        """Get all vertices of a cube"""

    @property
    def anchor_vertex(self) -> list[int]:
        """Get anchor vertex of a cube"""

    @property
    def id(self) -> int:
        """User ID of a cube"""

    @id.setter
    def id(self, arg: int, /) -> None:
        ...

    @property
    def domain(self) -> GridDomain_3D:
        ...

    def boundary(self) -> list[CombinatorialCube_3D]:
        """boundary of a cube"""

    def coboundary(self) -> list[CombinatorialCube_3D]:
        """coboundary of a cube"""

    def top_cofaces(self) -> list[CombinatorialCube_3D]:
        """top cofaces of a cube"""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, arg: CombinatorialCube_3D, /) -> bool:
        ...

    def __ne__(self, arg: CombinatorialCube_3D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[int, int, GridDomain_3D]:
        ...

    def __setstate__(self, arg: tuple[int, int, GridDomain_3D], /) -> None:
        ...

class CombinatorialCube_4D:

    @overload
    def __init__(self, domain: GridDomain_4D, x: int) -> None:
        ...

    @overload
    def __init__(self, anchor_vertex: Sequence[int], spanning_dims: Sequence[int], domain: GridDomain_4D) -> None:
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def uid(self) -> int:
        """Get UID of a cube"""

    @property
    def vertices(self) -> list[list[int]]:
        """Get all vertices of a cube"""

    @property
    def anchor_vertex(self) -> list[int]:
        """Get anchor vertex of a cube"""

    @property
    def id(self) -> int:
        """User ID of a cube"""

    @id.setter
    def id(self, arg: int, /) -> None:
        ...

    @property
    def domain(self) -> GridDomain_4D:
        ...

    def boundary(self) -> list[CombinatorialCube_4D]:
        """boundary of a cube"""

    def coboundary(self) -> list[CombinatorialCube_4D]:
        """coboundary of a cube"""

    def top_cofaces(self) -> list[CombinatorialCube_4D]:
        """top cofaces of a cube"""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, arg: CombinatorialCube_4D, /) -> bool:
        ...

    def __ne__(self, arg: CombinatorialCube_4D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[int, int, GridDomain_4D]:
        ...

    def __setstate__(self, arg: tuple[int, int, GridDomain_4D], /) -> None:
        ...

class Cube_1D:

    @overload
    def __init__(self, domain: GridDomain_1D, x: int, value: float) -> None:
        ...

    @overload
    def __init__(self, anchor_vertex: Sequence[int], spanning_dims: Sequence[int], domain: GridDomain_1D, value: float) -> None:
        ...

    @property
    def dim(self) -> int:
        """Topological dimension of the cube."""

    @property
    def uid(self) -> int:
        """Get UID of a cube"""

    @property
    def value(self) -> float:
        """Filtration value assigned to this cube."""

    @value.setter
    def value(self, arg: float, /) -> None:
        ...

    @property
    def vertices(self) -> list[list[int]]:
        """Get all vertices of a cube"""

    def boundary(self) -> list[int]:
        """boundary of a cube"""

    def coboundary(self) -> list[CombinatorialCube_1D]:
        """coboundary of a cube"""

    def top_cofaces(self) -> list[CombinatorialCube_1D]:
        """top-dim cofaces of a cube"""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, arg: Cube_1D, /) -> bool:
        ...

    def __ne__(self, arg: Cube_1D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[float, int, CombinatorialCube_1D]:
        ...

    def __setstate__(self, arg: tuple[float, int, CombinatorialCube_1D], /) -> None:
        ...

class Cube_2D:

    @overload
    def __init__(self, domain: GridDomain_2D, x: int, value: float) -> None:
        ...

    @overload
    def __init__(self, anchor_vertex: Sequence[int], spanning_dims: Sequence[int], domain: GridDomain_2D, value: float) -> None:
        ...

    @property
    def dim(self) -> int:
        """Topological dimension of the cube."""

    @property
    def uid(self) -> int:
        """Get UID of a cube"""

    @property
    def value(self) -> float:
        """Filtration value assigned to this cube."""

    @value.setter
    def value(self, arg: float, /) -> None:
        ...

    @property
    def vertices(self) -> list[list[int]]:
        """Get all vertices of a cube"""

    def boundary(self) -> list[int]:
        """boundary of a cube"""

    def coboundary(self) -> list[CombinatorialCube_2D]:
        """coboundary of a cube"""

    def top_cofaces(self) -> list[CombinatorialCube_2D]:
        """top-dim cofaces of a cube"""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, arg: Cube_2D, /) -> bool:
        ...

    def __ne__(self, arg: Cube_2D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[float, int, CombinatorialCube_2D]:
        ...

    def __setstate__(self, arg: tuple[float, int, CombinatorialCube_2D], /) -> None:
        ...

class Cube_3D:

    @overload
    def __init__(self, domain: GridDomain_3D, x: int, value: float) -> None:
        ...

    @overload
    def __init__(self, anchor_vertex: Sequence[int], spanning_dims: Sequence[int], domain: GridDomain_3D, value: float) -> None:
        ...

    @property
    def dim(self) -> int:
        """Topological dimension of the cube."""

    @property
    def uid(self) -> int:
        """Get UID of a cube"""

    @property
    def value(self) -> float:
        """Filtration value assigned to this cube."""

    @value.setter
    def value(self, arg: float, /) -> None:
        ...

    @property
    def vertices(self) -> list[list[int]]:
        """Get all vertices of a cube"""

    def boundary(self) -> list[int]:
        """boundary of a cube"""

    def coboundary(self) -> list[CombinatorialCube_3D]:
        """coboundary of a cube"""

    def top_cofaces(self) -> list[CombinatorialCube_3D]:
        """top-dim cofaces of a cube"""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, arg: Cube_3D, /) -> bool:
        ...

    def __ne__(self, arg: Cube_3D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[float, int, CombinatorialCube_3D]:
        ...

    def __setstate__(self, arg: tuple[float, int, CombinatorialCube_3D], /) -> None:
        ...

class Cube_4D:

    @overload
    def __init__(self, domain: GridDomain_4D, x: int, value: float) -> None:
        ...

    @overload
    def __init__(self, anchor_vertex: Sequence[int], spanning_dims: Sequence[int], domain: GridDomain_4D, value: float) -> None:
        ...

    @property
    def dim(self) -> int:
        """Topological dimension of the cube."""

    @property
    def uid(self) -> int:
        """Get UID of a cube"""

    @property
    def value(self) -> float:
        """Filtration value assigned to this cube."""

    @value.setter
    def value(self, arg: float, /) -> None:
        ...

    @property
    def vertices(self) -> list[list[int]]:
        """Get all vertices of a cube"""

    def boundary(self) -> list[int]:
        """boundary of a cube"""

    def coboundary(self) -> list[CombinatorialCube_4D]:
        """coboundary of a cube"""

    def top_cofaces(self) -> list[CombinatorialCube_4D]:
        """top-dim cofaces of a cube"""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, arg: Cube_4D, /) -> bool:
        ...

    def __ne__(self, arg: Cube_4D, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __getstate__(self) -> tuple[float, int, CombinatorialCube_4D]:
        ...

    def __setstate__(self, arg: tuple[float, int, CombinatorialCube_4D], /) -> None:
        ...

class Grid_1D:
    """1D grid with data"""

    def __init__(self, data: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu')], wrap: bool=False, values_on: str='vertices') -> None:
        ...

    @property
    def data_location(self) -> str:
        """Data location mode ('vertices' or 'cells')."""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def cube_filtration(self, max_dim: int=1, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a slim cubical filtration up to cell dimension max_dim."""

    def cube_filtration_and_critical_indices(self, max_dim: int=1, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a slim cubical filtration and flat array indices of critical vertices/top-cells.
        """

    def freudenthal_filtration(self, max_dim: int=1, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a fat-simplex Freudenthal triangulation filtration."""

    def freudenthal_filtration_and_critical_vertices(self, max_dim: int=1, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a fat-simplex Freudenthal filtration and critical vertex indices.
        """

    def freudenthal_filtration_slim(self, max_dim: int=1, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a compact slim-encoded Freudenthal filtration."""

    def freudenthal_filtration_and_critical_vertices_slim(self, max_dim: int=1, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a compact slim-encoded Freudenthal filtration and critical vertex indices.
        """

class Grid_2D:
    """2D grid with data"""

    def __init__(self, data: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu')], wrap: bool=False, values_on: str='vertices') -> None:
        ...

    @property
    def data_location(self) -> str:
        """Data location mode ('vertices' or 'cells')."""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def cube_filtration(self, max_dim: int=2, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a slim cubical filtration up to cell dimension max_dim."""

    def cube_filtration_and_critical_indices(self, max_dim: int=2, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a slim cubical filtration and flat array indices of critical vertices/top-cells.
        """

    def freudenthal_filtration(self, max_dim: int=2, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a fat-simplex Freudenthal triangulation filtration."""

    def freudenthal_filtration_and_critical_vertices(self, max_dim: int=2, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a fat-simplex Freudenthal filtration and critical vertex indices.
        """

    def freudenthal_filtration_slim(self, max_dim: int=2, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a compact slim-encoded Freudenthal filtration."""

    def freudenthal_filtration_and_critical_vertices_slim(self, max_dim: int=2, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a compact slim-encoded Freudenthal filtration and critical vertex indices.
        """

class Grid_3D:
    """3D grid with data"""

    def __init__(self, data: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu')], wrap: bool=False, values_on: str='vertices') -> None:
        ...

    @property
    def data_location(self) -> str:
        """Data location mode ('vertices' or 'cells')."""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def cube_filtration(self, max_dim: int=3, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a slim cubical filtration up to cell dimension max_dim."""

    def cube_filtration_and_critical_indices(self, max_dim: int=3, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a slim cubical filtration and flat array indices of critical vertices/top-cells.
        """

    def freudenthal_filtration(self, max_dim: int=3, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a fat-simplex Freudenthal triangulation filtration."""

    def freudenthal_filtration_and_critical_vertices(self, max_dim: int=3, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a fat-simplex Freudenthal filtration and critical vertex indices.
        """

    def freudenthal_filtration_slim(self, max_dim: int=3, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a compact slim-encoded Freudenthal filtration."""

    def freudenthal_filtration_and_critical_vertices_slim(self, max_dim: int=3, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a compact slim-encoded Freudenthal filtration and critical vertex indices.
        """

class Grid_4D:
    """4D grid with data"""

    def __init__(self, data: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu')], wrap: bool=False, values_on: str='vertices') -> None:
        ...

    @property
    def data_location(self) -> str:
        """Data location mode ('vertices' or 'cells')."""

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def cube_filtration(self, max_dim: int=4, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a slim cubical filtration up to cell dimension max_dim."""

    def cube_filtration_and_critical_indices(self, max_dim: int=4, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a slim cubical filtration and flat array indices of critical vertices/top-cells.
        """

    def freudenthal_filtration(self, max_dim: int=4, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a fat-simplex Freudenthal triangulation filtration."""

    def freudenthal_filtration_and_critical_vertices(self, max_dim: int=4, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a fat-simplex Freudenthal filtration and critical vertex indices.
        """

    def freudenthal_filtration_slim(self, max_dim: int=4, negate: bool=False, n_threads: int=1) -> Filtration:
        """Build a compact slim-encoded Freudenthal filtration."""

    def freudenthal_filtration_and_critical_vertices_slim(self, max_dim: int=4, negate: bool=False, n_threads: int=1) -> tuple[Filtration, list[int]]:
        """
        Build a compact slim-encoded Freudenthal filtration and critical vertex indices.
        """

class Filtration:
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

    @overload
    def __init__(self, cells: Sequence[Simplex | ProdSimplex | Cube_1D | Cube_2D | Cube_3D | Cube_4D], negate: bool=False, n_threads: int=1) -> None:
        ...

    @overload
    def __init__(self, vertices_values: Sequence[tuple[Sequence[int], float]], n_threads: int=1, duplicates_possible: bool=False) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Simplex]:
        ...

    def __getitem__(self, arg: int, /) -> Simplex:
        ...

    @property
    def negate(self) -> bool:
        """True when filtration order is descending (superlevel sets)."""

    def infinity(self) -> float:
        """filtration-order +inf: a value strictly later than every cell"""

    def neg_infinity(self) -> float:
        """filtration-order -inf: a value strictly earlier than every cell"""

    def fil_min(self, a: float, b: float) -> float:
        """filtration-order min: the value that enters earlier"""

    def fil_max(self, a: float, b: float) -> float:
        """filtration-order max: the value that enters later"""

    @property
    def max_dim(self) -> int:
        """maximal dimension of a cell in filtration"""

    def cells(self) -> list[Simplex]:
        """copy of all cells in filtration order"""

    def simplices(self) -> list[Simplex]:
        """copy of all simplices (cells) in filtration order"""

    def size(self) -> int:
        """number of cells in filtration"""

    def size_in_dimension(self, dim: int) -> int:
        """number of cells of dimension dim"""

    def n_vertices(self) -> int:
        """Number of 0-dimensional cells (vertices) in the filtration."""

    def simplex_value_by_sorted_id(self, sorted_id: int) -> float:
        """Filtration value of the cell at filtration index sorted_id."""

    def cell_value_by_sorted_id(self, sorted_id: int) -> float:
        """Filtration value of the cell at filtration index sorted_id."""

    def id_by_sorted_id(self, sorted_id: int) -> int:
        """Original user/input cell id corresponding to sorted_id."""

    def sorted_id_by_id(self, id: int) -> int:
        """Filtration index (sorted_id) of the cell with user id."""

    def cell(self, i: int) -> Simplex:
        """Return the valued cell at filtration index i."""

    def simplex(self, i: int) -> Simplex:
        """Return the valued simplex at filtration index i."""

    @property
    def dim_first(self) -> list[int]:
        """
        First sorted_id in each cell dimension (when grouped) or per-dimension bounds.
        """

    @property
    def dim_last(self) -> list[int]:
        """Last sorted_id (inclusive) in each cell dimension."""

    def sorting_permutation(self) -> list[int]:
        """Map from input cell id to filtration sorted_id."""

    def inv_sorting_permutation(self) -> list[int]:
        """Map from filtration sorted_id back to input cell id."""

    def value_by_uid(self, uid: int) -> float:
        """Filtration value of the cell with combinatorial uid."""

    def sorted_id_by_uid(self, uid: int) -> int:
        """Filtration sorted_id of the cell with combinatorial uid."""

    def cell_by_uid(self, uid: int) -> Simplex:
        """Valued cell with the given combinatorial uid."""

    def boundary_matrix(self, n_threads: int=1) -> list[list[int]]:
        """
        Return the Z_2 boundary matrix D as a list of sorted row-index columns.
        """

    def boundary_matrix_in_dimension(self, dim: int, n_threads: int=1) -> list[list[int]]:
        """Return the Z_2 boundary matrix block for cells of dimension dim."""

    def coboundary_matrix(self, n_threads: int=1) -> list[list[int]]:
        """Return the anti-transposed Z_2 coboundary matrix D^perp."""

    def boundary_matrix_rel(self, arg: Set[int], /) -> list[list[int]]:
        """
        Return the relative Z_2 boundary matrix zeroing columns in the given subcomplex.
        """

    def star_closure(self, seed_sorted_ids: Sequence[int], n_threads: int=1) -> list[int]:
        """
        Coface up-closure (union of stars) of the given cells (sorted_ids); pass to Decomposition.remove_simplices / Filtration.without_cells.
        """

    def is_up_closed(self, cells: Sequence[int], n_threads: int=1) -> bool:
        """
        True if the cells (sorted_ids) are closed under cofaces (removable as a filtration).
        """

    def without_cells(self, cells_to_remove: Sequence[int]) -> Filtration:
        """
        Subfiltration with the given cells (sorted_ids) removed; survivors keep order.
        """

    def reset_ids_to_sorted_ids(self) -> None:
        """Overwrite each cell's user id with its current sorted_id."""

    @overload
    def set_values(self, new_values: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], n_threads: int=1) -> None:
        """
        Update cell values (indexed by user id) and re-sort into filtration order.
        """

    @overload
    def set_values(self, new_values: Sequence[float], n_threads: int=1) -> None:
        ...

    def subfiltration(self, predicate: Callable[[CombinatorialSimplex], bool]) -> Filtration:
        """
        Return the subfiltration of cells satisfying predicate(combinatorial_simplex).
        """

    def __eq__(self, arg: Filtration, /) -> bool:
        ...

    def __ne__(self, arg: Filtration, /) -> bool:
        ...

    def get_vertices(self) -> NDArray[numpy.uint64]:
        """
        Return 0-simplices in filtration order as an (n_0, 1) NumPy array of vertex ids.
        """

    def get_edges(self) -> NDArray[numpy.uint64]:
        """
        Return 1-simplices in filtration order as an (n_1, 2) NumPy array of vertex ids.
        """

    def get_triangles(self) -> NDArray[numpy.uint64]:
        """
        Return 2-simplices in filtration order as an (n_2, 3) NumPy array of vertex ids.
        """

    def get_tetrahedra(self) -> NDArray[numpy.uint64]:
        """
        Return 3-simplices in filtration order as an (n_3, 4) NumPy array of vertex ids.
        """

    def get_simplices_as_arr(self, simplex_dim: int) -> NDArray[numpy.uint64]:
        """
        Return k-simplices in filtration order as an (n_k, k+1) NumPy array of vertex ids.
        """

    def __repr__(self) -> str:
        ...

    def to_str_debug(self) -> str:
        """
        Full per-cell dump grouped by dimension. Unbounded -- for debugging small filtrations only; repr() shows a bounded summary.
        """

    @property
    def kind(self) -> FiltrationKind:
        """
        FiltrationKind tag set by the constructor that built this filtration (or User for hand-built ones).
        """

    @kind.setter
    def kind(self, arg: FiltrationKind, /) -> None:
        ...

class ProdFiltration:
    """Marker for product-cell filtrations -- those whose cells are oineus.ProdSimplex
    (ProductCell), e.g. the result of mapping_cylinder or multiply_filtration.

    Use it for membership tests only::

        isinstance(fil, oineus.ProdFiltration)   # True for any product-cell filtration

    It is a cell-type marker, distinct from the filtration's FiltrationKind (which records how
    the filtration was built). It is NOT a constructor; build a product filtration through the
    unified facade::

        oineus.Filtration([oineus.ProdSimplex([0], [0], 0.0), ...])
    """

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[ProdSimplex]:
        ...

    def __getitem__(self, arg: int, /) -> ProdSimplex:
        ...

    @property
    def negate(self) -> bool:
        ...

    def infinity(self) -> float:
        """filtration-order +inf: a value strictly later than every cell"""

    def neg_infinity(self) -> float:
        """filtration-order -inf: a value strictly earlier than every cell"""

    def fil_min(self, a: float, b: float) -> float:
        """filtration-order min: the value that enters earlier"""

    def fil_max(self, a: float, b: float) -> float:
        """filtration-order max: the value that enters later"""

    @property
    def max_dim(self) -> int:
        """maximal dimension of a cell in filtration"""

    def cells(self) -> list[ProdSimplex]:
        """copy of all cells in filtration order"""

    def size(self) -> int:
        """number of cells in filtration"""

    @property
    def dim_first(self) -> list[int]:
        ...

    @property
    def dim_last(self) -> list[int]:
        ...

    def size_in_dimension(self, dim: int) -> int:
        """number of cells of dimension dim"""

    def cell_value_by_sorted_id(self, sorted_id: int) -> float:
        ...

    def get_id_by_sorted_id(self, sorted_id: int) -> int:
        ...

    def get_sorted_id_by_id(self, id: int) -> int:
        ...

    def get_cell(self, i: int) -> ProdSimplex:
        ...

    def get_sorting_permutation(self) -> list[int]:
        ...

    def get_inv_sorting_permutation(self) -> list[int]:
        ...

    def id_by_sorted_id(self, sorted_id: int) -> int:
        ...

    def sorted_id_by_id(self, id: int) -> int:
        ...

    def cell(self, i: int) -> ProdSimplex:
        ...

    def sorting_permutation(self) -> list[int]:
        ...

    def inv_sorting_permutation(self) -> list[int]:
        ...

    def boundary_matrix(self, n_threads: int) -> list[list[int]]:
        ...

    def boundary_matrix_in_dimension(self, dim: int, n_threads: int) -> list[list[int]]:
        ...

    def coboundary_matrix(self, n_threads: int) -> list[list[int]]:
        ...

    def star_closure(self, seed_sorted_ids: Sequence[int], n_threads: int=1) -> list[int]:
        """Coface up-closure (union of stars) of the given cells (sorted_ids)."""

    def is_up_closed(self, cells: Sequence[int], n_threads: int=1) -> bool:
        """True if the cells (sorted_ids) are closed under cofaces."""

    def without_cells(self, cells_to_remove: Sequence[int]) -> ProdFiltration:
        """
        Subfiltration with the given cells (sorted_ids) removed; survivors keep order.
        """

    def reset_ids_to_sorted_ids(self) -> None:
        ...

    def set_values(self, new_values: Sequence[float], n_threads: int=1) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def to_str_debug(self) -> str:
        """
        Full per-cell dump grouped by dimension. Unbounded -- for debugging small filtrations only; repr() shows a bounded summary.
        """

    def __eq__(self, arg: ProdFiltration, /) -> bool:
        ...

    def __ne__(self, arg: ProdFiltration, /) -> bool:
        ...

    @property
    def kind(self) -> FiltrationKind:
        """
        FiltrationKind tag set by the constructor that built this filtration (or User for hand-built ones).
        """

    @kind.setter
    def kind(self, arg: FiltrationKind, /) -> None:
        ...

class DiagramPoint:
    """
    Single persistence-diagram point with birth/death values and critical cell indices.
    """

    def __init__(self, birth: float, death: float) -> None:
        ...

    @property
    def birth(self) -> float:
        """Birth filtration value."""

    @birth.setter
    def birth(self, arg: float, /) -> None:
        ...

    @property
    def death(self) -> float:
        """Death filtration value (+inf or -inf for essential classes)."""

    @death.setter
    def death(self, arg: float, /) -> None:
        ...

    @property
    def birth_index(self) -> int:
        """Filtration sorted_id of the positive (birth) cell."""

    @birth_index.setter
    def birth_index(self, arg: int, /) -> None:
        ...

    @property
    def death_index(self) -> int:
        """Filtration sorted_id of the negative (death) cell."""

    @death_index.setter
    def death_index(self, arg: int, /) -> None:
        ...

    @property
    def persistence(self) -> float:
        """Absolute persistence ``|death - birth|``."""

    @property
    def index_persistence(self) -> int:
        """Index persistence ``|death_index - birth_index|``."""

    def is_diagonal(self) -> bool:
        """Return True if birth == death."""

    def is_inf(self) -> bool:
        """Return True if either coordinate is infinite (essential point)."""

    def __getitem__(self, i: int) -> float:
        """Return birth (i == 0) or death (i == 1)."""

    def __hash__(self) -> int:
        ...

    def __eq__(self, arg: DiagramPoint, /) -> bool:
        ...

    def __ne__(self, arg: DiagramPoint, /) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> tuple[float, float, int, int, int, int, int]:
        ...

    def __setstate__(self, arg: tuple[float, float, int, int, int, int, int], /) -> None:
        ...

class IndexDiagramPoint:
    """Persistence pairing point expressed in filtration sorted_id indices."""

    def __init__(self, birth: int, death: int) -> None:
        ...

    @property
    def birth(self) -> int:
        """Filtration sorted_id of the birth cell."""

    @birth.setter
    def birth(self, arg: int, /) -> None:
        ...

    @property
    def death(self) -> int:
        """Filtration sorted_id of the death cell."""

    @death.setter
    def death(self, arg: int, /) -> None:
        ...

    @property
    def persistence(self) -> int:
        """Index persistence ``|death - birth|``."""

    def __getitem__(self, i: int) -> int:
        """Return birth index (i == 0) or death index (i == 1)."""

    def __hash__(self) -> int:
        ...

    def __eq__(self, arg: IndexDiagramPoint, /) -> bool:
        ...

    def __ne__(self, arg: IndexDiagramPoint, /) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __getstate__(self) -> tuple[int, int, int, int, int, int, int]:
        ...

    def __setstate__(self, arg: tuple[int, int, int, int, int, int, int], /) -> None:
        ...

class Diagrams:
    """
    Container of persistence diagrams across homology dimensions 0..n_dims-1.
    """

    def __init__(self, max_dim: int) -> None:
        ...

    def in_dimension(self, dim: int, as_numpy: bool=True) -> NDArray[numpy.float64] | list[DiagramPoint]:
        """
        Return the persistence diagram in dimension dim: as an (n, 2) NumPy array when as_numpy is True (the default), or as a list of DiagramPoint objects (carrying birth_index and death_index) when as_numpy is False.
        """

    def index_diagram_in_dimension(self, dim: int, as_numpy: bool=True) -> NDArray[numpy.uint64] | list[IndexDiagramPoint]:
        """
        Return the persistence index pairing (including zero-persistence pairs) in dimension dim: as an (n, 2) size_t NumPy array of sorted_ids when as_numpy is True (the default), or as a list of IndexDiagramPoint objects when False.
        """

    def __getitem__(self, dim: int) -> NDArray[numpy.float64]:
        """Shorthand for in_dimension(dim, as_numpy=True)."""

    def __len__(self) -> int:
        """Number of stored homology dimensions."""

    def pad_to_dim(self, new_top_dim: int) -> None:
        """
        Extend diagrams to dimensions [0..new_top_dim] by appending empty diagrams as needed.
        """

    def trim_to_dim(self, max_dim: int) -> None:
        """Trim diagrams to dimensions [0..max_dim]."""

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: Diagrams, /) -> bool:
        ...

    def __ne__(self, arg: Diagrams, /) -> bool:
        ...

    def __getstate__(self) -> tuple[list[list[DiagramPoint]]]:
        ...

    def __setstate__(self, arg: tuple[Sequence[Sequence[DiagramPoint]]], /) -> None:
        ...

class Decomposition:
    """
    Z_2 matrix factorization R = D V (and U = V^{-1}) for persistent homology or cohomology, supporting parallel reduction, representative extraction, partial-U recovery, vineyards/warm-start updates, and SiRUP cell removal.
    """

    @overload
    def __init__(self, filtration: Filtration, dualize: bool=False, n_threads: int=4) -> None:
        """
        Build boundary (dualize=False) or coboundary (dualize=True) matrix from filtration.
        """

    @overload
    def __init__(self, filtration: ProdFiltration, dualize: bool=False, n_threads: int=4) -> None:
        """
        Build boundary (dualize=False) or coboundary (dualize=True) matrix from filtration.
        """

    @overload
    def __init__(self, d: Sequence[Sequence[int]], n_rows: int, dualize: bool=False, skip_check: bool=False) -> None:
        """Construct directly from a column-wise Z_2 boundary matrix d."""

    def clone(self) -> Decomposition:
        """Return a fast C++ deep copy of this Decomposition."""

    @property
    def r_data(self) -> list[list[int]]:
        """Reduced matrix R as a list of sorted row-index columns."""

    @r_data.setter
    def r_data(self, arg: Sequence[Sequence[int]], /) -> None:
        ...

    @property
    def v_data(self) -> list[list[int]]:
        """
        Change-of-basis matrix V (R = D V) as a list of sorted row-index columns.
        """

    @v_data.setter
    def v_data(self, arg: Sequence[Sequence[int]], /) -> None:
        ...

    @property
    def u_data_t(self) -> list[list[int]]:
        """
        Inverse matrix U = V^{-1} stored row-wise (list of sorted column-index rows).
        """

    @u_data_t.setter
    def u_data_t(self, arg: Sequence[Sequence[int]], /) -> None:
        ...

    @property
    def timings(self) -> ReductionTimings:
        """
        Per-phase wall-clock breakdown of the last reduce() call (ReductionTimings).
        """

    @property
    def u_timings(self) -> UComputeTimings:
        """
        Per-phase wall-clock breakdown of the last compute_u_* call (UComputeTimings).
        """

    @property
    def restore_thread_times(self) -> list[float]:
        """
        Diagnostic: per-worker wall-clock (seconds) of the last parallel ELZ-restore pass, to measure load imbalance of the equal-column-count split.
        """

    @property
    def d_data(self) -> list[list[int]]:
        """Input boundary/coboundary matrix D as column lists."""

    @property
    def is_reduced(self) -> bool:
        """True once reduce() has completed."""

    def has_matrix_v(self) -> bool:
        """True if matrix V is stored."""

    def has_matrix_u(self) -> bool:
        """True if any U row storage is allocated."""

    def has_full_matrix_u(self) -> bool:
        """True if a complete U = V^{-1} matrix is stored."""

    @property
    def factorization_valid(self) -> bool:
        """True if R = D V invariant holds."""

    def negative_v_elz_in_dim(self, dim: int) -> bool:
        """True if negative V columns in dimension dim are in canonical ELZ form."""

    def u_row(self, row: int) -> list[int]:
        """
        Raw internal U row storage. Unless has_full_matrix_u() is true, its presence does not imply a row of the complete inverse.
        """

    @property
    def n_computed_u_rows(self) -> int:
        """Number of U rows computed so far."""

    @property
    def n_valid_u_rows(self) -> int:
        """Number of currently valid U rows."""

    @property
    def lazy_restore_elz_time(self) -> float:
        """Wall-clock time (seconds) spent in lazy ELZ restoration."""

    def n_apparent_pairs(self) -> int:
        """
        Number of apparent pairs detected by the lean (use_apparent_pairs) reduction; 0 if the apparent path was not taken or the lean state was already dropped by a materializing access.
        """

    def r_as_csc(self) -> scipy.sparse.csc_matrix[int]:
        """Export reduced matrix R as a scipy.sparse.csc_matrix."""

    def v_as_csc(self) -> scipy.sparse.csc_matrix[int]:
        """Export change-of-basis matrix V as a scipy.sparse.csc_matrix."""

    def d_as_csc(self) -> scipy.sparse.csc_matrix[int]:
        """Export boundary/coboundary matrix D as a scipy.sparse.csc_matrix."""

    def u_as_csr(self) -> scipy.sparse.csr_matrix[int]:
        """Export complete inverse matrix U as a scipy.sparse.csr_matrix."""

    @property
    def dualize(self) -> bool:
        """
        True if this decomposition represents coboundary (cohomology) reduction.
        """

    @property
    def dim_first(self) -> list[int]:
        """First matrix column index per dimension."""

    @property
    def dim_last(self) -> list[int]:
        """Last matrix column index (inclusive) per dimension."""

    def make_dynamic(self, n_threads: int=1) -> None:
        """
        Build the row-incidence indices (in parallel) so subsequent manipulations cost O(changed support); optional, built on first use otherwise. Dropped by reduce().
        """

    def is_dynamic(self) -> bool:
        """True if dynamic row-incidence indices are currently active."""

    def transpose(self, i: int, stats: DecompositionManipStats | None=None) -> None:
        """Vineyards: transpose adjacent filtration positions i and i+1."""

    def transpose_to(self, new_to_old: Sequence[int], stats: DecompositionManipStats | None=None) -> int:
        """
        Vineyards: realize a target order as adjacent transpositions; returns the count.
        """

    def move(self, i: int, j: int, stats: DecompositionManipStats | None=None) -> None:
        """Move the cell at position i to position j."""

    def move_right(self, i: int, j: int, stats: DecompositionManipStats | None=None) -> None:
        """Move the cell at position i rightwards to j >= i."""

    def move_left(self, i: int, j: int, stats: DecompositionManipStats | None=None) -> None:
        """Move the cell at position i leftwards to j <= i."""

    def apply_move_schedule(self, new_to_old: Sequence[int], stats: DecompositionManipStats | None=None) -> int:
        """
        Move schedule: realize a target order as a sequence of moves; returns the move count.
        """

    def update_with_permutation(self, new_to_old: Sequence[int], stats: DecompositionManipStats | None=None) -> None:
        """Luo-Nelson Alg 2: warm-start update under a pure reorder."""

    def update_with_edits(self, new_to_old: Sequence[int], new_boundary: Sequence[Sequence[int]], new_dim_first: Sequence[int], new_dim_last: Sequence[int], stats: DecompositionManipStats | None=None) -> None:
        """
        Luo-Nelson Alg 3: warm-start update with cell insertion/deletion + reorder.
        """

    def remove_simplices(self, cells_to_remove: Sequence[int], stats: DecompositionManipStats | None=None, n_threads: int=1) -> None:
        """
        SiRUP (Giunti-Lazovskis): update the reduced R = D V decomposition in place after removing a coface-closed set of cells (sorted_ids; a union of stars, e.g. from Filtration.star_closure). Updates the barcode and representative cycles minimally; survivors keep filtration order, matching Filtration.without_cells. Homology only.
        """

    def is_reduced_consistent(self) -> bool:
        """
        True if R is reduced and D V == R (does not require V upper-triangular).
        """

    def is_elz(self, n_threads: int=8) -> bool:
        """
        True if V satisfies the canonical Edelsbrunner-Letscher-Zomorodian (ELZ) form.
        """

    def n_elz_violators(self, n_threads: int=8) -> int:
        """Count of V columns violating canonical ELZ form across all dimensions."""

    def n_elz_violators_in_dim(self, dim: int, n_threads: int=8) -> int:
        """Count of V columns violating canonical ELZ form in dimension dim."""

    def is_column_elz(self, column_idx: int) -> int:
        """True if V column column_idx is in canonical ELZ form."""

    def restore_elz(self, dim: int=18446744073709551615, v_only: bool=False, verbose: bool=False, n_threads: int=1) -> None:
        """
        Restore V (and U unless v_only=True) to canonical ELZ form in dimension dim (or all dims by default).
        """

    def compute_u_from_v(self, dim: int=18446744073709551615, n_threads: int=1, verbose: bool=False) -> None:
        """Compute U = V^{-1} from V via column-wise back-substitution."""

    def compute_u_from_v_1(self, dim: int=18446744073709551615, n_threads: int=1, verbose: bool=False) -> None:
        """Compute U = V^{-1} from V using the parallel column solver."""

    def densify_v_for_selinv(self, rows_to_invert: Set[int], n_threads: int=1) -> scipy.sparse.csr_matrix[float]:
        ...

    def densify_v_for_selinv_with_targets(self, filtration: Filtration, rows_to_invert: Sequence[int], targets: Sequence[float]) -> scipy.sparse.csr_matrix[float]:
        ...

    def sanity_check(self, d: Sequence[Sequence[int]]=[]) -> bool:
        """Verify reducedness and R = D V (and R U = D when U is present)."""

    @overload
    def diagram(self, fil: Filtration, include_inf_points: bool=True, n_threads: int=1) -> Diagrams:
        """
        Extract persistence diagrams (excluding zero-persistence pairs) in all dimensions.
        """

    @overload
    def diagram(self, fil: ProdFiltration, include_inf_points: bool=True, n_threads: int=1) -> Diagrams:
        ...

    @overload
    def diagram_serial(self, fil: Filtration, include_inf_points: bool=True) -> Diagrams:
        """
        Serial persistence diagram extractor (reference path matching diagram(..., n_threads=1)).
        """

    @overload
    def diagram_serial(self, fil: ProdFiltration, include_inf_points: bool=True) -> Diagrams:
        ...

    @overload
    def zero_pers_diagram(self, fil: Filtration, n_threads: int=1) -> Diagrams:
        """Extract zero-persistence pairs (birth == death) in all dimensions."""

    @overload
    def zero_pers_diagram(self, fil: ProdFiltration, n_threads: int=1) -> Diagrams:
        ...

    @overload
    def compute_partial_u_rows(self, filtration: Filtration, rows: Sequence[int], bounds: Sequence[float], dim: int, cmp: str='above', n_threads: int=1, verbose: bool=False) -> None:
        """
        Solve selected U rows via V^T U^T = I truncated at target filtration value bounds.
        """

    @overload
    def compute_partial_u_rows(self, filtration: ProdFiltration, rows: Sequence[int], bounds: Sequence[float], dim: int, cmp: str='above', n_threads: int=1, verbose: bool=False) -> None:
        ...

    @overload
    def compute_full_u_rows(self, filtration: Filtration, dim: int, n_threads: int=1, verbose: bool=False) -> None:
        """
        Solve all U rows in dimension dim via parallel row-wise forward substitution.
        """

    @overload
    def compute_full_u_rows(self, filtration: ProdFiltration, dim: int, n_threads: int=1, verbose: bool=False) -> None:
        ...

    def filtration_index(self, matrix_index: int) -> int:
        """
        Convert matrix column/row index to filtration sorted_id (reverses index when dualize is True).
        """

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def to_str_debug(self) -> str:
        """
        Full per-entry dump of the stored D, R, V, U matrices. Unbounded -- for debugging small decompositions only; repr() shows a bounded summary.
        """

    def __eq__(self, arg: Decomposition, /) -> bool:
        ...

    def __ne__(self, arg: Decomposition, /) -> bool:
        ...

    def __getstate__(self) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]], bool, bool, list[int], list[int], list[int], list[int], list[int], dict[int, bool], int, bool, int, ReductionTimings, UComputeTimings, list[float], dict[int, bool], bool, int, int, float]:
        ...

    def __setstate__(self, arg: tuple[Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[Sequence[int]], bool, bool, Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int], Mapping[int, bool], int, bool, int, ReductionTimings, UComputeTimings, Sequence[float], Mapping[int, bool], bool, int, int, float], /) -> None:
        ...

class KerImCokReduced:
    """
    Reduced kernel, image, and cokernel persistence state for an inclusion L -> K.
    """

    def __init__(self, K: Filtration, L: Filtration, params: KICRParams) -> None:
        """Reduce the inclusion L -> K according to KICRParams."""

    def domain_diagrams(self) -> Diagrams:
        """Persistence diagrams of the subcomplex L (domain of the inclusion)."""

    def codomain_diagrams(self) -> Diagrams:
        """
        Persistence diagrams of the full complex K (computed when params.codomain is True).
        """

    def kernel_diagrams(self) -> Diagrams:
        """
        Kernel persistence diagrams of H_*(L) -> H_*(K) (classes born in L and killed in K).
        """

    def cokernel_diagrams(self) -> Diagrams:
        """
        Cokernel persistence diagrams of H_*(L) -> H_*(K) (classes in K modulo the image of L).
        """

    def image_diagrams(self) -> Diagrams:
        """
        Image persistence diagrams of H_*(L) -> H_*(K) (classes born in L and surviving in K).
        """

    def old_order_to_new(self) -> list[int]:
        """
        Map from K filtration sorted_id to reordered inclusion-matrix column index.
        """

    def new_order_to_old(self) -> list[int]:
        """
        Map from reordered inclusion-matrix column index back to K filtration sorted_id.
        """

    def sorted_L_to_sorted_K(self) -> NDArray[numpy.int64]:
        """
        Int64 NumPy array mapping each sorted_id in L to its matching sorted_id in K.
        """

    @property
    def fil_K(self) -> Filtration:
        """Codomain filtration K."""

    @fil_K.setter
    def fil_K(self, arg: Filtration, /) -> None:
        ...

    @property
    def fil_L(self) -> Filtration:
        """Domain (subcomplex) filtration L."""

    @fil_L.setter
    def fil_L(self, arg: Filtration, /) -> None:
        ...

    @property
    def decomposition_f(self) -> Decomposition:
        """Decomposition of K (codomain)."""

    @decomposition_f.setter
    def decomposition_f(self, arg: Decomposition, /) -> None:
        ...

    @property
    def decomposition_g(self) -> Decomposition:
        """Decomposition of L (domain)."""

    @decomposition_g.setter
    def decomposition_g(self, arg: Decomposition, /) -> None:
        ...

    @property
    def decomposition_im(self) -> Decomposition:
        """Decomposition used to extract image persistence."""

    @decomposition_im.setter
    def decomposition_im(self, arg: Decomposition, /) -> None:
        ...

    @property
    def decomposition_ker(self) -> Decomposition:
        """Decomposition used to extract kernel persistence."""

    @decomposition_ker.setter
    def decomposition_ker(self, arg: Decomposition, /) -> None:
        ...

    @property
    def decomposition_cok(self) -> Decomposition:
        """Decomposition used to extract cokernel persistence."""

    @decomposition_cok.setter
    def decomposition_cok(self, arg: Decomposition, /) -> None:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: KerImCokReduced, /) -> bool:
        ...

    def __ne__(self, arg: KerImCokReduced, /) -> bool:
        ...

    def __getstate__(self) -> tuple[Filtration, Filtration, Decomposition, list[list[int]], Decomposition, Decomposition, Decomposition, Decomposition, int, list[list[DiagramPoint]], list[list[DiagramPoint]], list[list[DiagramPoint]], list[list[DiagramPoint]], list[list[DiagramPoint]], list[int], list[int], list[int], list[int], list[int], KICRParams]:
        ...

    def __setstate__(self, arg: tuple[Filtration, Filtration, Decomposition, Sequence[Sequence[int]], Decomposition, Decomposition, Decomposition, Decomposition, int, Sequence[Sequence[DiagramPoint]], Sequence[Sequence[DiagramPoint]], Sequence[Sequence[DiagramPoint]], Sequence[Sequence[DiagramPoint]], Sequence[Sequence[DiagramPoint]], Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int], KICRParams], /) -> None:
        ...

class KerImCokReducedProd:
    """
    Reduced kernel, image, and cokernel persistence state for an inclusion L -> K.
    """

    def __init__(self, K: ProdFiltration, L: ProdFiltration, params: KICRParams) -> None:
        """Reduce the inclusion L -> K according to KICRParams."""

    def domain_diagrams(self) -> Diagrams:
        """Persistence diagrams of the subcomplex L (domain of the inclusion)."""

    def codomain_diagrams(self) -> Diagrams:
        """
        Persistence diagrams of the full complex K (computed when params.codomain is True).
        """

    def kernel_diagrams(self) -> Diagrams:
        """
        Kernel persistence diagrams of H_*(L) -> H_*(K) (classes born in L and killed in K).
        """

    def cokernel_diagrams(self) -> Diagrams:
        """
        Cokernel persistence diagrams of H_*(L) -> H_*(K) (classes in K modulo the image of L).
        """

    def image_diagrams(self) -> Diagrams:
        """
        Image persistence diagrams of H_*(L) -> H_*(K) (classes born in L and surviving in K).
        """

    def old_order_to_new(self) -> list[int]:
        """
        Map from K filtration sorted_id to reordered inclusion-matrix column index.
        """

    def new_order_to_old(self) -> list[int]:
        """
        Map from reordered inclusion-matrix column index back to K filtration sorted_id.
        """

    def sorted_L_to_sorted_K(self) -> NDArray[numpy.int64]:
        """
        Int64 NumPy array mapping each sorted_id in L to its matching sorted_id in K.
        """

    @property
    def fil_K(self) -> ProdFiltration:
        """Codomain filtration K."""

    @fil_K.setter
    def fil_K(self, arg: ProdFiltration, /) -> None:
        ...

    @property
    def fil_L(self) -> ProdFiltration:
        """Domain (subcomplex) filtration L."""

    @fil_L.setter
    def fil_L(self, arg: ProdFiltration, /) -> None:
        ...

    @property
    def decomposition_f(self) -> Decomposition:
        """Decomposition of K (codomain)."""

    @decomposition_f.setter
    def decomposition_f(self, arg: Decomposition, /) -> None:
        ...

    @property
    def decomposition_g(self) -> Decomposition:
        """Decomposition of L (domain)."""

    @decomposition_g.setter
    def decomposition_g(self, arg: Decomposition, /) -> None:
        ...

    @property
    def decomposition_im(self) -> Decomposition:
        """Decomposition used to extract image persistence."""

    @decomposition_im.setter
    def decomposition_im(self, arg: Decomposition, /) -> None:
        ...

    @property
    def decomposition_ker(self) -> Decomposition:
        """Decomposition used to extract kernel persistence."""

    @decomposition_ker.setter
    def decomposition_ker(self, arg: Decomposition, /) -> None:
        ...

    @property
    def decomposition_cok(self) -> Decomposition:
        """Decomposition used to extract cokernel persistence."""

    @decomposition_cok.setter
    def decomposition_cok(self, arg: Decomposition, /) -> None:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: KerImCokReducedProd, /) -> bool:
        ...

    def __ne__(self, arg: KerImCokReducedProd, /) -> bool:
        ...

    def __getstate__(self) -> tuple[ProdFiltration, ProdFiltration, Decomposition, list[list[int]], Decomposition, Decomposition, Decomposition, Decomposition, int, list[list[DiagramPoint]], list[list[DiagramPoint]], list[list[DiagramPoint]], list[list[DiagramPoint]], list[list[DiagramPoint]], list[int], list[int], list[int], list[int], list[int], KICRParams]:
        ...

    def __setstate__(self, arg: tuple[ProdFiltration, ProdFiltration, Decomposition, Sequence[Sequence[int]], Decomposition, Decomposition, Decomposition, Decomposition, int, Sequence[Sequence[DiagramPoint]], Sequence[Sequence[DiagramPoint]], Sequence[Sequence[DiagramPoint]], Sequence[Sequence[DiagramPoint]], Sequence[Sequence[DiagramPoint]], Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int], KICRParams], /) -> None:
        ...

class IndicesValues:
    """
    Pair of (cell sorted_ids, target filtration values) produced by TopologyOptimizer.
    """

    def __getitem__(self, arg: int, /) -> list[int] | list[float]:
        ...

    def indices_array(self) -> NDArray[numpy.int64]:
        """
        Zero-copy numpy view over the indices vector. The view is invalidated by any operation that mutates the underlying IndicesValues; copy via np.array(...) to detach.
        """

    def values_array(self) -> NDArray[numpy.float64]:
        """Zero-copy numpy view over the values vector."""

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: IndicesValues, /) -> bool:
        ...

    def __ne__(self, arg: IndicesValues, /) -> bool:
        ...

    def __getstate__(self) -> tuple[list[int], list[float]]:
        ...

    def __setstate__(self, arg: tuple[Sequence[int], Sequence[float]], /) -> None:
        ...

class IndicesValuesProd:
    """
    Pair of (cell sorted_ids, target filtration values) produced by TopologyOptimizer.
    """

    def __getitem__(self, arg: int, /) -> list[int] | list[float]:
        ...

    def indices_array(self) -> NDArray[numpy.int64]:
        """
        Zero-copy numpy view over the indices vector. The view is invalidated by any operation that mutates the underlying IndicesValues; copy via np.array(...) to detach.
        """

    def values_array(self) -> NDArray[numpy.float64]:
        """Zero-copy numpy view over the values vector."""

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: IndicesValuesProd, /) -> bool:
        ...

    def __ne__(self, arg: IndicesValuesProd, /) -> bool:
        ...

    def __getstate__(self) -> tuple[list[int], list[float]]:
        ...

    def __setstate__(self, arg: tuple[Sequence[int], Sequence[float]], /) -> None:
        ...

class TopologyOptimizer:

    def __init__(self, fil: Filtration, with_crit_sets: bool=True, dims_to_restore_elz: Sequence[int]=[], n_threads: int=1, u_strategy: UStrategy=UStrategy.Auto) -> None:
        """
        Construct a TopologyOptimizer for one autograd backward. with_crit_sets=false sets up the cheapest reduction (R only, no V, no U) for diagram-loss; with_crit_sets=true sets up V with clearing and treats dims_to_restore_elz as an allowlist for lazy negative-V restoration. ensure_has_u_* solves only the selected target rows. u_strategy picks which equation to solve for U; LegacyInBand forces a serial in-band U during the reduction.
        """

    def compute_diagram(self, include_inf_points: bool) -> Diagrams:
        """compute diagrams in all dimensions"""

    def simplify(self, epsilon: float, strategy: DenoiseStrategy, dim: int) -> IndicesValues:
        """make points with persistence less than epsilon go to the diagonal"""

    def get_nth_persistence(self, dim: int, n: int) -> float:
        """return n-th persistence value in d-dimensional persistence diagram"""

    def match(self, template_dgm: Sequence[DiagramPoint], dim: int, q: float=1.0, wasserstein_delta: float=0.01, return_wasserstein_distance: bool=False, dualize: bool=False, internal_p: float=-1.0) -> IndicesValues | tuple[IndicesValues, float]:
        """return target from Wasserstein matching"""

    @property
    def homology_decomposition(self) -> Decomposition:
        """Copy of the internal homology Decomposition."""

    @property
    def cohomology_decomposition(self) -> Decomposition:
        """Copy of the internal cohomology Decomposition."""

    def homology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal homology Decomposition (must be built first).
        """

    def cohomology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal cohomology Decomposition (must be built first).
        """

    @property
    def is_hom_built(self) -> bool:
        """
        True iff the homology Decomposition has been materialized from the cached boundary matrix. Set by the first ensure_hom_built / ensure_hom_reduced call.
        """

    @property
    def is_coh_built(self) -> bool:
        """
        True iff the cohomology Decomposition has been materialized. Counterpart of is_hom_built.
        """

    def singleton(self, idx: int, new_value: float) -> tuple[float, list[int]]:
        """Return critical set for moving cell idx (sorted_id) to new_value."""

    def singletons(self, indices: Sequence[int], values: Sequence[float]) -> list[tuple[float, list[int]]]:
        """Return list of critical sets for each (index, target_value) pair."""

    def reduce_all(self) -> None:
        """Eagerly build and reduce both homology and cohomology decompositions."""

    def increase_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing death to inf"""

    def decrease_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing death to -inf"""

    def increase_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing birth to inf"""

    def decrease_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing birth to -inf"""

    def combine_loss(self, critical_sets: Sequence[tuple[float, Sequence[int]]], strategy: ConflictStrategy) -> IndicesValues:
        """
        combine critical sets into well-defined assignment of new values to indices
        """

    def crit_sets_apply(self, indices: Sequence[int], values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Fused per-pair critical-set walk + conflict resolution in one C++ call. Internally calls ensure_hom_reduced (for the per-pair is_negative dispatch) and ensure_has_u_* on the relevant side when U is needed. Returns IndicesValues; use .indices_array() / .values_array() for zero-copy numpy.
        """

    def crit_sets_apply_typed(self, birth_indices: Sequence[int], birth_values: Sequence[float], death_indices: Sequence[int], death_values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Role-aware critical-set preparation and application for persistence-diagram backward. Only non-empty birth/death target lists cause their corresponding side to be reduced.
        """

    def ensure_hom_built(self) -> None:
        """
        Materialize the homology Decomposition from the cached boundary matrix (no reduction). Idempotent. Use this when you need access to the decomposition object but not yet to a reduced state. Most callers want ensure_hom_reduced.
        """

    def ensure_coh_built(self) -> None:
        """Cohomology counterpart of ensure_hom_built."""

    def ensure_hom_reduced(self) -> None:
        """
        Reduce the homology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent: no-op if already reduced.
        """

    def ensure_coh_reduced(self) -> None:
        """
        Reduce the cohomology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent.
        """

    def ensure_has_u_hom(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Ensure U-row data is available on the hom side over the given filtration row indices (each in geometric `dim`) with the matching value bounds. No-op for LegacyInBand (U is already built in-band during reduction).
        """

    def ensure_has_u_coh(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Cohomology-side counterpart of ensure_has_u_hom. The row indices are passed as filtration indices; conversion to matrix layout (fil_size - 1 - i) is done internally.
        """

    def update(self, new_values: Sequence[float], n_threads: int=1) -> None:
        """
        Update filtration values (indexed by user id), re-sort, and invalidate cached reductions.
        """

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: TopologyOptimizer, /) -> bool:
        ...

    def __ne__(self, arg: TopologyOptimizer, /) -> bool:
        ...

    def __getstate__(self) -> tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, list[int], UStrategy, bool, bool]:
        ...

    def __setstate__(self, arg: tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, Sequence[int], UStrategy, bool, bool], /) -> None:
        ...

class TopologyOptimizerProd:

    def __init__(self, fil: ProdFiltration, with_crit_sets: bool=True, dims_to_restore_elz: Sequence[int]=[], n_threads: int=1, u_strategy: UStrategy=UStrategy.Auto) -> None:
        """
        Construct a TopologyOptimizer for one autograd backward. with_crit_sets=false sets up the cheapest reduction (R only, no V, no U) for diagram-loss; with_crit_sets=true sets up V with clearing and treats dims_to_restore_elz as an allowlist for lazy negative-V restoration. ensure_has_u_* solves only the selected target rows. u_strategy picks which equation to solve for U; LegacyInBand forces a serial in-band U during the reduction.
        """

    def compute_diagram(self, include_inf_points: bool) -> Diagrams:
        """compute diagrams in all dimensions"""

    def simplify(self, epsilon: float, strategy: DenoiseStrategy, dim: int) -> IndicesValuesProd:
        """make points with persistence less than epsilon go to the diagonal"""

    def get_nth_persistence(self, dim: int, n: int) -> float:
        """return n-th persistence value in d-dimensional persistence diagram"""

    def match(self, template_dgm: Sequence[DiagramPoint], dim: int, q: float=1.0, wasserstein_delta: float=0.01, return_wasserstein_distance: bool=False, dualize: bool=False, internal_p: float=-1.0) -> IndicesValuesProd | tuple[IndicesValuesProd, float]:
        """return target from Wasserstein matching"""

    @property
    def homology_decomposition(self) -> Decomposition:
        """Copy of the internal homology Decomposition."""

    @property
    def cohomology_decomposition(self) -> Decomposition:
        """Copy of the internal cohomology Decomposition."""

    def homology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal homology Decomposition (must be built first).
        """

    def cohomology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal cohomology Decomposition (must be built first).
        """

    @property
    def is_hom_built(self) -> bool:
        """
        True iff the homology Decomposition has been materialized from the cached boundary matrix. Set by the first ensure_hom_built / ensure_hom_reduced call.
        """

    @property
    def is_coh_built(self) -> bool:
        """
        True iff the cohomology Decomposition has been materialized. Counterpart of is_hom_built.
        """

    def singleton(self, idx: int, new_value: float) -> tuple[float, list[int]]:
        """Return critical set for moving cell idx (sorted_id) to new_value."""

    def singletons(self, indices: Sequence[int], values: Sequence[float]) -> list[tuple[float, list[int]]]:
        """Return list of critical sets for each (index, target_value) pair."""

    def reduce_all(self) -> None:
        """Eagerly build and reduce both homology and cohomology decompositions."""

    def increase_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing death to inf"""

    def decrease_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing death to -inf"""

    def increase_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing birth to inf"""

    def decrease_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing birth to -inf"""

    def combine_loss(self, critical_sets: Sequence[tuple[float, Sequence[int]]], strategy: ConflictStrategy) -> IndicesValuesProd:
        """
        combine critical sets into well-defined assignment of new values to indices
        """

    def crit_sets_apply(self, indices: Sequence[int], values: Sequence[float], strategy: ConflictStrategy) -> IndicesValuesProd:
        """
        Fused per-pair critical-set walk + conflict resolution in one C++ call. Internally calls ensure_hom_reduced (for the per-pair is_negative dispatch) and ensure_has_u_* on the relevant side when U is needed. Returns IndicesValues; use .indices_array() / .values_array() for zero-copy numpy.
        """

    def crit_sets_apply_typed(self, birth_indices: Sequence[int], birth_values: Sequence[float], death_indices: Sequence[int], death_values: Sequence[float], strategy: ConflictStrategy) -> IndicesValuesProd:
        """
        Role-aware critical-set preparation and application for persistence-diagram backward. Only non-empty birth/death target lists cause their corresponding side to be reduced.
        """

    def ensure_hom_built(self) -> None:
        """
        Materialize the homology Decomposition from the cached boundary matrix (no reduction). Idempotent. Use this when you need access to the decomposition object but not yet to a reduced state. Most callers want ensure_hom_reduced.
        """

    def ensure_coh_built(self) -> None:
        """Cohomology counterpart of ensure_hom_built."""

    def ensure_hom_reduced(self) -> None:
        """
        Reduce the homology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent: no-op if already reduced.
        """

    def ensure_coh_reduced(self) -> None:
        """
        Reduce the cohomology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent.
        """

    def ensure_has_u_hom(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Ensure U-row data is available on the hom side over the given filtration row indices (each in geometric `dim`) with the matching value bounds. No-op for LegacyInBand (U is already built in-band during reduction).
        """

    def ensure_has_u_coh(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Cohomology-side counterpart of ensure_has_u_hom. The row indices are passed as filtration indices; conversion to matrix layout (fil_size - 1 - i) is done internally.
        """

    def update(self, new_values: Sequence[float], n_threads: int=1) -> None:
        """
        Update filtration values (indexed by user id), re-sort, and invalidate cached reductions.
        """

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: TopologyOptimizerProd, /) -> bool:
        ...

    def __ne__(self, arg: TopologyOptimizerProd, /) -> bool:
        ...

    def __getstate__(self) -> tuple[bool, ProdFiltration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, list[int], UStrategy, bool, bool]:
        ...

    def __setstate__(self, arg: tuple[bool, ProdFiltration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, Sequence[int], UStrategy, bool, bool], /) -> None:
        ...

class TopologyOptimizerCube_1D:

    def __init__(self, fil: Filtration, with_crit_sets: bool=True, dims_to_restore_elz: Sequence[int]=[], n_threads: int=1, u_strategy: UStrategy=UStrategy.Auto) -> None:
        """
        Construct a TopologyOptimizer for one autograd backward. with_crit_sets=false sets up the cheapest reduction (R only, no V, no U) for diagram-loss; with_crit_sets=true sets up V with clearing and treats dims_to_restore_elz as an allowlist for lazy negative-V restoration. ensure_has_u_* solves only the selected target rows. u_strategy picks which equation to solve for U; LegacyInBand forces a serial in-band U during the reduction.
        """

    def compute_diagram(self, include_inf_points: bool) -> Diagrams:
        """compute diagrams in all dimensions"""

    def simplify(self, epsilon: float, strategy: DenoiseStrategy, dim: int) -> IndicesValues:
        """make points with persistence less than epsilon go to the diagonal"""

    def get_nth_persistence(self, dim: int, n: int) -> float:
        """return n-th persistence value in d-dimensional persistence diagram"""

    def match(self, template_dgm: Sequence[DiagramPoint], dim: int, q: float=1.0, wasserstein_delta: float=0.01, return_wasserstein_distance: bool=False, dualize: bool=False, internal_p: float=-1.0) -> IndicesValues | tuple[IndicesValues, float]:
        """return target from Wasserstein matching"""

    @property
    def homology_decomposition(self) -> Decomposition:
        """Copy of the internal homology Decomposition."""

    @property
    def cohomology_decomposition(self) -> Decomposition:
        """Copy of the internal cohomology Decomposition."""

    def homology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal homology Decomposition (must be built first).
        """

    def cohomology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal cohomology Decomposition (must be built first).
        """

    @property
    def is_hom_built(self) -> bool:
        """
        True iff the homology Decomposition has been materialized from the cached boundary matrix. Set by the first ensure_hom_built / ensure_hom_reduced call.
        """

    @property
    def is_coh_built(self) -> bool:
        """
        True iff the cohomology Decomposition has been materialized. Counterpart of is_hom_built.
        """

    def singleton(self, idx: int, new_value: float) -> tuple[float, list[int]]:
        """Return critical set for moving cell idx (sorted_id) to new_value."""

    def singletons(self, indices: Sequence[int], values: Sequence[float]) -> list[tuple[float, list[int]]]:
        """Return list of critical sets for each (index, target_value) pair."""

    def reduce_all(self) -> None:
        """Eagerly build and reduce both homology and cohomology decompositions."""

    def increase_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing death to inf"""

    def decrease_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing death to -inf"""

    def increase_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing birth to inf"""

    def decrease_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing birth to -inf"""

    def combine_loss(self, critical_sets: Sequence[tuple[float, Sequence[int]]], strategy: ConflictStrategy) -> IndicesValues:
        """
        combine critical sets into well-defined assignment of new values to indices
        """

    def crit_sets_apply(self, indices: Sequence[int], values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Fused per-pair critical-set walk + conflict resolution in one C++ call. Internally calls ensure_hom_reduced (for the per-pair is_negative dispatch) and ensure_has_u_* on the relevant side when U is needed. Returns IndicesValues; use .indices_array() / .values_array() for zero-copy numpy.
        """

    def crit_sets_apply_typed(self, birth_indices: Sequence[int], birth_values: Sequence[float], death_indices: Sequence[int], death_values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Role-aware critical-set preparation and application for persistence-diagram backward. Only non-empty birth/death target lists cause their corresponding side to be reduced.
        """

    def ensure_hom_built(self) -> None:
        """
        Materialize the homology Decomposition from the cached boundary matrix (no reduction). Idempotent. Use this when you need access to the decomposition object but not yet to a reduced state. Most callers want ensure_hom_reduced.
        """

    def ensure_coh_built(self) -> None:
        """Cohomology counterpart of ensure_hom_built."""

    def ensure_hom_reduced(self) -> None:
        """
        Reduce the homology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent: no-op if already reduced.
        """

    def ensure_coh_reduced(self) -> None:
        """
        Reduce the cohomology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent.
        """

    def ensure_has_u_hom(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Ensure U-row data is available on the hom side over the given filtration row indices (each in geometric `dim`) with the matching value bounds. No-op for LegacyInBand (U is already built in-band during reduction).
        """

    def ensure_has_u_coh(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Cohomology-side counterpart of ensure_has_u_hom. The row indices are passed as filtration indices; conversion to matrix layout (fil_size - 1 - i) is done internally.
        """

    def update(self, new_values: Sequence[float], n_threads: int=1) -> None:
        """
        Update filtration values (indexed by user id), re-sort, and invalidate cached reductions.
        """

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: TopologyOptimizerCube_1D, /) -> bool:
        ...

    def __ne__(self, arg: TopologyOptimizerCube_1D, /) -> bool:
        ...

    def __getstate__(self) -> tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, list[int], UStrategy, bool, bool]:
        ...

    def __setstate__(self, arg: tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, Sequence[int], UStrategy, bool, bool], /) -> None:
        ...

class TopologyOptimizerCube_2D:

    def __init__(self, fil: Filtration, with_crit_sets: bool=True, dims_to_restore_elz: Sequence[int]=[], n_threads: int=1, u_strategy: UStrategy=UStrategy.Auto) -> None:
        """
        Construct a TopologyOptimizer for one autograd backward. with_crit_sets=false sets up the cheapest reduction (R only, no V, no U) for diagram-loss; with_crit_sets=true sets up V with clearing and treats dims_to_restore_elz as an allowlist for lazy negative-V restoration. ensure_has_u_* solves only the selected target rows. u_strategy picks which equation to solve for U; LegacyInBand forces a serial in-band U during the reduction.
        """

    def compute_diagram(self, include_inf_points: bool) -> Diagrams:
        """compute diagrams in all dimensions"""

    def simplify(self, epsilon: float, strategy: DenoiseStrategy, dim: int) -> IndicesValues:
        """make points with persistence less than epsilon go to the diagonal"""

    def get_nth_persistence(self, dim: int, n: int) -> float:
        """return n-th persistence value in d-dimensional persistence diagram"""

    def match(self, template_dgm: Sequence[DiagramPoint], dim: int, q: float=1.0, wasserstein_delta: float=0.01, return_wasserstein_distance: bool=False, dualize: bool=False, internal_p: float=-1.0) -> IndicesValues | tuple[IndicesValues, float]:
        """return target from Wasserstein matching"""

    @property
    def homology_decomposition(self) -> Decomposition:
        """Copy of the internal homology Decomposition."""

    @property
    def cohomology_decomposition(self) -> Decomposition:
        """Copy of the internal cohomology Decomposition."""

    def homology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal homology Decomposition (must be built first).
        """

    def cohomology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal cohomology Decomposition (must be built first).
        """

    @property
    def is_hom_built(self) -> bool:
        """
        True iff the homology Decomposition has been materialized from the cached boundary matrix. Set by the first ensure_hom_built / ensure_hom_reduced call.
        """

    @property
    def is_coh_built(self) -> bool:
        """
        True iff the cohomology Decomposition has been materialized. Counterpart of is_hom_built.
        """

    def singleton(self, idx: int, new_value: float) -> tuple[float, list[int]]:
        """Return critical set for moving cell idx (sorted_id) to new_value."""

    def singletons(self, indices: Sequence[int], values: Sequence[float]) -> list[tuple[float, list[int]]]:
        """Return list of critical sets for each (index, target_value) pair."""

    def reduce_all(self) -> None:
        """Eagerly build and reduce both homology and cohomology decompositions."""

    def increase_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing death to inf"""

    def decrease_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing death to -inf"""

    def increase_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing birth to inf"""

    def decrease_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing birth to -inf"""

    def combine_loss(self, critical_sets: Sequence[tuple[float, Sequence[int]]], strategy: ConflictStrategy) -> IndicesValues:
        """
        combine critical sets into well-defined assignment of new values to indices
        """

    def crit_sets_apply(self, indices: Sequence[int], values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Fused per-pair critical-set walk + conflict resolution in one C++ call. Internally calls ensure_hom_reduced (for the per-pair is_negative dispatch) and ensure_has_u_* on the relevant side when U is needed. Returns IndicesValues; use .indices_array() / .values_array() for zero-copy numpy.
        """

    def crit_sets_apply_typed(self, birth_indices: Sequence[int], birth_values: Sequence[float], death_indices: Sequence[int], death_values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Role-aware critical-set preparation and application for persistence-diagram backward. Only non-empty birth/death target lists cause their corresponding side to be reduced.
        """

    def ensure_hom_built(self) -> None:
        """
        Materialize the homology Decomposition from the cached boundary matrix (no reduction). Idempotent. Use this when you need access to the decomposition object but not yet to a reduced state. Most callers want ensure_hom_reduced.
        """

    def ensure_coh_built(self) -> None:
        """Cohomology counterpart of ensure_hom_built."""

    def ensure_hom_reduced(self) -> None:
        """
        Reduce the homology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent: no-op if already reduced.
        """

    def ensure_coh_reduced(self) -> None:
        """
        Reduce the cohomology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent.
        """

    def ensure_has_u_hom(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Ensure U-row data is available on the hom side over the given filtration row indices (each in geometric `dim`) with the matching value bounds. No-op for LegacyInBand (U is already built in-band during reduction).
        """

    def ensure_has_u_coh(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Cohomology-side counterpart of ensure_has_u_hom. The row indices are passed as filtration indices; conversion to matrix layout (fil_size - 1 - i) is done internally.
        """

    def update(self, new_values: Sequence[float], n_threads: int=1) -> None:
        """
        Update filtration values (indexed by user id), re-sort, and invalidate cached reductions.
        """

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: TopologyOptimizerCube_2D, /) -> bool:
        ...

    def __ne__(self, arg: TopologyOptimizerCube_2D, /) -> bool:
        ...

    def __getstate__(self) -> tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, list[int], UStrategy, bool, bool]:
        ...

    def __setstate__(self, arg: tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, Sequence[int], UStrategy, bool, bool], /) -> None:
        ...

class TopologyOptimizerCube_3D:

    def __init__(self, fil: Filtration, with_crit_sets: bool=True, dims_to_restore_elz: Sequence[int]=[], n_threads: int=1, u_strategy: UStrategy=UStrategy.Auto) -> None:
        """
        Construct a TopologyOptimizer for one autograd backward. with_crit_sets=false sets up the cheapest reduction (R only, no V, no U) for diagram-loss; with_crit_sets=true sets up V with clearing and treats dims_to_restore_elz as an allowlist for lazy negative-V restoration. ensure_has_u_* solves only the selected target rows. u_strategy picks which equation to solve for U; LegacyInBand forces a serial in-band U during the reduction.
        """

    def compute_diagram(self, include_inf_points: bool) -> Diagrams:
        """compute diagrams in all dimensions"""

    def simplify(self, epsilon: float, strategy: DenoiseStrategy, dim: int) -> IndicesValues:
        """make points with persistence less than epsilon go to the diagonal"""

    def get_nth_persistence(self, dim: int, n: int) -> float:
        """return n-th persistence value in d-dimensional persistence diagram"""

    def match(self, template_dgm: Sequence[DiagramPoint], dim: int, q: float=1.0, wasserstein_delta: float=0.01, return_wasserstein_distance: bool=False, dualize: bool=False, internal_p: float=-1.0) -> IndicesValues | tuple[IndicesValues, float]:
        """return target from Wasserstein matching"""

    @property
    def homology_decomposition(self) -> Decomposition:
        """Copy of the internal homology Decomposition."""

    @property
    def cohomology_decomposition(self) -> Decomposition:
        """Copy of the internal cohomology Decomposition."""

    def homology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal homology Decomposition (must be built first).
        """

    def cohomology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal cohomology Decomposition (must be built first).
        """

    @property
    def is_hom_built(self) -> bool:
        """
        True iff the homology Decomposition has been materialized from the cached boundary matrix. Set by the first ensure_hom_built / ensure_hom_reduced call.
        """

    @property
    def is_coh_built(self) -> bool:
        """
        True iff the cohomology Decomposition has been materialized. Counterpart of is_hom_built.
        """

    def singleton(self, idx: int, new_value: float) -> tuple[float, list[int]]:
        """Return critical set for moving cell idx (sorted_id) to new_value."""

    def singletons(self, indices: Sequence[int], values: Sequence[float]) -> list[tuple[float, list[int]]]:
        """Return list of critical sets for each (index, target_value) pair."""

    def reduce_all(self) -> None:
        """Eagerly build and reduce both homology and cohomology decompositions."""

    def increase_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing death to inf"""

    def decrease_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing death to -inf"""

    def increase_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing birth to inf"""

    def decrease_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing birth to -inf"""

    def combine_loss(self, critical_sets: Sequence[tuple[float, Sequence[int]]], strategy: ConflictStrategy) -> IndicesValues:
        """
        combine critical sets into well-defined assignment of new values to indices
        """

    def crit_sets_apply(self, indices: Sequence[int], values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Fused per-pair critical-set walk + conflict resolution in one C++ call. Internally calls ensure_hom_reduced (for the per-pair is_negative dispatch) and ensure_has_u_* on the relevant side when U is needed. Returns IndicesValues; use .indices_array() / .values_array() for zero-copy numpy.
        """

    def crit_sets_apply_typed(self, birth_indices: Sequence[int], birth_values: Sequence[float], death_indices: Sequence[int], death_values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Role-aware critical-set preparation and application for persistence-diagram backward. Only non-empty birth/death target lists cause their corresponding side to be reduced.
        """

    def ensure_hom_built(self) -> None:
        """
        Materialize the homology Decomposition from the cached boundary matrix (no reduction). Idempotent. Use this when you need access to the decomposition object but not yet to a reduced state. Most callers want ensure_hom_reduced.
        """

    def ensure_coh_built(self) -> None:
        """Cohomology counterpart of ensure_hom_built."""

    def ensure_hom_reduced(self) -> None:
        """
        Reduce the homology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent: no-op if already reduced.
        """

    def ensure_coh_reduced(self) -> None:
        """
        Reduce the cohomology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent.
        """

    def ensure_has_u_hom(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Ensure U-row data is available on the hom side over the given filtration row indices (each in geometric `dim`) with the matching value bounds. No-op for LegacyInBand (U is already built in-band during reduction).
        """

    def ensure_has_u_coh(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Cohomology-side counterpart of ensure_has_u_hom. The row indices are passed as filtration indices; conversion to matrix layout (fil_size - 1 - i) is done internally.
        """

    def update(self, new_values: Sequence[float], n_threads: int=1) -> None:
        """
        Update filtration values (indexed by user id), re-sort, and invalidate cached reductions.
        """

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: TopologyOptimizerCube_3D, /) -> bool:
        ...

    def __ne__(self, arg: TopologyOptimizerCube_3D, /) -> bool:
        ...

    def __getstate__(self) -> tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, list[int], UStrategy, bool, bool]:
        ...

    def __setstate__(self, arg: tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, Sequence[int], UStrategy, bool, bool], /) -> None:
        ...

class TopologyOptimizerCube_4D:

    def __init__(self, fil: Filtration, with_crit_sets: bool=True, dims_to_restore_elz: Sequence[int]=[], n_threads: int=1, u_strategy: UStrategy=UStrategy.Auto) -> None:
        """
        Construct a TopologyOptimizer for one autograd backward. with_crit_sets=false sets up the cheapest reduction (R only, no V, no U) for diagram-loss; with_crit_sets=true sets up V with clearing and treats dims_to_restore_elz as an allowlist for lazy negative-V restoration. ensure_has_u_* solves only the selected target rows. u_strategy picks which equation to solve for U; LegacyInBand forces a serial in-band U during the reduction.
        """

    def compute_diagram(self, include_inf_points: bool) -> Diagrams:
        """compute diagrams in all dimensions"""

    def simplify(self, epsilon: float, strategy: DenoiseStrategy, dim: int) -> IndicesValues:
        """make points with persistence less than epsilon go to the diagonal"""

    def get_nth_persistence(self, dim: int, n: int) -> float:
        """return n-th persistence value in d-dimensional persistence diagram"""

    def match(self, template_dgm: Sequence[DiagramPoint], dim: int, q: float=1.0, wasserstein_delta: float=0.01, return_wasserstein_distance: bool=False, dualize: bool=False, internal_p: float=-1.0) -> IndicesValues | tuple[IndicesValues, float]:
        """return target from Wasserstein matching"""

    @property
    def homology_decomposition(self) -> Decomposition:
        """Copy of the internal homology Decomposition."""

    @property
    def cohomology_decomposition(self) -> Decomposition:
        """Copy of the internal cohomology Decomposition."""

    def homology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal homology Decomposition (must be built first).
        """

    def cohomology_decomposition_ref(self) -> Decomposition:
        """
        Mutable reference to the internal cohomology Decomposition (must be built first).
        """

    @property
    def is_hom_built(self) -> bool:
        """
        True iff the homology Decomposition has been materialized from the cached boundary matrix. Set by the first ensure_hom_built / ensure_hom_reduced call.
        """

    @property
    def is_coh_built(self) -> bool:
        """
        True iff the cohomology Decomposition has been materialized. Counterpart of is_hom_built.
        """

    def singleton(self, idx: int, new_value: float) -> tuple[float, list[int]]:
        """Return critical set for moving cell idx (sorted_id) to new_value."""

    def singletons(self, indices: Sequence[int], values: Sequence[float]) -> list[tuple[float, list[int]]]:
        """Return list of critical sets for each (index, target_value) pair."""

    def reduce_all(self) -> None:
        """Eagerly build and reduce both homology and cohomology decompositions."""

    def increase_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing death to inf"""

    def decrease_death(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing death to -inf"""

    def increase_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for increasing birth to inf"""

    def decrease_birth(self, negative_simplex_idx: int) -> list[int]:
        """return critical set for decreasing birth to -inf"""

    def combine_loss(self, critical_sets: Sequence[tuple[float, Sequence[int]]], strategy: ConflictStrategy) -> IndicesValues:
        """
        combine critical sets into well-defined assignment of new values to indices
        """

    def crit_sets_apply(self, indices: Sequence[int], values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Fused per-pair critical-set walk + conflict resolution in one C++ call. Internally calls ensure_hom_reduced (for the per-pair is_negative dispatch) and ensure_has_u_* on the relevant side when U is needed. Returns IndicesValues; use .indices_array() / .values_array() for zero-copy numpy.
        """

    def crit_sets_apply_typed(self, birth_indices: Sequence[int], birth_values: Sequence[float], death_indices: Sequence[int], death_values: Sequence[float], strategy: ConflictStrategy) -> IndicesValues:
        """
        Role-aware critical-set preparation and application for persistence-diagram backward. Only non-empty birth/death target lists cause their corresponding side to be reduced.
        """

    def ensure_hom_built(self) -> None:
        """
        Materialize the homology Decomposition from the cached boundary matrix (no reduction). Idempotent. Use this when you need access to the decomposition object but not yet to a reduced state. Most callers want ensure_hom_reduced.
        """

    def ensure_coh_built(self) -> None:
        """Cohomology counterpart of ensure_hom_built."""

    def ensure_hom_reduced(self) -> None:
        """
        Reduce the homology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent: no-op if already reduced.
        """

    def ensure_coh_reduced(self) -> None:
        """
        Reduce the cohomology side with the recipe baked in at construction time. Builds the decomposition first if needed. Idempotent.
        """

    def ensure_has_u_hom(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Ensure U-row data is available on the hom side over the given filtration row indices (each in geometric `dim`) with the matching value bounds. No-op for LegacyInBand (U is already built in-band during reduction).
        """

    def ensure_has_u_coh(self, dim: int, rows_fil: Sequence[int], bounds: Sequence[float]) -> None:
        """
        Cohomology-side counterpart of ensure_has_u_hom. The row indices are passed as filtration indices; conversion to matrix layout (fil_size - 1 - i) is done internally.
        """

    def update(self, new_values: Sequence[float], n_threads: int=1) -> None:
        """
        Update filtration values (indexed by user id), re-sort, and invalidate cached reductions.
        """

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: TopologyOptimizerCube_4D, /) -> bool:
        ...

    def __ne__(self, arg: TopologyOptimizerCube_4D, /) -> bool:
        ...

    def __getstate__(self) -> tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, list[int], UStrategy, bool, bool]:
        ...

    def __setstate__(self, arg: tuple[bool, Filtration, Decomposition, Decomposition, ReductionParams, ReductionParams, bool, int, Sequence[int], UStrategy, bool, bool], /) -> None:
        ...

class InfKind(enum.Enum):
    """The four families of essential (infinite-coordinate) diagram points."""
    INF_DEATH = 0
    '(finite, +inf) -- homology class born finitely, never dies.'
    NEG_INF_DEATH = 1
    '(finite, -inf).'
    INF_BIRTH = 2
    '(+inf, finite).'
    NEG_INF_BIRTH = 3
    '(-inf, finite).'

class EssentialMatches:
    """Grouped view over essential-point matches by family."""

    @property
    def inf_death(self) -> NDArray[numpy.int64]:
        ...

    @property
    def neg_inf_death(self) -> NDArray[numpy.int64]:
        ...

    @property
    def inf_birth(self) -> NDArray[numpy.int64]:
        ...

    @property
    def neg_inf_birth(self) -> NDArray[numpy.int64]:
        ...

    def __getitem__(self, arg: object, /) -> NDArray[numpy.int64]:
        ...

    def __contains__(self, arg: object, /) -> bool:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def keys(self) -> object:
        ...

    def values(self) -> list:
        ...

    def items(self) -> list:
        ...

    def __repr__(self) -> str:
        ...

class EssentialLongestEdges:
    """Grouped view over per-family tied-longest edges."""

    @property
    def inf_death(self) -> list[EssentialLongestEdge]:
        ...

    @property
    def neg_inf_death(self) -> list[EssentialLongestEdge]:
        ...

    @property
    def inf_birth(self) -> list[EssentialLongestEdge]:
        ...

    @property
    def neg_inf_birth(self) -> list[EssentialLongestEdge]:
        ...

    def __getitem__(self, arg: object, /) -> list[EssentialLongestEdge]:
        ...

    def __contains__(self, arg: object, /) -> bool:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...

    def keys(self) -> object:
        ...

    def values(self) -> list:
        ...

    def items(self) -> list:
        ...

    def __repr__(self) -> str:
        ...

class LongestEdges:
    """Bottleneck longest-edge data, split into finite and essential parts."""

    @property
    def finite(self) -> list[FiniteLongestEdge]:
        ...

    @property
    def essential(self) -> EssentialLongestEdges:
        ...

    def __repr__(self) -> str:
        ...

class FiniteLongestEdge:
    """
    One edge tied for the bottleneck distance in the finite part of a matching. ``idx_a`` or ``idx_b`` is ``None`` if that endpoint is a diagonal projection.
    """

    @property
    def length(self) -> float:
        ...

    @property
    def idx_a(self) -> object:
        ...

    @property
    def idx_b(self) -> object:
        ...

    @property
    def point_a(self) -> tuple:
        ...

    @property
    def point_b(self) -> tuple:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: FiniteLongestEdge, /) -> bool:
        ...

    def __ne__(self, arg: FiniteLongestEdge, /) -> bool:
        ...

    def __getstate__(self) -> tuple[float, int, int, float, float, float, float]:
        ...

    def __setstate__(self, arg: tuple, /) -> None:
        ...

class EssentialLongestEdge:
    """
    One edge tied for the bottleneck distance within a single essential family.
    """

    @property
    def length(self) -> float:
        ...

    @property
    def idx_a(self) -> int:
        ...

    @property
    def idx_b(self) -> int:
        ...

    @property
    def coord_a(self) -> float:
        ...

    @property
    def coord_b(self) -> float:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: EssentialLongestEdge, /) -> bool:
        ...

    def __ne__(self, arg: EssentialLongestEdge, /) -> bool:
        ...

    def __getstate__(self) -> tuple[float, int, int, float, float]:
        ...

    def __setstate__(self, arg: tuple, /) -> None:
        ...

class DiagramMatching:
    """Optimal Wasserstein matching: bucketed pair indices, cost, distance."""

    @property
    def finite_to_finite(self) -> NDArray[numpy.int64]:
        """
        (n, 2) int64 ndarray of (idx_a, idx_b) pairs for finite-to-finite matches.
        """

    @property
    def a_to_diagonal(self) -> NDArray[numpy.int64]:
        """1-D int64 ndarray of indices in dgm_a matched to the diagonal."""

    @property
    def b_to_diagonal(self) -> NDArray[numpy.int64]:
        """1-D int64 ndarray of indices in dgm_b matched to the diagonal."""

    @property
    def essential(self) -> EssentialMatches:
        """Grouped view of essential-point matches per family."""

    @property
    def cost(self) -> float:
        """Total Wasserstein cost (== distance ** q)."""

    @property
    def distance(self) -> float:
        """Wasserstein distance."""

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: DiagramMatching, /) -> bool:
        ...

    def __ne__(self, arg: DiagramMatching, /) -> bool:
        ...

    def __getstate__(self) -> tuple[list[tuple[int, int]], list[int], list[int], list[list[tuple[int, int]]], float, float]:
        ...

    def __setstate__(self, arg: tuple, /) -> None:
        ...

class BottleneckMatching(DiagramMatching):
    """
    Optimal bottleneck matching: inherits the DiagramMatching interface and adds longest-edge data via the .longest view.
    """

    @property
    def longest(self) -> LongestEdges:
        """
        View over tied-longest edges, with `.finite` (a list of FiniteLongestEdge) and `.essential` (an EssentialLongestEdges grouped view).
        """

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, arg: BottleneckMatching, /) -> bool:
        ...

    def __ne__(self, arg: BottleneckMatching, /) -> bool:
        ...

    def __getstate__(self) -> tuple[list[tuple[int, int]], list[int], list[int], list[list[tuple[int, int]]], float, float, list[FiniteLongestEdge], list[list[EssentialLongestEdge]]]:
        ...

    def __setstate__(self, arg: tuple, /) -> None:
        ...

class MixupBarcodes:
    """Mixup barcodes of an inclusion L -> K, one barcode per homological degree."""
    max_dim: int
    def __init__(self, triples: dict[int, tuple[NDArray[numpy.float64], NDArray[numpy.int64], NDArray[numpy.float64]]], max_dim: int) -> None: ...
    @classmethod
    def empty(cls, max_dim: int) -> MixupBarcodes: ...
    def in_dimension(self, dim: int) -> NDArray[numpy.float64]: ...
    def __getitem__(self, dim: int) -> NDArray[numpy.float64]: ...
    def __contains__(self, dim: int) -> bool: ...
    def keys(self) -> list[int]: ...
    def index_triples_in_dimension(self, dim: int) -> NDArray[numpy.int64]: ...
    def essential_in_dimension(self, dim: int) -> NDArray[numpy.float64]: ...
    def persistence_barcode(self, dim: int) -> NDArray[numpy.float64]: ...
    def image_sub_barcode(self, dim: int) -> NDArray[numpy.float64]: ...
    def mixup_sub_barcode(self, dim: int) -> NDArray[numpy.float64]: ...
    def total_persistence(self, dim: int) -> float: ...
    def total_mixup(self, dim: int) -> float: ...
    def mixup_percentages(self, dim: int) -> NDArray[numpy.float64]: ...
    def total_mixup_percentage(self, dim: int) -> float: ...
    def mean_mixup_percentage(self, dim: int) -> float: ...

__version__: str
REAL_DTYPE: numpy.dtype[Any]

def as_real_numpy(arr: Any, dtype: Any = None) -> NDArray[numpy.floating[Any]]: ...
def reduce(filtration: Filtration, params: ReductionParams | None = None, dualize: bool = False, **kwargs: Any) -> Decomposition: ...
def get_boundary_matrix(vertices: Sequence[Sequence[int]], values: Sequence[float] | None = None, negate: bool = False, n_threads: int = 1) -> tuple[list[list[int]], list[float]]: ...
def compute_relative_diagrams(fil: Filtration, rel: Filtration, include_inf_points: bool = True) -> Diagrams: ...
def get_denoise_target(dim: int, fil: Filtration, rv_matrix: Decomposition, eps: float, strategy: DenoiseStrategy) -> Any: ...
def get_induced_matching(included_filtration: Filtration, containing_filtration: Filtration, dim: int = -1, n_threads: int = 1) -> Any: ...
def get_nth_persistence(fil: Filtration, rv_matrix: Decomposition, dim: int, n: int) -> float: ...
def get_permutation(target_matching: Mapping[int, float], fil: Filtration) -> list[int]: ...
def get_permutation_dtv(diagram_to_values: Any, fil: Filtration) -> list[int]: ...
def remove_simplices(fil: Filtration, dcmp: Decomposition, seeds: Sequence[int], *, close_star: bool = True, stats: DecompositionManipStats | None = None, n_threads: int = 1) -> Filtration: ...

def point_to_diagonal(dgm: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], indices: Sequence[int] | NDArray[numpy.integer[Any]] | None = None) -> NDArray[numpy.floating[Any]] | list[tuple[float, float]]: ...
def bottleneck_distance(dgm_1: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], dgm_2: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], delta: float = 0.01) -> float: ...
def wasserstein_distance(dgm_1: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], dgm_2: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], q: float = 1.0, delta: float = 0.01, internal_p: float = ..., wasserstein_q: float | None = None, check_for_zero: bool = True) -> float: ...
def wasserstein_matching(dgm_1: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], dgm_2: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], *, q: float = 2.0, delta: float = 0.01, internal_p: float = ..., ignore_inf_points: bool = True) -> DiagramMatching: ...
def bottleneck_matching(dgm_1: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], dgm_2: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], *, delta: float = 0.01, ignore_inf_points: bool = True) -> BottleneckMatching: ...
def sliced_wasserstein_distance(dgm_1: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], dgm_2: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], n_directions: int = 100, ignore_inf_points: bool = False, seed: int | None = None, directions: NDArray[numpy.floating[Any]] | None = None) -> float: ...
def sliced_wasserstein_distance_diag_corrected(dgm_1: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], dgm_2: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], n_directions: int = 100, ignore_inf_points: bool = False, seed: int | None = None, directions: NDArray[numpy.floating[Any]] | None = None) -> float: ...

def init_frechet_mean_first_diagram(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]]) -> NDArray[numpy.floating[Any]]: ...
def init_frechet_mean_random_diagram(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, domain: DiagramPlaneDomain = DiagramPlaneDomain.AboveDiagonal, random_noise_scale: float = 1.0, random_seed: int = 42) -> NDArray[numpy.floating[Any]]: ...
def init_frechet_mean_medoid_diagram(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, n_threads: int = 1) -> NDArray[numpy.floating[Any]]: ...
def init_frechet_mean_diagonal_grid(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, domain: DiagramPlaneDomain = DiagramPlaneDomain.AboveDiagonal, grid_n_x_bins: int = 16, grid_n_y_bins: int = 16) -> NDArray[numpy.floating[Any]]: ...
def frechet_mean_objective(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], barycenter: NDArray[numpy.floating[Any]] | Sequence[DiagramPoint], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, wasserstein_delta: float = 0.01, internal_p: float = ..., n_threads: int = 1) -> float: ...
def make_frechet_mean_persistence_schedule(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, initial_threshold_fraction: float = 0.5, max_active_growth: float = 0.10, min_persistence: float = 0.0) -> list[float]: ...
def frechet_mean_newborn_points_from_newly_active(newly_active_diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None) -> NDArray[numpy.floating[Any]]: ...
def frechet_mean(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, max_iter: int = 100, tol: float = 1e-7, wasserstein_delta: float = 0.01, internal_p: float = ..., init_strategy: FrechetMeanInit = FrechetMeanInit.Grid, domain: DiagramPlaneDomain = DiagramPlaneDomain.AboveDiagonal, ignore_infinite_points: bool = False, random_noise_scale: float = 1.0, random_seed: int = 42, grid_n_x_bins: int = 16, grid_n_y_bins: int = 16, custom_initial_barycenter: NDArray[numpy.floating[Any]] | None = None, n_threads: int = 1) -> NDArray[numpy.floating[Any]]: ...
@overload
def frechet_mean_multistart(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, starts: Sequence[Any] = ..., return_details: Literal[False] = False, n_threads: int = 1, **kwargs: Any) -> NDArray[numpy.floating[Any]]: ...
@overload
def frechet_mean_multistart(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, starts: Sequence[Any] = ..., return_details: Literal[True], n_threads: int = 1, **kwargs: Any) -> tuple[NDArray[numpy.floating[Any]], dict[str, Any]]: ...
@overload
def progressive_frechet_mean(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, thresholds: Sequence[float] | None = None, initial_threshold_fraction: float = 0.5, max_active_growth: float = 0.10, min_persistence: float = 0.0, initial_seed: Any = "medoid", support_update_predicate: Callable[..., bool] | None = None, support_update_fn: Callable[..., NDArray[numpy.floating[Any]] | None] | None = None, return_details: Literal[False] = False, n_threads: int = 1, **kwargs: Any) -> NDArray[numpy.floating[Any]]: ...
@overload
def progressive_frechet_mean(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, thresholds: Sequence[float] | None = None, initial_threshold_fraction: float = 0.5, max_active_growth: float = 0.10, min_persistence: float = 0.0, initial_seed: Any = "medoid", support_update_predicate: Callable[..., bool] | None = None, support_update_fn: Callable[..., NDArray[numpy.floating[Any]] | None] | None = None, return_details: Literal[True], n_threads: int = 1, **kwargs: Any) -> tuple[NDArray[numpy.floating[Any]], dict[str, Any]]: ...
@overload
def progressive_frechet_mean_multistart(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, starts: Sequence[Any] = ..., return_details: Literal[False] = False, n_threads: int = 1, **kwargs: Any) -> NDArray[numpy.floating[Any]]: ...
@overload
def progressive_frechet_mean_multistart(diagrams: Sequence[NDArray[numpy.floating[Any]] | Sequence[DiagramPoint]], *, weights: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, starts: Sequence[Any] = ..., return_details: Literal[True], n_threads: int = 1, **kwargs: Any) -> tuple[NDArray[numpy.floating[Any]], dict[str, Any]]: ...

def to_scipy_matrix(sparse_cols: Sequence[Sequence[int]], shape: tuple[int, int] | None = None) -> scipy.sparse.csc_matrix: ...
def max_distance(data: NDArray[numpy.floating[Any]], from_pwdists: bool = False) -> float: ...
def is_reduced(a: NDArray[numpy.integer[Any]]) -> bool: ...

@overload
def freudenthal_filtration(data: NDArray[numpy.floating[Any]], negate: bool = False, wrap: bool = False, max_dim: int = 3, with_critical_vertices: Literal[False] = False, *, slim: bool = True, n_threads: int = 1) -> Filtration: ...
@overload
def freudenthal_filtration(data: NDArray[numpy.floating[Any]], negate: bool = False, wrap: bool = False, max_dim: int = 3, with_critical_vertices: Literal[True] = ..., *, slim: bool = True, n_threads: int = 1) -> tuple[Filtration, NDArray[numpy.int64]]: ...

@overload
def vr_filtration(data: NDArray[numpy.floating[Any]], from_pwdists: bool = False, max_dim: int = -1, max_diameter: float = -1.0, with_critical_edges: Literal[False] = False, *, packed: bool = True, n_threads: int = 1) -> Filtration: ...
@overload
def vr_filtration(data: NDArray[numpy.floating[Any]], from_pwdists: bool = False, max_dim: int = -1, max_diameter: float = -1.0, with_critical_edges: Literal[True] = ..., *, packed: bool = True, n_threads: int = 1) -> tuple[Filtration, NDArray[numpy.int64]]: ...

def alpha_filtration(points: NDArray[numpy.floating[Any]], weights: NDArray[numpy.floating[Any]] | None = None, exact: bool = False, periodic: bool = False, compute_bounding_box: bool = True, bbox_min: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, bbox_max: Sequence[float] | NDArray[numpy.floating[Any]] | None = None, *, packed: bool = True, n_threads: int = 1) -> Filtration: ...
def cech_filtration(points: NDArray[numpy.floating[Any]], max_dim: int = -1, max_radius: float = -1.0, *, vertex_ids: NDArray[numpy.integer[Any]] | None = None, eps: float = 0.0, n_threads: int = 1) -> Filtration: ...
def cube_filtration(data: NDArray[numpy.floating[Any]], negate: bool = False, wrap: bool = False, max_dim: int | None = None, values_on: str = "vertices", n_threads: int = 1) -> Filtration: ...
def list_to_filtration(data: Sequence[tuple[int, Sequence[int], float]]) -> Filtration: ...

@overload
def mapping_cylinder(fil_domain: Filtration, fil_codomain: Filtration, v_domain: Simplex | ProdSimplex | CombinatorialSimplex | CombinatorialProdSimplex, v_codomain: Simplex | ProdSimplex | CombinatorialSimplex | CombinatorialProdSimplex, v_domain_value: float | None = None, v_codomain_value: float | None = None, with_indices: Literal[False] = False) -> ProdFiltration: ...
@overload
def mapping_cylinder(fil_domain: Filtration, fil_codomain: Filtration, v_domain: Simplex | ProdSimplex | CombinatorialSimplex | CombinatorialProdSimplex, v_codomain: Simplex | ProdSimplex | CombinatorialSimplex | CombinatorialProdSimplex, v_domain_value: float | None = None, v_codomain_value: float | None = None, with_indices: Literal[True] = ...) -> tuple[ProdFiltration, list[int]]: ...

def multiply_filtration(fil: Filtration, sigma: Simplex | CombinatorialSimplex, sigma_value: float | None = None) -> ProdFiltration: ...

@overload
def min_filtration(fil_1: Filtration, fil_2: Filtration, with_indices: Literal[False] = False) -> Filtration: ...
@overload
def min_filtration(fil_1: Filtration, fil_2: Filtration, with_indices: Literal[True] = ...) -> tuple[Filtration, list[int], list[int]]: ...

def compute_diagrams_ls(data: NDArray[numpy.floating[Any]], negate: bool = False, wrap: bool = False, max_dim: int | None = None, params: ReductionParams | None = None, include_inf_points: bool = True, dualize: bool = False) -> Diagrams: ...
def compute_diagrams_vr(data: NDArray[numpy.floating[Any]], from_pwdists: bool = False, max_dim: int = -1, max_diameter: float = -1.0, params: ReductionParams | None = None, include_inf_points: bool = True, dualize: bool = True) -> Diagrams: ...
def compute_diagrams_alpha(points: NDArray[numpy.floating[Any]], weights: NDArray[numpy.floating[Any]] | None = None, params: ReductionParams | None = None, include_inf_points: bool = True, dualize: bool = False, exact: bool = False, periodic: bool = False, compute_bounding_box: bool = True, bbox_min: NDArray[numpy.floating[Any]] | Sequence[float] | None = None, bbox_max: NDArray[numpy.floating[Any]] | Sequence[float] | None = None) -> Diagrams: ...

def compute_kernel_image_cokernel_reduction(K: Filtration | list[tuple[int, list[int], float]], L: Filtration | list[tuple[int, list[int], float]], params: KICRParams | None = None, reduction_params: ReductionParams | None = None) -> KerImCokReduced: ...
def compute_ker_cok_reduction_cyl(fil_2: Filtration, fil_3: Filtration) -> KerImCokReducedProd: ...

def mixup_barcodes(A: NDArray[numpy.floating[Any]], B: NDArray[numpy.floating[Any]] | None, max_dim: int = 1, max_diameter: float | None = None, n_threads: int = 1) -> MixupBarcodes: ...
def mixup_barcodes_of_filtrations(K: Filtration, L: Filtration, max_dim: int | None = None, n_threads: int = 1) -> MixupBarcodes: ...
def function_delaunay_bifiltration(points: NDArray[numpy.floating[Any]], labels: NDArray[Any] | Sequence[Any], *, a_value: Any = None, max_dim: int = -1, squared: bool = True, jitter: float = 0.0, seed: int | None = None, dummy_scale: float = 20.0, conflict_method: str = "bfs", eps: float = 1e-10, n_threads: int = 1) -> tuple[Filtration, Filtration]: ...

from .vis import plot_chain as plot_chain
from .vis import plot_diagram as plot_diagram
from .vis import plot_diagram_gradient as plot_diagram_gradient
from .vis import plot_matching as plot_matching
from .vis import DEFAULT_CHAIN_EDGE_STYLE as DEFAULT_CHAIN_EDGE_STYLE
from .vis import DEFAULT_CHAIN_TETRAHEDRON_STYLE as DEFAULT_CHAIN_TETRAHEDRON_STYLE
from .vis import DEFAULT_CHAIN_TRIANGLE_STYLE as DEFAULT_CHAIN_TRIANGLE_STYLE
from .vis import DEFAULT_CHAIN_VERTEX_STYLE as DEFAULT_CHAIN_VERTEX_STYLE
from .vis import DEFAULT_DENSITY_STYLE as DEFAULT_DENSITY_STYLE
from .vis import DEFAULT_DENSITY_THRESHOLD as DEFAULT_DENSITY_THRESHOLD
from .vis import DEFAULT_DIAGONAL_PROJECTION_A_STYLE as DEFAULT_DIAGONAL_PROJECTION_A_STYLE
from .vis import DEFAULT_DIAGONAL_PROJECTION_B_STYLE as DEFAULT_DIAGONAL_PROJECTION_B_STYLE
from .vis import DEFAULT_DIAGONAL_STYLE as DEFAULT_DIAGONAL_STYLE
from .vis import DEFAULT_DIAGRAM_A_POINT_STYLE as DEFAULT_DIAGRAM_A_POINT_STYLE
from .vis import DEFAULT_DIAGRAM_B_POINT_STYLE as DEFAULT_DIAGRAM_B_POINT_STYLE
from .vis import DEFAULT_DIAGRAM_GRADIENT_STYLE as DEFAULT_DIAGRAM_GRADIENT_STYLE
from .vis import DEFAULT_GRADIENT_TOP_K_ARROWS as DEFAULT_GRADIENT_TOP_K_ARROWS
from .vis import DEFAULT_GRID_STYLE as DEFAULT_GRID_STYLE
from .vis import DEFAULT_INF_LINE_STYLE as DEFAULT_INF_LINE_STYLE
from .vis import DEFAULT_INF_POINT_STYLE as DEFAULT_INF_POINT_STYLE
from .vis import DEFAULT_LONGEST_EDGE_STYLE as DEFAULT_LONGEST_EDGE_STYLE
from .vis import DEFAULT_MATCHING_EDGE_QUANTILE as DEFAULT_MATCHING_EDGE_QUANTILE
from .vis import DEFAULT_MATCHING_EDGE_STYLE as DEFAULT_MATCHING_EDGE_STYLE
from .vis import DEFAULT_POINT_CLOUD_STYLE as DEFAULT_POINT_CLOUD_STYLE
from .vis import DEFAULT_POINT_STYLE as DEFAULT_POINT_STYLE
from .vis import OKABE_ITO_BLUE as OKABE_ITO_BLUE
from .vis import OKABE_ITO_VERMILLION as OKABE_ITO_VERMILLION
from .vis import default_chain_edge_style as default_chain_edge_style
from .vis import default_chain_tetrahedron_style as default_chain_tetrahedron_style
from .vis import default_chain_triangle_style as default_chain_triangle_style
from .vis import default_chain_vertex_style as default_chain_vertex_style
from .vis import default_density_style as default_density_style
from .vis import default_diagonal_projection_a_style as default_diagonal_projection_a_style
from .vis import default_diagonal_projection_b_style as default_diagonal_projection_b_style
from .vis import default_diagonal_style as default_diagonal_style
from .vis import default_diagram_a_point_style as default_diagram_a_point_style
from .vis import default_diagram_b_point_style as default_diagram_b_point_style
from .vis import default_diagram_gradient_style as default_diagram_gradient_style
from .vis import default_grid_style as default_grid_style
from .vis import default_inf_line_style as default_inf_line_style
from .vis import default_inf_point_style as default_inf_point_style
from .vis import default_longest_edge_style as default_longest_edge_style
from .vis import default_matching_edge_style as default_matching_edge_style
from .vis import default_point_cloud_style as default_point_cloud_style
from .vis import default_point_style as default_point_style

__all__: list[str]
