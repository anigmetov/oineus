# Strategy enums the diff optimizer API consumes (combine_loss/crit_sets_apply take
# ConflictStrategy, simplify takes DenoiseStrategy, the optimizer takes UStrategy);
# re-exported so oineus.diff is self-contained without reaching into the top-level oineus.
from .._oineus import ConflictStrategy, UStrategy, DenoiseStrategy
from .diff_filtration import DiffFiltration
from .top_optimizer import TopologyOptimizer
from .freudenthal import freudenthal_filtration
from .cubical import cube_filtration
from .vietoris_rips import vr_filtration
from .mapping_cylinder import mapping_cylinder_filtration
from .min_filtration import min_filtration
from .persistence_diagram import PersistenceDiagrams, persistence_diagram
from .alpha import alpha_filtration
from .alpha_utils import edge_circumradius_sq, triangle_circumradius_sq, tetrahedron_circumradius_sq
from .weak_alpha import weak_alpha_filtration
# torch-only for now: these import without torch and raise a clear
# ImportError/TypeError when called without torch / with non-torch arrays
from .cech_delaunay import triangle_meb, tetrahedron_meb, cech_delaunay_filtration
from .sliced_wasserstein import sliced_wasserstein_distance, sliced_wasserstein_distance_diag_corrected
from .wasserstein import wasserstein_cost

# find_spec keeps torch optional: oineus.diff imports (and the jax paths work)
# without torch installed. When torch IS installed, the torch-only modules
# above import it eagerly; the pd_torch adapter is imported lazily on dispatch.
import importlib.util as _importlib_util
try:
    TORCH_AVAILABLE = _importlib_util.find_spec("torch") is not None
except (ImportError, ValueError):
    TORCH_AVAILABLE = False
