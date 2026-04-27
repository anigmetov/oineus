from .diff_filtration import DiffFiltration
from .top_optimizer import TopologyOptimizer
from .freudenthal import freudenthal_filtration
from .cubical import cube_filtration
from .vietoris_rips import vr_filtration
from .mapping_cylinder import mapping_cylinder_filtration
from .min_filtration import min_filtration

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .cech_delaunay import triangle_meb, tetrahedron_meb, cech_delaunay_filtration
    from .weak_alpha import weak_alpha_filtration
    from .persistence_diagram import PersistenceDiagrams, persistence_diagram
    from .sliced_wasserstein import sliced_wasserstein_distance, sliced_wasserstein_distance_diag_corrected
    from .wasserstein import wasserstein_cost
