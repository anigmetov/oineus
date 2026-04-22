import eagerpy as epy

from .. import vr_filtration as non_diff_vr_filtration
from .diff_filtration import DiffFiltration
from ._tensor_utils import tensor_to_real_numpy


def vr_filtration(data, from_pwdists: bool = False, max_dim: int = -1,
                  max_diameter: float = -1.0, eps=1e-6, n_threads=8) -> DiffFiltration:
    data = epy.astensor(data)
    data_np = tensor_to_real_numpy(data)
    assert data.ndim == 2

    fil, edges = non_diff_vr_filtration(
        data=data_np,
        from_pwdists=from_pwdists,
        with_critical_edges=True,
        max_dim=max_dim,
        max_diameter=max_diameter,
        n_threads=n_threads,
    )

    if not from_pwdists:
        sqdists = epy.sum((data[edges[:, 0].flatten()] - data[edges[:, 1].flatten()]) ** 2, axis=1) + eps
        diff_dists = epy.sqrt(sqdists).raw
        return DiffFiltration(fil, diff_dists)
    else:
        edges = epy.astensor(edges)
        diff_dists = data[edges[:, 0], edges[:, 1]].raw
        return DiffFiltration(fil, diff_dists)
