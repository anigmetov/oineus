"""Internal helpers shared by differentiable filtration constructors."""
import numpy as np
import eagerpy as epy


def to_float64_numpy(data):
    """Return (eagerpy tensor, C-contiguous float64 numpy view) for a tensor-like input.

    The numpy copy is owning so that nanobind bindings which declare strict
    ``nb::ndarray<float64, c_contig>`` accept it — a non-owning view produced
    by ``eagerpy.numpy()`` on a torch tensor is otherwise rejected.
    """
    tensor = epy.astensor(data)
    np_data = np.array(tensor.float64().numpy(), dtype=np.float64, order="C")
    return tensor, np_data


def gather_values(tensor, critical_indices):
    """Gather values at flat indices while preserving the autograd graph.

    `tensor` is the eagerpy-wrapped differentiable input; `critical_indices`
    is a sequence (or numpy array) of flat indices into `tensor.flatten()`.
    Returns the raw framework tensor (unwrapped from eagerpy).
    """
    idx = np.asarray(critical_indices, dtype=np.int64)
    return tensor.flatten()[idx].raw
