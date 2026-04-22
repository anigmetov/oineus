"""Internal helpers shared by differentiable filtration constructors."""
import numpy as np
import eagerpy as epy

from .._dtype import REAL_DTYPE


def tensor_to_real_numpy(tensor):
    """Produce a contiguous Real-dtype numpy copy from an eagerpy tensor.

    The numpy copy is owning so that nanobind bindings which declare strict
    ``nb::ndarray<Real, c_contig>`` accept it — a non-owning view produced
    by ``eagerpy.numpy()`` on a torch tensor is otherwise rejected. The dtype
    matches how ``_oineus`` was compiled (``float32`` if Real=float,
    ``float64`` if Real=double).
    """
    if REAL_DTYPE == np.float32:
        casted = tensor.float32()
    else:
        casted = tensor.float64()
    return np.array(casted.numpy(), dtype=REAL_DTYPE, order="C")


def gather_values(tensor, critical_indices):
    """Gather values at flat indices while preserving the autograd graph.

    `tensor` is the eagerpy-wrapped differentiable input; `critical_indices`
    is a sequence (or numpy array) of flat indices into `tensor.flatten()`.
    Returns the raw framework tensor (unwrapped from eagerpy).
    """
    idx = np.asarray(critical_indices, dtype=np.int64)
    return tensor.flatten()[idx].raw
