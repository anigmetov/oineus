"""Internal helpers shared by differentiable filtration constructors."""
import numpy as np
import eagerpy as epy

from .._dtype import REAL_DTYPE, DEFAULT_REAL_DTYPE, REAL_MODULES

_F32 = np.dtype("float32")


def real_dtype_for_tensor(tensor):
    """The oineus Real dtype to use for an eagerpy tensor: float32 if it is a
    float32 tensor AND a float32 backend is compiled in, else float64 (the
    default). Lets a float32 torch model build a genuine float32 filtration."""
    if "float32" in str(tensor.dtype) and _F32 in REAL_MODULES:
        return _F32
    return DEFAULT_REAL_DTYPE


def tensor_to_real_numpy(tensor, dtype=None):
    """Produce a contiguous Real-dtype numpy copy from an eagerpy tensor.

    The numpy copy is owning so that nanobind bindings which declare strict
    ``nb::ndarray<Real, c_contig>`` accept it -- a non-owning view produced by
    ``eagerpy.numpy()`` on a torch tensor is otherwise rejected. ``dtype`` selects
    the target Real (default: the tensor's matching Real via real_dtype_for_tensor).
    """
    if dtype is None:
        dtype = real_dtype_for_tensor(tensor)
    casted = tensor.float32() if np.dtype(dtype) == _F32 else tensor.float64()
    return np.array(casted.numpy(), dtype=dtype, order="C")


def gather_values(tensor, critical_indices):
    """Gather values at flat indices while preserving the autograd graph.

    `tensor` is the eagerpy-wrapped differentiable input; `critical_indices`
    is a sequence (or numpy array) of flat indices into `tensor.flatten()`.
    Returns the raw framework tensor (unwrapped from eagerpy).
    """
    idx = np.asarray(critical_indices, dtype=np.int64)
    return tensor.flatten()[idx].raw
