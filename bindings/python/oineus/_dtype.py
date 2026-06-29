"""Real-dtype routing for the oineus Python package.

Oineus instantiates its Real-dependent C++ types for two dtypes. float64 (the
default) lives on the top ``_oineus`` module; float32, when compiled, lives in the
``_oineus._f32`` submodule with identical class names (the dtype is the submodule
path, hidden behind the public ``oineus.*`` facade). This module maps a numpy /
torch dtype to the right submodule so the factory functions can build float32
filtrations, diagrams and optimizers without any user-visible ``_f32``.

If the extension was built float64-only, ``_f32`` is absent and everything routes
to the top module (a float32 array is then upcast to float64).
"""

import numpy as np

from . import _oineus

_F64 = np.dtype("float64")
_F32 = np.dtype("float32")

# numpy dtype -> the C++ submodule that owns the Real-dependent types for it.
# float64 is the top module itself; float32 is the _f32 submodule when present.
REAL_MODULES = {_F64: _oineus}
if hasattr(_oineus, "_f32"):
    REAL_MODULES[_F32] = _oineus._f32

REAL_DTYPES = tuple(REAL_MODULES.keys())
DEFAULT_REAL_DTYPE = _F64

# Back-compat: historically the single compiled Real dtype. Kept so existing
# imports of REAL_DTYPE keep working; it is the default (float64).
REAL_DTYPE = DEFAULT_REAL_DTYPE


def detect_real_dtype(x):
    """Return the oineus Real dtype to use for ``x``.

    float32 if ``x`` is a float32 numpy array / torch tensor / jax array AND a
    float32 backend is compiled in; otherwise float64 (the default). Anything that
    is not a recognizable float32 array falls back to float64.
    """
    dt = None
    if isinstance(x, np.ndarray):
        dt = x.dtype
    else:
        tdt = getattr(x, "dtype", None)
        if tdt is not None:
            s = str(tdt)
            if "float32" in s:
                dt = _F32
            elif "float64" in s:
                dt = _F64
    if dt == _F32 and _F32 in REAL_MODULES:
        return _F32
    return DEFAULT_REAL_DTYPE


def real_module_for(x):
    """The C++ (sub)module whose Real-dependent classes match ``x``'s dtype."""
    return REAL_MODULES[detect_real_dtype(x)]


def dtype_of_oineus_obj(obj):
    """The numpy Real dtype of an oineus C++ object's backend -- float32 for a
    float32 filtration/diagram/optimizer (when a float32 build is present), else
    float64. For shaping a buffer (e.g. set_values input) to match an already-built
    object's Real type; module_of_oineus_obj is the module-valued counterpart."""
    if type(obj).__module__.endswith("._f32") and _F32 in REAL_MODULES:
        return _F32
    return DEFAULT_REAL_DTYPE


def module_of_oineus_obj(obj):
    """The C++ (sub)module that owns ``obj``'s class -- ``_oineus._f32`` for a
    float32 filtration/diagram/optimizer, else the top ``_oineus`` module. Used to
    route Real-dependent free functions (e.g. _mapping_cylinder, _min_filtration) to
    the backend matching an already-built C++ object."""
    return REAL_MODULES[dtype_of_oineus_obj(obj)]


def as_real_numpy(arr, dtype=None):
    """C-contiguous view of ``arr`` in a Real dtype (default float64).

    ``dtype`` selects the target Real (e.g. the one chosen by detect_real_dtype);
    casting/copying happens only when needed. Non-ndarray inputs pass through.
    """
    if dtype is None:
        dtype = DEFAULT_REAL_DTYPE
    if isinstance(arr, np.ndarray):
        if arr.dtype != dtype or not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=dtype)
    return arr
