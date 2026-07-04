"""Backend detection and concrete-value extraction for oineus.diff.

The differentiable functions in oineus.diff accept torch tensors or jax
arrays and return the matching framework's tensors, gradients flowing.
This module holds the (tiny) dispatch machinery:

- infer_backend classifies an array by the module of its type;
- concrete_numpy produces a concrete numpy copy of a framework tensor,
  including jax tracers encountered inside jax.grad (via an identity
  jax.custom_vjp whose forward receives the concrete primal values);
- require_torch guards the functions that are still torch-only.

Neither torch nor jax is imported at module level: each framework is
imported lazily only when one of its arrays is actually seen, so
oineus.diff imports and works with only torch installed, only jax
installed, or neither.
"""

import numpy as np


def infer_backend(x):
    """Classify ``x``: "torch" for torch tensors, "jax" for jax arrays and
    tracers, None otherwise (numpy arrays, lists, ...)."""
    mod = type(x).__module__.split(".")[0]
    if mod == "torch":
        return "torch"
    if mod in ("jax", "jaxlib"):
        return "jax"
    return None


def jax_concrete_numpy(x):
    """Concrete numpy value of a jax array or eager-AD tracer.

    Inside jax.grad / jax.vjp (without jit) the primal computation runs
    eagerly, and a jax.custom_vjp forward receives concrete primal values;
    this helper exploits that to read the value wrapped by a tracer. Under
    jax.jit the values are abstract and jax raises
    TracerArrayConversionError: oineus.diff is an eager boundary and cannot
    be jit-traced through (diagram sizes are data-dependent).
    """
    import jax

    if not isinstance(x, jax.core.Tracer):
        return np.asarray(x)

    out = []

    @jax.custom_vjp
    def peek(v):
        out.append(np.asarray(v))
        return v

    def peek_fwd(v):
        # under AD transforms only this branch runs; v is the concrete primal
        out.append(np.asarray(v))
        return v, None

    def peek_bwd(_, ct):
        return (ct,)

    peek.defvjp(peek_fwd, peek_bwd)
    peek(x)
    return out[0]


def concrete_numpy(x):
    """Concrete numpy copy of a framework tensor (or array-like).

    torch tensors are detached and moved to host; jax arrays are converted
    directly; jax tracers are peeked via jax_concrete_numpy (eager AD only).
    Anything else goes through np.asarray.
    """
    backend = infer_backend(x)
    if backend == "torch":
        return x.detach().cpu().numpy()
    if backend == "jax":
        return jax_concrete_numpy(x)
    return np.asarray(x)


def require_torch(x, what):
    """Raise a clear error unless ``x`` is a torch tensor.

    Used by the functions that are still torch-only (wasserstein costs,
    sliced wasserstein, cech_delaunay): ImportError when torch is not
    installed, TypeError when a non-torch (e.g. jax) array is passed.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            f"{what} requires torch and torch is not installed; "
            f"{what} is torch-only for now") from None
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{what} is torch-only for now; got {type(x).__name__}, "
            f"pass torch tensors")
