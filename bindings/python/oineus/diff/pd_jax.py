"""JAX adapter for differentiable persistence diagrams.

Two paths over the framework-neutral pd_core:

- dgm-loss is a plain gather ``values[index_dgm]``: jax's native VJP of a
  gather is the scatter, identical to the torch dgm-loss backward, so no
  custom gradient is needed.

- crit-sets needs a real custom VJP (its backward set is larger than the
  gather's support). The backward re-reduces from the residual values with
  a fresh TopologyOptimizer -- pure (no live C++ state crosses the
  forward/backward seam; the residual is just the values array) at the
  cost of a second reduction per backward, acceptable given crit-sets'
  far lower step count.

oineus.diff is an eager boundary: diagram sizes are data-dependent, so
these calls cannot sit inside jax.jit / jax.vmap. Use them inside the
function you pass to jax.grad and jit the surrounding network/loss only.

Without ``jax.config.update("jax_enable_x64", True)`` jax arrays are
float32; the filtration constructors then build float32 filtrations
(routed to the float32 oineus backend when it is compiled in) and the
diagrams and gradients stay float32. Enable x64 for float64 end to end.
"""

import numpy as np
import jax
import jax.numpy as jnp

from . import pd_core


def _gather(values, index_dgm):
    if index_dgm.size == 0:
        return jnp.zeros((0, 2), dtype=values.dtype)
    return values[jnp.asarray(index_dgm)]


def jax_diagram(fil_values, fwd, dim):
    """Diagram array (n_d, 2) for one dimension, differentiable w.r.t.
    fil_values. fwd is the PDForward of the shared eager reduction."""
    index_dgm = fwd.index_dgm[dim]

    if fwd.method == "dgm-loss":
        # native gather; jax differentiates it into the scatter for free
        return _gather(fil_values, index_dgm)
    if fwd.method != "crit-sets":
        raise RuntimeError(f"Unknown gradient method: {fwd.method}")

    under_fil = fwd.top_opt.under_fil

    @jax.custom_vjp
    def crit_diagram(values):
        return _gather(values, index_dgm)

    def crit_fwd(values):
        # values is concrete here: custom_vjp forwards receive primals
        return _gather(values, index_dgm), values

    def crit_bwd(values, grad_output):
        # Pure re-reduce: rebuild the optimizer from the saved values
        # instead of reusing the (stateful) one from the eager forward.
        values_np = np.asarray(values)
        re_fwd = pd_core.pd_forward(
            under_fil, values_np,
            dualize=fwd.dualize,
            method=fwd.method,
            dims_to_backprop=fwd.dims_to_backprop,
            n_threads=fwd.n_threads,
            u_strategy=fwd.u_strategy,
            conflict_strategy=fwd.strategy,
            step_size=fwd.step_size,
            max_dim=fwd.max_dim,
        )
        grad_np = pd_core.pd_backward(
            re_fwd, dim, np.asarray(grad_output, dtype=values_np.dtype))
        return (jnp.asarray(grad_np),)

    crit_diagram.defvjp(crit_fwd, crit_bwd)
    return crit_diagram(fil_values)
