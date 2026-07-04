"""Torch adapter for differentiable persistence diagrams.

One thin autograd.Function over the framework-neutral pd_core: forward
gathers the filtration values at the (birth, death) index diagram computed
by pd_core.pd_forward; backward hands the diagram gradient to
pd_core.pd_backward and wraps the resulting values-gradient back into a
torch tensor.
"""

import torch

from . import pd_core


class _TorchPDFunction(torch.autograd.Function):
    """One autograd Function per dimension. Forward subscripts fil values
    at birth/death indices; backward delegates to pd_core (dgm-loss scatter
    or crit-sets)."""

    @staticmethod
    def forward(ctx, fil_values, fwd, dim):
        index_dgm = fwd.index_dgm[dim]
        if index_dgm.size == 0:
            diagram = torch.zeros((0, 2), dtype=fil_values.dtype,
                                  device=fil_values.device)
        else:
            diagram = fil_values[torch.from_numpy(index_dgm).to(fil_values.device)]
        ctx.save_for_backward(fil_values)
        ctx.fwd = fwd
        ctx.dim = dim
        return diagram

    @staticmethod
    def backward(ctx, grad_output):
        (fil_values,) = ctx.saved_tensors
        grad_np = pd_core.pd_backward(ctx.fwd, ctx.dim,
                                      grad_output.detach().cpu().numpy())
        grad_vals = torch.from_numpy(grad_np).to(dtype=fil_values.dtype,
                                                 device=fil_values.device)
        return grad_vals, None, None


def torch_diagram(fil_values, fwd, dim):
    """Diagram tensor (n_d, 2) for one dimension, differentiable w.r.t.
    fil_values via the custom backward in pd_core."""
    return _TorchPDFunction.apply(fil_values, fwd, dim)
