#!/usr/bin/env python3
"""
Minimal example: differentiable persistence diagram + crit-sets backward.

Builds a small lower-star filtration on a random 2-D grid, takes
gradient steps that try to push the most-persistent H1 pair upward,
and prints the diagram trajectory.

Run:
    PYTHONPATH=build/bindings/python python examples/python/example_persistence_optim.py
"""

import numpy as np
import torch

import oineus.diff as oin_diff


def main():
    np.random.seed(7)
    torch.manual_seed(7)

    # 1) Build a differentiable filtration on a 16x16 grid of random
    #    values. The filtration's `values` is a leaf torch.Tensor, so
    #    gradients flow back through it.
    arr = torch.rand(16, 16, dtype=torch.float64, requires_grad=True)

    # 2) Optimization loop: at each step, rebuild the filtration on
    #    the current values, extract the diagram, and construct a
    #    loss that pushes the most-persistent H1 pair further from
    #    the diagonal. The crit-sets backward propagates gradients
    #    to the input tensor.
    optimizer = torch.optim.SGD([arr], lr=0.1)
    for step in range(5):
        # Rebuild the filtration on the current values; the previous
        # iteration's autograd graph is no longer reachable.
        fil = oin_diff.freudenthal_filtration(
            arr, max_dim=2, negate=False, wrap=False, n_threads=1)
        # u_strategy='auto' picks the production-default U-computation
        # path (currently row_partial with a partial-vs-full threshold
        # dispatch). Pin u_strategy='legacy_in_band' to use the
        # original in-band U as a cross-check; pin 'row_partial' to
        # bypass the threshold dispatch and always use partial.
        dgms = oin_diff.persistence_diagram(
            fil,
            gradient_method="crit-sets",
            conflict_strategy="avg",
            n_threads=4,
            # u_strategy="auto",  # default
        )
        dgm1 = dgms.in_dimension(1)
        if dgm1.shape[0] == 0:
            print(f"step {step}: no H1 pair, stopping")
            break

        # Pick the most persistent pair and push its death up by 0.1.
        persistence = dgm1[:, 1] - dgm1[:, 0]
        i = int(persistence.argmax().item())
        target_death = dgm1[i, 1].detach() + 0.1
        loss = (dgm1[i, 1] - target_death) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b, d = dgm1[i].detach().tolist()
        print(f"step {step}: H1 pairs = {dgm1.shape[0]}, "
              f"top-persistence pair = ({b:.3f}, {d:.3f}), "
              f"loss = {float(loss):.4e}")


if __name__ == "__main__":
    main()
