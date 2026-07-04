#!python3
"""Mixup barcodes on synthetic data (Wagner et al., arXiv:2402.15058).

Script-style copy of examples/python/mixup_example.py, run by ctest as
py-example-mixup; keep the two in sync.

Two configurations of a noisy circle A and a second cloud B:

1. B far away from A: no interaction, all mixup statistics are zero.
2. B sprinkled inside the circle: the points of B fill the hole of A,
   the degree-1 bar gets a long mixup sub-bar and the mean mixup
   percentage jumps.

Also shows the differentiable variant when torch is available: the
gradient of the total mixup with respect to the coordinates of B points
inward -- following it would push B deeper into the hole of A.
"""

import numpy as np
import oineus as oin

rng = np.random.default_rng(42)


def noisy_circle(n, radius=1.0, noise=0.05):
    angles = rng.uniform(0, 2 * np.pi, n)
    pts = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return pts + rng.normal(0, noise, pts.shape)


A = noisy_circle(30)
B_far = rng.uniform(4.0, 5.0, (10, 2))       # far away from A
B_in = rng.uniform(-0.4, 0.4, (10, 2))       # inside the hole of A

for name, B in (("B far away", B_far), ("B inside the hole", B_in)):
    mb = oin.mixup_barcodes(A, B, max_dim=1)
    print(f"--- {name} ---")
    print(mb)
    print(f"deg 0: total mixup {mb.total_mixup(0):.4f}, "
          f"mean mixup % {mb.mean_mixup_percentage(0):.4f}")
    print(f"deg 1: total mixup {mb.total_mixup(1):.4f}, "
          f"mean mixup % {mb.mean_mixup_percentage(1):.4f}")
    longest = max(mb.in_dimension(1), key=lambda t: t[2] - t[0])
    b, dp, d = longest
    print(f"most persistent hole of A: born {b:.3f}, dies {d:.3f}, "
          f"premature death {dp:.3f} (mixup {d - dp:.3f})\n")

far = oin.mixup_barcodes(A, B_far, max_dim=1)
inside = oin.mixup_barcodes(A, B_in, max_dim=1)
# B far away does not interact with A at all
assert far.total_mixup(0) == 0.0 and far.total_mixup(1) == 0.0
# B inside the hole fills it early (encirclement, degree 1) but does not
# touch A itself, so the degree-0 mixup (overlap) stays zero
assert inside.total_mixup(1) > 0.1
assert inside.mean_mixup_percentage(1) > far.mean_mixup_percentage(1)
assert inside.total_mixup(0) == 0.0

# subsampling lemma: fewer B points, at most the same mixup
half = oin.mixup_barcodes(A, B_in[:5], max_dim=1)
assert half.total_mixup(1) <= inside.total_mixup(1) + 1e-12

# differentiable variant (torch): gradients reach both A and B
try:
    import torch
    import oineus.diff
except ImportError:
    print("torch not available, skipping the differentiable part")
else:
    At = torch.tensor(A, requires_grad=True, dtype=torch.float64)
    Bt = torch.tensor(B_in, requires_grad=True, dtype=torch.float64)
    dmb = oineus.diff.mixup_barcodes(At, Bt, max_dim=1)
    loss = -dmb.total_mixup(1)  # maximize the entanglement
    loss.backward()
    assert At.grad is not None and Bt.grad is not None
    assert Bt.grad.abs().sum() > 0
    print(f"diff total mixup (deg 1): {float(dmb.total_mixup(1).detach()):.4f}, "
          f"|grad B| = {float(Bt.grad.norm()):.4f}")

print("mixup_example: all checks passed")
