"""Torch-optionality of oineus.diff.

Runs a subprocess in which a sys.meta_path blocker makes torch
unimportable, then checks that (a) oineus and oineus.diff import,
(b) a jax gradient flows end to end through vr_filtration +
persistence_diagram, and (c) the torch-only functions raise a clear
ImportError. A subprocess is required because torch may already be
imported (and cached in sys.modules) in the main pytest process.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("jax")


BLOCKED_SCRIPT = r"""
import sys

class TorchBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked for this test")
        return None

sys.meta_path.insert(0, TorchBlocker())

import oineus
import oineus.diff as od
assert od.TORCH_AVAILABLE is False
assert "torch" not in sys.modules

import numpy as np
import jax
import jax.numpy as jnp

pts = jnp.asarray(np.random.default_rng(0).uniform(
    -1.0, 1.0, size=(5, 2)).astype(np.float32))

def loss(x):
    fil = od.vr_filtration(x, max_dim=1, max_diameter=10.0)
    d0 = od.persistence_diagram(fil, dualize=True)[0]
    return ((d0[:, 1] - d0[:, 0]) ** 2).sum()

g = np.asarray(jax.grad(loss)(pts))
assert np.isfinite(g).all()
assert (g ** 2).sum() > 0

# torch-only functions must raise a clear ImportError, not crash obscurely
for fn, args in [
    (od.wasserstein_cost, (jnp.zeros((2, 2)), jnp.zeros((2, 2)))),
    (od.sliced_wasserstein_distance, (jnp.zeros((2, 2)), jnp.zeros((2, 2)))),
    (od.cech_delaunay_filtration, (pts,)),
]:
    try:
        fn(*args)
    except ImportError as e:
        assert "torch" in str(e), e
    else:
        raise AssertionError(f"{fn.__name__} did not raise without torch")

assert "torch" not in sys.modules
print("TORCH-OPTIONAL-OK")
"""


def test_diff_works_without_torch():
    result = subprocess.run(
        [sys.executable, "-c", BLOCKED_SCRIPT],
        capture_output=True, text=True, cwd=os.getcwd(), timeout=300,
    )
    assert result.returncode == 0, (
        f"subprocess failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "TORCH-OPTIONAL-OK" in result.stdout
