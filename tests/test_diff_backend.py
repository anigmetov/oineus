"""Backend detection in oineus.diff._backend.

infer_backend's fast path classifies by the module root of type(x), which
misses user-defined torch.Tensor subclasses (their module is the user's);
the isinstance fallback must catch them so persistence_diagram works on a
DiffFiltration whose values are a subclass instead of raising a misleading
TypeError.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import oineus as oin
import oineus.diff as od
from oineus.diff._backend import infer_backend


class MyTensor(torch.Tensor):
    pass


def test_infer_backend_torch_subclass():
    t = torch.zeros(3, dtype=torch.float64).as_subclass(MyTensor)
    assert type(t).__module__.split(".")[0] != "torch"  # fast path misses it
    assert infer_backend(t) == "torch"


def test_infer_backend_non_framework_types_stay_none():
    assert infer_backend(np.zeros(3)) is None
    assert infer_backend([1.0, 2.0]) is None
    assert infer_backend(None) is None


def _segment_diff_fil(values):
    """DiffFiltration over vertex 0, vertex 1, edge [0,1]; values must be
    strictly increasing so sorted order matches the input order."""
    vals = values.detach().cpu().numpy().tolist()
    simps = [(0, [0], vals[0]), (1, [1], vals[1]), (2, [0, 1], vals[2])]
    return od.DiffFiltration(oin.list_to_filtration(simps), values)


def test_persistence_diagram_works_on_torch_subclass_values():
    values = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64).as_subclass(MyTensor)
    values.requires_grad_()
    df = _segment_diff_fil(values)

    dgms = od.persistence_diagram(df, dualize=False)
    d0 = dgms[0]
    assert d0.shape == (1, 2)
    ((d0[:, 1] - d0[:, 0]) ** 2).sum().backward()

    # pair is (vertex 1, edge): loss gradient 2*(death-birth)*(-1, +1)
    g = values.grad.detach().cpu().numpy()
    np.testing.assert_allclose(g, [0.0, -0.2, 0.2], atol=1e-12)
