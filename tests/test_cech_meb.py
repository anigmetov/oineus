"""Randomized + adversarial correctness tests for the MEB routines.

Both the torch versions (oineus.diff.cech_delaunay.triangle_meb /
tetrahedron_meb) and the numpy mirrors used by oineus.cech_filtration are
checked against an independent subset-circumball oracle, and against the
miniball package when it is installed. Full Cech complexes contain obtuse,
needle-like and (near-)coplanar simplices that the Delaunay path never
produces, hence the adversarial families.
"""

import numpy as np
import pytest

import oineus as oin
from oineus import _triangle_meb_sq_np, _tetrahedron_meb_sq_np

from meb_oracle import meb_radius_sq_oracle

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import miniball
    HAS_MINIBALL = True
except ImportError:
    HAS_MINIBALL = False


def random_triangles(rng, n, d, scale=1.0):
    return scale * rng.standard_normal((n, 3, d))


def adversarial_triangles(rng, d):
    out = []
    for eps in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        # near-collinear
        t = np.zeros((3, d))
        t[1, 0] = 1.0
        t[2, 0] = 2.0
        t[2, 1] = eps
        out.append(t + rng.standard_normal(d) * 0.0)
        # needle (very obtuse)
        t2 = np.zeros((3, d))
        t2[1, 0] = 1.0
        t2[2, 0] = 0.5
        t2[2, 1] = eps
        out.append(t2)
    # exactly collinear
    t3 = np.zeros((3, d))
    t3[1, 0] = 1.0
    t3[2, 0] = 3.0
    out.append(t3)
    # right triangle (obtuse-test tie)
    t4 = np.zeros((3, d))
    t4[1, 0] = 1.0
    t4[2, 1] = 1.0
    out.append(t4)
    return np.array(out)


def adversarial_tets(rng):
    out = []
    for eps in (1e-3, 1e-5, 1e-7, 1e-9, 0.0):
        # near/exactly coplanar square
        out.append(np.array([[0, 0, 0], [1, 0, 0], [1, 1, eps], [0, 1, 0]], dtype=np.float64))
        # near/exactly coplanar, obtuse
        out.append(np.array([[0, 0, 0], [3, 0, 0], [1, 0.2, eps], [2, 0.1, 0]], dtype=np.float64))
        # needle
        out.append(np.array([[0, 0, 0], [1, eps, 0], [2, 0, eps], [3, 0, 0]], dtype=np.float64))
        # sliver: two long skew edges
        out.append(np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, eps], [0.5, -1, eps]], dtype=np.float64))
    # 4 points on a circle (cocircular, coplanar)
    ang = np.array([0.3, 1.7, 3.1, 5.0])
    out.append(np.column_stack([np.cos(ang), np.sin(ang), np.zeros(4)]))
    return np.array(out)


def check_triangles(tris, rtol=1e-7):
    p0, p1, p2 = tris[:, 0], tris[:, 1], tris[:, 2]
    got = _triangle_meb_sq_np(p0, p1, p2)
    for i, tri in enumerate(tris):
        expected = meb_radius_sq_oracle(tri)
        assert got[i] == pytest.approx(expected, rel=rtol, abs=1e-14), (i, tri)
    return got


def check_tets(tets, rtol=1e-7):
    p = [tets[:, k] for k in range(4)]
    got = _tetrahedron_meb_sq_np(*p)
    for i, tet in enumerate(tets):
        expected = meb_radius_sq_oracle(tet)
        assert got[i] == pytest.approx(expected, rel=rtol, abs=1e-14), (i, tet)
    return got


@pytest.mark.parametrize("d", [2, 3])
def test_triangle_meb_random(d):
    rng = np.random.default_rng(42 + d)
    check_triangles(random_triangles(rng, 200, d))
    check_triangles(random_triangles(rng, 50, d, scale=1e-3))
    check_triangles(random_triangles(rng, 50, d, scale=1e3))


@pytest.mark.parametrize("d", [2, 3])
def test_triangle_meb_adversarial(d):
    rng = np.random.default_rng(7)
    check_triangles(adversarial_triangles(rng, d))


def test_tetrahedron_meb_random():
    rng = np.random.default_rng(43)
    check_tets(rng.standard_normal((200, 4, 3)))
    check_tets(1e-3 * rng.standard_normal((50, 4, 3)))
    check_tets(1e3 * rng.standard_normal((50, 4, 3)))


def test_tetrahedron_meb_adversarial():
    rng = np.random.default_rng(8)
    check_tets(adversarial_tets(rng))


@pytest.mark.skipif(not HAS_MINIBALL, reason="requires miniball")
def test_meb_vs_miniball():
    rng = np.random.default_rng(44)
    tets = rng.standard_normal((100, 4, 3))
    got = _tetrahedron_meb_sq_np(*(tets[:, k] for k in range(4)))
    for i, tet in enumerate(tets):
        _, r_sq = miniball.get_bounding_ball(tet)
        assert got[i] == pytest.approx(r_sq, rel=1e-6), i
    tris = rng.standard_normal((100, 3, 2))
    got = _triangle_meb_sq_np(tris[:, 0], tris[:, 1], tris[:, 2])
    for i, tri in enumerate(tris):
        _, r_sq = miniball.get_bounding_ball(tri)
        assert got[i] == pytest.approx(r_sq, rel=1e-6), i


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_numpy_consistency():
    from oineus.diff.cech_delaunay import tetrahedron_meb, triangle_meb
    rng = np.random.default_rng(45)

    tets = np.concatenate([rng.standard_normal((100, 4, 3)), adversarial_tets(rng)])
    tt = torch.from_numpy(tets)
    _, r_torch = tetrahedron_meb(tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3], eps=0.0)
    r_np = _tetrahedron_meb_sq_np(*(tets[:, k] for k in range(4)), eps=0.0)
    assert np.allclose(r_torch.numpy(), r_np, rtol=1e-12, atol=1e-14)

    for d in (2, 3):
        tris = np.concatenate([rng.standard_normal((100, 3, d)), adversarial_triangles(rng, d)])
        tt = torch.from_numpy(tris)
        _, r_torch = triangle_meb(tt[:, 0], tt[:, 1], tt[:, 2], eps=0.0)
        r_np = _triangle_meb_sq_np(tris[:, 0], tris[:, 1], tris[:, 2], eps=0.0)
        assert np.allclose(r_torch.numpy(), r_np, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_meb_gradients_finite_on_adversarial():
    from oineus.diff.cech_delaunay import tetrahedron_meb
    tets = torch.from_numpy(adversarial_tets(np.random.default_rng(9))).requires_grad_(True)
    _, r = tetrahedron_meb(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3], eps=0.0)
    r.sum().backward()
    assert torch.isfinite(tets.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
