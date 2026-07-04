"""Closed-form differentiable squared-circumradius helpers used by the alpha
filtration. Each function returns a 1-D tensor of squared circumradii in the
input's framework (torch or jax, via eagerpy). They assume the inputs are
*Gabriel* simplices, so no MEB fallback is taken.
"""
import eagerpy as epy


def _cross3(a, b):
    """Cross product of (n, 3) eagerpy tensors, written componentwise so it
    works for every backend (eagerpy has no cross)."""
    return epy.stack([
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    ], axis=1)


def edge_circumradius_sq(p0, p1):
    """Squared circumradius of an edge (= half-length squared).

    Args:
        p0, p1: tensors of shape ``(n, d)``.

    Returns:
        Tensor of shape ``(n,)`` in the input framework.
    """
    p0, p1 = epy.astensor(p0), epy.astensor(p1)
    return (0.25 * ((p0 - p1) ** 2).sum(axis=-1)).raw


def triangle_circumradius_sq(p0, p1, p2, eps=1e-12):
    """Squared circumradius of triangles in 2D or 3D.

    Args:
        p0, p1, p2: tensors of shape ``(n, d)`` with ``d in {2, 3}``.
        eps: numerical-stability term.

    Returns:
        Tensor of shape ``(n,)`` in the input framework.
    """
    p0, p1, p2 = epy.astensor(p0), epy.astensor(p1), epy.astensor(p2)
    a = p1 - p0
    b = p2 - p0
    c = p2 - p1

    a_sq = (a ** 2).sum(axis=1)
    b_sq = (b ** 2).sum(axis=1)
    c_sq = (c ** 2).sum(axis=1)

    d = p0.shape[1]
    if d == 2:
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        area_2_sq = cross ** 2
    else:
        cross = _cross3(a, b)
        area_2_sq = (cross ** 2).sum(axis=1)

    return ((a_sq * b_sq * c_sq + eps) / (4 * area_2_sq + eps)).raw


def tetrahedron_circumradius_sq(p0, p1, p2, p3, eps=1e-12):
    """Squared circumradius of tetrahedra in 3D.

    Args:
        p0, p1, p2, p3: tensors of shape ``(n, 3)``.
        eps: numerical-stability term.

    Returns:
        Tensor of shape ``(n,)`` in the input framework.
    """
    p0, p1, p2, p3 = (epy.astensor(p0), epy.astensor(p1),
                      epy.astensor(p2), epy.astensor(p3))
    a = p1 - p0
    b = p2 - p0
    c = p3 - p0

    a_sq = (a ** 2).sum(axis=1, keepdims=True)
    b_sq = (b ** 2).sum(axis=1, keepdims=True)
    c_sq = (c ** 2).sum(axis=1, keepdims=True)

    cross_bc = _cross3(b, c)
    cross_ca = _cross3(c, a)
    cross_ab = _cross3(a, b)

    volume_6 = (a * cross_bc).sum(axis=1)

    numerator_vec = a_sq * cross_bc + b_sq * cross_ca + c_sq * cross_ab
    circum_disp = numerator_vec / (2 * epy.expand_dims(volume_6, axis=1) + eps)
    return (circum_disp ** 2).sum(axis=1).raw
