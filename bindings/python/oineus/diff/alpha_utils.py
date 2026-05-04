"""Closed-form differentiable squared-circumradius helpers used by the alpha
filtration. Each function returns a 1-D tensor of squared circumradii. They
assume the inputs are *Gabriel* simplices, so no MEB fallback is taken.
"""
import torch


def edge_circumradius_sq(p0, p1):
    """Squared circumradius of an edge (= half-length squared).

    Args:
        p0, p1: tensors of shape ``(n, d)``.

    Returns:
        Tensor of shape ``(n,)``.
    """
    return 0.25 * torch.sum((p0 - p1) ** 2, dim=-1)


def triangle_circumradius_sq(p0, p1, p2, eps=1e-12):
    """Squared circumradius of triangles in 2D or 3D.

    Args:
        p0, p1, p2: tensors of shape ``(n, d)`` with ``d in {2, 3}``.
        eps: numerical-stability term.

    Returns:
        Tensor of shape ``(n,)``.
    """
    a = p1 - p0
    b = p2 - p0
    c = p2 - p1

    a_sq = torch.sum(a ** 2, dim=1)
    b_sq = torch.sum(b ** 2, dim=1)
    c_sq = torch.sum(c ** 2, dim=1)

    d = p0.shape[1]
    if d == 2:
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        area_2_sq = cross ** 2
    else:
        cross = torch.cross(a, b, dim=1)
        area_2_sq = torch.sum(cross ** 2, dim=1)

    return (a_sq * b_sq * c_sq + eps) / (4 * area_2_sq + eps)


def tetrahedron_circumradius_sq(p0, p1, p2, p3, eps=1e-12):
    """Squared circumradius of tetrahedra in 3D.

    Args:
        p0, p1, p2, p3: tensors of shape ``(n, 3)``.
        eps: numerical-stability term.

    Returns:
        Tensor of shape ``(n,)``.
    """
    a = p1 - p0
    b = p2 - p0
    c = p3 - p0

    a_sq = torch.sum(a ** 2, dim=1, keepdim=True)
    b_sq = torch.sum(b ** 2, dim=1, keepdim=True)
    c_sq = torch.sum(c ** 2, dim=1, keepdim=True)

    cross_bc = torch.cross(b, c, dim=1)
    cross_ca = torch.cross(c, a, dim=1)
    cross_ab = torch.cross(a, b, dim=1)

    volume_6 = torch.sum(a * cross_bc, dim=1)

    numerator_vec = a_sq * cross_bc + b_sq * cross_ca + c_sq * cross_ab
    circum_disp = numerator_vec / (2 * volume_6.unsqueeze(1) + eps)
    return torch.sum(circum_disp ** 2, dim=1)
