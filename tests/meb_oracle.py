"""Shared brute-force MEB oracle for the Cech tests.

Independent of the production MEB code: the MEB of <= d+2 points in R^d is
the circumball of some subset of the points, so minimize over circumballs
of all vertex subsets that contain the whole set.
"""

from itertools import combinations

import numpy as np


def circumball(points):
    """Center and squared radius of the smallest ball with all points ON it.

    Solves the linearized equidistance system relative to points[0]; the
    min-norm solution lies in the affine hull of the points, which makes
    the ball the smallest one through all of them.
    """
    p0 = points[0]
    rows = 2 * (points[1:] - p0)
    rhs = np.sum((points[1:] - p0) ** 2, axis=1)
    y, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    return p0 + y, np.sum(y * y)


def meb_radius_sq_oracle(points, tol=1e-12):
    """MEB squared radius of up to d+2 points via subset circumballs."""
    n = len(points)
    best = np.inf
    for k in range(2, n + 1):
        for subset in combinations(range(n), k):
            center, r_sq = circumball(points[list(subset)])
            dists_sq = np.sum((points - center) ** 2, axis=1)
            if np.all(dists_sq <= r_sq * (1 + 1e-9) + tol):
                best = min(best, r_sq)
    return best
