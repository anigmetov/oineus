import numpy as np
import typing
from typing import Optional, List, Tuple


def generate_two_circles(n_points_small=5, n_points_large=7,
                        radius_small=1.0, radius_large=3.0,
                        noise_level=0.5):
    """Generate points sampled from two concentric circles with noise."""

    # Small circle
    theta_small = np.linspace(0, 2*np.pi, n_points_small, endpoint=False)
    small_circle = np.column_stack([
        radius_small * np.cos(theta_small),
        radius_small * np.sin(theta_small)
    ])
    # Add noise
    small_circle += np.random.normal(0, noise_level * radius_small, small_circle.shape)

    # Large circle - offset center
    x_displacement = 9.0
    theta_large = np.linspace(0, 2*np.pi, n_points_large, endpoint=False)
    large_circle = np.column_stack([
        radius_large * np.cos(theta_large) + x_displacement,
        radius_large * np.sin(theta_large)
    ])
    # Add noise
    large_circle += np.random.normal(0, noise_level * radius_large, large_circle.shape)

    # Combine both circles
    points = np.vstack([small_circle, large_circle])

    print("generated points")

    return points



def generate_holey_square(n_points: int, radii: List[float], max_attempts: int = 5000) -> np.ndarray:
    """
    Sample a point cloud from a square with circular holes.

    Parameters:
    -----------
    n_points : int
        Number of points to sample
    radii : List[float]
        List of radii for the holes (corresponding to death values in 1D persistence)
    max_attempts : int
        Maximum number of attempts to place each circle center

    Returns:
    --------
    points : np.ndarray
        Array of shape (n_points, 2) containing the sampled points
    """
    # Step a: Sample circle centers ensuring they don't overlap
    centers = []

    for i, r in enumerate(radii):
        attempts = 0
        while attempts < max_attempts:
            # Sample a random center in the unit square [0, 1]^2
            # Keep it away from boundaries to ensure circle fits
            margin = r + 0.05
            if margin >= 0.5:
                raise ValueError(f"Radius {r} is too large to fit in unit square with separation")

            center = np.random.uniform(margin, 1 - margin, 2)

            # Check if this center is too close to existing centers
            valid = True
            for j, (c, prev_r) in enumerate(centers):
                # Required distance: sum of radii plus some separation
                min_dist = r + prev_r + 0.05
                dist = np.linalg.norm(center - c)
                if dist < min_dist:
                    valid = False
                    break

            if valid:
                centers.append((center, r))
                break

            attempts += 1

        if attempts >= max_attempts:
            raise RuntimeError(f"Could not place circle {i+1} with radius {r} after {max_attempts} attempts")

    # Step b: Sample points from a larger square, excluding the circles
    k = 10  # Oversample factor
    outer_size = 1.01  # Size of outer square
    offset = (outer_size - 1.0) / 2  # Offset to center the unit square

    # Sample many points
    candidates = np.random.uniform(-offset, 1 + offset, (k * n_points, 2))

    # Remove points inside any of the circles
    valid_mask = np.ones(len(candidates), dtype=bool)
    for center, r in centers:
        distances = np.linalg.norm(candidates - center, axis=1)
        valid_mask &= (distances > r)

    valid_points = candidates[valid_mask]

    # Check if we have enough points
    if len(valid_points) < n_points:
        raise RuntimeError(f"Could not sample enough points. Got {len(valid_points)}, needed {n_points}. Try larger k or smaller radii.")

    # Randomly select exactly n_points
    indices = np.random.choice(len(valid_points), n_points, replace=False)
    points = valid_points[indices]

    return points, centers




def sample_spheres(
    n_spheres: int,
    n_points_per_sphere: int,
    dim: int = 2,
    centers: Optional[np.ndarray] = None,
    radii: Optional[List[float]] = None,
    sigma: float = 0.05
) -> np.ndarray:
    """
    Sample points from n_spheres spheres in dim-dimensional space.

    Args:
        n_spheres: Number of spheres
        n_points_per_sphere: Points to sample per sphere
        dim: Dimension of the ambient space (sphere is (dim-1)-dimensional)
        centers: Array of shape (n_spheres, dim) with sphere centers
        radii: List of n_spheres radii
        sigma: Standard deviation of Gaussian noise to add

    Returns:
        Array of shape (n_spheres * n_points_per_sphere, dim)
    """
    if centers is None:
        centers = np.random.uniform(-2, 2, size=(n_spheres, dim))

    if radii is None:
        radii = np.random.uniform(0.5, 1.5, size=n_spheres)

    all_points = []

    for i in range(n_spheres):
        # Sample from standard normal and normalize to get uniform distribution on sphere
        points = np.random.randn(n_points_per_sphere, dim)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Scale by radius and translate to center
        points = points * radii[i] + centers[i]

        # Add noise
        points += np.random.randn(n_points_per_sphere, dim) * sigma

        all_points.append(points)

    return np.vstack(all_points)


def sample_tori(
    n_tori: int,
    n_points_per_torus: int,
    major_radii: Optional[List[float]] = None,
    minor_radii: Optional[List[float]] = None,
    centers: Optional[np.ndarray] = None,
    sigma: float = 0.05,
    separate: bool = False
) -> np.ndarray:
    """
    Sample points from n_tori 2D tori embedded in 3D space.

    Args:
        n_tori: Number of tori
        n_points_per_torus: Points to sample per torus
        major_radii: List of major radii (R)
        minor_radii: List of minor radii (r)
        centers: Array of shape (n_tori, 3) with torus centers
        sigma: Standard deviation of Gaussian noise
        separate: If True, ensure tori are non-intersecting with significant separation

    Returns:
        Array of shape (n_tori * n_points_per_torus, 3)
    """
    if major_radii is None:
        major_radii = np.random.uniform(1.5, 2.5, size=n_tori)

    if minor_radii is None:
        minor_radii = np.random.uniform(0.3, 0.8, size=n_tori)

    if centers is None:
        if separate:
            # Generate centers with guaranteed separation
            centers = np.zeros((n_tori, 3))
            max_outer_radius = np.max(major_radii) + np.max(minor_radii)
            min_separation = 3 * max_outer_radius

            for i in range(n_tori):
                if i == 0:
                    centers[i] = np.random.uniform(-3, 3, size=3)
                else:
                    # Keep generating random positions until we find one that's far enough
                    max_attempts = 1000
                    for _ in range(max_attempts):
                        candidate = np.random.uniform(-10, 10, size=3)
                        distances = np.linalg.norm(centers[:i] - candidate, axis=1)
                        if np.all(distances >= min_separation):
                            centers[i] = candidate
                            break
                    else:
                        # If we can't find a good position, use a grid-based approach
                        angle = 2 * np.pi * i / n_tori
                        radius = min_separation * 1.5
                        centers[i] = np.array([
                            radius * np.cos(angle),
                            radius * np.sin(angle),
                            (i - n_tori/2) * min_separation / n_tori
                        ])
        else:
            centers = np.random.uniform(-3, 3, size=(n_tori, 3))

    all_points = []

    for i in range(n_tori):
        # Sample angles uniformly
        theta = np.random.uniform(0, 2*np.pi, n_points_per_torus)
        phi = np.random.uniform(0, 2*np.pi, n_points_per_torus)

        R = major_radii[i]
        r = minor_radii[i]

        # Parametric equations for torus
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)

        points = np.column_stack([x, y, z]) + centers[i]

        # Add noise
        points += np.random.randn(n_points_per_torus, 3) * sigma

        all_points.append(points)

    return np.vstack(all_points)


def sample_random(
    n_points: int,
    dim: int,
    distr: str = "normal",
    bounds: Tuple[float, float] = (-3, 3)
) -> np.ndarray:
    """
    Sample random points from a distribution.

    Args:
        n_points: Number of points to sample
        dim: Dimension of the space
        distr: Distribution type ("normal" or "uniform")
        bounds: Bounds for uniform distribution

    Returns:
        Array of shape (n_points, dim)
    """
    if distr == "normal":
        return np.random.randn(n_points, dim)
    elif distr == "uniform":
        return np.random.uniform(bounds[0], bounds[1], size=(n_points, dim))
    else:
        raise ValueError(f"Unknown distribution: {distr}")


def sample_annulus(
    n_points: int,
    inner_radius: float = 1.0,
    outer_radius: float = 2.0,
    center: Optional[np.ndarray] = None,
    sigma: float = 0.05
) -> np.ndarray:
    """
    Sample points from an annulus (2D ring) in the plane.

    Args:
        n_points: Number of points to sample
        inner_radius: Inner radius of annulus
        outer_radius: Outer radius of annulus
        center: Center of annulus (default [0, 0])
        sigma: Standard deviation of Gaussian noise

    Returns:
        Array of shape (n_points, 2)
    """
    if center is None:
        center = np.zeros(2)

    # Sample radius using inverse transform sampling for uniform area distribution
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, n_points))
    theta = np.random.uniform(0, 2*np.pi, n_points)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    points = np.column_stack([x, y]) + center
    points += np.random.randn(n_points, 2) * sigma

    return points


def sample_figure_eight(
    n_points: int,
    radius: float = 1.0,
    sigma: float = 0.05
) -> np.ndarray:
    """
    Sample points from a figure-eight (two circles touching at origin).

    Args:
        n_points: Number of points to sample
        radius: Radius of each circle
        sigma: Standard deviation of Gaussian noise

    Returns:
        Array of shape (n_points, 2)
    """
    # Split points between two circles
    n_per_circle = n_points // 2

    # Left circle centered at (-radius, 0)
    theta1 = np.random.uniform(0, 2*np.pi, n_per_circle)
    x1 = -radius + radius * np.cos(theta1)
    y1 = radius * np.sin(theta1)

    # Right circle centered at (radius, 0)
    theta2 = np.random.uniform(0, 2*np.pi, n_points - n_per_circle)
    x2 = radius + radius * np.cos(theta2)
    y2 = radius * np.sin(theta2)

    points = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])

    points += np.random.randn(n_points, 2) * sigma

    return points


def sample_swiss_roll(
    n_points: int,
    noise: float = 0.1,
    length: float = 15.0
) -> np.ndarray:
    """
    Sample points from a Swiss roll manifold.

    Args:
        n_points: Number of points
        noise: Amount of noise to add
        length: Length of the roll

    Returns:
        Array of shape (n_points, 3)
    """
    t = np.random.uniform(1.5*np.pi, 4.5*np.pi, n_points)
    h = np.random.uniform(0, length, n_points)

    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)

    points = np.column_stack([x, y, z])
    points += np.random.randn(n_points, 3) * noise

    return points


def sample_trefoil_knot(
    n_points: int,
    sigma: float = 0.05
) -> np.ndarray:
    """
    Sample points from a trefoil knot.

    Args:
        n_points: Number of points
        sigma: Standard deviation of Gaussian noise

    Returns:
        Array of shape (n_points, 3)
    """
    t = np.random.uniform(0, 2*np.pi, n_points)

    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)

    points = np.column_stack([x, y, z])
    points += np.random.randn(n_points, 3) * sigma

    return points


def sample_linked_circles(
    n_points_per_circle: int,
    radius: float = 1.0,
    separation: float = 2.0,
    sigma: float = 0.05
) -> np.ndarray:
    """
    Sample points from two linked circles (simplified Hopf link).

    Args:
        n_points_per_circle: Number of points per circle
        radius: Radius of each circle
        separation: Distance between circle centers
        sigma: Standard deviation of Gaussian noise

    Returns:
        Array of shape (2 * n_points_per_circle, 3)
    """
    # First circle in xy-plane
    theta1 = np.random.uniform(0, 2*np.pi, n_points_per_circle)
    circle1 = np.column_stack([
        radius * np.cos(theta1),
        radius * np.sin(theta1),
        np.zeros(n_points_per_circle)
    ])

    # Second circle in yz-plane, shifted along x-axis
    theta2 = np.random.uniform(0, 2*np.pi, n_points_per_circle)
    circle2 = np.column_stack([
        np.full(n_points_per_circle, separation),
        radius * np.cos(theta2),
        radius * np.sin(theta2)
    ])

    points = np.vstack([circle1, circle2])
    points += np.random.randn(2 * n_points_per_circle, 3) * sigma

    return points


def sample_clusters(
    n_clusters: int,
    n_points_per_cluster: int,
    dim: int = 2,
    cluster_std: float = 0.3,
    centers: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Sample points from well-separated Gaussian clusters.

    Args:
        n_clusters: Number of clusters
        n_points_per_cluster: Points per cluster
        dim: Dimension of the space
        cluster_std: Standard deviation within each cluster
        centers: Array of shape (n_clusters, dim) with cluster centers

    Returns:
        Array of shape (n_clusters * n_points_per_cluster, dim)
    """
    if centers is None:
        centers = np.random.uniform(-5, 5, size=(n_clusters, dim))

    all_points = []

    for i in range(n_clusters):
        points = np.random.randn(n_points_per_cluster, dim) * cluster_std
        points += centers[i]
        all_points.append(points)

    return np.vstack(all_points)


def sample_mobius_strip(
    n_points: int,
    radius: float = 2.0,
    width: float = 1.0,
    sigma: float = 0.05
) -> np.ndarray:
    """
    Sample points from a Möbius strip.

    Args:
        n_points: Number of points
        radius: Radius of the strip's centerline
        width: Width of the strip
        sigma: Standard deviation of Gaussian noise

    Returns:
        Array of shape (n_points, 3)
    """
    t = np.random.uniform(0, 2*np.pi, n_points)
    s = np.random.uniform(-width/2, width/2, n_points)

    x = (radius + s * np.cos(t/2)) * np.cos(t)
    y = (radius + s * np.cos(t/2)) * np.sin(t)
    z = s * np.sin(t/2)

    points = np.column_stack([x, y, z])
    points += np.random.randn(n_points, 3) * sigma

    return points


def sample_sierpinski_triangle(
    n_points: int,
    iterations: int = 10000
) -> np.ndarray:
    """
    Sample points from a Sierpinski triangle using the chaos game.

    Args:
        n_points: Number of points (will be <= iterations)
        iterations: Number of iterations to run

    Returns:
        Array of shape (n_points, 2)
    """
    # Triangle vertices
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ])

    # Start at random point
    point = np.random.rand(2)
    points = []

    for i in range(iterations):
        # Choose random vertex
        vertex = vertices[np.random.randint(3)]
        # Move halfway to that vertex
        point = (point + vertex) / 2

        if i > 100:  # Skip initial transient
            points.append(point.copy())

        if len(points) >= n_points:
            break

    return np.array(points[:n_points])


# Functions for Lower-star


def rosenbrock_2d(nx, ny, a=1, b=100):
    """Rosenbrock function: classic optimization test function."""
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-1, 3, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return (a - X)**2 + b * (Y - X**2)**2


def rastrigin_2d(nx, ny):
    """Rastrigin function: many local minima."""
    x = np.linspace(-5.12, 5.12, nx)
    y = np.linspace(-5.12, 5.12, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return 20 + X**2 + Y**2 - 10 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))



# ============================================================================
# 1D Functions
# ============================================================================

def random_gaussian_1d(n, complexity=3):
    """Sum of random Gaussians in 1D.

    Args:
        n: Number of grid points
        complexity: Number of Gaussian peaks
    """
    # Random domain
    xmin, xmax = np.random.uniform(-2, 0), np.random.uniform(1, 3)
    x = np.linspace(xmin, xmax, n)

    # Random centers and sigmas
    centers = np.random.uniform(xmin, xmax, complexity)
    sigmas = np.random.uniform(0.05, 0.3, complexity) * (xmax - xmin)
    amplitudes = np.random.uniform(0.5, 2.0, complexity)

    result = np.zeros(n)
    for c, s, a in zip(centers, sigmas, amplitudes):
        result += a * np.exp(-((x - c) ** 2) / (2 * s ** 2))

    return result


def random_poly_1d(n, complexity=4):
    """Random polynomial with specified degree.

    Args:
        n: Number of grid points
        complexity: Degree of polynomial
    """
    # Random domain
    xmin, xmax = np.random.uniform(-3, -1), np.random.uniform(1, 3)
    x = np.linspace(xmin, xmax, n)

    # Random coefficients (higher degrees get smaller coefficients for stability)
    coeffs = np.random.uniform(-1, 1, complexity + 1)
    for i in range(complexity + 1):
        coeffs[i] /= (i + 1) ** 0.5

    result = np.zeros(n)
    for i, c in enumerate(coeffs):
        result += c * x ** i

    return result


def random_trig_1d(n, complexity=3):
    """Random trigonometric sum.

    Args:
        n: Number of grid points
        complexity: Highest frequency
    """
    # Random domain
    domain_length = np.random.uniform(2 * np.pi, 4 * np.pi)
    x = np.linspace(0, domain_length, n)

    result = np.zeros(n)
    # Include frequencies from 1 to complexity
    for k in range(1, complexity + 1):
        amp_sin = np.random.uniform(-1, 1)
        amp_cos = np.random.uniform(-1, 1)
        phase = np.random.uniform(0, 2 * np.pi)
        result += amp_sin * np.sin(k * x + phase) + amp_cos * np.cos(k * x + phase)

    return result


# ============================================================================
# 2D Functions
# ============================================================================

def random_gaussian_2d(nx, ny, complexity=5):
    """Sum of random 2D Gaussian peaks.

    Args:
        nx, ny: Number of grid points
        complexity: Number of Gaussian peaks
    """
    # Random domain
    xmin, xmax = np.random.uniform(-2, 0), np.random.uniform(1, 3)
    ymin, ymax = np.random.uniform(-2, 0), np.random.uniform(1, 3)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    result = np.zeros((nx, ny))

    for _ in range(complexity):
        cx = np.random.uniform(xmin, xmax)
        cy = np.random.uniform(ymin, ymax)
        sigma = np.random.uniform(0.05, 0.2) * min(xmax - xmin, ymax - ymin)
        amplitude = np.random.uniform(0.5, 2.0)

        result += amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    return result


def random_poly_2d(nx, ny, complexity=3):
    """Random polynomial in 2D with total degree = complexity.

    Args:
        nx, ny: Number of grid points
        complexity: Maximum total degree (i + j <= complexity)
    """
    # Random domain
    xmin, xmax = np.random.uniform(-2, 0), np.random.uniform(1, 3)
    ymin, ymax = np.random.uniform(-2, 0), np.random.uniform(1, 3)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    result = np.zeros((nx, ny))

    for i in range(complexity + 1):
        for j in range(complexity + 1 - i):
            if i + j <= complexity:
                coeff = np.random.uniform(-1, 1) / ((i + j + 1) ** 0.5)
                result += coeff * (X ** i) * (Y ** j)

    return result


def random_trig_2d(nx, ny, complexity=3):
    """Random trigonometric polynomial in 2D.

    Args:
        nx, ny: Number of grid points
        complexity: Highest frequency in each direction
    """
    # Random domain
    x_length = np.random.uniform(2 * np.pi, 4 * np.pi)
    y_length = np.random.uniform(2 * np.pi, 4 * np.pi)

    x = np.linspace(0, x_length, nx)
    y = np.linspace(0, y_length, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    result = np.zeros((nx, ny))

    for kx in range(1, complexity + 1):
        for ky in range(1, complexity + 1):
            amp = np.random.uniform(-1, 1)
            phase_x = np.random.uniform(0, 2 * np.pi)
            phase_y = np.random.uniform(0, 2 * np.pi)
            result += amp * np.sin(kx * X + phase_x) * np.cos(ky * Y + phase_y)

    return result


# ============================================================================
# 3D Functions
# ============================================================================

def random_gaussian_3d(nx, ny, nz, complexity=5):
    """Sum of random 3D Gaussian peaks.

    Args:
        nx, ny, nz: Number of grid points
        complexity: Number of Gaussian peaks
    """
    # Random domain
    xmin, xmax = np.random.uniform(-2, 0), np.random.uniform(1, 3)
    ymin, ymax = np.random.uniform(-2, 0), np.random.uniform(1, 3)
    zmin, zmax = np.random.uniform(-2, 0), np.random.uniform(1, 3)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    result = np.zeros((nx, ny, nz))

    for _ in range(complexity):
        cx = np.random.uniform(xmin, xmax)
        cy = np.random.uniform(ymin, ymax)
        cz = np.random.uniform(zmin, zmax)
        sigma = np.random.uniform(0.05, 0.2) * min(xmax - xmin, ymax - ymin, zmax - zmin)
        amplitude = np.random.uniform(0.5, 2.0)

        result += amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * sigma**2))

    return result


def random_poly_3d(nx, ny, nz, complexity=3):
    """Random polynomial in 3D with total degree = complexity.

    Args:
        nx, ny, nz: Number of grid points
        complexity: Maximum total degree (i + j + k <= complexity)
    """
    # Random domain
    xmin, xmax = np.random.uniform(-2, 0), np.random.uniform(1, 3)
    ymin, ymax = np.random.uniform(-2, 0), np.random.uniform(1, 3)
    zmin, zmax = np.random.uniform(-2, 0), np.random.uniform(1, 3)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    result = np.zeros((nx, ny, nz))

    for i in range(complexity + 1):
        for j in range(complexity + 1 - i):
            for k in range(complexity + 1 - i - j):
                if i + j + k <= complexity:
                    coeff = np.random.uniform(-1, 1) / ((i + j + k + 1) ** 0.5)
                    result += coeff * (X ** i) * (Y ** j) * (Z ** k)

    return result


def random_trig_3d(nx, ny, nz, complexity=3):
    """Random trigonometric polynomial in 3D.

    Args:
        nx, ny, nz: Number of grid points
        complexity: Highest frequency in each direction
    """
    # Random domain
    x_length = np.random.uniform(2 * np.pi, 4 * np.pi)
    y_length = np.random.uniform(2 * np.pi, 4 * np.pi)
    z_length = np.random.uniform(2 * np.pi, 4 * np.pi)

    x = np.linspace(0, x_length, nx)
    y = np.linspace(0, y_length, ny)
    z = np.linspace(0, z_length, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    result = np.zeros((nx, ny, nz))

    for kx in range(1, complexity + 1):
        for ky in range(1, complexity + 1):
            for kz in range(1, complexity + 1):
                amp = np.random.uniform(-1, 1)
                phase_x = np.random.uniform(0, 2 * np.pi)
                phase_y = np.random.uniform(0, 2 * np.pi)
                phase_z = np.random.uniform(0, 2 * np.pi)
                result += amp * np.sin(kx * X + phase_x) * np.cos(ky * Y + phase_y) * np.sin(kz * Z + phase_z)

    return result


# ============================================================================
# Sphere Distance Functions
# ============================================================================

def union_of_spheres_signed_distance(nx, ny, nz, complexity=5):
    """Signed distance to union of random spheres.

    For a union of shapes, the signed distance is the minimum of individual distances.
    Negative inside, positive outside, zero on boundary.

    Args:
        nx, ny, nz: Number of grid points
        complexity: Number of random spheres
    """
    # Random domain
    xmin, xmax = np.random.uniform(-3, -1), np.random.uniform(1, 3)
    ymin, ymax = np.random.uniform(-3, -1), np.random.uniform(1, 3)
    zmin, zmax = np.random.uniform(-3, -1), np.random.uniform(1, 3)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Sample random spheres
    centers = []
    radii = []
    domain_size = min(xmax - xmin, ymax - ymin, zmax - zmin)

    for _ in range(complexity):
        cx = np.random.uniform(xmin, xmax)
        cy = np.random.uniform(ymin, ymax)
        cz = np.random.uniform(zmin, zmax)
        r = np.random.uniform(0.1, 0.4) * domain_size
        centers.append((cx, cy, cz))
        radii.append(r)

    # Compute signed distance to each sphere: dist_to_center - radius
    # Stack all distances and take minimum (union)
    distances = np.full((complexity, nx, ny, nz), np.inf)

    for i, ((cx, cy, cz), r) in enumerate(zip(centers, radii)):
        dist_to_center = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        distances[i] = dist_to_center - r

    # Union: minimum signed distance
    result = np.min(distances, axis=0)

    return result


def union_of_spheres_unsigned_distance(nx, ny, nz, complexity=5):
    """Unsigned distance to union of random spheres.

    Always non-negative. Zero on boundary and inside, positive outside.

    Args:
        nx, ny, nz: Number of grid points
        complexity: Number of random spheres
    """
    # Random domain
    xmin, xmax = np.random.uniform(-3, -1), np.random.uniform(1, 3)
    ymin, ymax = np.random.uniform(-3, -1), np.random.uniform(1, 3)
    zmin, zmax = np.random.uniform(-3, -1), np.random.uniform(1, 3)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Sample random spheres
    centers = []
    radii = []
    domain_size = min(xmax - xmin, ymax - ymin, zmax - zmin)

    for _ in range(complexity):
        cx = np.random.uniform(xmin, xmax)
        cy = np.random.uniform(ymin, ymax)
        cz = np.random.uniform(zmin, zmax)
        r = np.random.uniform(0.1, 0.4) * domain_size
        centers.append((cx, cy, cz))
        radii.append(r)

    # Compute signed distance to each sphere
    distances = np.full((complexity, nx, ny, nz), np.inf)

    for i, ((cx, cy, cz), r) in enumerate(zip(centers, radii)):
        dist_to_center = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        distances[i] = dist_to_center - r

    # Union: minimum signed distance, then clip to non-negative
    signed_dist = np.min(distances, axis=0)
    result = np.maximum(signed_dist, 0)

    return result



def get_lower_star_data(dataset_name: int, dim: int, nx: int, ny: int, nz: int, complexity: int):
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "rastrigin":
        x = rastrigin_2d(nx, ny)
    elif dataset_name == "rosenbrock":
        x = rosenbrock_2d(nx, ny)
    elif dataset_name == "gaussian":
        if dim == 1:
            x = random_gaussian_1d(nx, complexity=complexity)
        elif dim == 2:
            x = random_gaussian_2d(nx, ny, complexity=complexity)
        elif dim == 3:
            x = random_gaussian_3d(nx, ny, nz, complexity=complexity)
    elif dataset_name == "trig":
        if dim == 1:
            x = random_trig_1d(nx, complexity=complexity)
        elif dim == 2:
            x = random_trig_2d(nx, ny, complexity=complexity)
        elif dim == 3:
            x = random_trig_3d(nx, ny, nz, complexity=complexity)
    elif dataset_name == "poly":
        if dim == 1:
            x = random_poly_1d(nx, complexity=complexity)
        elif dim == 2:
            x = random_poly_2d(nx, ny, complexity=complexity)
        elif dim == 3:
            x = random_poly_3d(nx, ny, nz, complexity=complexity)
    elif dataset_name == "spheres_signed":
        x = union_of_spheres_signed_distance(nx, ny, nz, complexity=complexity)
    elif dataset_name == "spheres_unsigned":
        x = union_of_spheres_unsigned_distance(nx, ny, nz, complexity=complexity)
    else:
        raise RuntimeError(f"Unknown lower-star dataset: {dataset_name}")
    print(f"{x.shape=}")
    return x


def get_pointcloud_data(dataset_name: str, dim: int, n_units: int, n_points_per_unit: int):
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "spheres":
        pts = sample_spheres(n_spheres=n_units, n_points_per_sphere=n_points_per_unit, dim=dim)
    elif dataset_name == "tori":
        pts = sample_tori(n_tori=n_units, n_points_per_torus=n_points_per_unit, separate=False)
    elif dataset_name == "tori_separate":
        pts = sample_tori(n_tori=n_units, n_points_per_torus=n_points_per_unit, separate=True)
    elif dataset_name == "two_circles":
        pts = generate_two_circles(n_points_small=n_points_per_unit, n_points_large=n_points_per_unit)
    elif dataset_name == "random_normal":
        pts = sample_random(n_points=n_points_per_unit, dim=dim)
    elif dataset_name == "annulus":
        pts = sample_annulus(n_points=n_points_per_unit)
    elif dataset_name == "figure8":
        pts = sample_figure_eight(n_points=n_points_per_unit)
    elif dataset_name == "swiss_roll":
        pts = sample_swiss_roll(n_points=n_points_per_unit)
    elif dataset_name == "trefoil":
        pts = sample_trefoil_knot(n_points=n_points_per_unit)
    elif dataset_name == "linked":
        pts = sample_linked_circles(n_points_per_circle=n_points_per_unit)
    elif dataset_name == "clusters":
        pts = sample_clusters(n_clusters=n_units, n_points_per_cluster=n_points_per_unit, dim=dim)
    elif dataset_name == "mobius":
        pts = sample_mobius_strip(n_points=n_points_per_unit)
    elif dataset_name == "sierpinski":
        pts = sample_sierpinski_triangle(n_points=n_points_per_unit)
    return pts


def is_pointcloud(dataset_name: str):
    dataset_name = dataset_name.lower().strip()
    return dataset_name in ["spheres", "tori", "tori_separate", "random_normal",
                            "annulus", "figure8", "swiss_roll", "trefoil",
                            "linked", "clusters", "mobius", "sierpinski",
                            "two_circles"]


# Example usage
if __name__ == "__main__":
    # Generate various point clouds
    spheres = sample_spheres(n_spheres=3, n_points_per_sphere=100, dim=3)
    tori = sample_tori(n_tori=2, n_points_per_torus=200)
    random_normal = sample_random(n_points=100, dim=2, distr="normal")
    annulus = sample_annulus(n_points=150)
    figure8 = sample_figure_eight(n_points=200)
    swiss_roll = sample_swiss_roll(n_points=500)
    trefoil = sample_trefoil_knot(n_points=300)
    linked = sample_linked_circles(n_points_per_circle=100)
    clusters = sample_clusters(n_clusters=5, n_points_per_cluster=50, dim=3)
    mobius = sample_mobius_strip(n_points=300)
    sierpinski = sample_sierpinski_triangle(n_points=1000)

    print(f"Spheres shape: {spheres.shape}")
    print(f"Tori shape: {tori.shape}")
    print(f"Random points shape: {random_normal.shape}")
    print(f"Annulus shape: {annulus.shape}")
    print(f"Figure-8 shape: {figure8.shape}")
    print(f"Swiss roll shape: {swiss_roll.shape}")
    print(f"Trefoil knot shape: {trefoil.shape}")
    print(f"Linked circles shape: {linked.shape}")
    print(f"Clusters shape: {clusters.shape}")
    print(f"Möbius strip shape: {mobius.shape}")
    print(f"Sierpinski triangle shape: {sierpinski.shape}")
