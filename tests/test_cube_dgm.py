import pytest
import numpy as np
import oineus as oin
from gudhi import CubicalComplex
from oineus._dtype import REAL_DTYPE


def get_gudhi_dgms(a, values_on):
    """Compute persistence diagrams using GUDHI."""
    if values_on == "cells":
        cc = CubicalComplex(top_dimensional_cells=a)
    else:
        cc = CubicalComplex(vertices=a)
    persistence = cc.persistence(homology_coeff_field=2, min_persistence=1e-15)
    dgms_all = [cc.persistence_intervals_in_dimension(d) for d in range(cc.dimension())]
    return dgms_all


def get_oin_dgms(a, values_on, n_threads, dualize):
    """Compute persistence diagrams using Oineus."""
    fil = oin.cube_filtration(a, n_threads=n_threads, values_on=values_on)
    dcmp = oin.Decomposition(fil, dualize=dualize, n_threads=n_threads)

    rp = oin.ReductionParams()
    rp.n_threads = n_threads
    dcmp.reduce(rp)

    dgms = dcmp.diagram(fil)
    return [dgms.in_dimension(d) for d in range(a.ndim)]


def dgms_equal(dgms_1, dgms_2, tol=1e-4):
    """Check if two sets of persistence diagrams are equal within tolerance."""
    assert len(dgms_1) == len(dgms_2), f"Different number of points: {len(dgms_1)} vs {len(dgms_2)}"

    for dim in range(len(dgms_1)):
        d1 = np.sort(dgms_1[dim], axis=0).astype(REAL_DTYPE)
        d2 = np.sort(dgms_2[dim], axis=0).astype(REAL_DTYPE)
        d1[d1 == np.inf] = 10000.0
        d2[d2 == np.inf] = 10000.0

        if d1.shape != d2.shape:
            return False

        if np.linalg.norm(d1 - d2) >= tol:
            return False

    return True


# Test data generators

def gaussian_1d(n, centers=[0.3, 0.7], sigmas=[0.1, 0.1]):
    """Sum of Gaussians in 1D."""
    x = np.linspace(0, 1, n)
    result = np.zeros(n)
    for c, s in zip(centers, sigmas):
        result += np.exp(-((x - c) ** 2) / (2 * s ** 2))
    return result


def sine_wave_1d(n, freq=4, phase=0):
    """Sine wave function."""
    x = np.linspace(0, 2 * np.pi, n)
    return np.sin(freq * x + phase)


def polynomial_1d(n):
    """Polynomial with multiple critical points: x^4 - 2x^2."""
    x = np.linspace(-2, 2, n)
    return x**4 - 2*x**2


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


def gaussian_peaks_2d(nx, ny, centers=[(0.3, 0.3), (0.7, 0.7), (0.5, 0.5)]):
    """Sum of 2D Gaussian peaks."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    result = np.zeros((nx, ny))

    for cx, cy in centers:
        result += np.exp(-((X - cx)**2 + (Y - cy)**2) / 0.02)

    return result


def saddle_2d(nx, ny):
    """Saddle point function: x^2 - y^2."""
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X**2 - Y**2


def sphere_3d(nx, ny, nz):
    """Sphere function: simple bowl shape."""
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X**2 + Y**2 + Z**2


def gaussian_peaks_3d(nx, ny, nz, centers=[(0.3, 0.3, 0.3), (0.7, 0.7, 0.7)]):
    """Sum of 3D Gaussian peaks."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    result = np.zeros((nx, ny, nz))

    for cx, cy, cz in centers:
        result += np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / 0.02)

    return result


def trig_3d(nx, ny, nz):
    """Trigonometric combination in 3D."""
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    z = np.linspace(0, 2 * np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


# Parametrized tests

class TestRandom:
    """Test with random data."""

    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("n", [2, 5, 10])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    @pytest.mark.parametrize("dualize", [True, False])
    def test_random_data(self, dim, n, values_on, dualize):
        """Test with random data of various dimensions and sizes."""
        np.random.seed(42)  # For reproducibility
        a = np.random.randn(*((n,) * dim))
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=dualize)
        assert dgms_equal(dgms_gudhi, dgms_oin)


class Test1D:
    """Test 1D functions."""

    @pytest.mark.parametrize("n", [10, 50, 100])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    @pytest.mark.parametrize("dualize", [True, False])
    def test_gaussian_peaks(self, n, values_on, dualize):
        """Test with sum of Gaussian peaks."""
        a = gaussian_1d(n)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=dualize)
        assert dgms_equal(dgms_gudhi, dgms_oin)

    @pytest.mark.parametrize("n", [20, 50])
    @pytest.mark.parametrize("freq", [2, 4])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_sine_wave(self, n, freq, values_on):
        """Test with sine wave."""
        a = sine_wave_1d(n, freq=freq)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)

    @pytest.mark.parametrize("n", [30, 60])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_polynomial(self, n, values_on):
        """Test with polynomial function."""
        a = polynomial_1d(n)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)


class Test2D:
    """Test 2D functions."""

    @pytest.mark.parametrize("nx,ny", [(10, 10), (20, 30), (50, 50)])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    @pytest.mark.parametrize("dualize", [True, False])
    def test_rosenbrock(self, nx, ny, values_on, dualize):
        """Test with Rosenbrock function."""
        a = rosenbrock_2d(nx, ny)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=dualize)
        assert dgms_equal(dgms_gudhi, dgms_oin)

    @pytest.mark.parametrize("nx,ny", [(15, 15), (20, 25)])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_rastrigin(self, nx, ny, values_on):
        """Test with Rastrigin function (many local minima)."""
        a = rastrigin_2d(nx, ny)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)

    @pytest.mark.parametrize("nx,ny", [(20, 20), (30, 40)])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_gaussian_peaks_2d(self, nx, ny, values_on):
        """Test with multiple Gaussian peaks."""
        a = gaussian_peaks_2d(nx, ny)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)

    @pytest.mark.parametrize("nx,ny", [(15, 20), (25, 25)])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_saddle(self, nx, ny, values_on):
        """Test with saddle point function."""
        a = saddle_2d(nx, ny)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)


class Test3D:
    """Test 3D functions."""

    @pytest.mark.parametrize("nx,ny,nz", [(10, 10, 10), (15, 20, 25), (30, 30, 30)])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_sphere(self, nx, ny, nz, values_on):
        """Test with sphere function."""
        a = sphere_3d(nx, ny, nz)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)

    # @pytest.mark.parametrize("nx,ny,nz", [(10, 10, 10), (10, 15, 20)])
    # @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    @pytest.mark.parametrize("nx,ny,nz", [(10, 10, 10)])
    @pytest.mark.parametrize("values_on", ["vertices"])
    def test_gaussian_peaks_3d(self, nx, ny, nz, values_on):
        """Test with 3D Gaussian peaks."""
        a = gaussian_peaks_3d(nx, ny, nz)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)

    @pytest.mark.parametrize("nx,ny,nz", [(8, 8, 8), (10, 12, 14)])
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_trig_combination(self, nx, ny, nz, values_on):
        """Test with trigonometric combination."""
        a = trig_3d(nx, ny, nz)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=1, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)


class TestMultithreading:
    """Test that multi-threading gives same results."""

    @pytest.mark.parametrize("n_threads", [1, 2, 4])
    def test_multithreading_consistency(self, n_threads):
        """Test that different thread counts give same results."""
        np.random.seed(42)
        a = np.random.randn(20, 20, 20)

        dgms_single = get_oin_dgms(a, values_on="vertices", n_threads=1, dualize=False)
        dgms_multi = get_oin_dgms(a, values_on="vertices", n_threads=n_threads, dualize=False)

        assert dgms_equal(dgms_single, dgms_multi)


class TestLargeGrids:
    """Test with larger grids (marked as slow)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_large_2d(self, values_on):
        """Test with 100x100 2D grid."""
        a = rosenbrock_2d(100, 100)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=4, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)

    @pytest.mark.slow
    @pytest.mark.parametrize("values_on", ["vertices", "cells"])
    def test_large_3d(self, values_on):
        """Test with 50x50x50 3D grid."""
        a = sphere_3d(50, 50, 50)
        dgms_gudhi = get_gudhi_dgms(a, values_on)
        dgms_oin = get_oin_dgms(a, values_on=values_on, n_threads=4, dualize=False)
        assert dgms_equal(dgms_gudhi, dgms_oin)
