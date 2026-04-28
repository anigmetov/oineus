"""Smoke tests for plot_diagram_gradient (matplotlib Agg backend)."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import oineus  # noqa: E402


def _fresh_ax():
    fig, ax = plt.subplots()
    return fig, ax


def _quivers(ax):
    from matplotlib.quiver import Quiver
    return [c for c in ax.collections if isinstance(c, Quiver)]


class TestNumpyGradient:
    def test_basic_render(self):
        dgm = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        grad = np.array([[0.1, -0.1], [0.0, 0.2], [-0.05, 0.05]])

        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax)

        qs = _quivers(ax)
        assert len(qs) == 1
        assert qs[0].N == 3
        np.testing.assert_allclose(qs[0].U, grad[:, 0])
        np.testing.assert_allclose(qs[0].V, grad[:, 1])
        plt.close(fig)

    def test_descent_flips_signs(self):
        dgm = np.array([[0.0, 1.0]])
        grad = np.array([[0.3, -0.4]])

        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax, descent=True)

        q = _quivers(ax)[0]
        np.testing.assert_allclose(q.U, -grad[:, 0])
        np.testing.assert_allclose(q.V, -grad[:, 1])
        plt.close(fig)

    def test_inf_rows_skipped(self):
        dgm = np.array([[0.0, 1.0], [0.5, np.inf], [1.0, 3.0]])
        grad = np.array([[0.1, 0.0], [0.0, 0.0], [-0.1, 0.2]])

        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax)

        q = _quivers(ax)[0]
        assert q.N == 2
        np.testing.assert_allclose(q.U, np.array([0.1, -0.1]))
        np.testing.assert_allclose(q.V, np.array([0.0, 0.2]))
        plt.close(fig)

    def test_missing_gradient_raises(self):
        dgm = np.array([[0.0, 1.0]])
        with pytest.raises(ValueError):
            oineus.plot_diagram_gradient(dgm)

    def test_shape_mismatch_raises(self):
        dgm = np.array([[0.0, 1.0], [0.5, 2.0]])
        grad = np.array([[0.1, 0.1]])
        fig, ax = _fresh_ax()
        with pytest.raises(ValueError):
            oineus.plot_diagram_gradient(dgm, grad, ax=ax)
        plt.close(fig)


class TestDictInput:
    def test_per_dim_dict(self):
        dgms = {0: np.array([[0.0, 1.0]]), 1: np.array([[0.5, 2.0], [0.7, 1.8]])}
        grads = {0: np.array([[0.1, 0.0]]), 1: np.array([[0.0, 0.1], [-0.1, 0.0]])}

        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgms, grads, ax=ax)
        qs = _quivers(ax)
        assert len(qs) == 2
        # Sorted by dim, so first quiver has 1 arrow, second has 2.
        assert sorted(q.N for q in qs) == [1, 2]
        plt.close(fig)


class TestTorchGradient:
    def test_torch_tensor_with_grad(self):
        torch = pytest.importorskip("torch")
        dgm = torch.tensor([[0.0, 1.0], [0.5, 2.0]],
                           dtype=torch.float64, requires_grad=True)
        loss = (dgm[:, 1] - 2.0).pow(2).sum() + (dgm[:, 0] - 1.0).pow(2).sum()
        loss.backward()
        expected = dgm.grad.detach().cpu().numpy()

        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, ax=ax)
        q = _quivers(ax)[0]
        np.testing.assert_allclose(q.U, expected[:, 0])
        np.testing.assert_allclose(q.V, expected[:, 1])
        plt.close(fig)

    def test_torch_tensor_without_grad_raises(self):
        torch = pytest.importorskip("torch")
        dgm = torch.tensor([[0.0, 1.0]], dtype=torch.float64,
                           requires_grad=True)
        # No backward called -- .grad is None.
        with pytest.raises(ValueError, match="retain_grad|grad"):
            oineus.plot_diagram_gradient(dgm)

    def test_torch_explicit_gradient_overrides(self):
        torch = pytest.importorskip("torch")
        dgm = torch.tensor([[0.0, 1.0]], dtype=torch.float64,
                           requires_grad=True)
        (dgm.sum()).backward()  # populates .grad with ones
        override = np.array([[0.7, -0.3]])

        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, override, ax=ax)
        q = _quivers(ax)[0]
        np.testing.assert_allclose(q.U, override[:, 0])
        np.testing.assert_allclose(q.V, override[:, 1])
        plt.close(fig)


class TestDensityMode:
    """Above DEFAULT_DENSITY_THRESHOLD, gradient switches to density-of-points
    plus top-K arrows."""

    def _big_diagram_and_grad(self):
        rng = np.random.default_rng(2)
        n = oineus.DEFAULT_DENSITY_THRESHOLD + 500
        births = rng.uniform(0.0, 1.0, n)
        deaths = births + rng.exponential(0.01, n)
        dgm = np.column_stack([births, deaths])
        grad = rng.standard_normal((n, 2))
        return dgm, grad

    def test_top_k_default_caps_arrows(self):
        pytest.importorskip("mpl_scatter_density")
        dgm, grad = self._big_diagram_and_grad()
        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax)
        # The default cap is the module constant.
        cap = oineus.DEFAULT_GRADIENT_TOP_K_ARROWS
        total_arrows = sum(q.N for q in _quivers(ax))
        assert total_arrows == cap, f"expected {cap} arrows, got {total_arrows}"
        plt.close(fig)

    def test_explicit_top_k_overrides_default(self):
        pytest.importorskip("mpl_scatter_density")
        dgm, grad = self._big_diagram_and_grad()
        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax, top_k_arrows=42)
        total_arrows = sum(q.N for q in _quivers(ax))
        assert total_arrows == 42
        plt.close(fig)

    def test_top_k_picks_largest_magnitudes(self):
        pytest.importorskip("mpl_scatter_density")
        # Construct a diagram where exactly 5 points have huge gradient and
        # the rest have ~zero. Top-K=5 must select exactly those.
        n = oineus.DEFAULT_DENSITY_THRESHOLD + 100
        rng = np.random.default_rng(3)
        births = rng.uniform(0.0, 1.0, n)
        deaths = births + 0.2
        dgm = np.column_stack([births, deaths])
        grad = 1e-6 * rng.standard_normal((n, 2))
        big_idx = np.array([7, 100, 500, 5000, n - 1])
        grad[big_idx] = np.array([[10.0, 0.0]] * 5) * np.array([[1, 1, 1, 1, 1]]).T

        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax, top_k_arrows=5)
        q = _quivers(ax)[0]
        assert q.N == 5
        # Quiver tail X coordinates must equal the births of the chosen points.
        sel_x = sorted(np.asarray(q.X).tolist())
        expected_x = sorted(births[big_idx].tolist())
        np.testing.assert_allclose(sel_x, expected_x)
        plt.close(fig)

    def test_below_threshold_no_arrow_cap(self):
        # No density mode, no cap by default, every arrow drawn.
        n = max(10, oineus.DEFAULT_DENSITY_THRESHOLD // 100)
        dgm = np.column_stack([np.linspace(0, 1, n), np.linspace(0, 1, n) + 0.3])
        grad = np.ones((n, 2))
        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax)
        total_arrows = sum(q.N for q in _quivers(ax))
        assert total_arrows == n
        plt.close(fig)

    def test_use_density_false_disables_cap(self):
        # When density is explicitly disabled, no top-K cap kicks in.
        dgm, grad = self._big_diagram_and_grad()
        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax, use_density=False)
        total_arrows = sum(q.N for q in _quivers(ax))
        assert total_arrows == dgm.shape[0]
        plt.close(fig)


class TestStylePlumbing:
    def test_default_style_getter_returns_copy(self):
        s = oineus.default_diagram_gradient_style()
        s["color"] = "totally-new-color"
        assert oineus.DEFAULT_DIAGRAM_GRADIENT_STYLE["color"] != "totally-new-color"

    def test_custom_quiver_style(self):
        dgm = np.array([[0.0, 1.0]])
        grad = np.array([[0.1, 0.1]])

        style = oineus.default_diagram_gradient_style()
        style["color"] = "magenta"

        fig, ax = _fresh_ax()
        oineus.plot_diagram_gradient(dgm, grad, ax=ax, quiver_style=style)
        q = _quivers(ax)[0]
        # facecolors stored as RGBA; magenta = (1, 0, 1, alpha).
        fc = q.get_facecolor()
        assert fc.shape[0] >= 1
        assert abs(fc[0, 0] - 1.0) < 1e-6 and abs(fc[0, 1] - 0.0) < 1e-6 and abs(fc[0, 2] - 1.0) < 1e-6
        plt.close(fig)
