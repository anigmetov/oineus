"""Smoke tests for plot_matching and plot_diagram (matplotlib Agg backend)."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import oineus  # noqa: E402


def _fresh_ax():
    fig, ax = plt.subplots()
    return fig, ax


class TestPlotDiagram:
    def test_single_dim_numpy(self):
        dgm = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        fig, ax = _fresh_ax()
        oineus.plot_diagram(dgm, ax=ax)
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_with_infinite_points(self):
        dgm = np.array([[0.0, 1.0], [0.5, np.inf]])
        fig, ax = _fresh_ax()
        oineus.plot_diagram(dgm, ax=ax)
        plt.close(fig)

    def test_custom_point_style_via_dict(self):
        dgm = np.array([[0.0, 1.0]])
        fig, ax = _fresh_ax()
        style = oineus.default_point_style()
        style["s"] = 100.0
        oineus.plot_diagram(dgm, ax=ax, point_style=style)
        plt.close(fig)

    def test_per_dim_color(self):
        dgms = {0: np.array([[0.0, 1.0]]), 1: np.array([[0.5, 2.0]])}
        fig, ax = _fresh_ax()
        oineus.plot_diagram(dgms, ax=ax, color={0: "tab:red", 1: "tab:blue"})
        plt.close(fig)


class TestPlotMatchingWasserstein:
    def test_basic_render(self):
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
        dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])
        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)
        fig, ax = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax)
        # Expect a LineCollection for ordinary edges plus scatters for points and diagonal.
        has_line_collection = any(isinstance(c, LineCollection) for c in ax.collections)
        assert has_line_collection
        plt.close(fig)

    def test_category_flags(self):
        dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
        dgm_b = np.array([[0.1, 0.9]])  # forces an a_to_diagonal match
        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)

        fig, ax = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax, plot_a_to_diagonal=False)
        plt.close(fig)

    def test_custom_edge_style_applied(self):
        dgm_a = np.array([[0.0, 1.0]])
        dgm_b = np.array([[0.1, 0.9]])
        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)

        style = oineus.default_matching_edge_style()
        style["linewidth"] = 9.0
        style["color"] = "magenta"

        fig, ax = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax, ordinary_edge_style=style)

        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(lcs) >= 1
        # Verify our override landed on at least one collection.
        matched = False
        for lc in lcs:
            widths = lc.get_linewidths()
            if len(widths) and abs(float(widths[0]) - 9.0) < 1e-9:
                matched = True
                break
        assert matched, "expected a LineCollection with linewidth=9.0"
        plt.close(fig)


class TestPlotMatchingBottleneck:
    def test_bottleneck_highlights_longest(self):
        # Two points per side; ties produce 4 longest edges under L_inf.
        dgm_a = np.array([[0.0, 1.0], [10.0, 11.0]])
        dgm_b = np.array([[1.0, 2.0], [11.0, 12.0]])
        m = oineus.bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert len(m.longest.finite) == 4

        fig, ax = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax)

        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        # At least one LineCollection for the longest-edge overlay.
        assert len(lcs) >= 1
        plt.close(fig)

    def test_bottleneck_defaults_suppress_diag_edges(self):
        dgm_a = np.array([[0.0, 1.0]])
        dgm_b = np.array([[5.0, 6.0]])  # forces both to their own diagonals
        m = oineus.bottleneck_matching(dgm_a, dgm_b, delta=0.0)
        assert len(m.a_to_diagonal) == 1
        assert len(m.b_to_diagonal) == 1

        fig, ax = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax)  # defaults: no diag edges
        # Diag edges are off by default for bottleneck; only finite-to-finite
        # (empty here) and longest-edge highlights remain.
        plt.close(fig)

    def test_bottleneck_opt_in_diag_edges(self):
        dgm_a = np.array([[0.0, 1.0]])
        dgm_b = np.array([[5.0, 6.0]])
        m = oineus.bottleneck_matching(dgm_a, dgm_b, delta=0.0)

        fig, ax = _fresh_ax()
        oineus.plot_matching(
            dgm_a, dgm_b, m, ax=ax,
            plot_a_to_diagonal=True, plot_b_to_diagonal=True,
        )
        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        # Expect the ordinary-edge LineCollection plus a longest-edge overlay.
        assert len(lcs) >= 1
        plt.close(fig)


class TestStylePlumbing:
    def test_getter_returns_copy(self):
        s = oineus.default_matching_edge_style()
        s["color"] = "totally-new-color"
        # Module-level dict unchanged.
        assert oineus.DEFAULT_MATCHING_EDGE_STYLE["color"] != "totally-new-color"

    def test_global_mutation_propagates(self):
        original = oineus.DEFAULT_MATCHING_EDGE_STYLE["linewidth"]
        try:
            oineus.DEFAULT_MATCHING_EDGE_STYLE["linewidth"] = 7.0
            dgm_a = np.array([[0.0, 1.0]])
            dgm_b = np.array([[0.1, 0.9]])
            m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)

            fig, ax = _fresh_ax()
            oineus.plot_matching(dgm_a, dgm_b, m, ax=ax)
            lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
            assert any(
                abs(float(lc.get_linewidths()[0]) - 7.0) < 1e-9
                for lc in lcs if len(lc.get_linewidths())
            )
            plt.close(fig)
        finally:
            oineus.DEFAULT_MATCHING_EDGE_STYLE["linewidth"] = original
