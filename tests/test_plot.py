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


def _has_density_artist(ax):
    # ScatterDensityArtist is registered as a non-Collection artist, so we
    # walk the artists list rather than ax.collections.
    from mpl_scatter_density import ScatterDensityArtist
    return any(isinstance(a, ScatterDensityArtist) for a in ax.get_children())


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


class TestPlotDiagramDensity:
    """Density rendering kicks in above DEFAULT_DENSITY_THRESHOLD."""

    @pytest.fixture
    def big_diagram(self):
        # Most points sit near the diagonal (noise); a handful are outliers.
        rng = np.random.default_rng(0)
        n = oineus.DEFAULT_DENSITY_THRESHOLD + 500
        births = rng.uniform(0.0, 1.0, n)
        deaths = births + rng.exponential(0.01, n)  # mostly low persistence
        outlier_idx = rng.choice(n, 20, replace=False)
        deaths[outlier_idx] = births[outlier_idx] + 1.0  # high-persistence
        return np.column_stack([births, deaths])

    def test_density_artist_added_above_threshold(self, big_diagram):
        pytest.importorskip("mpl_scatter_density")
        fig, ax = _fresh_ax()
        oineus.plot_diagram(big_diagram, ax=ax)
        assert _has_density_artist(ax), "expected a ScatterDensityArtist for big diagram"
        plt.close(fig)

    def test_no_density_below_threshold(self):
        # Stay well under the threshold.
        n = max(10, oineus.DEFAULT_DENSITY_THRESHOLD // 100)
        dgm = np.column_stack([np.linspace(0, 1, n), np.linspace(0, 1, n) + 0.5])
        fig, ax = _fresh_ax()
        oineus.plot_diagram(dgm, ax=ax)
        # Below threshold there should be only ordinary scatter.
        if oineus.vis_utils._HAS_MPL_SCATTER_DENSITY:
            assert not _has_density_artist(ax)
        plt.close(fig)

    def test_outliers_still_scatter_in_density_mode(self, big_diagram):
        pytest.importorskip("mpl_scatter_density")
        fig, ax = _fresh_ax()
        oineus.plot_diagram(big_diagram, ax=ax)
        # Outliers go to a regular scatter even when bulk is density.
        assert any(c.__class__.__name__ == "PathCollection" for c in ax.collections)
        plt.close(fig)

    def test_user_axes_work_without_projection(self, big_diagram):
        # The whole point of dropping the projection requirement: user can
        # pass any plain matplotlib axes.
        pytest.importorskip("mpl_scatter_density")
        fig, ax = plt.subplots()  # vanilla axes, no projection
        oineus.plot_diagram(big_diagram, ax=ax)
        assert _has_density_artist(ax)
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


class TestPlotMatchingDensity:
    """Above threshold, matching switches to density bulk + top-quantile edges."""

    def _big_pair(self):
        rng = np.random.default_rng(1)
        n = oineus.DEFAULT_DENSITY_THRESHOLD // 2 + 500
        a_b = rng.uniform(0.0, 1.0, n)
        a_d = a_b + rng.exponential(0.01, n)
        b_b = rng.uniform(0.0, 1.0, n)
        b_d = b_b + rng.exponential(0.01, n)
        return np.column_stack([a_b, a_d]), np.column_stack([b_b, b_d])

    def test_density_artist_added_above_threshold(self):
        pytest.importorskip("mpl_scatter_density")
        dgm_a, dgm_b = self._big_pair()
        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)
        fig, ax = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax)
        assert _has_density_artist(ax)
        plt.close(fig)

    def test_edge_quantile_filters_ordinary_edges(self):
        pytest.importorskip("mpl_scatter_density")
        dgm_a, dgm_b = self._big_pair()
        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)

        fig, ax_quantile = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax_quantile, edge_quantile=0.99)
        n_quantile = sum(
            len(c.get_segments())
            for c in ax_quantile.collections
            if isinstance(c, LineCollection)
        )
        plt.close(ax_quantile.figure)

        fig, ax_no_filter = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax_no_filter, use_density=False)
        n_no_filter = sum(
            len(c.get_segments())
            for c in ax_no_filter.collections
            if isinstance(c, LineCollection)
        )
        plt.close(ax_no_filter.figure)

        # Quantile mode keeps only the longest few percent.
        assert 0 < n_quantile < n_no_filter

    def test_use_density_false_disables_density(self):
        dgm_a, dgm_b = self._big_pair()
        m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)
        fig, ax = _fresh_ax()
        oineus.plot_matching(dgm_a, dgm_b, m, ax=ax, use_density=False)
        if oineus.vis_utils._HAS_MPL_SCATTER_DENSITY:
            assert not _has_density_artist(ax)
        plt.close(fig)


class TestStylePlumbing:
    def test_getter_returns_copy(self):
        # color was promoted to a top-level match_color arg; the dict
        # only carries the non-color keys now.
        s = oineus.default_matching_edge_style()
        s["linewidth"] = 99.0
        # Module-level dict unchanged.
        assert oineus.DEFAULT_MATCHING_EDGE_STYLE["linewidth"] != 99.0

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
