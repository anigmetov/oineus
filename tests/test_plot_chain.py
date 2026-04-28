"""Smoke tests for plot_chain (matplotlib Agg backend).

Covers Phase 1 (2D point clouds, simplicial) and Phase 2 (3D point clouds,
2D / 3D scalar fields with cubical or Freudenthal filtrations).
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

import oineus  # noqa: E402


def _fresh_ax():
    fig, ax = plt.subplots()
    return fig, ax


def _square_points():
    # Four corners of a unit square -- enough to build a non-trivial VR
    # filtration with 0/1/2-cells and predictable vertex indexing.
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def _build_vr(points, max_dim=2):
    fil = oineus.vr_filtration(points, max_dim=max_dim, n_threads=1)
    dcmp = oineus.Decomposition(fil, dualize=False)
    params = oineus.ReductionParams()
    params.compute_v = True
    dcmp.reduce(params)
    return fil, dcmp


def _cells_by_dim(fil, dim):
    return [i for i in range(fil.size()) if len(fil[i].vertices) == dim + 1]


class TestPlotChainBasic:
    def test_edges_only_chain(self):
        points = _square_points()
        fil, _ = _build_vr(points, max_dim=1)
        edge_ids = _cells_by_dim(fil, 1)
        assert len(edge_ids) >= 3

        fig, ax = _fresh_ax()
        oineus.plot_chain(points, fil, edge_ids, ax=ax)

        # One LineCollection holds all edges.
        line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(line_collections) == 1
        assert len(line_collections[0].get_segments()) == len(edge_ids)
        # No PolyCollection because the chain has no triangles.
        assert not any(isinstance(c, PolyCollection) for c in ax.collections)
        plt.close(fig)

    def test_triangle_only_chain(self):
        points = _square_points()
        fil, _ = _build_vr(points, max_dim=2)
        tri_ids = _cells_by_dim(fil, 2)
        if not tri_ids:
            pytest.skip("VR did not produce 2-cells for this point set")

        fig, ax = _fresh_ax()
        oineus.plot_chain(points, fil, tri_ids[:1], ax=ax)
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) == 1
        # Triangle vertices must come from points[simplex.vertices].
        rendered = polys[0].get_paths()[0].vertices
        cell = fil[tri_ids[0]]
        expected_vertex_set = {tuple(points[v]) for v in cell.vertices}
        rendered_vertex_set = {tuple(np.round(v, 12)) for v in rendered[:3]}
        # Round-trip in either order: rendered vertices should be exactly the
        # cell's points (Path.vertices closes the polygon by repeating the first).
        assert expected_vertex_set == rendered_vertex_set
        plt.close(fig)

    def test_mixed_dim_chain(self):
        points = _square_points()
        fil, _ = _build_vr(points, max_dim=2)
        edge_ids = _cells_by_dim(fil, 1)[:2]
        tri_ids = _cells_by_dim(fil, 2)[:1]
        if not edge_ids or not tri_ids:
            pytest.skip("VR did not produce both edges and triangles")

        fig, ax = _fresh_ax()
        oineus.plot_chain(points, fil, list(edge_ids) + list(tri_ids), ax=ax)
        lines = [c for c in ax.collections if isinstance(c, LineCollection)]
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(lines) == 1 and len(lines[0].get_segments()) == len(edge_ids)
        assert len(polys) == 1
        plt.close(fig)

    def test_empty_chain_just_renders_points(self):
        points = _square_points()
        fil, _ = _build_vr(points, max_dim=1)
        fig, ax = _fresh_ax()
        oineus.plot_chain(points, fil, [], ax=ax)
        # No LineCollection or PolyCollection -- only the underlying scatter.
        assert not any(isinstance(c, (LineCollection, PolyCollection)) for c in ax.collections)
        # PathCollection (scatter) is expected for the point cloud itself.
        assert any(c.__class__.__name__ == "PathCollection" for c in ax.collections)
        plt.close(fig)


class TestChainCoercion:
    def test_chain_as_numpy_array(self):
        points = _square_points()
        fil, _ = _build_vr(points, max_dim=1)
        edge_ids = np.asarray(_cells_by_dim(fil, 1)[:3], dtype=np.int64)
        fig, ax = _fresh_ax()
        oineus.plot_chain(points, fil, edge_ids, ax=ax)
        lines = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(lines[0].get_segments()) == len(edge_ids)
        plt.close(fig)

    def test_chain_as_range(self):
        points = _square_points()
        fil, _ = _build_vr(points, max_dim=1)
        # First 4 cells in a square's VR are guaranteed to be the 4 vertex
        # cells (sorted_id 0..3). They render via the vertex_style scatter.
        fig, ax = _fresh_ax()
        oineus.plot_chain(points, fil, range(4), ax=ax)
        # Two scatters: the underlying point cloud + the chain-vertex scatter.
        scatters = [c for c in ax.collections if c.__class__.__name__ == "PathCollection"]
        assert len(scatters) == 2
        plt.close(fig)

    def test_chain_as_scipy_sparse_column(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        points = _square_points()
        fil, dcmp = _build_vr(points, max_dim=2)

        # Use a V-matrix column directly: pick a column with at least one
        # nonzero entry.
        v_csc = dcmp.v_as_csc()
        nonempty_col = next(
            (j for j in range(v_csc.shape[1]) if v_csc.indptr[j + 1] - v_csc.indptr[j] > 0),
            None,
        )
        if nonempty_col is None:
            pytest.skip("All V columns are empty")
        column = v_csc[:, nonempty_col]

        # Equivalent dense list for cross-check.
        dense_ids = list(dcmp.v_data[nonempty_col])

        fig, ax_sparse = _fresh_ax()
        oineus.plot_chain(points, fil, column, ax=ax_sparse)
        sparse_collections = sorted(
            (c.__class__.__name__, len(c.get_segments()) if isinstance(c, LineCollection)
             else len(c.get_paths()) if isinstance(c, PolyCollection)
             else 0)
            for c in ax_sparse.collections
        )
        plt.close(ax_sparse.figure)

        fig, ax_dense = _fresh_ax()
        oineus.plot_chain(points, fil, dense_ids, ax=ax_dense)
        dense_collections = sorted(
            (c.__class__.__name__, len(c.get_segments()) if isinstance(c, LineCollection)
             else len(c.get_paths()) if isinstance(c, PolyCollection)
             else 0)
            for c in ax_dense.collections
        )
        plt.close(ax_dense.figure)

        assert sparse_collections == dense_collections


class TestPlotChainSourceKind:
    def test_invalid_source_kind_raises(self):
        points = _square_points()
        fil, _ = _build_vr(points, max_dim=1)
        with pytest.raises(ValueError, match="source_kind"):
            oineus.plot_chain(points, fil, [], source_kind="bogus")

    def test_field_2d_with_wrong_shape_raises(self):
        field = np.arange(16, dtype=float).reshape((4, 4))
        fil = oineus.cube_filtration(field, max_dim=2)
        # 1D source for a field is malformed.
        with pytest.raises(ValueError, match="2D|3D"):
            oineus.plot_chain(np.arange(5, dtype=float), fil, [])


class TestPlotChain3DPoints:
    def _build_3d_vr(self):
        # Points at the 4 vertices of a tetrahedron + interior to ensure
        # we get a 3-cell.
        rng = np.random.default_rng(7)
        base = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        points = np.vstack([base, base.mean(axis=0)[None, :] + 0.05 * rng.standard_normal((4, 3))])
        fil = oineus.vr_filtration(points, max_dim=3, n_threads=1)
        return points, fil

    def test_3d_edges_render_as_line3dcollection(self):
        points, fil = self._build_3d_vr()
        edge_ids = [i for i in range(fil.size()) if len(fil[i].vertices) == 2][:3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        oineus.plot_chain(points, fil, edge_ids, ax=ax)
        line3d = [c for c in ax.collections if isinstance(c, Line3DCollection)]
        assert len(line3d) == 1
        plt.close(fig)

    def test_3d_tetrahedron_renders_four_triangle_faces(self):
        points, fil = self._build_3d_vr()
        tet_ids = [i for i in range(fil.size()) if len(fil[i].vertices) == 4]
        if not tet_ids:
            pytest.skip("No 3-cells produced by VR")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        oineus.plot_chain(points, fil, tet_ids[:1], ax=ax)
        poly3d = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(poly3d) == 1
        # get_paths() is empty until the figure is drawn (3D projection
        # populates paths lazily during rendering).
        fig.canvas.draw()
        assert len(poly3d[0].get_paths()) == 4
        plt.close(fig)

    def test_3d_auto_creates_3d_axes(self):
        points, fil = self._build_3d_vr()
        edge_ids = [i for i in range(fil.size()) if len(fil[i].vertices) == 2][:1]
        ax = oineus.plot_chain(points, fil, edge_ids)
        assert getattr(ax, "name", "") == "3d"
        plt.close(ax.figure)

    def test_3d_nonmatching_axes_raises(self):
        points, fil = self._build_3d_vr()
        fig, ax2d = plt.subplots()
        with pytest.raises(ValueError, match="3D"):
            oineus.plot_chain(points, fil, [], ax=ax2d)
        plt.close(fig)


class TestPlotChainField2D:
    def _build_2d_field(self):
        rng = np.random.default_rng(11)
        return rng.uniform(0.0, 1.0, (6, 8)).astype(np.float64)

    def test_cubical_2cube_renders_polycollection(self):
        field = self._build_2d_field()
        fil = oineus.cube_filtration(field, max_dim=2)
        cube2_ids = [i for i in range(fil.size()) if len(fil[i].vertices) == 4][:1]
        fig, ax = _fresh_ax()
        oineus.plot_chain(field, fil, cube2_ids, ax=ax)
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) == 1
        plt.close(fig)

    def test_cubical_field_auto_routes_to_field_kind(self):
        # No explicit source_kind: cube filtration must auto-route to "field".
        field = self._build_2d_field()
        fil = oineus.cube_filtration(field, max_dim=2)
        edge_ids = [i for i in range(fil.size()) if len(fil[i].vertices) == 2][:2]
        fig, ax = _fresh_ax()
        oineus.plot_chain(field, fil, edge_ids, ax=ax)
        # imshow leaves an AxesImage on the axes.
        assert any(c.__class__.__name__ == "AxesImage" for c in ax.images)
        plt.close(fig)

    def test_freudenthal_field_with_explicit_source_kind(self):
        field = self._build_2d_field()
        fil = oineus.freudenthal_filtration(field, max_dim=2)
        # Find a 2-simplex (triangle) and verify the rendered triangle
        # corners match the expected (j, i) of its grid-id vertices.
        H, W = field.shape
        tri_id = next(
            (i for i in range(fil.size()) if len(fil[i].vertices) == 3),
            None,
        )
        if tri_id is None:
            pytest.skip("Freudenthal produced no 2-simplices")

        fig, ax = _fresh_ax()
        oineus.plot_chain(field, fil, [tri_id], ax=ax, source_kind="field")
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) == 1

        rendered = polys[0].get_paths()[0].vertices
        cell = fil[tri_id]
        expected_xy = []
        for vid in cell.vertices:
            i = vid // W
            j = vid % W
            expected_xy.append((j, i))
        rendered_xy = {(round(v[0], 9), round(v[1], 9)) for v in rendered[:3]}
        assert set(expected_xy) == rendered_xy
        plt.close(fig)


class TestPlotChainField3D:
    def _build_3d_field(self):
        rng = np.random.default_rng(13)
        return rng.uniform(0.0, 1.0, (3, 4, 5)).astype(np.float64)

    def test_cubical_3cube_renders_six_faces(self):
        field = self._build_3d_field()
        fil = oineus.cube_filtration(field, max_dim=3)
        cube3_ids = [i for i in range(fil.size()) if len(fil[i].vertices) == 8][:1]
        if not cube3_ids:
            pytest.skip("No 3-cubes in this filtration")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        oineus.plot_chain(field, fil, cube3_ids, ax=ax, plot_source=False)
        poly3d = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(poly3d) == 1
        fig.canvas.draw()
        assert len(poly3d[0].get_paths()) == 6
        plt.close(fig)

    def test_freudenthal_3d_tet(self):
        field = self._build_3d_field()
        fil = oineus.freudenthal_filtration(field, max_dim=3)
        tet_ids = [i for i in range(fil.size()) if len(fil[i].vertices) == 4][:1]
        if not tet_ids:
            pytest.skip("No 3-simplices in 3D Freudenthal filtration")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        oineus.plot_chain(field, fil, tet_ids, ax=ax, source_kind="field", plot_source=False)
        poly3d = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        # One collection of 4 triangle faces.
        assert len(poly3d) == 1
        plt.close(fig)


class TestDualizeIndexTranslation:
    """Cohomology V/U matrices are indexed in matrix space, not filtration
    space. plot_chain's `dualize` kwarg handles the translation."""

    def _setup(self):
        # Small 2D field; sublevel filtration on -field puts peaks first.
        rng = np.random.default_rng(17)
        field = rng.uniform(0.0, 1.0, (8, 10)).astype(np.float64)
        fil = oineus.freudenthal_filtration(-field, max_dim=2)
        return field, fil

    def test_dualize_true_renders_same_as_pretranslated(self):
        field, fil = self._setup()
        n = fil.size()

        dcmp = oineus.Decomposition(fil, dualize=True)
        params = oineus.ReductionParams()
        params.compute_v = True
        dcmp.reduce(params)

        # Pick a cohomology column with at least 2 entries to make the test
        # meaningful (a 1-cell mass).
        col_idx = next(
            (j for j in range(n) if len(dcmp.v_data[j]) >= 2),
            None,
        )
        if col_idx is None:
            pytest.skip("No cohomology V column with >= 2 entries")
        matrix_chain = list(dcmp.v_data[col_idx])
        translated = [n - 1 - i for i in matrix_chain]

        # Render via dualize=True (let plot_chain translate)
        fig, ax_dual = _fresh_ax()
        oineus.plot_chain(field, fil, matrix_chain, ax=ax_dual,
                          source_kind="field", dualize=True)
        a_collections = [type(c).__name__ for c in ax_dual.collections]
        plt.close(ax_dual.figure)

        # Render with the user-translated chain (no dualize flag)
        fig, ax_pre = _fresh_ax()
        oineus.plot_chain(field, fil, translated, ax=ax_pre,
                          source_kind="field")
        b_collections = [type(c).__name__ for c in ax_pre.collections]
        plt.close(ax_pre.figure)

        assert a_collections == b_collections

    def test_dualize_can_be_a_decomposition(self):
        # Convenience: passing the Decomposition itself reads its .dualize
        # attribute. Saves the user from having to remember a flag.
        field, fil = self._setup()

        dcmp = oineus.Decomposition(fil, dualize=True)
        params = oineus.ReductionParams()
        params.compute_v = True
        dcmp.reduce(params)

        col_idx = next(
            (j for j in range(fil.size()) if len(dcmp.v_data[j]) >= 2),
            None,
        )
        if col_idx is None:
            pytest.skip("No cohomology V column with >= 2 entries")

        fig, ax_a = _fresh_ax()
        oineus.plot_chain(field, fil, dcmp.v_data[col_idx], ax=ax_a,
                          source_kind="field", dualize=dcmp)
        a_collections = [type(c).__name__ for c in ax_a.collections]
        plt.close(ax_a.figure)

        fig, ax_b = _fresh_ax()
        oineus.plot_chain(field, fil, dcmp.v_data[col_idx], ax=ax_b,
                          source_kind="field", dualize=True)
        b_collections = [type(c).__name__ for c in ax_b.collections]
        plt.close(ax_b.figure)

        assert a_collections == b_collections

    def test_homology_decomposition_dualize_attr_is_false(self):
        # If user passes a homology decomposition we should NOT translate.
        field, fil = self._setup()
        dcmp_hom = oineus.Decomposition(fil, dualize=False)
        params = oineus.ReductionParams()
        params.compute_v = True
        dcmp_hom.reduce(params)

        col_idx = next(
            (j for j in range(fil.size()) if len(dcmp_hom.v_data[j]) >= 1),
            None,
        )
        if col_idx is None:
            pytest.skip("No homology V column with content")

        fig, ax_via_dcmp = _fresh_ax()
        oineus.plot_chain(field, fil, dcmp_hom.v_data[col_idx], ax=ax_via_dcmp,
                          source_kind="field", dualize=dcmp_hom)
        via_dcmp_types = [type(c).__name__ for c in ax_via_dcmp.collections]
        plt.close(ax_via_dcmp.figure)

        fig, ax_no_flag = _fresh_ax()
        oineus.plot_chain(field, fil, dcmp_hom.v_data[col_idx], ax=ax_no_flag,
                          source_kind="field")
        no_flag_types = [type(c).__name__ for c in ax_no_flag.collections]
        plt.close(ax_no_flag.figure)

        assert via_dcmp_types == no_flag_types


class TestStylePlumbing:
    def test_chain_style_getter_returns_copy(self):
        s = oineus.default_chain_edge_style()
        s["color"] = "totally-new"
        assert oineus.DEFAULT_CHAIN_EDGE_STYLE["color"] != "totally-new"

    def test_custom_edge_style_applied(self):
        points = _square_points()
        fil, _ = _build_vr(points, max_dim=1)
        edge_ids = _cells_by_dim(fil, 1)[:1]

        style = oineus.default_chain_edge_style()
        style["linewidth"] = 9.0
        fig, ax = _fresh_ax()
        oineus.plot_chain(points, fil, edge_ids, ax=ax, edge_style=style)
        lines = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert any(abs(float(lc.get_linewidths()[0]) - 9.0) < 1e-9 for lc in lines)
        plt.close(fig)
