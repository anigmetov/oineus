import oineus as oin
from icecream import ic


import pytest
import oineus as oin


class Test1DGrid:
    def test_shape_1_grid(self):
        """Test 1D grid with shape [1] (1 vertex, 0 edges)."""
        dom = oin.GridDomain_1D(1)

        # Only one vertex exists
        v0 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)

        # Test the single vertex
        assert v0.boundary() == []  # vertices have no boundary
        assert v0.coboundary() == []  # no edges exist
        assert v0.top_cofaces() == []  # no edges exist

    def test_shape_2_grid(self):
        """Test 1D grid with shape [2] (2 vertices, 1 edge)."""
        dom = oin.GridDomain_1D(2)

        # Create all cubes
        v0 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)
        v1 = oin.Cube_1D(anchor_vertex=[1], spanning_dims=[], domain=dom)
        e01 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom)

        # Test vertex at 0
        assert v0.boundary() == []
        assert set(v0.coboundary()) == {e01}
        assert set(v0.top_cofaces()) == {e01}

        # Test vertex at 1
        assert v1.boundary() == []
        assert set(v1.coboundary()) == {e01}
        assert set(v1.top_cofaces()) == {e01}

        # Test edge [0,1]
        assert set(e01.boundary()) == {v0, v1}
        assert e01.coboundary() == []  # already top-dimensional
        assert set(e01.top_cofaces()) == {e01}  # itself


    def test_shape_3_grid(self):
        """Test 1D grid with shape [3] (3 vertices, 2 edges)."""
        dom = oin.GridDomain_1D(3)

        # Create all cubes
        v0 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)
        v1 = oin.Cube_1D(anchor_vertex=[1], spanning_dims=[], domain=dom)
        v2 = oin.Cube_1D(anchor_vertex=[2], spanning_dims=[], domain=dom)
        e01 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom)
        e12 = oin.Cube_1D(anchor_vertex=[1], spanning_dims=[0], domain=dom)

        # Test boundary vertex at 0
        assert v0.boundary() == []
        assert set(v0.coboundary()) == {e01}
        assert set(v0.top_cofaces()) == {e01}

        # Test interior vertex at 1
        assert v1.boundary() == []
        assert set(v1.coboundary()) == {e01, e12}
        assert set(v1.top_cofaces()) == {e01, e12}

        # Test boundary vertex at 2
        assert v2.boundary() == []
        assert set(v2.coboundary()) == {e12}
        assert set(v2.top_cofaces()) == {e12}

        # Test edge [0,1]
        assert set(e01.boundary()) == {v0, v1}
        assert e01.coboundary() == []
        assert set(e01.top_cofaces()) == {e01}

        # Test edge [1,2]
        assert set(e12.boundary()) == {v1, v2}
        assert e12.coboundary() == []
        assert set(e12.top_cofaces()) == {e12}

    def test_shape_4_grid(self):
        """Test 1D grid with shape [4] (4 vertices, 3 edges)."""
        dom = oin.GridDomain_1D(4)

        # Create all cubes
        v0 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)
        v1 = oin.Cube_1D(anchor_vertex=[1], spanning_dims=[], domain=dom)
        v2 = oin.Cube_1D(anchor_vertex=[2], spanning_dims=[], domain=dom)
        v3 = oin.Cube_1D(anchor_vertex=[3], spanning_dims=[], domain=dom)
        e01 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom)
        e12 = oin.Cube_1D(anchor_vertex=[1], spanning_dims=[0], domain=dom)
        e23 = oin.Cube_1D(anchor_vertex=[2], spanning_dims=[0], domain=dom)

        # Test boundary vertex at 0
        assert v0.boundary() == []
        assert set(v0.coboundary()) == {e01}
        assert set(v0.top_cofaces()) == {e01}

        # Test interior vertex at 1
        assert v1.boundary() == []
        assert set(v1.coboundary()) == {e01, e12}
        assert set(v1.top_cofaces()) == {e01, e12}

        # Test interior vertex at 2
        assert v2.boundary() == []
        assert set(v2.coboundary()) == {e12, e23}
        assert set(v2.top_cofaces()) == {e12, e23}

        # Test boundary vertex at 3
        assert v3.boundary() == []
        assert set(v3.coboundary()) == {e23}
        assert set(v3.top_cofaces()) == {e23}

        # Test edge [0,1]
        assert set(e01.boundary()) == {v0, v1}
        assert e01.coboundary() == []
        assert set(e01.top_cofaces()) == {e01}

        # Test edge [1,2]
        assert set(e12.boundary()) == {v1, v2}
        assert e12.coboundary() == []
        assert set(e12.top_cofaces()) == {e12}

        # Test edge [2,3]
        assert set(e23.boundary()) == {v2, v3}
        assert e23.coboundary() == []
        assert set(e23.top_cofaces()) == {e23}



class Test2DGrid:
    def test_shape_1x1_grid(self):
        """Test 2D grid with shape [1, 1] (1 vertex, 0 edges, 0 squares)."""
        dom = oin.GridDomain_2D(1, 1)

        # Only one vertex exists
        v00 = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[], domain=dom)

        assert v00.boundary() == []
        assert v00.coboundary() == []
        assert v00.top_cofaces() == []

    def test_shape_2x2_grid(self):
        """Test 2D grid with shape [2, 2] (4 vertices, 4 edges, 1 square)."""
        dom = oin.GridDomain_2D(2, 2)

        # Vertices
        v00 = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[], domain=dom)
        v10 = oin.Cube_2D(anchor_vertex=[1, 0], spanning_dims=[], domain=dom)
        v01 = oin.Cube_2D(anchor_vertex=[0, 1], spanning_dims=[], domain=dom)
        v11 = oin.Cube_2D(anchor_vertex=[1, 1], spanning_dims=[], domain=dom)

        # Edges (horizontal: spanning dim 0, vertical: spanning dim 1)
        e00_h = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[0], domain=dom)  # [0,0] to [1,0]
        e01_h = oin.Cube_2D(anchor_vertex=[0, 1], spanning_dims=[0], domain=dom)  # [0,1] to [1,1]
        e00_v = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[1], domain=dom)  # [0,0] to [0,1]
        e10_v = oin.Cube_2D(anchor_vertex=[1, 0], spanning_dims=[1], domain=dom)  # [1,0] to [1,1]

        # Square
        s00 = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[0, 1], domain=dom)

        # Test corner vertex [0,0]
        assert v00.boundary() == []
        assert set(v00.coboundary()) == {e00_h, e00_v}
        assert set(v00.top_cofaces()) == {s00}

        # Test corner vertex [1,1]
        assert v11.boundary() == []
        assert set(v11.coboundary()) == {e01_h, e10_v}
        assert set(v11.top_cofaces()) == {s00}

        # Test horizontal edge
        assert set(e00_h.boundary()) == {v00, v10}
        assert set(e00_h.coboundary()) == {s00}
        assert set(e00_h.top_cofaces()) == {s00}

        # Test vertical edge
        assert set(e00_v.boundary()) == {v00, v01}
        assert set(e00_v.coboundary()) == {s00}
        assert set(e00_v.top_cofaces()) == {s00}

        # Test square
        assert set(s00.boundary()) == {e00_h, e01_h, e00_v, e10_v}
        assert s00.coboundary() == []
        assert set(s00.top_cofaces()) == {s00}


    def test_shape_3x5_grid(self):
        """Test 2D grid with shape [3, 5] - selective testing."""
        dom = oin.GridDomain_2D(3, 5)

        # Corner: vertex, edge, square at [0,0]
        v00 = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[], domain=dom)
        e00_h = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[0], domain=dom)
        e00_v = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[1], domain=dom)
        s00 = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[0, 1], domain=dom)

        # Corner vertex [0,0] - only 2 edges, 1 square
        assert v00.boundary() == []
        assert set(v00.coboundary()) == {e00_h, e00_v}
        assert set(v00.top_cofaces()) == {s00}

        # Corner edge and square
        v10 = oin.Cube_2D(anchor_vertex=[1, 0], spanning_dims=[], domain=dom)
        assert set(e00_h.boundary()) == {v00, v10}
        assert set(e00_h.coboundary()) == {s00}
        assert set(s00.top_cofaces()) == {s00}

        # Side: vertex at [0,2] (on left edge, not corner)
        v02 = oin.Cube_2D(anchor_vertex=[0, 2], spanning_dims=[], domain=dom)
        e02_h = oin.Cube_2D(anchor_vertex=[0, 2], spanning_dims=[0], domain=dom)
        e01_v = oin.Cube_2D(anchor_vertex=[0, 1], spanning_dims=[1], domain=dom)
        e02_v = oin.Cube_2D(anchor_vertex=[0, 2], spanning_dims=[1], domain=dom)
        s01 = oin.Cube_2D(anchor_vertex=[0, 1], spanning_dims=[0, 1], domain=dom)
        s02 = oin.Cube_2D(anchor_vertex=[0, 2], spanning_dims=[0, 1], domain=dom)

        # Side vertex - 3 edges, 2 squares
        assert v02.boundary() == []
        assert set(v02.coboundary()) == {e02_h, e01_v, e02_v}
        assert set(v02.top_cofaces()) == {s01, s02}

        # Side edge at [0,2] spanning dim 0 - horizontal edge, has 2 squares!
        v12 = oin.Cube_2D(anchor_vertex=[1, 2], spanning_dims=[], domain=dom)
        assert set(e02_h.boundary()) == {v02, v12}
        assert set(e02_h.coboundary()) == {s01, s02}  # two squares above and below

        # Interior: vertex at [1,2]
        e12_h = oin.Cube_2D(anchor_vertex=[1, 2], spanning_dims=[0], domain=dom)
        e11_v = oin.Cube_2D(anchor_vertex=[1, 1], spanning_dims=[1], domain=dom)
        e12_v = oin.Cube_2D(anchor_vertex=[1, 2], spanning_dims=[1], domain=dom)
        e02_h_from_1 = oin.Cube_2D(anchor_vertex=[0, 2], spanning_dims=[0], domain=dom)
        s11 = oin.Cube_2D(anchor_vertex=[1, 1], spanning_dims=[0, 1], domain=dom)
        s12 = oin.Cube_2D(anchor_vertex=[1, 2], spanning_dims=[0, 1], domain=dom)

        # Interior vertex - 4 edges, 4 squares
        assert v12.boundary() == []
        assert set(v12.coboundary()) == {e12_h, e11_v, e12_v, e02_h_from_1}
        assert set(v12.top_cofaces()) == {s01, s02, s11, s12}

        # Interior edge at [1,2] spanning dim 0
        v22 = oin.Cube_2D(anchor_vertex=[2, 2], spanning_dims=[], domain=dom)
        assert set(e12_h.boundary()) == {v12, v22}
        assert set(e12_h.coboundary()) == {s11, s12}  # two squares above and below

        # Interior square at [1,2]
        e13_h = oin.Cube_2D(anchor_vertex=[1, 3], spanning_dims=[0], domain=dom)
        e22_v = oin.Cube_2D(anchor_vertex=[2, 2], spanning_dims=[1], domain=dom)
        assert set(s12.boundary()) == {e12_h, e13_h, e12_v, e22_v}
        assert s12.coboundary() == []
        assert set(s12.top_cofaces()) == {s12}

# # 1D
# #
# dom_1 = oin.GridDomain_1D(10)

# vertex_1 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom_1)

# ic(vertex_1)
# ic(vertex_1.boundary())

# assert(len(vertex_1.boundary()) == 0)

# ic(vertex_1.coboundary())

# correct_coboundary = set( [oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom_1).uid] )
# ic(correct_coboundary)

# assert(set([c.uid for c in vertex_1.coboundary()]) == correct_coboundary)

# c1 = vertex_1.coboundary()[0]
# c2 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom_1)

# ic(c1 == c2)

# print(vertex_1.top_cofaces())


# vertex_2 = oin.Cube_1D(anchor_vertex=[3], spanning_dims=[], domain=dom_1)

# assert(len(vertex_2.boundary()) == 0)
