import oineus as oin


class Test1DGrid:
    def test_shape_1_grid(self):
        """Test 1D grid with shape [1] (1 vertex, 0 edges)."""
        dom = oin.GridDomain_1D(1)
        v0 = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)

        assert v0.boundary() == []
        assert v0.coboundary() == []
        assert v0.top_cofaces() == []

    def test_shape_2_grid(self):
        """Test 1D grid with shape [2] (2 vertices, 1 edge)."""
        dom = oin.GridDomain_1D(2)

        v0 = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)
        v1 = oin.CombinatorialCube_1D(anchor_vertex=[1], spanning_dims=[], domain=dom)
        e01 = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom)

        assert v0.boundary() == []
        assert set(v0.coboundary()) == {e01}
        assert set(v0.top_cofaces()) == {e01}

        assert v1.boundary() == []
        assert set(v1.coboundary()) == {e01}
        assert set(v1.top_cofaces()) == {e01}

        assert set(e01.boundary()) == {v0, v1}
        assert e01.coboundary() == []
        assert set(e01.top_cofaces()) == {e01}

    def test_shape_3_grid(self):
        """Test 1D grid with shape [3] (3 vertices, 2 edges)."""
        dom = oin.GridDomain_1D(3)

        v0 = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)
        v1 = oin.CombinatorialCube_1D(anchor_vertex=[1], spanning_dims=[], domain=dom)
        v2 = oin.CombinatorialCube_1D(anchor_vertex=[2], spanning_dims=[], domain=dom)
        e01 = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom)
        e12 = oin.CombinatorialCube_1D(anchor_vertex=[1], spanning_dims=[0], domain=dom)

        assert v0.boundary() == []
        assert set(v0.coboundary()) == {e01}
        assert set(v0.top_cofaces()) == {e01}

        assert v1.boundary() == []
        assert set(v1.coboundary()) == {e01, e12}
        assert set(v1.top_cofaces()) == {e01, e12}

        assert v2.boundary() == []
        assert set(v2.coboundary()) == {e12}
        assert set(v2.top_cofaces()) == {e12}

        assert set(e01.boundary()) == {v0, v1}
        assert e01.coboundary() == []
        assert set(e01.top_cofaces()) == {e01}

        assert set(e12.boundary()) == {v1, v2}
        assert e12.coboundary() == []
        assert set(e12.top_cofaces()) == {e12}

    def test_shape_4_grid(self):
        """Test 1D grid with shape [4] (4 vertices, 3 edges)."""
        dom = oin.GridDomain_1D(4)

        v0 = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)
        v1 = oin.CombinatorialCube_1D(anchor_vertex=[1], spanning_dims=[], domain=dom)
        v2 = oin.CombinatorialCube_1D(anchor_vertex=[2], spanning_dims=[], domain=dom)
        v3 = oin.CombinatorialCube_1D(anchor_vertex=[3], spanning_dims=[], domain=dom)
        e01 = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom)
        e12 = oin.CombinatorialCube_1D(anchor_vertex=[1], spanning_dims=[0], domain=dom)
        e23 = oin.CombinatorialCube_1D(anchor_vertex=[2], spanning_dims=[0], domain=dom)

        assert v0.boundary() == []
        assert set(v0.coboundary()) == {e01}
        assert set(v0.top_cofaces()) == {e01}

        assert v1.boundary() == []
        assert set(v1.coboundary()) == {e01, e12}
        assert set(v1.top_cofaces()) == {e01, e12}

        assert v2.boundary() == []
        assert set(v2.coboundary()) == {e12, e23}
        assert set(v2.top_cofaces()) == {e12, e23}

        assert v3.boundary() == []
        assert set(v3.coboundary()) == {e23}
        assert set(v3.top_cofaces()) == {e23}

        assert set(e01.boundary()) == {v0, v1}
        assert e01.coboundary() == []
        assert set(e01.top_cofaces()) == {e01}

        assert set(e12.boundary()) == {v1, v2}
        assert e12.coboundary() == []
        assert set(e12.top_cofaces()) == {e12}

        assert set(e23.boundary()) == {v2, v3}
        assert e23.coboundary() == []
        assert set(e23.top_cofaces()) == {e23}


class Test2DGrid:
    def test_shape_1x1_grid(self):
        """Test 2D grid with shape [1, 1] (1 vertex, 0 edges, 0 squares)."""
        dom = oin.GridDomain_2D(1, 1)
        v00 = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[], domain=dom)

        assert v00.boundary() == []
        assert v00.coboundary() == []
        assert v00.top_cofaces() == []

    def test_shape_2x2_grid(self):
        """Test 2D grid with shape [2, 2] (4 vertices, 4 edges, 1 square)."""
        dom = oin.GridDomain_2D(2, 2)

        v00 = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[], domain=dom)
        v10 = oin.CombinatorialCube_2D(anchor_vertex=[1, 0], spanning_dims=[], domain=dom)
        v01 = oin.CombinatorialCube_2D(anchor_vertex=[0, 1], spanning_dims=[], domain=dom)
        v11 = oin.CombinatorialCube_2D(anchor_vertex=[1, 1], spanning_dims=[], domain=dom)

        e00_h = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[0], domain=dom)
        e01_h = oin.CombinatorialCube_2D(anchor_vertex=[0, 1], spanning_dims=[0], domain=dom)
        e00_v = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[1], domain=dom)
        e10_v = oin.CombinatorialCube_2D(anchor_vertex=[1, 0], spanning_dims=[1], domain=dom)

        s00 = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[0, 1], domain=dom)

        assert v00.boundary() == []
        assert set(v00.coboundary()) == {e00_h, e00_v}
        assert set(v00.top_cofaces()) == {s00}

        assert v11.boundary() == []
        assert set(v11.coboundary()) == {e01_h, e10_v}
        assert set(v11.top_cofaces()) == {s00}

        assert set(e00_h.boundary()) == {v00, v10}
        assert set(e00_h.coboundary()) == {s00}
        assert set(e00_h.top_cofaces()) == {s00}

        assert set(e00_v.boundary()) == {v00, v01}
        assert set(e00_v.coboundary()) == {s00}
        assert set(e00_v.top_cofaces()) == {s00}

        assert set(s00.boundary()) == {e00_h, e01_h, e00_v, e10_v}
        assert s00.coboundary() == []
        assert set(s00.top_cofaces()) == {s00}

    def test_shape_3x5_grid(self):
        """Test 2D grid with shape [3, 5] - selective testing."""
        dom = oin.GridDomain_2D(3, 5)

        v00 = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[], domain=dom)
        e00_h = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[0], domain=dom)
        e00_v = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[1], domain=dom)
        s00 = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[0, 1], domain=dom)

        assert v00.boundary() == []
        assert set(v00.coboundary()) == {e00_h, e00_v}
        assert set(v00.top_cofaces()) == {s00}

        v10 = oin.CombinatorialCube_2D(anchor_vertex=[1, 0], spanning_dims=[], domain=dom)
        assert set(e00_h.boundary()) == {v00, v10}
        assert set(e00_h.coboundary()) == {s00}
        assert set(s00.top_cofaces()) == {s00}

        v02 = oin.CombinatorialCube_2D(anchor_vertex=[0, 2], spanning_dims=[], domain=dom)
        e02_h = oin.CombinatorialCube_2D(anchor_vertex=[0, 2], spanning_dims=[0], domain=dom)
        e01_v = oin.CombinatorialCube_2D(anchor_vertex=[0, 1], spanning_dims=[1], domain=dom)
        e02_v = oin.CombinatorialCube_2D(anchor_vertex=[0, 2], spanning_dims=[1], domain=dom)
        s01 = oin.CombinatorialCube_2D(anchor_vertex=[0, 1], spanning_dims=[0, 1], domain=dom)
        s02 = oin.CombinatorialCube_2D(anchor_vertex=[0, 2], spanning_dims=[0, 1], domain=dom)

        assert v02.boundary() == []
        assert set(v02.coboundary()) == {e02_h, e01_v, e02_v}
        assert set(v02.top_cofaces()) == {s01, s02}

        v12 = oin.CombinatorialCube_2D(anchor_vertex=[1, 2], spanning_dims=[], domain=dom)
        assert set(e02_h.boundary()) == {v02, v12}
        assert set(e02_h.coboundary()) == {s01, s02}

        e12_h = oin.CombinatorialCube_2D(anchor_vertex=[1, 2], spanning_dims=[0], domain=dom)
        e11_v = oin.CombinatorialCube_2D(anchor_vertex=[1, 1], spanning_dims=[1], domain=dom)
        e12_v = oin.CombinatorialCube_2D(anchor_vertex=[1, 2], spanning_dims=[1], domain=dom)
        e02_h_from_1 = oin.CombinatorialCube_2D(anchor_vertex=[0, 2], spanning_dims=[0], domain=dom)
        s11 = oin.CombinatorialCube_2D(anchor_vertex=[1, 1], spanning_dims=[0, 1], domain=dom)
        s12 = oin.CombinatorialCube_2D(anchor_vertex=[1, 2], spanning_dims=[0, 1], domain=dom)

        assert v12.boundary() == []
        assert set(v12.coboundary()) == {e12_h, e11_v, e12_v, e02_h_from_1}
        assert set(v12.top_cofaces()) == {s01, s02, s11, s12}

        v22 = oin.CombinatorialCube_2D(anchor_vertex=[2, 2], spanning_dims=[], domain=dom)
        assert set(e12_h.boundary()) == {v12, v22}
        assert set(e12_h.coboundary()) == {s11, s12}

        e13_h = oin.CombinatorialCube_2D(anchor_vertex=[1, 3], spanning_dims=[0], domain=dom)
        e22_v = oin.CombinatorialCube_2D(anchor_vertex=[2, 2], spanning_dims=[1], domain=dom)
        assert set(s12.boundary()) == {e12_h, e13_h, e12_v, e22_v}
        assert s12.coboundary() == []
        assert set(s12.top_cofaces()) == {s12}


def test_cube_value_wraps_combinatorial_methods_1d():
    """Value-cube API should expose combinatorial cube traversals in 1D."""
    dom = oin.GridDomain_1D(2)

    v0_val = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom, value=1.0)
    e01_val = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom, value=2.0)

    v0_comb = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom)
    e01_comb = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom)

    assert v0_val.boundary() == v0_comb.boundary() == []
    assert set(v0_val.coboundary()) == set(v0_comb.coboundary()) == {e01_comb}
    assert set(v0_val.top_cofaces()) == set(v0_comb.top_cofaces()) == {e01_comb}

    assert e01_val.boundary() == [c.uid for c in e01_comb.boundary()]
    assert set(e01_val.coboundary()) == set(e01_comb.coboundary()) == set()
    assert set(e01_val.top_cofaces()) == set(e01_comb.top_cofaces()) == {e01_comb}
