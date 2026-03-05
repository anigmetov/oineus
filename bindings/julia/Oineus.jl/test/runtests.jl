using Test
using .Oineus

@testset "VREdge" begin
    e = Oineus.make_vr_edge(3, 7)
    @test Oineus.x(e) == 3
    @test Oineus.y(e) == 7

    Oineus.set_x!(e, 11)
    Oineus.set_y!(e, 13)
    @test Oineus.x(e) == 11
    @test Oineus.y(e) == 13

    @test occursin("edge(", Oineus.repr(e))
end

@testset "Simplex + Filtration + Decomposition + Diagrams" begin
    c0 = Oineus.CombinatorialSimplex([0])
    c1 = Oineus.CombinatorialSimplex([1])
    c01 = Oineus.CombinatorialSimplex([0, 1])

    s0 = Oineus.Simplex(c0, 0.0)
    s1 = Oineus.Simplex(c1, 0.0)
    s01 = Oineus.Simplex(c01, 1.0)

    fil = Oineus.Filtration([s0, s1, s01], false, 1)
    @test Oineus.size(fil) == 3
    @test Oineus.max_dim(fil) == 1
    @test Oineus.n_vertices(fil) == 2

    dec = Oineus.Decomposition(fil, false, 1)
    p = Oineus.ReductionParams()
    Oineus.set_n_threads!(p, 1)
    Oineus.reduce!(dec, p)

    dgms = Oineus.diagram(dec, fil, true)
    pts0 = Oineus.diagram_in_dimension(dgms, 0)
    @test length(pts0) >= 1
    @test all(x -> Oineus.birth(x) <= Oineus.death(x), pts0)

    idx0 = Oineus.index_diagram_in_dimension(dgms, 0)
    @test length(idx0) == length(pts0)

    arr0 = Oineus.diagram_array(dgms, 0)
    @test size(arr0, 2) == 2
    @test size(arr0, 1) == length(pts0)

    iarr0 = Oineus.index_diagram_array(dgms, 0)
    @test size(iarr0, 2) == 2
    @test size(iarr0, 1) == length(idx0)
end
