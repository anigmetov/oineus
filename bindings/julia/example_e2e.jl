#!/usr/bin/env julia

include(joinpath(@__DIR__, "Oineus.jl", "src", "Oineus.jl"))
using .Oineus

# Minimal simplicial complex:
# vertices [0], [1] appear at filtration value 0.0
# edge [0, 1] appears at filtration value 1.0
s0 = Oineus.Simplex([0], 0.0)
s1 = Oineus.Simplex([1], 0.0)
s01 = Oineus.Simplex([0, 1], 1.0)

println("Simplices constructed")
fil = Oineus.Filtration([s0, s1, s01], false, 1)
println("Filtration constructed")
dec = Oineus.Decomposition(fil, false, 1)
println("Decomposition constructed")

params = Oineus.ReductionParams()
Oineus.set_n_threads!(params, 1)
Oineus.reduce!(dec, params)

dgms = Oineus.diagram(dec, fil, true)

println("H0 diagram as DiagramPoint objects:")
h0 = Oineus.diagram_in_dimension(dgms, 0)
for p in h0
    println(
        "  (birth=$(Oineus.birth(p)), death=$(Oineus.death(p)), " *
        "birth_index=$(Oineus.birth_index(p)), death_index=$(Oineus.death_index(p)))"
    )
end

println()
println("H0 diagram as Float64 matrix (n_points, 2):")
println(Oineus.diagram_array(dgms, 0))

println()
println("H0 index diagram as UInt64 matrix (n_points, 2):")
println(Oineus.index_diagram_array(dgms, 0))
