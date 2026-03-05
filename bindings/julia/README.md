# Julia bindings (`CxxWrap.jl`)

This directory contains Julia bindings for Oineus.

Current binding surface is intentionally minimal and focused on simplicial
diagram computation:
- `CombinatorialSimplex`, `Simplex`
- `Filtration` (simplicial)
- `Decomposition`
- `DiagramPoint` / `Diagrams`
- `VREdge`

## Build

From the repository root:

```bash
cmake -S . -B build -Doin_build_julia=ON
cmake --build build -j
```

`CMake` must be able to find `JlCxx` (`CxxWrap` C++ package), e.g. via
`CMAKE_PREFIX_PATH`.

## Julia package wrapper

The Julia package scaffold lives in:

`bindings/julia/Oineus.jl`

If the built shared library is not in one of the default probe paths, set:

```bash
export OINEUS_JULIA_LIBRARY=/absolute/path/to/liboineus_julia.<ext>
```

Then in Julia:

```julia
using Pkg
Pkg.develop(path="bindings/julia/Oineus.jl")
using Oineus
```

## End-to-end example

From the repository root:

```bash
cmake -S . -B build_julia -Doin_build_julia=ON
cmake --build build_julia -j --target oineus_julia
export OINEUS_JULIA_LIBRARY="$(pwd)/build_julia/bindings/julia/liboineus_julia.$(julia -e 'print(Libdl.dlext)')"
julia --compiled-modules=no bindings/julia/example_e2e.jl
```

This prints:
- diagram points as Julia `DiagramPoint` objects (with birth/death and index data),
- the same diagram as a numeric matrix of shape `(n_points, 2)`.
