[![CTest](https://github.com/anigmetov/oineus/actions/workflows/python-tests.yml/badge.svg)](https://github.com/anigmetov/oineus/actions/workflows/python-tests.yml) [![Build and Upload Wheels](https://github.com/anigmetov/oineus/actions/workflows/build_wheels.yaml/badge.svg)](https://github.com/anigmetov/oineus/actions/workflows/build_wheels.yaml)

# Oineus

Oineus is a high-performance persistent homology and topology-aware optimization
library for Python, powered by a parallel C++20 core and `nanobind` bindings
(with PEP 561 `.pyi` type stubs).

## Key Capabilities

- **Filtrations**:
  - Regular 1D–4D scalar fields via **Freudenthal** triangulations (including periodic `wrap=True` domains) and **cubical** complexes (`values_on="vertices"` or `"cells"`, sublevel or superlevel `negate=True`).
  - Point clouds via **Vietoris–Rips** (in-order generation), **Alpha** (weighted, periodic, exact predicates via `diode`), **Weak Alpha**, **Čech** (exact smallest enclosing balls), **Čech–Delaunay**, and **Function–Delaunay** bifiltrations.
  - Custom simplicial, cubical, and product complexes (`Filtration`, `ProdFiltration`, `list_to_filtration`, `mapping_cylinder`, `min_filtration`).
- **Parallel Matrix Reduction & Dynamic Persistence**:
  - Shared-memory lockfree persistence reduction (`R = D V` and selected-inversion `U = V^{-1}`) with clearing, apparent pairs, canonical ELZ restoration, slim & bit-packed encodings, `float32` and `float64` support, and zero-persistence diagrams.
  - Dynamic vineyards (`transpose`, `move`, `apply_move_schedule`), Luo–Nelson warm-started re-reduction (`update_with_permutation`, `update_with_edits`), and SiRUP simplex removal (`remove_simplices`).
- **Distances, Matchings & Barycenters**:
  - Exact and approximate **Bottleneck** and **$q$-Wasserstein** distances & explicit matchings (powered by Hera, including essential `InfKind` families), **Sliced Wasserstein** and diagram-corrected Sliced Wasserstein distances, and multistart / progressive **Wasserstein Fréchet means**.
- **Maps & Multi-Group Topology**:
  - Relative persistent homology, **Kernel / Image / Cokernel (KICR)** persistence (for inclusions and mapping cylinders), induced matchings, and two-population **Mixup Barcodes**.
- **Differentiable TDA (`oineus.diff`)**:
  - End-to-end differentiable filtrations, persistence diagrams, KICR diagrams, mixup barcodes, Wasserstein / Sliced Wasserstein losses, critical-set big-step optimization (`TopologyOptimizer`), and Weighill–Zhou **birth/death cochain content losses** (`oineus.diff.cochains`) supporting **PyTorch** and **JAX**.
- **Visualization (`oineus.vis`)**:
  - Publication-ready Matplotlib helpers (`plot_diagram`, `plot_matching`, `plot_diagram_gradient`, `plot_chain`) with Okabe–Ito palettes and hybrid scatter-density rendering for large diagrams.

## References

- D. Morozov and A. Nigmetov, *"Towards lockfree persistent homology,"* Proceedings of the 32nd ACM Symposium on Parallelism in Algorithms and Architectures (SPAA), 2020.
- A. Nigmetov and D. Morozov, [*"Topological Optimization with Big Steps,"*](https://arxiv.org/abs/2203.16748) arXiv:2203.16748, 2022.
- M. Vejdemo-Johansson, J. Matuszewski, and U. Bauer, [*"In-order generation of Vietoris-Rips Complexes,"*](https://arxiv.org/abs/2411.05495) arXiv:2411.05495, 2024.

## Pronunciation

Say it **OY-neh-oos**, IPA `/ˈɔɪneʊs/`, with stress on the first syllable:

- **Oi** as in **oil** or **coin**.
- **-eus** as in Orpheus, Theseus, or Prometheus, roughly "eh-oos".

## Documentation

Full executable tutorials, conceptual guides, and the complete Python API reference are hosted at **https://oineus.readthedocs.io**.

## Installation

Install prebuilt wheels from PyPI:

```shell
pip install oineus
```

Common optional extras depending on your workflow:

```shell
# Matplotlib visualization helpers (oineus.vis)
pip install "oineus[vis]"

# Alpha, Weak Alpha, Cech-Delaunay, and Function-Delaunay filtrations
pip install "diode>=1.2.2"

# Differentiable persistence & optimization (oineus.diff)
pip install torch   # or: pip install jax
```

## Building from Source

Oineus requires a C++20 compiler, CMake ($\ge 3.18$), Python ($\ge 3.9$), and Boost headers (`libboost-dev` on Ubuntu/Debian, `brew install boost` on macOS). All other C++ dependencies (`nanobind`, Taskflow, Hera, Eigen, Catch2) are vendored in `extern/` and `bindings/python/`.

```shell
git clone https://github.com/anigmetov/oineus.git
cd oineus
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(which python)"
cmake --build build -j4
```

To run the test suite from the build directory:

```shell
ctest --test-dir build --output-on-failure -j4
```

When importing the CMake-built module directly without installing a wheel, add `build/bindings/python` to `PYTHONPATH` (do not import from the source directory `bindings/python/`, which does not contain the compiled `_oineus` extension):

```shell
export PYTHONPATH="$PWD/build/bindings/python:$PYTHONPATH"
python -c "import oineus as oin; print(oin.__version__)"
```

## License

Oineus is distributed under a modified BSD license. See [legal.txt](legal.txt) for details.
