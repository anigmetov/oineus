# Installation

## Install from PyPI (Recommended)

Prebuilt wheels are published on PyPI for Linux (`x86_64`, `aarch64`) and macOS (`arm64`, `x86_64`) across Python 3.10–3.14:

```bash
pip install oineus
```

Core `oineus` depends only on `numpy`, `scipy`, and `eagerpy`. Install optional extras depending on which modules you use:

```bash
# Visualization helpers in oineus.vis (Matplotlib)
pip install "oineus[vis]"

# Hybrid density rendering for very large diagrams in plot_diagram(..., scatter_only=False)
pip install mpl-scatter-density

# Alpha, Weak Alpha, Cech-Delaunay, and Function-Delaunay filtrations (CGAL bindings)
pip install "diode>=1.2.2"

# Differentiable TDA in oineus.diff (install PyTorch, JAX, or both)
pip install torch
pip install jax

# Degree >= 1 cocycle lifting in oineus.diff.cochains (degree 0 needs no extra package)
pip install "dionysus>=2.1.8"
```

## Optional Dependency Matrix

| Package | Required For | Notes |
| --- | --- | --- |
| `matplotlib` (`oineus[vis]`) | {py:mod}`oineus.vis` (`plot_diagram`, `plot_matching`, `plot_diagram_gradient`, `plot_chain`) | Core `import oineus` works without it. |
| `mpl-scatter-density` | Hybrid pixel-density background in {py:func}`oineus.vis.plot_diagram` (`scatter_only=False`) | Required whenever `scatter_only=False` triggers density rendering. |
| `diode>=1.2.2` | {py:func}`oineus.alpha_filtration`, {py:func}`oineus.compute_diagrams_alpha`, {py:func}`oineus.function_delaunay_bifiltration`, and `oineus.diff.{alpha,weak_alpha,cech_delaunay}_filtration` | Version `1.2.2+` includes periodic Delaunay lifts. |
| `torch` | PyTorch autograd support in {py:mod}`oineus.diff` and all losses in `oineus.diff.cochains` | CPU build is sufficient (`torch` GPU tensors are moved cleanly across the C++ boundary). |
| `jax` | JAX `jit` / `grad` / `value_and_grad` support in {py:mod}`oineus.diff` | Works independently of `torch` (`TORCH_AVAILABLE=False` is supported). |
| `dionysus>=2.1.8` | Degree $\ge 1$ cocycle lifting in `oineus.diff.cochains` | Degree-0 birth/death cochains use union-find directly and do not require `dionysus`. |

:::{note}
Prebuilt **macOS Intel (`x86_64`)** wheels disable the vendored `jemalloc` allocator due to cross-compilation constraints on CI runners; numerical results are identical, while parallel reduction on Intel Macs can gain extra speed by building from source below. **Apple Silicon (`arm64`)** and **Linux** wheels link `jemalloc` out of the box.
:::

## Building from Source

Building from source requires:

- A **C++20** compiler (`clang++ >= 14` or `g++ >= 11`),
- **CMake** ($\ge 3.18$),
- **Python** ($\ge 3.9$),
- **Boost** headers (`sudo apt-get install libboost-dev` on Ubuntu/Debian, or `brew install boost` on macOS).

All other C++ libraries (`nanobind`, Taskflow, Hera, Eigen, Catch2) are vendored inside the repository—no system OpenMP or TBB installation is needed.

### Install into a Virtual Environment via `pip`

```bash
git clone https://github.com/anigmetov/oineus.git
cd oineus
pip install -v .
```

### Iterative CMake Build (Developer Workflow)

If you are developing on Oineus or running the CTest suite, build out-of-source with CMake and point `PYTHONPATH` to `build/bindings/python`:

```bash
git clone https://github.com/anigmetov/oineus.git
cd oineus

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(which python)"
cmake --build build -j4
ctest --test-dir build --output-on-failure -j4

export PYTHONPATH="$PWD/build/bindings/python:$PYTHONPATH"
```

:::{warning}
Do **not** run Python from inside `bindings/python/` or add the source directory `bindings/python/` to `PYTHONPATH`: that directory contains only `.py` and `.pyi` files without the compiled `_oineus` C++ extension, which causes an `ImportError: cannot import name '_oineus'`. Always import either from an installed wheel or from `build/bindings/python`.
:::

## Verify Your Installation

```python
import numpy as np
import oineus as oin

print("Oineus version:", oin.__version__)
img = np.random.default_rng(0).random((16, 16))
dgms = oin.compute_diagrams_ls(img)
print("H0 bars:", dgms.in_dimension(0).shape, "H1 bars:", dgms.in_dimension(1).shape)
```

Type-checkers (`mypy`, `pyright`, Pylance) automatically pick up `oineus/py.typed` and `.pyi` stubs with no extra configuration.
