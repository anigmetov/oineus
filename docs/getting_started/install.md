# Installation

## From PyPI (recommended)

```bash
pip install oineus
```

This installs pre-built wheels where available. Oineus has runtime
dependencies on NumPy, SciPy, and `eagerpy`. Plotting helpers in
{py:mod}`oineus.vis` need matplotlib, available as the `vis` extra; PyTorch is
**optional** and only needed for the differentiable API in
{py:mod}`oineus.diff`. `import oineus` works without either -- the relevant
helpers are simply unavailable until you install them.

```bash
pip install oineus[vis]    # plotting helpers (matplotlib)
pip install oineus torch   # if you want oineus.diff
```

## From source

Oineus builds with CMake. You need a C++20 compiler and **Boost**. All other
dependencies (including the taskflow library used for parallelism) are vendored
under `extern/`, so there is no system TBB or OpenMP requirement.

```bash
git clone https://github.com/anigmetov/oineus.git
cd oineus
pip install -v .
```

A `pip install .` invokes the CMake build under the hood and produces an
importable `oineus` package with both Python and C++ parts.

## Developer setup

For iterative development, build once with CMake and point `PYTHONPATH` at
the build tree so you can re-run scripts without reinstalling. The
recommended layout is a dedicated build directory and a plain venv:

```bash
python -m venv venv_build
source venv_build/bin/activate
pip install numpy scipy matplotlib

mkdir build_nanobind && cd build_nanobind
cmake ..
make -j4

export PYTHONPATH="$PWD/bindings/python:$PYTHONPATH"
python -c "import oineus; print(oineus.__version__)"
```

The important bit: the _source_ directory `bindings/python/` only contains
`.py` files, not the compiled extension. You always want to import from
`<build>/bindings/python/`, which carries both.

## Optional dependencies

| Package | Needed for |
| --- | --- |
| `torch` | differentiable filtrations and diagrams in {py:mod}`oineus.diff` |
| `diode` | alpha-shape filtrations via {py:func}`oineus.compute_diagrams_alpha` |
| `mpl_scatter_density` | density plots in {py:func}`oineus.plot_diagram` |
| `gudhi` | convenient loaders for example datasets (not an Oineus dependency) |

## Verifying the install

```python
import numpy as np
import oineus

img = np.random.rand(16, 16)
dgms = oineus.compute_diagrams_ls(img, max_dim=1)
print("H0:", dgms.in_dimension(0).shape)
print("H1:", dgms.in_dimension(1).shape)
```

If you get back two `(n, 2)` arrays, your install works.
