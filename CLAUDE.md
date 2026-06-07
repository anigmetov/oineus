# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oineus is a C++ library for computing persistent homology with Python bindings. It implements:
- Shared-memory parallel computation of persistent homology
- Critical set method for topological optimization
- Differentiable filtrations for gradient-based optimization
- Kernel, image, and cokernel persistence

Key features:
- Lower-star persistence on regular grids (1D, 2D, 3D)
- Vietoris-Rips filtrations
- User-defined filtrations
- Zero-persistence diagrams
- Fréchet mean and Wasserstein distances for persistence diagrams

## Build Commands

### Standard C++ Build

Any build directory works; the conventional one is `build/`.

```bash
cmake -S . -B build
cmake --build build -j4
```

Build type defaults to Release. To build in Debug mode:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

Debug notes:
- You may find existing build directories in the tree named `build_nanobind/`, `build_claude/`, etc. These are personal working copies; there is **nothing magical about any specific name** and they will be cleaned up before merging.
- The wheel-building pipeline (scikit-build-core via `pyproject.toml`) creates its own build dirs under `.venv` / the wheel cache and is independent of manual cmake builds.

### Preferred Local Development Workflow (uv + cmake)

`pyproject.toml` is split into two visible sections (look for the
`# === Wheel build ===` / `# === Local dev environment ===` headers):
the upper half is wheel/distribution metadata that CI and
cibuildwheel consume; the lower half is for working on oineus locally.
This section is about the lower half.

```bash
# One-time setup: install dev deps into .venv WITHOUT installing
# oineus itself. --no-install-project is load-bearing -- see callout
# below.
uv sync --group dev --no-install-project

# Build C++ (pick any build dir; `build/` is the convention).
cmake -S . -B build -DPython_EXECUTABLE="$(uv run --no-sync which python)"
cmake --build build -j4

# Run tests via ctest from the build dir.
cd build && ctest --output-on-failure -j4
```

**Rebuild loop** after a C++ change:
```bash
cmake --build build -j4
cd build && ctest -R py-diff --output-on-failure -j4   # or whatever filter
```

For one-off scripts:
```bash
PYTHONPATH="$PWD/build/bindings/python" uv run --no-sync python my_script.py
```

> **Do NOT install oineus into the venv.** The scikit-build-core
> editable install registers a meta-path finder at
> `sys.meta_path[0]` that intercepts `import oineus` and routes it
> to the installed copy under `.venv/lib/.../site-packages/oineus/`,
> overriding both the `build/tests/oineus/` copy that ctest uses
> and any `PYTHONPATH` you set. That kills fast iteration -- C++
> rebuilds stop being visible.
>
> Concretely:
> - **Always** pass `--no-install-project` to `uv sync`.
> - **Never** run plain `uv sync` or `uv add <pkg>` -- both will
>   reinstate the editable install. Use `uv pip install <pkg>`
>   instead (it doesn't trigger a project sync), or add to
>   `[dependency-groups] dev` and re-run `uv sync --group dev
>   --no-install-project`.
> - If you ever end up with oineus installed, undo with
>   `uv pip uninstall oineus`.

The dev dependency list in `[dependency-groups] dev` covers
everything ctest needs: pytest, matplotlib (oineus.vis imports it
unconditionally), torch (py-diff-* tests), gudhi + dionysus
(ground-truth oracles), diode (CGAL alpha-shape tests), icecream
(a couple of example scripts). Heavy install -- torch alone is
~1 GB on macOS arm64. There is no lighter variant; all of these
are needed to get ctest fully green.

### Legacy workflow (venv_build/ + build_nanobind/)

The previous workflow used a plain `python -m venv venv_build` plus a cmake build in `build_nanobind/`. This still works and is what older docs refer to. The uv workflow above supersedes it for new work.

### CMake Options

- `oin_build_tests` (ON): Build C++ and Python tests
- `oin_build_examples` (ON): Build example programs
- `oin_use_spdlog` (OFF): Enable spdlog for logging
- `oin_use_jemalloc` (ON): Link jemalloc (a HARD requirement by default). Its
  thread-caching allocator roughly halves the free-heavy copy-back phase of the
  parallel reduction (per-column frees + cross-thread hazard-pointer
  reclamation), ~30% faster parallel wall on large grids. If jemalloc is not
  found, configure FAILS with a message explaining the fixes. Point CMake at a
  non-standard install with `-DJEMALLOC_ROOT=/prefix` (or the `JEMALLOC_ROOT` env
  var); to build without it and use the system allocator, pass
  `-Doin_use_jemalloc=OFF`.
- `oin_build_julia` (OFF): Build Julia bindings
- `OINEUS_PYTHON_INT` ("long int"): Integer type for Python bindings
- `OINEUS_PYTHON_REAL` ("double"): Real type for Python bindings

### Dependencies

Required:
- C++17 compiler
- Boost
- pthreads (via CMake `find_package(Threads)`; standard on Linux/macOS)
- Python 3.10+ (for Python bindings)

Threading: oineus uses `std::thread` + `std::atomic` directly, plus
the header-only taskflow library vendored under `extern/taskflow/`.
There is no system TBB / oneTBB / OpenMP dependency.

Vendored (under `extern/` or `bindings/python/`, not git submodules):
- nanobind (Python bindings)
- taskflow (parallel execution for sparse_matrix.h, kernel.h)
- Catch2 (C++ testing)
- hera (Wasserstein / bottleneck distances)
- Eigen, spdlog, opts, icecream

## Testing

### Preferred: run the full suite with CTest

```bash
cd build
ctest --output-on-failure
```

CTest is the preferred way to run tests. It covers both the C++ binaries
(Catch2) and the Python tests in a single invocation, and — importantly —
it only runs the tests that are explicitly registered in
`tests/CMakeLists.txt`. Some tests are intentionally disabled there
(commented-out `add_test` blocks for example scripts that need API
updates). Running `pytest tests/` directly will pick those files up and
fail; CTest skips them.

Filter examples (run from the build directory):
```bash
ctest -R "^py-"        # only python tests
ctest -R py-api-cells  # one test suite
ctest --output-on-failure -j4
```

### Registering a new test

Any newly added Python test file `tests/test_<name>.py` must be added to
`tests/CMakeLists.txt` so CTest picks it up. There are two `foreach(NAME
IN ITEMS ...)` blocks:

- The **pytest-driven** loop — for normal `test_*.py` files using pytest
  (`def test_*` functions, fixtures, etc.). This is the common case.
- The **script-style** loop — for example scripts that are run as plain
  `python test_<name>.py` with no pytest discovery.

Append the dashed test name (e.g. `my-new-feature`) to the appropriate
loop. Naming convention: CTest case `<foo-bar>` runs
`test_<foo_bar>.py` — dashes in the test name become underscores in the
file name. This rule is also stated as a comment at the top of the
pytest loop in `tests/CMakeLists.txt`.

### Direct invocation (for iterating on a single test)

For fast inner-loop work on one specific test:

```bash
# C++
./build/tests/tests
./build/tests/tests_parallel_col_to_row

# Python (via uv + cmake build)
PYTHONPATH="$PWD/build/bindings/python" uv run --no-sync pytest tests/test_api_cells.py
PYTHONPATH="$PWD/build/bindings/python" uv run --no-sync pytest tests/test_api_cells.py::test_simplex_api
PYTHONPATH="$PWD/build/bindings/python" uv run --no-sync pytest -k "test_vr" tests/
```

If you used the one-shot install path (`uv sync --group dev`), drop the
`PYTHONPATH=...` prefix — oineus is already on `sys.path` via the editable
install.

For anything beyond a single iteration, switch back to `ctest` so disabled
tests stay disabled and the C++ side runs too.

Common test suites:
- `py-api-*`: API smoke tests for Python bindings (one per binding module)
- `py-example-*`: Example scripts as tests
- `py-cube-dgm`, `py-kicr`, `py-frechet`: Feature-specific tests

## Code Architecture

### C++ Core (include/oineus/)

The C++ library is header-only with templates. Key components:

**Core data structures:**
- `cell.h`, `simplex.h`: Base cell types
- `product_cell.h`: Product cells for mapping cylinders
- `cube.h`: Cubical complex cells
- `cell_with_value.h`: Cells with filtration values

**Filtration construction:**
- `filtration.h`: Main filtration class
- `vietoris_rips.h`: VR complex construction
- `grid.h`: Regular grid management
- `grid_domain.h`: Grid indexing and boundary operations

**Persistence computation:**
- `decomposition.h`: Core VRU decomposition (R=DV, RU=D) for persistent homology
- `sparse_matrix.h`: Sparse matrix operations for reduction
- `kernel.h`: Kernel/image/cokernel persistence

**Diagram operations:**
- `diagram.h`: Persistence diagram representation
- `frechet_mean.h`: Fréchet mean computation for diagrams

**Optimization:**
- `top_optimizer.h`: Topology optimization utilities
- `loss.h`: Loss functions for optimization

**Utilities:**
- `params.h`: Parameter structures (ReductionParams, KICRParams)
- `timer.h`: Performance timing
- `profile.h`: Profiling support

### Python Bindings (bindings/python/)

The bindings use **nanobind** (not pybind11). Recent migration from pybind11 to nanobind is complete.

**Binding modules (*.cpp files):**
- `oineus.cpp`: Main module setup
- `oineus_common.cpp`: Common types and enums
- `oineus_cells.cpp`: Cell/simplex bindings
- `oineus_filtration.cpp`: Filtration construction
- `oineus_decomposition.cpp`: VRU decomposition and reduction
- `oineus_diagram.cpp`: Diagram types and operations
- `oineus_kicr.cpp`: Kernel/image/cokernel
- `oineus_top_optimizer.cpp`: Topology optimizer
- `oineus_functions.cpp`: Utility functions

**Python package (bindings/python/oineus/):**
- `__init__.py`: Main module with convenience functions
- `vis_utils.py`: Visualization helpers

**Differentiable diagrams:**
The `oineus.diff` subpackage (defined in `oineus/diff/__init__.py`) provides differentiable filtrations for PyTorch/JAX optimization.

### Test Organization

**C++ tests (tests/*.cpp):**
- Use Catch2 framework
- `tests_reduction.cpp`: Core reduction algorithms
- `tests_sparse_matrix.cpp`: Sparse matrix operations
- `tests_parallel_col_to_row.cpp`: Parallel algorithms

**Python tests (tests/*.py):**
- `test_api_*.py`: API smoke tests, one per binding module (cells, common, dcmp, dgm, fil, kicr)
- `test_fr_dgm_*.py`: Larger-scale serial/parallel reduction integration tests cross-checked against dionysus ground truth (random grids, vertebra dataset)
- `test_kicr.py`: Kernel/image/cokernel functional tests
- `test_example_*.py`: Example scripts (also serve as tests)
- `test_frechet.py`, `test_dgm_dist.py`: Distance and Fréchet-mean tests
- `test_diff_*.py`: Differentiable filtration / gradient / Wasserstein tests

## Development Guidelines

### Adding New C++ Functionality

1. Add header to `include/oineus/`
2. Implement as template if needed for type flexibility
3. Update relevant Python bindings in `bindings/python/oineus_*.cpp`
4. Add C++ tests to appropriate file in `tests/`
5. Add Python API tests to `tests/test_api_*.py`

### Modifying Python Bindings

The bindings use **nanobind**, not pybind11. Key differences:
- Use `nb::` namespace, not `py::`
- Different type caster syntax
- More strict about ownership and lifetime

When modifying bindings:
1. Update appropriate `oineus_*.cpp` file
2. Rebuild: `make -j4` in build directory
3. Test with `python -m pytest ../tests/test_api_*.py`

### Type System

**Template parameters:**
- `Int`: Integer type (typically `long int` in Python bindings)
- `Real`: Floating-point type (typically `double` in Python bindings)

These are configured via CMake: `OINEUS_PYTHON_INT` and `OINEUS_PYTHON_REAL`.

**Important classes are templated:**
- `Filtration<Cell>`: Works with Simplex, ProdSimplex, Cube
- `Decomposition<Int, Real>`: VRU decomposition
- `TopologyOptimizer<Int, Real>`: Optimization utilities

### Printable types convention

Every public C++ struct/class should be printable, with two tiers:

- **`operator<<(std::ostream&, const T&)`** — human-readable, what a user
  would want to see when they `print(obj)`. Keep it concise: e.g. for a
  simplex, just its vertices and filtration value, not internal indices,
  not boundary lists. This is the analogue of Python's `__str__`.

- **`std::string to_str_debug(const T&)`** (free function or member) —
  full state dump, for debugging. Includes anything that's not in the
  pretty form: internal IDs, sorted/unsorted indices, cached state, etc.
  This is the analogue of Python's `__repr__`.

`operator<<` should practically always be defined; `to_str_debug` only
when the pretty form genuinely omits useful debugging information.

In nanobind bindings, expose both via `std::stringstream`:

```cpp
nb::class_<T>(m, "T")
    .def("__str__",  [](const T& x) { std::stringstream ss; ss << x; return ss.str(); })
    .def("__repr__", [](const T& x) { return to_str_debug(x); });
```

If a class has no separate debug form, point both to `operator<<`. Many
existing types in the codebase only wire up `__repr__`; new code should
follow the two-tier convention, and existing types should be upgraded
opportunistically.

### Comment and docstring style

The "don't bloat the diff" rule below **supersedes** the two style rules
above it. Apply the style rules only to prose you are *already* writing
or rewriting for an unrelated reason; never go fix style in passing.

- **ASCII only in docstrings.** Docstrings reach users in many
  environments (terminals, sphinx, IDE tooltips, JSON/REST exports), so
  use plain `--`, `...`, `'`, `->`, `*` instead of em-dashes, ellipses,
  smart quotes, arrows, fancy bullets, mathematical Unicode, etc. This
  rule applies *only* to docstrings; in-code comments may use whatever
  characters you like.
- **No trailing period on single-line, single-sentence comments.**
  Periods are for two-or-more-sentence comments and for proper
  docstrings. A comment like `# clamp negative values` ends without a
  period; `# clamp negative values. Caller already filtered NaNs.` keeps
  both.
- **Don't touch comments or docstrings that are already fine** unless the
  task specifically requires it (or the comment no longer matches the
  code). Drive-by style edits bloat the diff and make it harder to
  review the actual change. This is the priority rule: if a docstring
  has non-ASCII characters or a single-line comment ends with a period
  but the prose is otherwise correct and the task is unrelated, leave
  it alone.

### Persistence Computation Workflow

Standard workflow for computing persistence:
1. **Create filtration**: `vr_filtration()`, `freudenthal_filtration()`, or manual `Filtration(simplices)`
2. **Create decomposition**: `Decomposition(fil, dualize=False)`
3. **Set reduction parameters**: `ReductionParams()` (threads, clearing, compute V/U)
4. **Reduce**: `dcmp.reduce(params)`
5. **Extract diagrams**: `dcmp.diagram(fil)` or `dcmp.diagram(fil).in_dimension(d)`

For kernel/image/cokernel:
1. Create two filtrations K (complex) and L (subcomplex)
2. Use `compute_kernel_image_cokernel_reduction(K, L, params)`
3. Extract diagrams: `.kernel()`, `.image()`, `.cokernel()`

### Common Patterns

**Accessing diagram points:**
```python
# As NumPy array
dgm = dcmp.diagram(fil).in_dimension(d)  # shape (n, 2)

# As list of DiagramPoint objects
dgm = dcmp.diagram(fil).in_dimension(d, as_numpy=False)
for p in dgm:
    p.birth, p.death, p.birth_index, p.death_index
```

**Multi-threading:**
Most operations support parallel execution via `ReductionParams.n_threads` or `n_threads` parameter.

**Simplex IDs:**
- `id`: Index in original simplex list (preserved if `set_ids=False`)
- `sorted_id`: Index in filtration order
- Vertex `i` must have `id == i`

## Performance and parallelism

### Sizes to expect

- **Filtrations and boundary matrices** are the heavyweights: easily millions
  of cells. They drive memory and CPU cost.
- **Persistence diagrams** are smaller than the underlying filtration but can
  still be very large (low dimensions especially). Note the distinction:
  the *index diagram* contains every persistence pair (including
  zero-persistence pairs); the diagrams returned to users have
  zero-persistence pairs filtered out by default.
- **Essential points** in any one diagram are typically ≤ 1. The code handles
  the multi-essential case correctly (e.g. four families in
  `oineus.EssentialMatches`), but those code paths are *not* a performance
  concern — small Python loops are fine, no need to vectorize or parallelize
  them.
- **Bottleneck longest edges** are usually unique. Ties produce more entries
  in `BottleneckMatching.longest.finite` / `longest.essential.*` and are
  handled correctly, but again the per-edge Python loop is not a hot path.

### Empirical complexity

The textbook worst-case complexities here are misleading; design decisions
should be guided by what we actually observe:

- **Hera Wasserstein** uses a geometric auction (not Hungarian) and computes
  approximate distances. Empirical scaling on our diagrams is roughly
  `O(n^1.6)`, not `O(n^3)`.
- **Hera bottleneck** scales empirically around `O(n^1.2)`, not
  `O(n^{1.5} log n)`.
- **Persistence reduction** has a true `O(n^3)` worst case, and that bound is
  attainable on adversarial inputs, but typical filtrations encountered in
  practice are much closer to linear in the number of cells.

Implication: do **not** assume "the heavy stage dominates anyway, so a
quadratic pre-/post-processing step in Python is fine." For million-cell
inputs the dominant phase may itself be only mildly superlinear, and a
naive quadratic step around it will swamp it.

### Hera is in-tree

Hera (under `extern/hera/`) is developed by the same author as Oineus —
it is not an external pinned dependency. Modifying Hera to improve its
interface, structure, or performance is fair game whenever it cleans up
the Python boundary or removes redundant work. There is no upstream API
contract to preserve.

### Parallelism conventions

- For functions that process large objects (filtration construction,
  reduction, large-diagram Hera distances/matchings), parallelize when it's
  straightforward, and expose an explicit `n_threads` parameter so the
  caller controls thread count. Don't pick a default that grabs every core.
- Python-facing functions must be safe to call from user-level parallel
  code (`joblib`, `concurrent.futures`, `threading`, multiprocessing pools,
  …). Concretely:
  - **No global mutable state** that two simultaneous calls could race on.
    Caches, counters, logger buffers, etc. must be either thread-safe or
    per-call.
  - **Release the GIL** on any non-trivial C++ work via
    `nb::call_guard<nb::gil_scoped_release, oineus_python::SignalGuard>()`.
    Order matters: nanobind's `call_guard` is implemented via
    `detail::tuple<Ts...>` where the LAST template argument is destroyed
    LAST, so `SignalGuard` must come second so its destructor runs after
    the GIL has been reacquired. Reversing the order causes
    `PyErr_SetString` to fire without the GIL -- intermittent segfaults
    have been seen here. The Hera distance and matching bindings already
    do this; new C++ entry points should follow suit.
  - Don't rely on per-process singletons that would break under
    `multiprocessing` fork-then-spawn.

### Ctrl-C handling

Every Python entry point wrapped in `SignalGuard` (see
`bindings/python/oineus_signal_guard.h`) responds to Ctrl-C within tens
of milliseconds. The mechanism: a ref-counted C-level SIGINT handler
flips an inline global `oineus::g_stop_flag` (`volatile sig_atomic_t`,
the only async-signal-safe write); long-running C++ loops poll it via
`oineus::interrupted()` and throw `oineus::interrupted_exception`; a
nanobind exception translator converts that to `KeyboardInterrupt`.
Polling sites live in `include/oineus/decomposition.h`,
`grid.h`, `vietoris_rips_inorder.h`, `filtration.h`, and -- under the
`HERA_USE_OINEUS_INTERRUPT` macro -- in `extern/hera/wasserstein/` and
`extern/hera/bottleneck/`.

**Worker threads must NOT throw on interrupt.** A `std::thread` that
exits via an uncaught exception calls `std::terminate`. Inside
`parallel_reduction` (decomposition.h:299), the polling site returns
early; the orchestrating function (`reduce_parallel_r_only`,
`reduce_parallel_rv`) checks the flag after `join()` and throws on
the joining thread. New parallel code should follow the same pattern.

Pure-C++ users get free cooperative cancellation: install their own
SIGINT handler that calls `oineus::request_stop()` and the existing
polling sites will pick it up. The Python `SignalGuard` only installs
its own handler while a wrapped binding is on the stack.

## Common Issues

**Import errors:**
If you see `ImportError: cannot import name '_oineus'`, check:
1. `PYTHONPATH` includes `build/bindings/python`, not source `bindings/python`
2. The C++ module was built successfully
3. Python version matches the one CMake detected (check CMake output)

**Wrong Python version:**
If pybind11/nanobind picks wrong Python:
```bash
cmake .. -DPYTHON_EXECUTABLE=$(which python)
```

**Test failures in build directory:**
Python tests expect to run from `build` directory with pytest. CTest handles this automatically.

## Miscellania

- preferred short name in Python examples: `import oineus as oin`


## Key Algorithms

**Persistent homology reduction:**
- Algorithm from "Towards lockfree persistent homology" (Morozov & Nigmetov, SPAA 2020)
- Parallel reduction with lock-free data structures
- Optional clearing optimization

**Critical set method:**
- From "Topological Optimization with Big Steps" (Nigmetov & Morozov, 2022)
- Computes critical sets for singleton losses
- Resolves conflicts with Max/Avg strategies

**Differentiable filtrations:**
- Backpropagation through persistence diagrams
- Critical edge tracking for VR complexes
- Critical vertex tracking for grid filtrations
