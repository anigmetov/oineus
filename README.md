## About
Oineus is an implementation of shared-memory parallel
computation of persistent homology published in
*D. Morozov and A. Nigmetov.
"Towards lockfree persistent homology."
Proceedings of the 32nd ACM Symposium on Parallelism
in Algorithms and Architectures. 2020.*

It also contains an implementation of the critical set method
from
[*A. Nigmetov and D. Morozov,
"Topological Optimization with Big Steps."
arXiv preprint arXiv:2203.16748 (2022)."*](https://arxiv.org/abs/2203.16748)

Currently it supports computation of lower-star persistence
of scalar functions on regular grids, Vietoris-Rips filtrations or of user-defined
filtrations (where each simplex needs to be created manually).
It can compute image, kernel and cokernel persistence.
Oineus provides differentiable filtrations to simplify topological
optimization. It can compute zero-persistence diagrams. It provides
a convenience function for mapping cylinder construction.
It is written in C++ with python bindings (pybind11).

## Installation

```shell
pip install oineus
```

## Compilation

Oineus requires C++17 standard and python3.
`Pybind11` is included as a submodule.
Other dependencies are Boost and TBB,
install them before compiling Oineus.
E.g., on Ubuntu
```shell
$ sudo apt-get install libtbb-dev
$ sudo apt-get install libboost-all-dev
```


After that compilation is standard:

```shell
$ git clone --recurse-submodules git@github.com:anigmetov/oineus.git
$ cd oineus
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```
Compiled Oineus python package is located in `[build_directory]/python/bindings`.
The `oineus` directory there contains the `__init__.py` and the binary
with C++ bindings whose name is platform-dependent, say, `_oineus.cpython-311-x86_64-linux-gnu.so`.
If you get an error similar to
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/narn/code/oineus/bindings/python/oineus/__init__.py", line 7, in <module>
    from . import _oineus
ImportError: cannot import name '_oineus' from partially initialized module 'oineus' (most likely due to a circular import) (/home/narn/code/oineus/bindings/python/oineus/__init__.py)
```
you are most probably trying to import Oineus from the source directory `oineus/bindings/python`,
which contains only the `__init__.py`. Make sure that `PYTHONPATH` contains the directory `oineus/build/bindings/python`.

Another possibility can be that Pybind11 did not pick up the correct Python
executable. Suppose that you have system-wide Python, version 3.11,
but you want to compile for your virtual environment, `venv` or `conda`,
in which you have Python 3.9. Sometimes, even you activated your Python
virtual environment, pybind11 chooses a different Python (you can see which one
in the output of `cmake ..`; also, the name of the binary
`_oineus.cpython-311-x86_64-linux-gnu.so` contains the version).
In this case, remove CMakeCache.txt and run
```shell
$ cmake .. -DPYTHON_EXECUTABLE=path_to_correct_python
```
If the command `which python` shows you the right Python,
   the easiest is
```shell
$ cmake .. -DPYTHON_EXECUTABLE=$(which python)
```


Python packages needed by Oineus are `numpy` and `scipy`. Some of the examples
require `torch` for optimization and `matplotlib`, `plotly` and `dash` for
visualization. File `requirements.txt` contains all of these; if you do not
need to run examples, it is simpler to just `pip install numpy scipy` in your virtual
environment.


## Usage

See [Tutorial](doc/tutorial.md).

## License

Oineus is a free program distributed under modified
BSD license. See legal.txt for details.
