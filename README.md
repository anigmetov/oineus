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
of scalar functions on regular grids or of user-defined
filtrations (where each simplex needs to be created manually).
It is written in C++ with python bindings (pybind11).

## Compilation

Oineus requires C++17 standard and python3.
`Pybind11` is included as a submodule.
Compilation is standard:

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
which contains only the `__init__.py`. Make sure that `PYTHONPATH` contains the `oineus/build/bindings/python`.


## Usage

See [Tutorial](doc/tutorial.md).

## License

Oineus is a free program distributed under modified
BSD license. See legal.txt for details.
