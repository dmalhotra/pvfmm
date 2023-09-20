# Python bindings to PVFMM

This package provides Python bindings to the [C API](../include/pvfmm.h) using
[`ctypes`](https://docs.python.org/3/library/ctypes.html) and
[`mpi4py`](https://mpi4py.readthedocs.io/en/stable/index.html).

## Installation

Build and install the PVFMM library as usual, and then run `pip install .`
inside this directory.

## Usage

If you did not install PVFMM (see [INSTALL](../INSTALL)), you will need
to tell the Python library where to find the `libpvfmm.so` file with
`export PVFMM=/path/to/build/folder`.

Then, you should be able to run code which uses it by using
```
mpirun $other_args_here python -m mpi4py path/to/program.py
```

## Examples

See the `examples/` subdirectory for Python implementations of a few example programs.
The examples rely on [numba](https://numba.pydata.org/).
