# Contributing to PyCoMSA

For bug fixes or new features, please file an issue before submitting a
pull request. If the change isn't trivial, it may be best to wait for
feedback.

## Coding guidelines

### Versions

This project targets Python 3.7 or later.

Python objects should be typed; since it is not supported by Cython,
you must manually declare types in type stubs (`.pyi` files). In Python
files, you can add type annotations to function signatures (supported in
Python 3.5) and in variable assignments (supported only from Python
3.6 onward).

### Interfacing with C/C++

When interfacing with C or C++, and in particular with pointers, use assertions
everywhere you assume the pointer to be non-NULL. Also consider using
assertions when accessing raw C arrays, if applicable.

## Setting up a local repository

Make sure you clone the repository in recursive mode, so you also get the
wrapped code of Prodigal which is exposed as a ``git`` submodule:

```console
$ git clone --recursive https://github.com/althonos/pycomsa
```

## Running tests

Tests are written as usual Python unit tests with the `unittest` module of
the standard library. Running them requires the extension to be built
locally:

```console
$ python -m pip install -v -e . --no-build-isolation
$ python -m unittest -vv pycomsa.tests
```
