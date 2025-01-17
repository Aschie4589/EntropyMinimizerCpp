# ENTROPY MINIMIZER

This is my attempt at porting the EntropyMinimizer written in Python to C++.
Ideally, it should make use of LAPACK and CBLAS to perform matrix operations fast (and on multiple CPU cores!)

## Known issues
So far, none! Wheev...

## How to compile
To compile, open a terminal in the main folder and execute:
```batch
make FLAGS=value
```
Available flags are:
- `PLATFORM`. Can be `apple` or `linux`. Determines the compiler used, and the architecture (`arm64` or `x86_64`).
- `LAPACK`. Can be `accelerate`, `openblas` or `mkl`. Will make use of the Apple Accelerate, OpenBLAS or IntelMKL versions of BLAS and LAPACK.

**Important:** In order to compile, please make sure that the location of the relevant LAPACK libraries is set correctly (check `makefile`).

The compiled binary will be called `out`. Run the minimizer by calling:
``` batch
./out
```
in the relevant folder.

## Usage
*For now no command line options are available. To be updated...*


