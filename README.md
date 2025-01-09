# ENTROPY MINIMIZER

This is my attempt at porting the EntropyMinimizer written in Python to C++.
Ideally, it should make use of LAPACK and CBLAS to perform matrix operations fast (and on multiple CPU cores!)

**Known issues:**
So far, none! Wheev...

**How to compile**
To compile, open a terminal in the main folder and execute:
```batch
make
```
 
The compiled binary will be called `out`. Run the minimizer by calling:
``` batch
./out
```

In the relevant folder.

To compile the project, you'll need to update the `makefile` file, to reflect the location of your libraries.
*Note that it is important to link `openblas` before anything else! Otherwise `lapack` will not be multi-threaded!*

**Usage**

*For now no command line options are available. To be updated...*


