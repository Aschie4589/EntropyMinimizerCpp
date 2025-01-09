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


**TO-DO:**
- Implement logs (missing: integration wth EntropyMinimizer).
- Implement save states.
- Implement command line interaction.
    - This would ideally mean that the program can just be run from command line, something like:
    ```
    minimizer --kraus kraus.file --full-minimization --save-intermediate --save-final --discard-unpromising-branches --track-resource-usage ...
- Add prediction of entropy (that is, the actual entropy of the channel and not the epsilon entropy).
- Implement a full search of MOE by discarding the branches that are not promising, exploiting exponential convergence.
- Track resource usage in separate log file.

