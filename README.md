# ENTROPY MINIMIZER

This is my attempt at porting the EntropyMinimizer written in Python to C++.
Ideally, it should make use of LAPACK and CBLAS to perform matrix operations fast (and on multiple CPU cores!)

**Known issues:**
- The algorithm seems to not always improve the entropy, and indeed sometimes the numbers go up. Needs further investigation

**TO-DO:**
- Implement logs
- Implement save state
- Implement command line interaction
- Add prediction of entropy (that is, the actual entropy of the channel and not the epsilon entropy)
- Implement a full search of MOE by discarding the branches that are not promising, exploiting exponential convergence

