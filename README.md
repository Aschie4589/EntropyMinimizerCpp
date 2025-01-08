# ENTROPY MINIMIZER

This is my attempt at porting the EntropyMinimizer written in Python to C++.
Ideally, it should make use of LAPACK and CBLAS to perform matrix operations fast (and on multiple CPU cores!)

**Known issues:**
- The algorithm seems to not always improve the entropy, and indeed sometimes the numbers go up. Needs further investigation
- A possible place to start: why is it that applying a channel with d Kraus operators to a rank 1 operator produces a rank d+1 output? This is visible in the entropy calculation...

**TO-DO:**
- Implement logs
- Implement save states
- Implement command line interaction
    - This would ideally mean that the program can just be run from command line, something like:
    ```
    minimizer --kraus kraus.file --full-minimization --save-intermediate --save-final --discard-unpromising-branches --track-resource-usage ...
- Add prediction of entropy (that is, the actual entropy of the channel and not the epsilon entropy)
- Implement a full search of MOE by discarding the branches that are not promising, exploiting exponential convergence
- Track resource usage in separate log file

