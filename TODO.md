**TO-DO:**
[v] Implement logs (missing: integration with EntropyMinimizer).
- Implement save states.
- Implement command line interaction.
    - This would ideally mean that the program can just be run from command line, something like:
    ```
    minimizer --kraus kraus.file --full-minimization --save-intermediate --save-final --discard-unpromising-branches --track-resource-usage ...
- Add prediction of entropy (that is, the actual entropy of the channel and not the epsilon entropy).
- Implement a full search of MOE by discarding the branches that are not promising, exploiting exponential convergence.
- Track resource usage in separate log file.