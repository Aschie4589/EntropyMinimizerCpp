/*
CONFIG

Here all the configuration parameters for the minimizer are stored.
These are expanded at compile time, so they will be hard-coded in the program.
*/

#ifndef CONFIG_H
#define CONFIG_H

/* 
Minimizer parameters
*/
#define EPS 1.0f/1000                      // The default value of epsilon that is passed to the minimizer.

#define MAX_ITERATIONS 500000           // How many iterations of the algorithm to run before giving up. Has to be less than uint32_t range

#define CONVERGENCE_TOLERANCE 1e-15     // When running the algorithm, if the improvement is below this threshold value for CONVERGENCE_ITERS iterations, 
#define CONVERGENCE_ITERS 20            // 

/*
The Minimizer class provides a few methods for printing vectors and matrices (which are not too advanced!)
Here you can change their behavior.
*/
#define PRINT_PRECISION 15  // How many digits to print in print statements


#endif