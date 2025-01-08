#ifndef ENTROPY_MINIMIZER_H
#define ENTROPY_MINIMIZER_H

#include "Config.h"
#include "Minimizer.h"

class EntropyMinimizer {
public:
    EntropyMinimizer(std::vector<std::complex<double> >* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension,double eps=EPS);
    ~EntropyMinimizer();

    // Setup functions
    int initializeRun();                        // This starts a new run with a random vector
    int initializeRun(std::vector<std::complex<double> >* start_vector);    // This starts a new run but with a specified vector

    // Algorithm functions
    int stepMinimization();                     // Do one step of minimization, then check if we need to stop. Return 1 if we need to stop, 0 othwerise.
    int runMinimization();

    Minimizer* minimizer;

private:
    double entropy_buffer[CONVERGENCE_ITERS];   // This array keeps track of past iterations of entropy
    int current_iteration;                      // This is the index of the current iteration, also used for insertion and deletion of elements fromt eh queue

    double MOE;


};



#endif