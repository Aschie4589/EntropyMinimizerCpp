#ifndef ENTROPY_MINIMIZER_H
#define ENTROPY_MINIMIZER_H

#include "config.h"
#include "minimizer.h"
#include "message_handler.h"
#include "entropy_config.h"
class EntropyMinimizer {
public:
    EntropyMinimizer(std::vector<std::complex<double> >* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, EntropyConfig* conf);
    ~EntropyMinimizer();

    // Setup functions
    int initializeRun();                        // This starts a new run with a random vector
    int initializeRun(std::vector<std::complex<double> >* start_vector);    // This starts a new run but with a specified vector

    // Algorithm functions
    int stepMinimization();                     // Do one step of minimization, then check if we need to stop. Return 1 if we need to stop, 0 othwerise.
    int runMinimization();
    int findMOE();                              // This function finds the MOE of the channel

    // IO functions
    int saveState(); // Save the state of the minimizer to a file

    Minimizer* minimizer;
    EntropyConfig* config;

private:
    std::string run_id;                         // This is the id of the run
    std::string minimizer_id;                   // This is the id of the minimizer
    double entropy_buffer[CONVERGENCE_ITERS];   // This array keeps track of past iterations of entropy
    int current_iteration;                      // This is the index of the current iteration, also used for insertion and deletion of elements fromt eh queue
    std::ostringstream oss;                      // Useful for formatting certain strings
    MessageHandler* message_handler;            // This makes sure logs and messages are handled correctly.
    // Seralizer
    VectorSerializer* serializer;               // This is used to save the state of the vector
    // Entropy estimator
    EntropyEstimator* entropy_estimator;       // This is used to estimate the entropy of the state

    double MOE;


};



#endif