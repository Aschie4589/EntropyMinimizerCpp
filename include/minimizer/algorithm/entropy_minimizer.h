#ifndef ENTROPY_MINIMIZER_H
#define ENTROPY_MINIMIZER_H

#include <atomic>
#include <sstream>
#include <string>

#include "minimizer/config/compile_time_config.h"
#include "minimizer/config/entropy_config.h"

#include "minimizer/core/cuda_minimizer.h"

#include "utilities/messaging/message_handler.h"


class EntropyMinimizer {
public:
    EntropyMinimizer(cuDoubleComplex* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, EntropyConfig* conf, MessageHandler& msg_handler, CudaMinimizerStrategy strategy_preference = CudaMinimizerStrategy::AUTO_DETECT);
    ~EntropyMinimizer();

    // Setup functions
    cudaError_t initializeRun();                        // This starts a new run with a random vector
    cudaError_t initializeRun(cuDoubleComplex* start_vector);    // This starts a new run but with a specified vector. Vector is assumed to be on host.

    // Algorithm functions
    int stepMinimization();                     // Do one step of minimization, then check if we need to stop. Return 1 if we need to stop, 0 othwerise.
    int runMinimization();                      // Run one pass of the minimization algorithm. Requires a run to be initialized.
    int runMinimization(double target_entropy); // Run one pass of the minimization algorithm. Requires a run to be initialized.
    int findMOE();                              // This function finds the MOE of the channel

    // Minimizer interface
    double requestEntropy();
    cudaError_t requestStep();

    // IO functions
    //int saveState();                            // Save the state of the minimizer to a file
    //int saveState(std::string filename);        // Save the state of the minimizer to a file
    int saveVector();                           // Save the vector of the minimizer to a file
    int saveVector(std::string filename);       // Save the vector of the minimizer to a file
    
    // Graceful termination
    void requestTerminate();             // This is used to request the termination of the minimization algorithm
    bool shouldTerminate();              // This is used to check if the minimization algorithm should terminate
    static EntropyMinimizer* self;              // This is used to store the pointer to this instance, so that signal_handler can call the correct function
    static void signal_handler(int signal);            // This is the signal handler for the termination of the minimization algorithm

    // minimizer selection
    CudaMinimizerBase* minimizer;
    CudaMinimizer<double>* minimizer_d; // Double precision minimizer
    CudaMinimizer<float>* minimizer_f; // Float precision minimizer
    cudaError_t setMinimizer(CudaMinimizerBase* min); // Select the new minimizer to use

    EntropyConfig* config;

private:

    std::string run_id;                         // This is the id of the run
    std::string minimizer_id;                   // This is the id of the minimizer
    double entropy_buffer[CONVERGENCE_ITERS];   // This array keeps track of past iterations of entropy
    int current_iteration;                      // This is the index of the current iteration, also used for insertion and deletion of elements fromt eh queue
    std::ostringstream oss;                      // Useful for formatting certain strings
    MessageHandler& message_handler_;           // Non-owning reference to message handler
    // Seralizer
    VectorSerializer* serializer;               // This is used to save the state of the vector
    // Entropy estimator
    EntropyEstimator* entropy_estimator;       // This is used to estimate the entropy of the state

    double MOE;
    int input_dim;                              // Input dimension of the kraus operators
    int output_dim;                             // Output dimension of the kraus operators

    // Adaptive section
    cuComplex* kraus_ops_f;                     // Pointer to the kraus operators on device (only needed because this class is responsible for lifetime of object!)
    bool current_precision_float;               // Are we currently working with float precision?

    std::atomic<bool> terminate_requested{false};     // This is used to stop the minimization algorithm

};



#endif