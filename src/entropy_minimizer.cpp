#include "common_includes.h"

#include "entropy_minimizer.h"
#include "config.h"
#include "minimizer.h"
#include "message_handler.h"


EntropyMinimizer::EntropyMinimizer(std::vector<std::complex<double> >* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, EntropyConfig* conf){

    // Save configuration
    config = conf;

    minimizer = new Minimizer(kraus_ops, kraus_number, kraus_in_dimension, kraus_out_dimension, config->epsilon); // This avoids having to use initialize list

    // Setup logging and messages
    message_handler = new MessageHandler();
    message_handler->createPrinter();
    message_handler->createLogger(config->log_file);
    message_handler->setLogging(config->log);
    message_handler->setPrinting(config->print);  
//    std::ostringstream oss;  
    // Initialize the current iteration and current MOE
    current_iteration = 0;
    MOE = -1;


} 


int EntropyMinimizer::initializeRun(){
    message_handler->message("Initializing new run. No starting vector detected, generating random one...");
    int info = minimizer->initializeRandomVector();
    message_handler->message("Vector generated!");

    current_iteration = 0;

    // Compute the entropy of the start vector and save it in the buffer
    minimizer->calculateEntropy();
    entropy_buffer[0] = *minimizer->getEntropy();

    // Check if we have found a new MOE
    if (MOE < 0 || entropy_buffer[0] < MOE) {
        MOE = entropy_buffer[0];
    }

    return info;
}

int EntropyMinimizer::initializeRun(std::vector<std::complex<double> >* start_vector){
    message_handler->message("Initializing new run. A vector was passed as input...");

    int info = minimizer->initializeVector(start_vector);
    if (info == 0){
        message_handler->message("Successfully initialized run with given vector!");
    } else if (info==1) {
        message_handler->message("The vector passed did not match the right dimension. I have instead generated a random one!");
    } else if (info==2){
        message_handler->message("The vector passed was empty. I have instead generated a random one!");
    }

    current_iteration = 0;

    // Compute the entropy of the start vector and save it in the buffer
    minimizer->calculateEntropy();
    entropy_buffer[0] = *minimizer->getEntropy();

    // Check if we have found a new MOE
    if (MOE < 0 || entropy_buffer[0] < MOE) {
        MOE = entropy_buffer[0];
    }

    return info;
}

int EntropyMinimizer::stepMinimization(){
    // Step 1: step through the algorithm
    minimizer->step();
    current_iteration +=1;

    // Step 2: update the MOE if the newly found MOE is lower
    // 2.1: Compute the entropy of the state and save it in the buffer. Buffer is CONVERGENCE_ITERS long.
    minimizer->calculateEntropy();
    entropy_buffer[current_iteration % CONVERGENCE_ITERS] = *minimizer->getEntropy();
    // 2.2: Check if we have found a new MOE
    if (entropy_buffer[0] < MOE) {
        MOE = entropy_buffer[current_iteration % CONVERGENCE_ITERS];
    }

    // Step 3: check if we need to stop.
    if (current_iteration >= CONVERGENCE_ITERS){
        // 3.1: stop if the new improvement is negative - we have reached numerical instability!
        for (int i=0; i < CONVERGENCE_ITERS-1; i++){
            // Only run through CONVERGENCE_ITERS-1 because we want the deltas.
            // Compute entropy[i-1]-entropy[i] which needs to be positive. If negative: stop
            if (entropy_buffer[(current_iteration-i-1)%CONVERGENCE_ITERS]-entropy_buffer[(current_iteration-i)%CONVERGENCE_ITERS]<0){
                return 1; // We need to stop
            }
        }
        // 3.2: stop if all the improvements are small - we have converged.
        // Since all improvements are positive, just check the total improvement divided by CONVERGENGE_ITERS
        if ((entropy_buffer[(current_iteration+1)%CONVERGENCE_ITERS]-entropy_buffer[(current_iteration)%CONVERGENCE_ITERS]) / CONVERGENCE_ITERS < CONVERGENCE_TOLERANCE){
            return 1;
        }

    }
    return 0;

}

int EntropyMinimizer::runMinimization() {
    while (stepMinimization() == 0){
        // To print this message, clear the stream first
        oss.str("");
        oss << current_iteration << ": " <<std::fixed << std::setprecision(PRINT_PRECISION)<< *minimizer->getEntropy();
        message_handler->message(oss.str());
        // check that we haven't iterated to many times!
        if (current_iteration > config->max_iterations){
            message_handler->message("We reached the maximum number of iterations! Aborting...");
            return 1;
        }
    }
    message_handler->message("We reached the tolerance: we have converged!");

    return 0;
}



EntropyMinimizer::~EntropyMinimizer()
{
    delete minimizer;
}

