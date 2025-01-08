#include "common_includes.h"

#include "entropy_minimizer.h"
#include "config.h"
#include "minimizer.h"


EntropyMinimizer::EntropyMinimizer(std::vector<std::complex<double> >* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension,double eps){
    minimizer = new Minimizer(kraus_ops, kraus_number, kraus_in_dimension, kraus_out_dimension, eps); // This avoids having to use initialize list

    // Initialize the current iteration and current MOE
    current_iteration = 0;
    MOE = -1;
} 


int EntropyMinimizer::initializeRun(){
    int info = minimizer->initializeRandomVector();
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
    int info = minimizer->initializeVector(start_vector);
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
                //DEBUG: PRINT STATEMENT
                std::cout << "We need to stop: we have reached the numerical instability."<<std::endl;
                return 1; // We need to stop
            }
        }
        // 3.2: stop if all the improvements are small - we have converged.
        // Since all improvements are positive, just check the total improvement divided by CONVERGENGE_ITERS
        if ((entropy_buffer[(current_iteration+1)%CONVERGENCE_ITERS]-entropy_buffer[(current_iteration)%CONVERGENCE_ITERS]) / CONVERGENCE_ITERS < CONVERGENCE_TOLERANCE){
            //DEBUG : print statement
            std::cout << "We have converged, the tolerance has been reached!" <<std::endl;
            return 1;
        }

    }
    return 0;

}

int EntropyMinimizer::runMinimization() {
    while (stepMinimization() == 0){
        std::cout << current_iteration << ": " <<std::fixed << std::setprecision(PRINT_PRECISION)<< *minimizer->getEntropy() <<std::endl;

        // check that we haven't iterated to many times!
        if (current_iteration > MAX_ITERATIONS){
            return 1;
        }
    }
    return 0;
}



EntropyMinimizer::~EntropyMinimizer()
{
    delete minimizer;
}

