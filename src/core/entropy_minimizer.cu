#include "common_includes.h"

#include "core/entropy_minimizer.h"
#include "config/config.h"
#include "core/cuda_minimizer.h"
#include "helpers/message_handler.h"

#include "helpers/uuid.h"

#include "cuComplex.h"
#include <cuda_runtime.h>




__global__ void complex_doubles_to_complex_floats(const cuDoubleComplex* in, cuComplex* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = make_cuComplex(static_cast<float>(in[idx].x), 
                                  static_cast<float>(in[idx].y));
    }
}

__global__ void complex_floats_to_complex_doubles(const cuComplex* in, cuDoubleComplex* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = make_cuDoubleComplex(static_cast<double>(in[idx].x), 
                                  static_cast<double>(in[idx].y));
    }
}


EntropyMinimizer::EntropyMinimizer(cuDoubleComplex* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, EntropyConfig* conf){
/*
    Wrapper class for the minimization algorithm.
    This class handles the initialization of the minimizer, the configuration, and the logging.
    It also keeps track of entropy and iteration counts.
    It stops the minimization algorithm gracefully on SIGTERM.

    Arguments:
    - kraus_ops (device): pointer to the array of kraus operators on device
    - kraus_number: number of kraus operators
    - kraus_in_dimension: input dimension of the kraus operators
    - kraus_out_dimension: output dimension of the kraus operators
    - conf: pointer to the configuration object that contains the parameters for the minimization algorithm

*/
    // Store configuration
    config = conf;

    // Initialize minimizers
    minimizer_d = new CudaMinimizer<double>(kraus_ops, kraus_number, kraus_in_dimension, kraus_out_dimension, config->epsilon); // This avoids having to use initialize list
    // Allocate more space on device for single precision kraus_ops
    kraus_ops_f = nullptr;
    cudaError_t err = cudaMalloc((void**)&kraus_ops_f, kraus_number * kraus_in_dimension * kraus_out_dimension * sizeof(cuComplex));
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate memory for single precision kraus operators: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory for single precision kraus operators");
    }
    int threads_per_block = 256; // Number of threads per block
    int blocks = (kraus_number * kraus_in_dimension * kraus_out_dimension + threads_per_block - 1) / threads_per_block; // Calculate number of blocks needed
    complex_doubles_to_complex_floats<<<blocks, threads_per_block>>>(kraus_ops, kraus_ops_f, kraus_number * kraus_in_dimension * kraus_out_dimension);
    cudaDeviceSynchronize();
    minimizer_f = new CudaMinimizer<float>(kraus_ops_f, kraus_number, kraus_in_dimension, kraus_out_dimension, 1.5e-07f); // This avoids having to use initialize list


    input_dim = kraus_in_dimension;
    output_dim = kraus_out_dimension;
    // initialize the entropy estimator
    entropy_estimator = new EntropyEstimator();
    // Initialize the current iteration and current MOE
    current_iteration = 0;
    MOE = -1;

    // Setup logging and messages
    message_handler = new MessageHandler();
    message_handler->createPrinter();
    if (config->use_custom_log_file){
        message_handler->createLogger(config->log_file);
    } else {
        message_handler->createLogger();
    }
    message_handler->setLogging(config->log);
    message_handler->setPrinting(config->print);  

    if (config->MOE_variable_precision){
        message_handler->message("Using adaptive precision for MOE calculations. Starting with float precision.");
        current_precision_float = true;
        setMinimizer(minimizer_f);
    } else {
        message_handler->message("Using double precision for MOE calculations.");
        current_precision_float = false; // Use full precision already now.
        setMinimizer(minimizer_d);
    }


    // get uuid
    minimizer_id = generate_uuid_v4();

    // initialize a serializer object
    serializer = new VectorSerializer();

    // Initialize the signaling stuff (for graceful termination)
    self = this;
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);
} 

double EntropyMinimizer::requestEntropy(){
    /*
    This function returns the current entropy of the selected minimizer.
    It is a wrapper for minimizer->getEntropy() and also handles the entropy calculation.
    
    Returns:
    - double: current entropy of the minimizer.
    
    Note: 
    - Minimizer should be selected correctly before requesting entropy.    
    */
    minimizer -> calculateEpsilonEntropy();
    return minimizer -> getEntropy();
}

cudaError_t EntropyMinimizer::requestStep(){
    /*
    This function performs one step of the minimization algorithm
    It is a wrapper for minimizer->step()
    
    Returns:
    - double: current entropy of the minimizer after the step.
    
    Note: 
    - Minimizer should be correctly set elsewhere!
    
    */
    cudaError_t err = minimizer->step();
    if (err != cudaSuccess) {
        message_handler->message("Failed to requestStep: " + std::string(cudaGetErrorString(err)));
    }
    return err;
}

cudaError_t EntropyMinimizer::setMinimizer(CudaMinimizerBase* min){
    /*
    This function selects the minimizer to use for the algorithm.
    It is used to switch between double and float precision minimizers.
    
    Input:
    - min: pointer to the new minimizer to use. Should be either a CudaMinimizer<double> or CudaMinimizer<float>.
    
    Returns:
    - cudaError_t: cudaSuccess on successful selection of the minimizer.
    
    Note: 
    - The minimizer should be initialized before calling this function.
    
    */
    if (min == nullptr) {
        std::cerr<<"Cannot select a null minimizer."<<std::endl;
        return cudaErrorInvalidValue;
    }
    
    minimizer = min;    
    return cudaSuccess;
}


cudaError_t EntropyMinimizer::initializeRun(){
    /*
    Initializes a new "run", which is one algorithm pass starting with a random vector, on which the algorithm can be run until convergence.
    This function is responsible for initializing the inner minimizer with ta random start_vector, and for initializing the necessary variables for the run to their correct state.
    
    Returns:
    - cudaError_t: cudaSuccess on successful initalization.

    Note: 
    - The start vector is generated randomly on the device, so no input is required.

    */

    message_handler -> message("Initializing new run. Selecting the appropriate precision for the minimizer...");
    if (config->MOE_variable_precision) {
        message_handler->message("Using float precision for minimizer initialization.");
        current_precision_float = true; // Start with float precision
        setMinimizer(minimizer_f); // Use float precision minimizer
    } else {
        message_handler->message("Using double precision for minimizer initialization.");
        current_precision_float = false; // Use double precision
        setMinimizer(minimizer_d); // Use double precision minimizer
    }

    message_handler->message("Generating a random vector...");
    cudaError_t err = minimizer -> initializeRandomVector();
    if (err != cudaSuccess){
        message_handler->message("Failed to initialize random vector: " + std::string(cudaGetErrorString(err)));
        return err;
    }

    // get new uuid
    run_id = generate_uuid_v4();

    current_iteration = 0;

    // Compute the entropy of the start vector and save it in the buffer
    double curr_en = requestEntropy();
    entropy_buffer[0] = curr_en;

    // Reset the entropy estimator
    entropy_estimator->reset();
    entropy_estimator->appendEntropy(curr_en);

    // Check if we have found a new MOE
    if (MOE < 0 || curr_en < MOE) {
        MOE = curr_en;
    }

    return err;
}


cudaError_t EntropyMinimizer::initializeRun(cuDoubleComplex* start_vector){
    /*
    Initializes a new "run", which is one algorithm pass starting with a given vector, on which the algorithm can be run until convergence.
    This function is responsible for initializing the inner minimizer with the given start_vector, and for initializing the necessary variables for the run to their correct state.
    
    Input:
    - start_vector (host): pointer to the start vector on host.
    
    Returns:
    - cudaError_t: cudaSuccess on successful initalization.

    Note: 
    - The passed vector is assumed to be of the correct size. Undefined behavior to be expected if dimensions don't match.
    - The passed vector is copied to the device memory.
    
    */
    message_handler -> message("Initializing new run. Selecting the appropriate precision for the minimizer...");
    if (config->MOE_variable_precision) {
        message_handler->message("Using float precision for minimizer initialization.");
        current_precision_float = true; // Start with float precision
        setMinimizer(minimizer_f); // Use float precision minimizer
    } else {
        message_handler->message("Using double precision for minimizer initialization.");
        current_precision_float = false; // Use double precision
        setMinimizer(minimizer_d); // Use double precision minimizer
    }

    // Depending on the type it might be necessary to cast this...
    if (current_precision_float) {
        cuComplex* tmp_vector = new cuComplex[input_dim];
        for (int i = 0; i < input_dim; i++) {
            tmp_vector[i] = make_cuComplex(static_cast<float>(start_vector[i].x), 
                                            static_cast<float>(start_vector[i].y));
        }
        // Initialize device...
        message_handler->message("Copying start vector to device in float precision.");
        cudaError_t err = minimizer_f->initializeVectorFromHost(static_cast<void*>(tmp_vector));
        delete[] tmp_vector; // Free the temporary vector
        if (err != cudaSuccess) {
            message_handler->message("Failed to initialize vector from host: " + std::string(cudaGetErrorString(err)));
            return err;
        }

    } else {
        message_handler->message("Copying start vector to device in double precision.");
        // Initialize device with the given start vector
        cudaError_t err = minimizer_d->initializeVectorFromHost(static_cast<void*>(start_vector));
        if (err != cudaSuccess) {
            message_handler->message("Failed to initialize vector from host: " + std::string(cudaGetErrorString(err)));
            return err;
        }
    }


    // get new uuid
    run_id = generate_uuid_v4();
    
    current_iteration = 0;
    if (config->MOE_prediction_tolerance){
        message_handler->message("MOE prediction tolerance is set to " + std::to_string(config->MOE_prediction_tolerance) + ". Using adaptive precision for MOE calculations.");
        current_precision_float = true; // Start with float precision
    } else {
        current_precision_float = false; // Use double precision
    }

    // Compute the entropy of the start vector and save it in the buffer

    double curr_en = requestEntropy();
    entropy_buffer[0] = curr_en;

    // Reset the entropy estimator
    entropy_estimator->reset();
    entropy_estimator->appendEntropy(curr_en);

    // Check if we have found a new MOE
    if (MOE < 0 || curr_en < MOE) {
        MOE = curr_en;
    }

    return cudaSuccess;
}

int EntropyMinimizer::stepMinimization(){
    /*
    
    This is a wrapper function for minimizer->step() and adds the following functionality:
    1. It keeps track of the iterations and checks for convergence.
    2. It keeps track of the MOE found in the current run.
    3. (optional) It attempts to predict a final MOE upon convergence, using the entropy_estimator.

    Returns:
    - int: 
        - ENTROPY_MINIMIZER_CONVERGED if the algorithm has converged
        - ENTROPY_MINIMIZER_MAX_ITERS if the maximum number of iterations has been reached
        - ENTROPY_MINIMIZER_NUMERICAL_INST if the algorithm has reached numerical instability (improvement is negative)

    */
    // Step 1: step through the algorithm
    requestStep();
    current_iteration +=1;

    // Step 2: update the MOE if the newly found MOE is lower. No need to calculate the entropy, since it already is done when minimizer steps
    // 2.1: Compute the entropy of the state and save it in the buffer. Buffer is CONVERGENCE_ITERS long. Also update the entropy estimator buffer
    double ent = requestEntropy();
    entropy_buffer[current_iteration % CONVERGENCE_ITERS] = ent;
    entropy_estimator->appendEntropy(ent);
    // 2.2: Check if we have found a new MOE
    if (ent < MOE) MOE = ent;

    // Step 3: check if we need to stop.
    if (current_iteration >= CONVERGENCE_ITERS){
        // 3.1: stop if the new improvement is negative - we have reached numerical instability!
        for (int i=0; i < CONVERGENCE_ITERS-1; i++){
            // Only run through CONVERGENCE_ITERS-1 because we want the deltas.
            // Compute entropy[i-1]-entropy[i] which needs to be positive. If negative: stop
            if (entropy_buffer[(current_iteration-i-1)%CONVERGENCE_ITERS]-entropy_buffer[(current_iteration-i)%CONVERGENCE_ITERS]<0){
                return ENTROPY_MINIMIZER_NUMERICAL_INST; // Stop
            }
        }
        // 3.2: stop if all the improvements are small - we have converged.
        // Since all improvements are positive, just check the total improvement divided by CONVERGENGE_ITERS
        if ((entropy_buffer[(current_iteration+1)%CONVERGENCE_ITERS]-entropy_buffer[(current_iteration)%CONVERGENCE_ITERS]) / CONVERGENCE_ITERS < CONVERGENCE_TOLERANCE){
            return ENTROPY_MINIMIZER_CONVERGED;
        }

    }
    return ENTROPY_MINIMIZER_CONTINUE;

}

int EntropyMinimizer::runMinimization(){
    /*
    
    Run a full minimization pass on the current minimizer state.

    Returns:
    - int: 
        - ENTROPY_MINIMIZER_CONVERGED if the algorithm has converged
        - ENTROPY_MINIMIZER_MAX_ITERS if the maximum number of iterations has been reached
        - ENTROPY_MINIMIZER_NUMERICAL_INST if the algorithm has reached numerical instability (improvement is negative)
        - ENTROPY_MINIMIZER_TERMINATED if the minimization was terminated by the user
        - ENTROPY_MINIMIZER_MOE_PREDICTION if the MOE prediction was used to stop the minimization.
    Note:
    - Before running, initialize the EntropyMinimizer with initializeRun() or initializeRun(cuDoubleComplex* start_vector).
    */



    message_handler->message("Running single minimization pass with no entropy prediction.");
    // Perform the minimization
    message_handler->message("Starting minimization...");
    int status = ENTROPY_MINIMIZER_CONTINUE; // Initialize status
    while (status == ENTROPY_MINIMIZER_CONTINUE && current_iteration < config->max_iterations && !shouldTerminate()){
        status = stepMinimization();
        // Print the current entropy from this run. 
        oss.str("");
        oss << "[Iteration " << current_iteration << "] Entropy: " << std::fixed << std::setprecision(PRINT_PRECISION) << requestEntropy();
        message_handler->message(oss.str());

        if (config->save_checkpoint && current_iteration % config->checkpoint_interval == 0){
            // Save the state
            oss.str("");
            oss << "[Iteration " << current_iteration << "] Checkpoint reached. Saving current state..";
            message_handler->message(oss.str());
            if (config->use_custom_checkpoint_file){
                saveVector(config->checkpoint_file);
            } else {
                saveVector();
            }
        }

    }
    if (terminate_requested){
        message_handler->message("Minimization stopped: termination requested.");
        status = ENTROPY_MINIMIZER_TERMINATED;
        if (config->save_checkpoint){
            message_handler->message("Checkpoints are enabled. Saving last checkpoint...");
            if (config->use_custom_checkpoint_file){
                saveVector(config->checkpoint_file);
            } else {
                saveVector();
            }
        }
    }
    else if (current_iteration >= config->max_iterations){
            message_handler->message("We reached the maximum number of iterations! Aborting...");
    } else {
        if (config->MOE_variable_precision && current_precision_float){
            // If we are using variable precision, we can switch to double precision minimizer
            message_handler->message("Switching to double precision minimizer for the final iterations.");
            setMinimizer(minimizer_d);
            current_precision_float = false; // Switch to double precision

            // Initialize minimizer_d with the current state of minimizer_f
            cuDoubleComplex* tmp_vec = nullptr;
            cudaMalloc((void**)&tmp_vec, input_dim * sizeof(cuDoubleComplex)); // Allocate temporary vector on device
            // Convert to double precision
            int threads_per_block = 256; // Number of threads per block
            int blocks = (input_dim + threads_per_block - 1) / threads_per_block; // Calculate number of blocks needed
            complex_floats_to_complex_doubles<<<blocks, threads_per_block>>>(minimizer_f->getVector(), tmp_vec, input_dim);
            cudaDeviceSynchronize();
            // Check for errors in the conversion
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                message_handler->message("Failed to convert float vector to double precision: " + std::string(cudaGetErrorString(err)));
                cudaFree(tmp_vec); // Free the temporary vector
                return err;
            }
            // Initialize the double precision minimizer with the converted vector
            err = minimizer_d->initializeVectorFromDevice(static_cast<void*>(tmp_vec));
            if (err != cudaSuccess) {
                message_handler->message("Failed to initialize double precision vector from host: " + std::string(cudaGetErrorString(err)));
                return err;
            }
            // Free the temporary vector
            cudaFree(tmp_vec);

            // Run minimization again!
            current_iteration = 0;
            runMinimization();

        } else {
            // If we are not using variable precision, we just stop
            message_handler->message("We reached the tolerance: we have converged!");
        }
    }

    // We have finished the minimization attempts. Print the final MOE
    oss.str("");
    oss << "Final entropy: " << requestEntropy();
    message_handler->message(oss.str());

    return status;
}

int EntropyMinimizer::runMinimization(double target_entropy){
    /*
    
    Run a full minimization pass on the current minimizer state.

    Returns:
    - int: 
        - ENTROPY_MINIMIZER_CONVERGED if the algorithm has converged
        - ENTROPY_MINIMIZER_MAX_ITERS if the maximum number of iterations has been reached
        - ENTROPY_MINIMIZER_NUMERICAL_INST if the algorithm has reached numerical instability (improvement is negative)
        - ENTROPY_MINIMIZER_TERMINATED if the minimization was terminated by the user
        - ENTROPY_MINIMIZER_MOE_PREDICTION if the MOE prediction was used to stop the minimization.
    Note:
    - Before running, initialize the EntropyMinimizer with initializeRun() or initializeRun(cuDoubleComplex* start_vector).
    */

    // Print message
    oss.str("");
    oss << "Running single minimization pass with target entropy " << target_entropy << ".";
    message_handler->message(oss.str());

    // Perform the minimization
    message_handler->message("Starting minimization...");
    // Initialize a flag that, if MOE prediction is used, will stop the minimization
    int status = ENTROPY_MINIMIZER_CONTINUE; // Initialize status
    bool predict_stop = false;
    while (status == ENTROPY_MINIMIZER_CONTINUE && current_iteration < config->max_iterations && !predict_stop && !shouldTerminate()){
        status = stepMinimization();
        // Print the current entropy from this run. 
        if (current_iteration % 20 == 0){
            oss.str("");
            oss << "[Iteration " << current_iteration << "] Entropy: " << std::fixed << std::setprecision(PRINT_PRECISION) << requestEntropy();
            message_handler->message(oss.str());
        }
        // Check if we need to stop because of final entropy prediction
        if (config->MOE_use_prediction){
            // First update the model
            double Rsquared = entropy_estimator->exponentialFit();

            // If fit is good, and the slope points the right way, predict the final entropy
            if (Rsquared > RSQUARED_THRESHOLD && entropy_estimator->model_params[1] < 0){
                // Predict the final entropy and the number of steps
                double predicted_entropy = entropy_estimator->predictFinalEntropy();
                int predicted_steps = entropy_estimator->predictFinalSteps();

                // Entropy is positive. A negative prediction means that the model is not valid.
                if (predicted_entropy>0){
                    // Print log message, use many digits
                    oss.str("");
                    oss << "Predicted final entropy: " << std::fixed << std::setprecision(PRINT_PRECISION) << predicted_entropy << " at iteration " << predicted_steps;
                    message_handler->message(oss.str());
                }
                if (predicted_entropy - target_entropy > config->MOE_prediction_tolerance){
                    predict_stop = true;
                }

            }                

        }
        if (predict_stop){
            message_handler->message("Predicted final entropy is above target entropy. Stopping minimization.");
            status = ENTROPY_MINIMIZER_MOE_PREDICTION;
        }

        if (config->save_checkpoint && current_iteration % config->checkpoint_interval == 0){
            // Save the state
            oss.str("");
            oss << "[Iteration " << current_iteration << "] Checkpoint reached. Saving current state..";
            message_handler->message(oss.str());
            if (config->use_custom_checkpoint_file){
                saveVector(config->checkpoint_file);
            } else {
                saveVector();
            }
        }
    }
    if (terminate_requested){
        message_handler->message("Minimization stopped: termination requested.");
        status = ENTROPY_MINIMIZER_TERMINATED;
        if (config->save_checkpoint){
            oss.str("");
            oss << "Checkpoints are enabled. Saving last checkpoint...";
            message_handler->message(oss.str());
            if (config->use_custom_checkpoint_file){
                saveVector(config->checkpoint_file);
            } else {
                saveVector();
            }
        }
    }
    else if (current_iteration >= config->max_iterations){
        message_handler->message("We reached the maximum number of iterations! Aborting...");
    } else if (predict_stop){
        message_handler->message("Minimization stopped: predicted MOE is above target entropy.");
    } else {
        message_handler->message("We reached the tolerance: we have converged!");
    }

    // We have finished the minimization attempts. Print the final MOE
    oss.str("");
    oss << "Final entropy: " << requestEntropy();
    message_handler->message(oss.str());

    return status;
}

int EntropyMinimizer::findMOE(){
    /*
    This function runs the minimization algorithm multiple times, each time starting with a random vector, until it finds the MOE of the channel.
    It uses the MOE prediction to stop the minimization if the predicted final entropy is above the current MOE.

    Returns:
    - int: 
        0 if the process terminated normally
        1 if the process was terminated by the user (SIGTERM)

    */
    // Print message
    oss.str("");
    oss << "Will try to find MOE. Running" << config->minimization_attempts << " minimization attempts.";
    message_handler->message(oss.str());

    // Step through the minimization attempts
    for (int i=0; i<config->minimization_attempts; i++){
        if (shouldTerminate()){
            message_handler->message("Termination requested. Aborting...");
            return 1;
        }
        // Initialize a new run
        initializeRun();
        // Print message
        oss.str("");
        oss << "Initializing minimization attempt " << i+1 << " of " << config->minimization_attempts << ".";
        message_handler->message(oss.str());

        // Perform the minimization
        message_handler->message("Starting minimization...");
        // Initialize a flag that, if MOE prediction is used, will stop the minimization
        bool predict_stop = false;
        while (stepMinimization() == 0 && current_iteration < config->max_iterations && !predict_stop && !shouldTerminate()){
            // Print the current entropy from this run. 
            if (current_iteration % 20 == 0){
                // Print every 20 iterations
                oss.str("");
                oss << "[Iteration " << current_iteration << "] Entropy: " << std::fixed << std::setprecision(PRINT_PRECISION) << requestEntropy();
                message_handler->message(oss.str());
            }
            // If necessary, update the MOE
            double new_entropy = requestEntropy();
            if (new_entropy < MOE){
                MOE = new_entropy;
            }
            // Also print the current MOE
            if (current_iteration % 20 == 0){
            oss.str("");
            oss << "Current MOE: " << MOE;
            message_handler->message(oss.str());
            }
            // Check if we need to stop because of MOE prediction
            if (config->MOE_use_prediction){
                // First update the model
                double Rsquared = entropy_estimator->exponentialFit();

                // If fit is good, and the slope points the right way, predict the final entropy
                if (Rsquared > RSQUARED_THRESHOLD && entropy_estimator->model_params[1] < 0){
                    // Predict the final entropy and the number of steps
                    double predicted_entropy = entropy_estimator->predictFinalEntropy();
                    int predicted_steps = entropy_estimator->predictFinalSteps();

                    // Entropy is negative. A negative prediction means that the model is not valid.
                    if (predicted_entropy>0){
                        // Print log message, use many digits
                        oss.str("");
                        oss << "Predicted final entropy: " << std::fixed << std::setprecision(PRINT_PRECISION) << predicted_entropy << " at iteration " << predicted_steps;
                        message_handler->message(oss.str());
                    }
                    if (predicted_entropy - MOE > config->MOE_prediction_tolerance){
                        predict_stop = true;
                    }

                }                

            }
        }
        if (shouldTerminate()){
            message_handler->message("Termination requested. Aborting...");
            return 1;
        }
        else if (current_iteration >= config->max_iterations){
            message_handler->message("We reached the maximum number of iterations! Aborting...");
        } else {
            message_handler->message("We reached the tolerance: we have converged!");
        }

    }

    // We have finished the minimization attempts. Print the final MOE
    oss.str("");
    oss << "Final MOE: " << MOE;
    message_handler->message(oss.str());
    

    return 0;
}


int EntropyMinimizer::saveVector(std::string filename){
    // First, get the vector from the minimizer
    std::vector<std::complex<double> >* vec = new std::vector<std::complex<double> >(input_dim);
    // Copy from GPU. If necessary, cast to complex<double>
    if (current_precision_float) {
        // If we are using float precision, we need to copy from the float vector
        cuComplex* tmp_vec = new cuComplex[input_dim];
        cudaError_t err = cudaMemcpy(tmp_vec, minimizer_f->getVector(), input_dim*sizeof(cuComplex), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
            message_handler->message("Failed to copy vector state from device: " + std::string(cudaGetErrorString(err)));
            delete[] tmp_vec; // Clean up
            return -1;
        }
        // Now, convert to complex<double>
        for (int i = 0; i < input_dim; i++) {
            (*vec)[i] = std::complex<double>(static_cast<double>(tmp_vec[i].x), 
                                              static_cast<double>(tmp_vec[i].y));
        }
        delete[] tmp_vec; // Clean up
    } else {
        // If we are using double precision, we can copy directly
        cudaError_t err = cudaMemcpy(vec->data(), minimizer_d->getVector(), input_dim*sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
            message_handler->message("Failed to copy vector state from device: " + std::string(cudaGetErrorString(err)));
            return -1;
        }    
    }
    // Create temporary filename (make operation atomic)
    std::string tmp_filename = filename + ".tmp";
    // Then, serialize the vector.    
    serializer->serialize("vector", tmp_filename, *vec, "Save state, custom path", 1, input_dim);
    // Rename the file
    std::filesystem::rename(tmp_filename, filename);
    // Print message
    oss.str("");
    oss << "Vector saved to " << filename;
    message_handler->message(oss.str());
    // Clean up
    delete vec;
    // Return success
    return 0;

}

int EntropyMinimizer::saveVector(){
    // Save the vector from the minimizer to a file.
    // First, get the vector from the minimizer
    std::vector<std::complex<double> >* vec = new std::vector<std::complex<double> >(input_dim);
    // Copy from GPU. If necessary, cast to complex<double>
    if (current_precision_float) {
        // If we are using float precision, we need to copy from the float vector
        cuComplex* tmp_vec = new cuComplex[input_dim];
        cudaError_t err = cudaMemcpy(tmp_vec, minimizer_f->getVector(), input_dim*sizeof(cuComplex), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
            message_handler->message("Failed to copy vector state from device: " + std::string(cudaGetErrorString(err)));
            delete[] tmp_vec; // Clean up
            return -1;
        }
        // Now, convert to complex<double>
        for (int i = 0; i < input_dim; i++) {
            (*vec)[i] = std::complex<double>(static_cast<double>(tmp_vec[i].x), 
                                              static_cast<double>(tmp_vec[i].y));
        }
        delete[] tmp_vec; // Clean up
    } else {
        // If we are using double precision, we can copy directly
        cudaError_t err = cudaMemcpy(vec->data(), minimizer_d->getVector(), input_dim*sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
            message_handler->message("Failed to copy vector state from device: " + std::string(cudaGetErrorString(err)));
            return -1;
        }    
    }    
    // Now, serialize the state
    // File is is SAVE_DIRECTORY/VECTORS_DIRECTORY/minimizer_id/run_id/state_timestamp.dat
    // use a path object then convert to string
    std::filesystem::path save_path = std::filesystem::path(SAVE_DIRECTORY) / std::filesystem::path(VECTORS_DIRECTORY) / std::filesystem::path(minimizer_id) / std::filesystem::path(run_id);
    // make sure the directory exists
    std::filesystem::create_directories(save_path);
    // Get the current time as a time_point
    auto now = std::chrono::system_clock::now();
    // Convert to time_t (the type used for time)
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // Format the time as a string
    std::tm tm = *std::localtime(&now_c);
    // Create a stringstream to format the time in a custom format
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    std::string timestamp = oss.str();
    // create the filename
    std::string filename = save_path.string() + "/vec_" + timestamp + ".dat";

    // create the tmp filename (make the operation atomic)
    std::string tmp_filename = save_path.string() + "/vec_" + timestamp + ".tmp";
    // serialize    
    serializer->serialize("vector", tmp_filename, *vec,"Save state", 1, input_dim);
    // rename the file
    std::filesystem::rename(tmp_filename, filename);

    // Print message
    oss.str("");
    oss << "Vector saved to " << filename;
    message_handler->message(oss.str());
    // Clean up
    delete vec;
    // Return success
    return 0;
}

void EntropyMinimizer::requestTerminate(){
    terminate_requested.store(true);
}

bool EntropyMinimizer::shouldTerminate(){
    return terminate_requested.load();
}

void EntropyMinimizer::signal_handler(int signal){
    if (signal == SIGTERM){
        self->requestTerminate();
    }
}

EntropyMinimizer* EntropyMinimizer::self = nullptr;

EntropyMinimizer::~EntropyMinimizer()
{
    delete minimizer_f;
    delete minimizer_d;

    delete message_handler;


    delete serializer;
    delete entropy_estimator;

    cudaFree(kraus_ops_f); // Free the single precision kraus operators
    kraus_ops_f = nullptr;
}

