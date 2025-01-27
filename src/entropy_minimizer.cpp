#include "common_includes.h"

#include "entropy_minimizer.h"
#include "config.h"
#include "minimizer.h"
#include "message_handler.h"

#include "uuid.h"


EntropyMinimizer::EntropyMinimizer(std::vector<std::complex<double> >* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, EntropyConfig* conf){

    // Save configuration
    config = conf;

    minimizer = new Minimizer(kraus_ops, kraus_number, kraus_in_dimension, kraus_out_dimension, config->epsilon); // This avoids having to use initialize list

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

    // get uuid for minimizer
    minimizer_id = generate_uuid_v4();

    // INITIALIZATION OF SERIALIZER
    serializer = new VectorSerializer();

    // INITIALIZATION OF ENTROPY ESTIMATOR 
    entropy_estimator = new EntropyEstimator();


    // Initialize the current iteration and current MOE
    current_iteration = 0;
    MOE = -1;


} 


int EntropyMinimizer::initializeRun(){
    message_handler->message("Initializing new run. No starting vector detected, generating random one...");
    int info = minimizer->initializeRandomVector();
    // print info
    if (info == 0){
        message_handler->message("Successfully initialized run with random vector!");
    } else if (info==1) {
        message_handler->message("The vector passed did not match the right dimension. I have instead generated a random one!");
    } else if (info==2){
        message_handler->message("The vector passed was empty. I have instead generated a random one!");
    }
    // print vector
    std::vector<std::complex<double> >* vector = minimizer->getState();
    std::string vector_string = "Vector: ";
    for (int i = 0; i < vector->size(); i++){
        vector_string += std::to_string(vector->at(i).real()) + " + " + std::to_string(vector->at(i).imag()) + "i, ";
    }
    message_handler->message(vector_string);


    // get new uuid
    run_id = generate_uuid_v4();

    current_iteration = 0;

    // Compute the entropy of the start vector and save it in the buffer
    minimizer->calculateEntropy();
    entropy_buffer[0] = *minimizer->getEntropy();

    // Reset the entropy estimator
    entropy_estimator->reset();
    entropy_estimator->appendEntropy(*minimizer->getEntropy());

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

    // get new uuid
    run_id = generate_uuid_v4();
    
    current_iteration = 0;

    // Compute the entropy of the start vector and save it in the buffer
    minimizer->calculateEntropy();
    entropy_buffer[0] = *minimizer->getEntropy();

    // Reset the entropy estimator
    entropy_estimator->reset();
    entropy_estimator->appendEntropy(*minimizer->getEntropy());


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

    // Step 2: update the MOE if the newly found MOE is lower. No need to calculate the entropy, since it already is done when minimizer steps
    // 2.1: Compute the entropy of the state and save it in the buffer. Buffer is CONVERGENCE_ITERS long. Also update the entropy estimator buffer
    entropy_buffer[current_iteration % CONVERGENCE_ITERS] = *minimizer->getEntropy();
    entropy_estimator->appendEntropy(*minimizer->getEntropy());
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


int EntropyMinimizer::findMOE(){
    // Print message
    oss.str("");
    oss << "Will try to find MOE. Running" << config->minimization_attempts << " minimization attempts.";
    message_handler->message(oss.str());

    // Step through the minimization attempts
    for (int i=0; i<config->minimization_attempts; i++){
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
        while (stepMinimization() == 0 && current_iteration < config->max_iterations && !predict_stop){
            // Print the current entropy from this run. 
            oss.str("");
            oss << "[Iteration " << current_iteration << "] Entropy: " << std::fixed << std::setprecision(PRINT_PRECISION) << *minimizer->getEntropy();
            message_handler->message(oss.str());
            // If necessary, update the MOE
            double new_entropy = *minimizer->getEntropy();
            if (new_entropy < MOE){
                MOE = new_entropy;
            }
            // Also print the current MOE
            oss.str("");
            oss << "Current MOE: " << MOE;
            message_handler->message(oss.str());
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
        if (current_iteration >= config->max_iterations){
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

int EntropyMinimizer::saveState(){
    // Save the state of the minimizer to a file
    // First, get the state of the minimizer
    std::vector<std::complex<double> >* state = minimizer->getState();
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
    std::string filename = save_path.string() + "/state_" + timestamp + ".dat";
    // serialize    
    serializer->serialize("vector",filename, *state,"Save state", minimizer->getD(), minimizer->getN());
    // Print message
    oss.str("");
    oss << "State saved to " << filename;
    message_handler->message(oss.str());
    return 0;
}


EntropyMinimizer::~EntropyMinimizer()
{
    delete minimizer;
    delete serializer;
    delete entropy_estimator;

}

