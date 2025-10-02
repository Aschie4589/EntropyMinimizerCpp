// My headers
#include "common_includes.h"
#include "config/config.h"

#include "core/matrix_operations.h"

#include "core/cuda_minimizer.h"
#include "core/entropy_minimizer.h"
#include "helpers/vector_serializer.h"
#include "core/entropy_estimator.h"

#include "helpers/message_handler.h"
#include "helpers/logger.h"

#include "core/kraus/random_generator.h"
#include "core/generate_random_vector.h"

#include "helpers/uuid.h"

#include "helpers/parse_arguments.h"
#include "libs/argparse/argparse.hpp"


#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

// Forward declarations for strategy classes
template<typename T> class HighMemoryStrategy;
template<typename T> class LowMemoryStrategy;
template<typename T> class LowMemoryStrategyProb;



int main(int argc, char** argv){

    // Get general purpose message handler
    MessageHandler* message_handler = new MessageHandler();
    // Step 0: Parse command line arguments. Parser is created with new, so it must be deleted.
    argparse::ArgumentParser* parser = parse_arguments(argc, argv);
    int N, d;
    
    

    // Option 1: kraus was called
    if (parser->is_subcommand_used("kraus")){
        // Case 1: Haar was called
        if (parser->at<argparse::ArgumentParser>("kraus").is_subcommand_used("haar")){
            // check printing and logging options and create logger or printer accordingly. Don't give file names or anything.
            if (parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<bool>("-l")){
                message_handler->createLogger();
            } 
            if (!parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<bool>("-s")){
                message_handler->createPrinter();
            }

            // now also print to message handler
            // first print the full command line
            std::string full_command = "Command called: ";
            for (int i = 0; i < argc; i++){
                full_command += argv[i];
                full_command += " ";
            }
            message_handler->message(full_command);
            // Now explicitly print the options
            message_handler->message("Parsed option N: " + std::to_string(parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<int>("-N")));
            message_handler->message("Parsed option k: " + std::to_string(parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<int>("-d")));
            message_handler->message("Parsed output: " + parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<std::string>("-o"));
            // print logging and printing options,
            message_handler->message("Logging is: " + std::to_string(parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<bool>("-l")));
            message_handler->message("Printing is: " + std::to_string(parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<bool>("-s") ));
            // save into N, d and output
            N = parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<int>("-N");
            d = parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<int>("-d");
            std::string output = parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<std::string>("-o");

            // Get GPU number to use
            int gpu_number = parser->at<argparse::ArgumentParser>("kraus").at<argparse::ArgumentParser>("haar").get<int>("--gpu");
            // Set the device to use
            cudaSetDevice(gpu_number);
            // log the GPU number
            message_handler->message("Using GPU number: " + std::to_string(gpu_number));
            // check that the output directory exists
            //first get the directory
            std::string output_directory = output.substr(0, output.find_last_of("/"));
            // check if the directory exists
            if (!std::filesystem::exists(output_directory)){
                // throw an error
                message_handler->message("Output directory does not exist. Please create it first.");
                return 1;
            }


            std::vector<std::complex<double> >* kraus_operators_host = new std::vector<std::complex<double> >(d * N * N);
            // Create a generator, first config 
            RandomGeneratorConfig config = RandomGeneratorConfig();
            config.kraus_number = d;
            config.kraus_in_dimension = N;
            config.kraus_out_dimension = N;
            config.message_handler = message_handler;

            RandomGenerator* generator = new RandomGenerator(config);
            // generate the kraus operators
            if ( generator->generate(kraus_operators_host) != 0 ){
                message_handler->message("Error generating Kraus operators.");
                return 1;
            }
            delete generator;

            // save the kraus operators
            VectorSerializer serializer = VectorSerializer();
            serializer.serialize("kraus", output, *kraus_operators_host, "Kraus operators for a random unitary channel", d, N);
            // Clean up
            delete kraus_operators_host;

            // print exit message
            message_handler->message("Kraus operators saved to " + output + ".");
            return 0;
        }

    } 

    // If condition to decide if any of "singleshot" or "multishot" was called
    if (parser->is_subcommand_used("singleshot")){
        // get the selected subparser
        argparse::ArgumentParser* subparser;
        subparser = &parser->at<argparse::ArgumentParser>("singleshot");

        // check printing and logging options and create logger or printer accordingly. Don't give file names or anything.
        if (subparser->get<bool>("-l")){
            message_handler->createLogger();
        } 
        if (!subparser->get<bool>("-s")){
            message_handler->createPrinter();
        }

        // DEBUG
        // print to message handler
        // first print the full command line
        std::string full_command = "Command called: ";
        for (int i = 0; i < argc; i++){
            full_command += argv[i];
            full_command += " ";
        }
        message_handler->message(full_command);
        // Now explicitly print the options
        message_handler->message("Parsed Kraus operators: " + subparser->get<std::string>("-k"));
        // Also print logging and printing options
        message_handler->message("Logging is: " + std::to_string(subparser->get<bool>("-l")));
        message_handler->message("Printing is: " + std::to_string(subparser->get<bool>("-s") ));
        // Get GPU number to use
        int gpu_number = subparser->get<int>("--gpu");
        // Set the device to use
        cudaSetDevice(gpu_number);
        // log the GPU number
        message_handler->message("Using GPU number: " + std::to_string(gpu_number));

        cuDoubleComplex *d_kraus_operators;
        VectorSerializer serializer = VectorSerializer();


        { // Scoped so that memory is deallocated once we leave this section - deserialized data is heavy!
        // Get the kraus operators from file
        DeserializedData deserialized_data = serializer.deserialize(subparser->get<std::string>("-k"));
        // Copy the kraus operators to GPU
        message_handler->message("Kraus operators loaded from " + subparser->get<std::string>("-k") + ".");
        
        cudaError_t errmalloc = cudaMalloc(&d_kraus_operators, deserialized_data.d*deserialized_data.N*deserialized_data.N * sizeof(std::complex<double>));
        if (errmalloc != cudaSuccess) {
            message_handler->message("Error allocating memory for Kraus operators: " + std::string(cudaGetErrorString(errmalloc)));
            return 1;
        }
        message_handler->message("Successfully allocated " + std::to_string(deserialized_data.vectorData.size() * sizeof(std::complex<double>)) + " bytes or " + std::to_string(deserialized_data.vectorData.size() * sizeof(std::complex<double>)/1024.0/1024/1024) + " GiB for Kraus operators on device.");
        // Copy the kraus operators to GPU
        cudaError_t errcopy = cudaMemcpy(d_kraus_operators, deserialized_data.vectorData.data(), deserialized_data.vectorData.size() * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
        if (errcopy != cudaSuccess) {
            message_handler->message("Error copying Kraus operators to device: " + std::string(cudaGetErrorString(errcopy)));
            return 1;
        }
        message_handler->message("Successfully copied " + std::to_string(deserialized_data.vectorData.size() * sizeof(std::complex<double>)) + " bytes or " + std::to_string(deserialized_data.vectorData.size() * sizeof(std::complex<double>)/1024.0/1024/1024) + " GiB of Kraus operators to device.");
        // Get N and d from metadata
        N = deserialized_data.N;
        d = deserialized_data.d;
        // Log the N and d
        message_handler->message("N: " + std::to_string(N));
        message_handler->message("d: " + std::to_string(d));
        }

        // Initialize configuration
        EntropyConfig config = EntropyConfig();
        // if present, set the number of iterations
        if (subparser->is_used("-i")){
            config.setMaxIterations(subparser->get<int>("-i"));
        }
        // set logging and printing
        config.setLogging(subparser->get<bool>("-l"));
        config.setPrinting(!subparser->get<bool>("-s"));
        // set prediction
        config.setMOEUsePrediction(subparser->get<bool>("--predict"));
        // set checkpointing
        config.setCheckpointing(subparser->get<bool>("-c"));
        if (subparser->is_used("-cf")){
            config.setCheckpointFile(subparser->get<std::string>("-cf"));
            // debug
            message_handler->message("Checkpoint file set to " + subparser->get<std::string>("-cf"));
        }
        if (subparser->is_used("-ci")){
            config.setCheckpointInterval(subparser->get<int>("-ci"));
            // debug
            message_handler->message("Checkpoint interval set to " + std::to_string(subparser->get<int>("-ci")));
        }

        // Parse strategy preference
        CudaMinimizerStrategy strategy = parseStrategyString(subparser->get<std::string>("--strategy"));
        message_handler->message("Using strategy: " + subparser->get<std::string>("--strategy"));

        // finally, create a minimizer
        EntropyMinimizer* minimizer = new EntropyMinimizer(d_kraus_operators, d, N, N, &config, strategy);

        signal(SIGTERM, minimizer->signal_handler);
        signal(SIGINT, minimizer->signal_handler);

        // Initialize run
        // Check if we have a starting vector specified in the command line
        if (subparser->is_used("--vector")){
            cuDoubleComplex* d_start_vector;
            cudaError_t err_vec_all = cudaMalloc(&d_start_vector, N * sizeof(cuDoubleComplex));
            if (err_vec_all != cudaSuccess) {
                message_handler->message("Error allocating memory for starting vector: " + std::string(cudaGetErrorString(err_vec_all)));
                return 1;
            }
            message_handler->message("Successfully allocated " + std::to_string(N * sizeof(cuDoubleComplex)) + " bytes or " + std::to_string(N * sizeof(cuDoubleComplex)/1024.0/1024/1024) + " GiB for copying starting vector on device.");
            
            // load the vector
            DeserializedData deserialized_vector = serializer.deserialize(subparser->get<std::string>("--vector"));
            // Copy the vector to GPU
            cudaError_t err_vec_copy = cudaMemcpy(d_start_vector, deserialized_vector.vectorData.data(), deserialized_vector.vectorData.size() * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
            if (err_vec_copy != cudaSuccess) {
                message_handler->message("Error copying starting vector to device: " + std::string(cudaGetErrorString(err_vec_copy)));
                return 1;
            }
            message_handler->message("Successfully copied " + std::to_string(deserialized_vector.vectorData.size() * sizeof(std::complex<double>)) + " bytes or " + std::to_string(deserialized_vector.vectorData.size() * sizeof(std::complex<double>)/1024.0/1024/1024) + " GiB of starting vector to device.");


            // log the vector loaded message
            message_handler->message("Starting vector loaded from " + subparser->get<std::string>("--vector") + ".");
            // check dimension
            if (deserialized_vector.N != N || deserialized_vector.d != 1){
                message_handler->message("Starting vector has wrong dimensions. Expected N = " + std::to_string(N) + ", d = 1. Got N = " + std::to_string(deserialized_vector.N) + ", d = " + std::to_string(deserialized_vector.d) + ".");
                return 1;
            }
            // initialize run
            minimizer->initializeRun(d_start_vector);
            // clean up
            cudaError_t err_vec_free = cudaFree(d_start_vector);
            if (err_vec_free != cudaSuccess) {
                message_handler->message("Error freeing memory for starting vector: " + std::string(cudaGetErrorString(err_vec_free)));
                return 1;
            }
            message_handler->message("Successfully freed memory for starting vector.");

        } else {
            // if no vector is specified, initialize with a random vector
            message_handler->message("No starting vector detected, generating random one...");
            minimizer->initializeRun();
        }

        // If target is set, pass the number to runMininization
        if (subparser->is_used("--target_entropy")){
            // run single shot with target MOE to beat
            minimizer->runMinimization(subparser->get<double>("--target_entropy"));
        } else {
            // run single shot
            minimizer->runMinimization();
        }


        // save the state if selected

        if (subparser->is_used("-S")){
            // check if the output file is specified
            if (subparser->is_used("--output")){
                // log that custom path was detected
                message_handler->message("Custom path detected. Saving state to " + subparser->get<std::string>("--output"));
                minimizer->saveVector(subparser->get<std::string>("--output"));
            } else {
                minimizer->saveVector();
            }
        }
        delete minimizer;
    }

    if (parser->is_subcommand_used("multishot")){
        // get the selected parser
        argparse::ArgumentParser* subparser;
        subparser = &parser->at<argparse::ArgumentParser>("multishot");

        // check printing and logging options and create logger or printer accordingly. Don't give file names or anything.
        if (subparser->get<bool>("-l")){
            message_handler->createLogger();
        } 
        if (!subparser->get<bool>("-s")){
            message_handler->createPrinter();
        }

        // now also print to message handler
        // first print the full command line
        std::string full_command = "Command called: ";
        for (int i = 0; i < argc; i++){
            full_command += argv[i];
            full_command += " ";
        }
        message_handler->message(full_command);
        // Now explicitly print the options
        message_handler->message("Parsed Kraus operators: " + subparser->get<std::string>("-k"));
        // print logging and printing options,
        message_handler->message("Logging is: " + std::to_string(subparser->get<bool>("-l")));
        message_handler->message("Printing is: " + std::to_string(subparser->get<bool>("-s") ));

        // Get GPU number to use
        int gpu_number = subparser->get<int>("--gpu");
        // Set the device to use
        cudaSetDevice(gpu_number);
        // log the GPU number
        message_handler->message("Using GPU number: " + std::to_string(gpu_number));

        // Allocate memory for kraus operators on device
        cuDoubleComplex *d_kraus_operators;

        VectorSerializer serializer = VectorSerializer();
        { // Scoped so that memory is deallocated once we leave this section - deserialized data is heavy!
        // Get the kraus operators from file
        DeserializedData deserialized_data = serializer.deserialize(subparser->get<std::string>("-k"));
        // Get N and d from metadata
        N = deserialized_data.N;
        d = deserialized_data.d;

        // Allocate memory
        cudaError_t errmalloc = cudaMalloc(&d_kraus_operators, N*N*d * sizeof(std::complex<double>));
        if (errmalloc != cudaSuccess) {
            message_handler->message("Error allocating memory for Kraus operators: " + std::string(cudaGetErrorString(errmalloc)));
            return 1;
        }
        message_handler->message("Successfully allocated " + std::to_string(d*N*N * sizeof(std::complex<double>)) + " bytes or " + std::to_string(d*N*N * sizeof(std::complex<double>)/1024.0/1024/1024) + " GiB for Kraus operators on device.");

        // Copy the kraus operators to GPU
        cudaError_t errcopy = cudaMemcpy(d_kraus_operators, deserialized_data.vectorData.data(), deserialized_data.vectorData.size() * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
        if (errcopy != cudaSuccess) {
            message_handler->message("Error copying Kraus operators to device: " + std::string(cudaGetErrorString(errcopy)));
            return 1;
        }
        message_handler->message("Successfully copied " + std::to_string(deserialized_data.vectorData.size() * sizeof(std::complex<double>)) + " bytes or " + std::to_string(deserialized_data.vectorData.size() * sizeof(std::complex<double>)/1024.0/1024/1024) + " GiB of Kraus operators to device.");
        // Print exit message
        message_handler->message("Kraus operators loaded from " + subparser->get<std::string>("-k") + ".");
        // Log the N and d
        message_handler->message("N: " + std::to_string(N));
        message_handler->message("d: " + std::to_string(d));
        }

        // initialize configuration
        EntropyConfig config = EntropyConfig();
        // if present, set the number of iterations
        if (subparser->is_used("-i")){
            config.setMaxIterations(subparser->get<int>("-i"));
        }
        // if present, set the number of minimization attempts
        if (subparser->is_used("-a")){
            config.setMinimizationAttempts(subparser->get<int>("-a"));
        }
        // set logging and printing
        config.setLogging(subparser->get<bool>("-l"));
        config.setPrinting(!subparser->get<bool>("-s"));

        // Parse strategy preference
        CudaMinimizerStrategy strategy = parseStrategyString(subparser->get<std::string>("--strategy"));
        message_handler->message("Using strategy: " + subparser->get<std::string>("--strategy"));

        // finally, create a minimizer
        EntropyMinimizer* minimizer = new EntropyMinimizer(d_kraus_operators, d, N, N, &config, strategy);


        signal(SIGTERM, minimizer->signal_handler);
        
        //log with message handler
        message_handler->message("Subcommand multishot was used");

        // run multishot
        minimizer->findMOE();
        // save the state if selected
        if (subparser->is_used("-S")){
            minimizer->saveVector();
        }

        delete minimizer;
        // Free the device memory
        cudaError_t errfree = cudaFree(d_kraus_operators);
        if (errfree != cudaSuccess) {
            message_handler->message("Error freeing memory for Kraus operators: " + std::string(cudaGetErrorString(errfree)));
            return 1;
        }
        message_handler->message("Successfully freed memory for Kraus operators.");
    }

    // Option 4: vector was called
    if (parser->is_subcommand_used("vector")){
        // get the selected parser
        argparse::ArgumentParser* subparser;
        subparser = &parser->at<argparse::ArgumentParser>("vector");

        // check printing and logging options and create logger or printer accordingly. Don't give file names or anything.
        if (subparser->get<bool>("-l")){
            message_handler->createLogger();
        } 
        if (!subparser->get<bool>("-s")){
            message_handler->createPrinter();
        }

        // now also print to message handler
        // first print the full command line
        std::string full_command = "Command called: ";
        for (int i = 0; i < argc; i++){
            full_command += argv[i];
            full_command += " ";
        }
        message_handler->message(full_command);
        // Now explicitly print the options
        message_handler->message("Parsed option N: " + std::to_string(subparser->get<int>("-N")));
        message_handler->message("Parsed output: " + subparser->get<std::string>("--output"));
        // print logging and printing options,
        message_handler->message("Logging is: " + std::to_string(subparser->get<bool>("-l")));
        message_handler->message("Printing is: " + std::to_string(subparser->get<bool>("-s") ));

        // Get GPU number to use
        int gpu_number = subparser->get<int>("--gpu");
        // Set the device to use
        cudaSetDevice(gpu_number);
        // log the GPU number
        message_handler->message("Using GPU number: " + std::to_string(gpu_number));

        // save into N, d and output
        N = subparser->get<int>("-N");
        std::string output = subparser->get<std::string>("--output");

        // check that the output directory exists
        //first get the directory
        std::string output_directory = output.substr(0, output.find_last_of("/"));
        // check if the directory exists
        if (!std::filesystem::exists(output_directory)){
            // throw an error
            message_handler->message("Output directory does not exist. Please create it first.");
            return 1;
        }

        // generate random vector
        std::vector<std::complex<double> >* random_vector = generateUniformRandomVector(N);

        // save the vector
        VectorSerializer serializer = VectorSerializer();
        serializer.serialize("vector", output, *random_vector, "Random vector for the minimization algorithm", 1, N);
        // print exit message
        message_handler->message("Random vector saved to " + output + ".");
        delete random_vector;
    }
    // Delete the parser
    delete parser;
    delete message_handler;
    return 0;


}

