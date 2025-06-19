// My headers
#include "common_includes.h"
#include "config/config.h"

#include "core/matrix_operations.h"

#include "core/minimizer.h"
#include "core/cuda_minimizer.h"
#include "core/entropy_minimizer.h"
#include "helpers/vector_serializer.h"
#include "core/entropy_estimator.h"

#include "helpers/message_handler.h"
#include "helpers/logger.h"

#include "core/generate_haar_unitary.h"
#include "core/generate_random_vector.h"

#include "helpers/uuid.h"

#include "helpers/parse_arguments.h"
#include "libs/argparse/argparse.hpp"


#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>



int main(int argc, char** argv){


    int N = 1024;
    int d = 32;

    // Set the device to use
    cudaSetDevice(0); // Use device 0

    // Create kraus operators
    cuDoubleComplex* kraus_operators;
    cudaError_t errmalloc = cudaMalloc(&kraus_operators, d * N * N * sizeof(cuDoubleComplex));
    if (errmalloc != cudaSuccess) {
        std::cerr << "Error allocating memory for Kraus operators: " << cudaGetErrorString(errmalloc) << std::endl;
        return 1;
    }
    std::cout << "Successfully allocated " << d * N * N * sizeof(cuDoubleComplex) << " bytes " << "or " << d * N * N * sizeof(cuDoubleComplex) /1024.0/1024/1024 <<" GB for Kraus operators." << std::endl;
    
    // Generate Haar random unitaries
    cudaError_t err = generateHaarRandomUnitaries(kraus_operators, N, d, 32);
    if (err != cudaSuccess) {
        std::cerr << "Error generating Haar random unitaries: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }



    // Create a minimizer
    CudaMinimizer* minimizer = new CudaMinimizer(kraus_operators, d, N, N, 1e-6);
    // Initialize vector
    cudaError_t errinit = minimizer->initializeRandomVector();
    if (errinit != cudaSuccess) {
        std::cerr << "Error initializing vector: " << cudaGetErrorString(errinit) << std::endl;
        return 1;
    }
    // Update projector
    cudaError_t errupdate = minimizer->updateProjector();
    if (errupdate != cudaSuccess) {
        std::cerr << "Error updating projector: " << cudaGetErrorString(errupdate) << std::endl;
        return 1;
    }

    // Next apply the channel

    cudaError_t errapply = minimizer->applyEpsilonChannel();
    if (errapply != cudaSuccess) {
        std::cerr << "Error applying channel: " << cudaGetErrorString(errapply) << std::endl;
        return 1;
    }

    // Retrieve and print the output matrix
    cuDoubleComplex* output_matrix = minimizer->getOutputState();
    // Copy to host memory for printing
    cuDoubleComplex* host_output_matrix = new cuDoubleComplex[N * N];
    cudaError_t errcopy_output = cudaMemcpy(host_output_matrix, output_matrix, N * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Calculate the trace
    cuDoubleComplex trace = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < N; i++) {
        trace.x += host_output_matrix[i * N + i].x; // Real part
        trace.y += host_output_matrix[i * N + i].y; // Imaginary part
    }
    std::cout << "Trace of the output matrix: " << trace.x << " + " << trace.y << "i" << std::endl;
   
    // Apply the dual channel
    cudaError_t errapply_dual = minimizer->applyEpsilonDualChannel();
    if (errapply_dual != cudaSuccess) {
        std::cerr << "Error applying dual channel: " << cudaGetErrorString(errapply_dual) << std::endl;
        return 1;
    }
    // Retrieve and print the output matrix
    cuDoubleComplex* output_matrix_dual = minimizer->getState();
    // Copy to host memory for printing
    cuDoubleComplex* host_output_matrix_dual = new cuDoubleComplex[N * N];
    cudaError_t errcopy_output_dual = cudaMemcpy(host_output_matrix_dual, output_matrix_dual, N * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (errcopy_output_dual != cudaSuccess) {
        std::cerr << "Error copying output matrix to host: " << cudaGetErrorString(errcopy_output_dual) << std::endl;
        return 1;
    }
    // Calculate the trace
    cuDoubleComplex trace_dual = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < N; i++) {
        trace_dual.x += host_output_matrix_dual[i * N + i].x; // Real part
        trace_dual.y += host_output_matrix_dual[i * N + i].y; // Imaginary part
    }
    std::cout << "Trace of the output matrix (dual): " << trace_dual.x << " + " << trace_dual.y << "i" << std::endl;

    return 0;
}






int main2(int argc, char** argv){

    // Get general purpose message handler
    MessageHandler* message_handler = new MessageHandler();
    // Step 0: Parse command line arguments. Parser is created with new, so it must be deleted.
    argparse::ArgumentParser* parser = parse_arguments(argc, argv);
    int N, d;
    
    cudaSetDevice(1); // Use device 1


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

            // check that the output directory exists
            //first get the directory
            std::string output_directory = output.substr(0, output.find_last_of("/"));
            // check if the directory exists
            if (!std::filesystem::exists(output_directory)){
                // throw an error
                message_handler->message("Output directory does not exist. Please create it first.");
                return 1;
            }


            // generate the Kraus operators
            std::vector<std::complex<double> >* kraus_operators = new std::vector<std::complex<double> >(d*N*N); // kraus_operators is the pointer.
            // Use the parallel GPU version of the Haar random unitary generator
            message_handler->message("Generating " + std::to_string(d) + " Haar random unitaries of size " + std::to_string(N) + "x" + std::to_string(N) + "...");
            // Generate the Haar random unitaries
            cuDoubleComplex* gpuUnitaries = nullptr;
            cudaMalloc(&gpuUnitaries, d * N * N * sizeof(cuDoubleComplex));
            cudaError_t err = generateHaarRandomUnitaries(gpuUnitaries, N, d, 1);
            if (err != cudaSuccess) {
                message_handler->message("Error generating Haar random unitaries: " + std::string(cudaGetErrorString(err)));
                return 1;
            }
            // Copy the generated unitaries to the kraus_operators vector
            cudaMemcpy(kraus_operators->data(), gpuUnitaries, d * N * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
            // Free the GPU memory
            cudaFree(gpuUnitaries);
            // Rescale the unitaries by 1/sqrt(d) to ensure they    are normalized
            for (int i = 0; i < d * N * N; i++)
            {
                kraus_operators->at(i) /= std::sqrt(double(d));
            }
            message_handler->message("Done generating Haar random unitaries!");
            /*
            for (int m = 0; m < d; m++){
                message_handler->message("Generating Haar random unitary " + std::to_string(m+1) + " of " + std::to_string(d) + "...");
                // append a new unitary at position i of the kraus operator.
                std::vector<std::complex<double> > new_haar_unitary = generateHaarRandomUnitary(N);
                for (int i = 0; i < N*N ;i++){
                    kraus_operators->at(m*N*N+i) = new_haar_unitary.at(i)/std::sqrt(double(d));
                }
                message_handler->message("Done!");
            }
                */
            

            // save the kraus operators
            VectorSerializer serializer = VectorSerializer();
            serializer.serialize("kraus", output, *kraus_operators, "Kraus operators for a random unitary channel", d, N);
            // print exit message
            message_handler->message("Kraus operators saved to " + output + ".");
            delete kraus_operators;
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

        // Get the kraus operators from file
        std::vector<std::complex<double> >* kraus_operators = new std::vector<std::complex<double> >();
        VectorSerializer serializer = VectorSerializer();
        DeserializedData deserialized_data = serializer.deserialize(subparser->get<std::string>("-k"));
        kraus_operators = &deserialized_data.vectorData;
        // Print exit message
        message_handler->message("Kraus operators loaded from " + subparser->get<std::string>("-k") + ".");
        // Get N and d from metadata
        N = deserialized_data.N;
        d = deserialized_data.d;
        // Log the N and d
        message_handler->message("N: " + std::to_string(N));
        message_handler->message("d: " + std::to_string(d));

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

        // finally, create a minimizer
        EntropyMinimizer* minimizer = new EntropyMinimizer(kraus_operators, d, N, N, &config);

        signal(SIGTERM, minimizer->signal_handler);

        // Initialize run
        // Check if we have a starting vector specified in the command line
        std::vector<std::complex<double> >* start_vector = new std::vector<std::complex<double> >();
        if (subparser->is_used("--vector")){
            // load the vector
            DeserializedData deserialized_vector = serializer.deserialize(subparser->get<std::string>("--vector"));
            start_vector = &deserialized_vector.vectorData;
            // log the vector loaded message
            message_handler->message("Starting vector loaded from " + subparser->get<std::string>("--vector") + ".");
            // check dimension
            if (deserialized_vector.N != N || deserialized_vector.d != 1){
                message_handler->message("Starting vector has wrong dimensions. Expected N = " + std::to_string(N) + ", d = 1. Got N = " + std::to_string(deserialized_vector.N) + ", d = " + std::to_string(deserialized_vector.d) + ".");
                return 1;
            }
            // initialize run
            minimizer->initializeRun(start_vector);
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

        // Try to load kraus
        std::vector<std::complex<double> >* kraus_operators = new std::vector<std::complex<double> >();
        VectorSerializer serializer = VectorSerializer();
        // get the kraus operators
        DeserializedData deserialized_data = serializer.deserialize(subparser->get<std::string>("-k"));
        kraus_operators = &deserialized_data.vectorData;
        // print exit message
        message_handler->message("Kraus operators loaded from " + subparser->get<std::string>("-k") + ".");
        // get N and d from metadata
        N = deserialized_data.N;
        d = deserialized_data.d;
        // log the N and d
        message_handler->message("N: " + std::to_string(N));
        message_handler->message("d: " + std::to_string(d));

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

        // finally, create a minimizer
        EntropyMinimizer* minimizer = new EntropyMinimizer(kraus_operators, d, N, N, &config);


        signal(SIGTERM, minimizer->signal_handler);
        
        //log with message handler
        message_handler->message("Subcommand multishot was used");

        // run multishot
        minimizer->findMOE();
        // save the state if selected
        if (subparser->is_used("-S")){
            minimizer->saveState();
        }

        delete minimizer;
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

