// My headers
#include "common_includes.h"
#include "config.h"

#include "matrix_operations.h"

#include "minimizer.h"
#include "entropy_minimizer.h"
#include "vector_serializer.h"
#include "entropy_estimator.h"

#include "message_handler.h"
#include "logger.h"

#include "generate_haar_unitary.h"
#include "generate_random_vector.h"

#include "uuid.h"

#include "parse_arguments.h"
#include "argparse/argparse.hpp"



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
            for (int m = 0; m < d; m++){
                message_handler->message("Generating Haar random unitary " + std::to_string(m+1) + " of " + std::to_string(d) + "...");
                // append a new unitary at position i of the kraus operator.
                std::vector<std::complex<double> >* new_haar_unitary = generateHaarRandomUnitary(N);
                for (int i = 0; i < N*N ;i++){
                    kraus_operators->at(m*N*N+i) = new_haar_unitary->at(i)/std::sqrt(double(d));
                }
                message_handler->message("Done!");
                delete new_haar_unitary;
            }

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
        // Also print logging and printing options,
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

