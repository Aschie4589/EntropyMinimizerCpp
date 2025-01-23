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
    if (parser->is_subcommand_used("singleshot") || parser->is_subcommand_used("multishot")){
        // get the selected parser
        argparse::ArgumentParser* subparser;
        if (parser->is_subcommand_used("singleshot")){
            subparser = &parser->at<argparse::ArgumentParser>("singleshot");
        } else {
            subparser = &parser->at<argparse::ArgumentParser>("multishot");
        }

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
        if (parser->is_subcommand_used("multishot") && subparser->is_used("-a")){
            config.setMinimizationAttempts(subparser->get<int>("-a"));
        }
        // set logging and printing
        config.setLogging(subparser->get<bool>("-l"));
        config.setPrinting(!subparser->get<bool>("-s"));

        // finally, create a minimizer
        EntropyMinimizer* minimizer = new EntropyMinimizer(kraus_operators, d, N, N, &config);
         // Option 2: singleshot was called
        if (parser->is_subcommand_used("singleshot")){
            //log with message handler
            message_handler->message("Subcommand singleshot was used");
            // run single shot
            minimizer->initializeRun();
            minimizer->runMinimization();
        }

        // Option 3: multishot was called
        if (parser->is_subcommand_used("multishot")){
            //log with message handler
            message_handler->message("Subcommand multishot was used");

            // run multishot
            minimizer->findMOE();
        }
        delete minimizer;
    }

    // Delete the parser
    delete parser;
    delete message_handler;
    return 0;


    // Step 1: Generate the Kraus operators of the random unitary channel
    std::vector<std::complex<double> >* kraus_operators = new std::vector<std::complex<double> >(d*N*N); // kraus_operators is the pointer.
    for (int m = 0; m < d; m++){
        std::cout << "Generating Haar random unitary " << m+1 << " of " << d << "... ";
        // append a new unitary at position i of the kraus operator.
        std::vector<std::complex<double> >* new_haar_unitary = generateHaarRandomUnitary(N);
        for (int i = 0; i < N*N ;i++){
            kraus_operators->at(m*N*N+i) = new_haar_unitary->at(i)/std::sqrt(double(d));
        }
        std::cout << "Done!" << std::endl;
        delete new_haar_unitary;
    }

}

