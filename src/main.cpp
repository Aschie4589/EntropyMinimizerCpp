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

    // Step 0: Parse command line arguments. Parser is created with new, so it must be deleted.
    argparse::ArgumentParser* parser = parse_arguments(argc, argv);

   if (parser->is_subcommand_used("kraus")){
    /// print
    std::cout << "Subcommand kraus was used" << std::endl;
    // was entropy also used as subparser of kraus?
    if (parser->at<argparse::ArgumentParser>("kraus").is_subcommand_used("entropy")){
        std::cout << "Subcommand entropy was used" << std::endl;
       }
    }
    // check if kraus was chosen


    // print N and d from command line
    std::cout << "N: " << parser->get<int>("-N") << std::endl;
    std::cout << "d: " << parser->get<int>("-d") << std::endl;
    // print the path to the kraus operators
    std::cout << "Kraus operators path: " << parser->get<std::string>("--kraus") << std::endl;
    // Assign the values of N and d to the global variables
    const int N = parser->get<int>("-N");
    const int d = parser->get<int>("-d");

    // Config setup
    EntropyConfig config = EntropyConfig();
    config.setLogging(parser->get<bool>("--logging"));
    config.setPrinting(!parser->get<bool>("--silent"));



    // get number of iterations
    int iterations = parser->get<int>("-i");
    std::cout << "Number of iterations: " << iterations << std::endl;

    // Delete the parser
    delete parser;


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

    // Step 2: initialize the minimizer that will take care of finding the MOE
    // Actual minimizer
    EntropyMinimizer minimizer = EntropyMinimizer(kraus_operators, d, N, N, &config); 

    // Step 3: get a random initial vector for the minimizer
    minimizer.minimizer->initializeRandomVector();


    // Step 4: run a minimization run
    minimizer.findMOE();

    // Step 5: save final state
    // minimizer.minimizer->saveState("final_state.dat");

    // Step 5: free memory
    delete kraus_operators;
    return 0;
}

