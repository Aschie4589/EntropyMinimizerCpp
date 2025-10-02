#include "libs/argparse/argparse.hpp"
#include "helpers/parse_arguments.h"
#include <stdexcept>
#include <algorithm>


CudaMinimizerStrategy parseStrategyString(const std::string& strategy_str) {
    std::string str = strategy_str;
    // Convert to lowercase for case-insensitive comparison
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    
    if (str == "auto" || str == "auto_detect") {
        return CudaMinimizerStrategy::AUTO_DETECT;
    } else if (str == "high" || str == "high_memory") {
        return CudaMinimizerStrategy::HIGH_MEMORY;
    } else if (str == "low" || str == "low_memory") {
        return CudaMinimizerStrategy::LOW_MEMORY;
    } else if (str == "low_prob" || str == "low_memory_prob" || str == "low_memory_probabilistic") {
        return CudaMinimizerStrategy::LOW_MEMORY_PROBABILISTIC;
    } else {
        throw std::invalid_argument("Invalid strategy: " + strategy_str + ". Valid options are: auto, high, low, low_prob");
    }
}


argparse::ArgumentParser* parse_arguments(int argc, char** argv){
    // Input command line arguments. See TODO.
    argparse::ArgumentParser* parser = new argparse::ArgumentParser("moe", "0.1", argparse::default_arguments::help);
    parser->add_description("Entropy minimization tools for quantum channels.");
    // Use subparsers for different commands

    /*
            SUBPARSER 1: Generate Kraus operators
    */

    // add a new subparser called kraus
    argparse::ArgumentParser* kraus_parser = new argparse::ArgumentParser("kraus", "0.1", argparse::default_arguments::help);
    kraus_parser->add_description("Generate Kraus operators for a channel using different methods");
    parser->add_subparser(*kraus_parser);

    // SUBSUBPARSER 1: haar unitary kraus operators
    argparse::ArgumentParser* haar_parser = new argparse::ArgumentParser("haar", "0.1", argparse::default_arguments::help);
    haar_parser->add_description("Generate Kraus operators using Haar random unitaries");
    kraus_parser->add_subparser(*haar_parser);
    // options are N and d for the Haar random unitaries
    haar_parser->add_argument("-N")
    .help("dimension of the Hilbert space")
    .required()
    .scan<'i', int>();
    haar_parser->add_argument("-d")
    .help("number of Kraus operators")
    .required()
    .scan<'i', int>();
    // also specify path to save the kraus operators
    haar_parser->add_argument("--output", "-o")
    .help("path to save the Kraus operators")
    .required();
    // logging?
    haar_parser->add_argument("--logging", "-l")
    .help("enable logging")
    .default_value(false)
    .implicit_value(true);
    // printing?
    haar_parser->add_argument("--silent", "-s")
    .help("disable printing")
    .default_value(false)
    .implicit_value(true);
    // GPU number to use
    haar_parser->add_argument("--gpu", "-g")
    .help("If multiple GPUs present, GPU number to use for the computation")
    .default_value(0)
    .scan<'i', int>();

    /*
            SUBPARSER 2: Single-shot entropy minimization
    */

    // add a new subparser called singleshot
    argparse::ArgumentParser* single_shot_parser = new argparse::ArgumentParser("singleshot", "0.1", argparse::default_arguments::help);
    single_shot_parser->add_description("SINGLE-SHOT ENTROPY MINIMIZATION.\n\nWith a randomly initialized vector, run the algorithm until convergence. If vector is specified, perform minimization with that vector.");
    parser->add_subparser(*single_shot_parser);  
    single_shot_parser->add_group("Required arguments");
    // option to load from file
    single_shot_parser->add_argument("-k", "--kraus")
    .help("path to stored Kraus operators")
    .required()
    .metavar("FILE");

    single_shot_parser->add_group("Other arguments");
    // save flag for final vector
    single_shot_parser->add_argument("--save", "-S")
    .help("save the final vector")
    .default_value(false)
    .implicit_value(true);

    // initial vector
    single_shot_parser->add_argument("--vector", "-v")
    .help("path to stored starting vector")
    .default_value(std::string(""))
    .metavar("FILE");

    // output file
    single_shot_parser->add_argument("--output", "-o")
    .help("path to save the final vector")
    .default_value(std::string(""))
    .metavar("FILE");

    // save checkpoint
    single_shot_parser->add_argument("--checkpoint", "-c")
    .help("save the current vector every checkpoint_interval iterations")
    .default_value(false)
    .implicit_value(true);

    // checkpoint interval
    single_shot_parser->add_argument("--checkpoint_interval", "-ci")
    .help("set the interval for saving the checkpoint file")
    .scan<'i', int>()
    .metavar("INT");

    // checkpoint file
    single_shot_parser->add_argument("--checkpoint_file", "-cf")
    .help("path to save the checkpoint file")
    .default_value(std::string(""))
    .metavar("FILE");



    // prediction flag
    single_shot_parser->add_argument("--predict", "-p")
    .help("predict the entropy and stop if it is not expected to beat the target")
    .default_value(false)
    .implicit_value(true);

    // target entropy to beat
    single_shot_parser->add_argument("--target_entropy", "-t")
    .help("target entropy to beat. This will make the minimizer predict the entropy and stop if this run is not expected to beat the target.")
    .scan<'g', double>()
    .metavar("FLOAT");

    // how many iterations to run the minimizer for
    single_shot_parser->add_argument("--iters", "-i")
    .help("max number of iterations")
    .scan<'i', int>()
    .metavar("INT");

    single_shot_parser->add_group("Printing arguments");
    // logging?
    single_shot_parser->add_argument("--logging", "-l")
    .help("enable logging")
    .default_value(false)
    .implicit_value(true);
    // printing?
    single_shot_parser->add_argument("--silent", "-s")
    .help("disable printing")
    .default_value(false)
    .implicit_value(true);

    // GPU number to use
    single_shot_parser->add_argument("--gpu", "-g")
    .help("If multiple GPUs present, GPU number to use for the computation")
    .default_value(0)
    .scan<'i', int>();

    // Memory strategy to use
    single_shot_parser->add_argument("--strategy")
    .help("Memory management strategy: auto (default), high, low, low_prob")
    .default_value(std::string("auto"))
    .metavar("STRATEGY");



    /*
            SUBPARSER 3: Multi-shot entropy minimization
    */

    // add a new subparser called multishot
    argparse::ArgumentParser* multi_shot_parser = new argparse::ArgumentParser("multishot", "0.1", argparse::default_arguments::help);
    multi_shot_parser->add_description("MULTI-SHOT ENTROPY MINIMIZATION.\n\nRun the algorithm multiple times with different starting vectors.");
    parser->add_subparser(*multi_shot_parser);

    multi_shot_parser->add_group("Required arguments");
    // add option to load kraus operators from file
    multi_shot_parser->add_argument("-k", "--kraus")
    .help("path to stored Kraus operators")
    .required()
    .metavar("FILE");
    multi_shot_parser->add_group("Other arguments");
    // save flag for final vector
    multi_shot_parser->add_argument("--save", "-S")
    .help("save the final vector")
    .default_value(false)
    .implicit_value(true);


    // how many iterations to run the minimizer for
    multi_shot_parser->add_argument("-i", "--iters")
    .help("number of iterations")
    .scan<'i', int>()
    .metavar("INT");
    // how many times to run the minimizer
    multi_shot_parser->add_argument("-a", "--atts")
    .help("number of minimization attempts")
    .scan<'i', int>()
    .metavar("INT");
    multi_shot_parser->add_group("Printing arguments");
    // logging?
    multi_shot_parser->add_argument("--logging", "-l")
    .help("enable logging")
    .default_value(false)
    .implicit_value(true);
    // printing?
    multi_shot_parser->add_argument("--silent", "-s")
    .help("disable printing")
    .default_value(false)
    .implicit_value(true);
    // GPU number to use
    multi_shot_parser->add_argument("--gpu", "-g")
    .help("If multiple GPUs present, GPU number to use for the computation")
    .default_value(0)
    .scan<'i', int>();

    // Memory strategy to use
    multi_shot_parser->add_argument("--strategy")
    .help("Memory management strategy: auto (default), high, low, low_prob")
    .default_value(std::string("auto"))
    .metavar("STRATEGY");

    /*
            SUBPARSER 4: Generate vector
    */

    // add a new subparser called vector_parser
    argparse::ArgumentParser* vector_parser = new argparse::ArgumentParser("vector", "0.1", argparse::default_arguments::help);
    vector_parser->add_description("Generate a random vector for the minimization algorithm.");
    parser->add_subparser(*vector_parser);

    vector_parser->add_group("Required arguments");
    // dimension of the vector
    vector_parser->add_argument("-N")
    .help("dimension of the Hilbert space")
    .required()
    .scan<'i', int>();

    // output file
    vector_parser->add_argument("--output", "-o")
    .help("path to save the final vector")
    .required()
    .metavar("FILE");

    vector_parser->add_group("Printing arguments");
    // logging?
    vector_parser->add_argument("--logging", "-l")
    .help("enable logging")
    .default_value(false)
    .implicit_value(true);
    // printing?
    vector_parser->add_argument("--silent", "-s")
    .help("disable printing")
    .default_value(false)
    .implicit_value(true);
    
    
    // GPU number to use
    vector_parser->add_argument("--gpu", "-g")
    .help("If multiple GPUs present, GPU number to use for the computation")
    .default_value(0)
    .scan<'i', int>();






    try {
    parser->parse_args(argc, argv);
    }
    catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
    }

    return parser;

}
