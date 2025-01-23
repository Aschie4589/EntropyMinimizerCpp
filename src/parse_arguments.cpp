#include "argparse/argparse.hpp"
#include "parse_arguments.h"


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


    // SUBPARSER 2: Single-shot entropy minimization
    argparse::ArgumentParser* single_shot_parser = new argparse::ArgumentParser("singleshot", "0.1", argparse::default_arguments::help);
    single_shot_parser->add_description("Single-shot entropy minimization");
    parser->add_subparser(*single_shot_parser);  
    // option to load from file
    single_shot_parser->add_argument("-k", "--kraus")
    .help("path to stored Kraus operators")
    .required();
    // how many iterations to run the minimizer for
    single_shot_parser->add_argument("-i", "--iters")
    .help("number of iterations")
    .scan<'i', int>();
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





    // SUBPARSER 3: Multi-shot entropy minimization
    argparse::ArgumentParser* multi_shot_parser = new argparse::ArgumentParser("multishot", "0.1", argparse::default_arguments::help);
    multi_shot_parser->add_description("Multi-shot entropy minimization");
    parser->add_subparser(*multi_shot_parser);
    // add option to load kraus operators from file
    multi_shot_parser->add_argument("-k", "--kraus")
    .help("path to stored Kraus operators")
    .required();
    // how many iterations to run the minimizer for
    multi_shot_parser->add_argument("-i", "--iters")
    .help("number of iterations")
    .scan<'i', int>();
    // how many times to run the minimizer
    multi_shot_parser->add_argument("-a", "--atts")
    .help("number of minimization attempts")
    .scan<'i', int>();
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



    if (false){
    // UNUSED: add option to specify starting vector for minimizer
    parser->add_argument("--vector")
    .help("path to stored starting vector")
    .default_value(std::string(""));
    }

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
