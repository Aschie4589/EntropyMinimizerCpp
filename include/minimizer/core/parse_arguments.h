#ifndef PARSE_ARGUMENTS_H
#define PARSE_ARGUMENTS_H

#include <argparse/argparse.hpp>
#include <minimizer/core/cuda_minimizer.h>
#include <string>

// Define a custom function to parse arguments
argparse::ArgumentParser* parse_arguments(int argc, char** argv);

// Helper function to parse strategy string to enum
CudaMinimizerStrategy parseStrategyString(const std::string& strategy_str);

#endif