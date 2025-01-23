#ifndef PARSE_ARGUMENTS_H
#define PARSE_ARGUMENTS_H

#include <argparse/argparse.hpp>

// Define a custom function to parse arguments
argparse::ArgumentParser* parse_arguments(int argc, char** argv);

#endif