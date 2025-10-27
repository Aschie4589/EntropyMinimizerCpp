#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#include <vector>
#include <complex>

#include "channel/generator/generator.h"
#include "utilities/messaging/message_handler.h"

// CG generator needs specific parameters
struct RandomGeneratorConfig{
    int kraus_number; // Number of Kraus operators to generate
    int kraus_in_dimension; // Input dimension of each Kraus operator
    int kraus_out_dimension; // Output dimension of each Kraus operator

    MessageHandler* message_handler; // Message handler for logging

};


// CG Generator class
class RandomGenerator : public Generator<RandomGeneratorConfig> {
public:
    RandomGenerator(const RandomGeneratorConfig& cfg) : Generator<RandomGeneratorConfig>(cfg) {}

    int generate(std::vector<std::complex<double>>* kraus) override;
};

#endif