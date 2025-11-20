#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#include <vector>
#include <complex>

#include "channel/generator/generator.h"
#include "utilities/messaging/message_handler.h"

// Configuration struct - pure data, no dependencies
struct RandomGeneratorConfig {
    int kraus_number; // Number of Kraus operators to generate
    int kraus_in_dimension; // Input dimension of each Kraus operator
    int kraus_out_dimension; // Output dimension of each Kraus operator
};

// Random Generator class
class RandomGenerator : public Generator<RandomGeneratorConfig> {
private:
    MessageHandler& msg_handler_; // Non-owning reference to message handler

public:
    // Constructor takes config and dependency separately
    RandomGenerator(const RandomGeneratorConfig& cfg, MessageHandler& msg_handler)
        : Generator<RandomGeneratorConfig>(cfg), msg_handler_(msg_handler) {}

    int generate(std::vector<std::complex<double>>* kraus) override;
};

#endif