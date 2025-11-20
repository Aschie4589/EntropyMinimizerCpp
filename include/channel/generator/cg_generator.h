#ifndef CG_GENERATOR_H
#define CG_GENERATOR_H

#include <vector>
#include <complex>

#include "channel/generator/generator.h"
#include "sud_cg.h"
#include "utilities/messaging/message_handler.h"

// Configuration struct - pure data, no dependencies
struct CGGeneratorConfig {
    int N; // N of SU(N)
    clebsch::weight in_rep; // Channel input irrep
    clebsch::weight out_rep; // Channel output irrep
    clebsch::weight anc_rep; // Ancilla irrep
    int alpha; // If outer multiplicity of in_rep within (anc_rep)^*(x)out_rep is greater than one, which one to use (0, 1, ...)
    double tol = 1e-12; // Tolerance for discarding small clebsch gordan coefficients

    // Constructor - only configuration parameters
    CGGeneratorConfig(int n, const clebsch::weight& in, const clebsch::weight& out, 
                      const clebsch::weight& anc, int alpha = 0, double tolerance = 1e-12)
        : N(n), in_rep(in), out_rep(out), anc_rep(anc), alpha(alpha), tol(tolerance) {}
};

// CG Generator class
class CGGenerator : public Generator<CGGeneratorConfig> {
private:
    MessageHandler& msg_handler_; // Non-owning reference to message handler

public:
    // Constructor takes config and dependency separately
    CGGenerator(const CGGeneratorConfig& cfg, MessageHandler& msg_handler) 
        : Generator<CGGeneratorConfig>(cfg), msg_handler_(msg_handler) {}

    int generate(std::vector<std::complex<double>>* kraus) override;
};

#endif