#ifndef CG_GENERATOR_H
#define CG_GENERATOR_H

#include "core/kraus/generator.h"
#include "helpers/sud_cg.h"
#include "helpers/message_handler.h"
// CG generator needs specific parameters
struct CGGeneratorConfig{
    int N; // N of SU(N)
    clebsch::weight in_rep; // Channel input irrep
    clebsch::weight out_rep; // Channel output irrep

    clebsch::weight anc_rep; // Ancilla irrep
    int anc_alpha; // If outer multiplicity of anc_rep within (in_rep)^*(x)out_rep is greater than one, which one to use (0, 1, ...)

    double tol = 1e-12; // Tolerance for discarding small clebsch gordan coefficients

    MessageHandler* message_handler;
};


// CG Generator class
class CGGenerator : public Generator<CGGeneratorConfig> {
public:
    CGGenerator(const CGGeneratorConfig& cfg) : Generator<CGGeneratorConfig>(cfg) {}

    int generate(std::vector<std::complex<double>>* kraus) override;
};

#endif