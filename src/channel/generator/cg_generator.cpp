#include "channel/generator/cg_generator.h"


int CGGenerator::generate(std::vector<std::complex<double>>* kraus) {
    /*
    Returns a vector of Kraus operators, each represented as a flat vector in column-major order
    Updates the input pointer kraus in place! Requires that the pointer is not null.
    */

    // Compute the dual representation weight of the ancilla representation
    clebsch::weight dual_anc_rep(config.N);
    for (int i = 1; i <= config.N; ++i) {
        dual_anc_rep(i) = config.anc_rep(1) - config.anc_rep(config.N + 1 - i); // Remember 1 indexing
    }
    // Print the dual ancillary irrep
    config.message_handler->message("Dual ancilla irrep weight: ( " + std::to_string(dual_anc_rep(1)));
    for (int i = 2; i <= config.N; ++i) {
        config.message_handler->message(", " + std::to_string(dual_anc_rep(i)));
    }
    config.message_handler->message(" )");

    // Step 1 (following arxiv paper): create a clebsch::decomposition decomp to decompose dual_anc_rep x out_rep
    const clebsch::decomposition decomp(dual_anc_rep, config.out_rep);
    config.message_handler->message("Decomposition of dual_anc_rep x out_rep has " + std::to_string(decomp.size()) + " irreps.");
    config.message_handler->message("The requested input irrep has outer multiplicity " + std::to_string(decomp.multiplicity(config.in_rep)) + ".");
    config.message_handler->message("The dimension of the ancilla irrep is " + std::to_string(config.anc_rep.dimension()) + ".");
    config.message_handler->message("Will generate " + std::to_string(config.anc_rep.dimension()) + " Kraus operators, each of dimension " + std::to_string(config.out_rep.dimension()) + " x " + std::to_string(config.in_rep.dimension()) + ".");
    // Step 2: calculate all CG coefficients for all ancillary reps of the same type as anc_rep
    const clebsch::coefficients C(config.in_rep, dual_anc_rep, config.out_rep); // find in_rep insinde dual_anc_rep(x)out_rep
    config.message_handler->message("Computed all Clebsch-Gordan coefficients for the requested channel.");
    
    // Step 3: construct the Kraus operators
    for (int l = 0; l < config.anc_rep.dimension(); ++l) {
        // Loop over input basis
        for (int in_vec_index = 0; in_vec_index < config.in_rep.dimension(); ++in_vec_index) {
            // Now loop over output basis
            for (int out_vec_index = 0; out_vec_index < config.out_rep.dimension(); ++out_vec_index) {
                // The Kraus operator matrix element (out_vec_index, in_vec_index) is
                double coeff = C(l, out_vec_index, config.alpha, in_vec_index);
                // Set the matrix element in K_l. K_l has to be in col major format
                kraus->at(l * config.out_rep.dimension() * config.in_rep.dimension() + in_vec_index * config.out_rep.dimension() + out_vec_index) = std::complex<double>(coeff, 0.0);

            }


        }
        
    }
    config.message_handler->message("Generated " + std::to_string(config.anc_rep.dimension()) + " Kraus operators.");

    return 0;

}