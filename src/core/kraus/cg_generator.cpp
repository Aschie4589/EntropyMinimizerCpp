#include "core/kraus/cg_generator.h"

int CGGenerator::generate(std::vector<std::complex<double>>* kraus) {
    /*
    Returns a vector of Kraus operators, each represented as a flat vector in column-major order
    Updates the input pointer kraus in place! Requires that the pointer is not null.
    */

    config.message_handler->message("Generating Kraus operators using Clebsch-Gordan coefficients...");
    // Compute the dual representation weight of the input representation
    clebsch::weight dual_in_rep(config.N);
    for (int i = 1; i <= config.N; ++i) {
        dual_in_rep(i) = config.in_rep(1) - config.in_rep(config.N + 1 - i); // Remember 1 indexing
    }

    // Step 1 (following arxiv paper): create a clebsch::decomposition decomp to decompose dual_in_rep x out_rep
    const clebsch::decomposition decomp(dual_in_rep, config.out_rep);
    config.message_handler->message("Decomposition of dual_in_rep x out_rep has " + std::to_string(decomp.size()) + " irreps.");
    config.message_handler->message("The requested ancilla irrep has outer multiplicity " + std::to_string(decomp.multiplicity(config.anc_rep)) + ".");
    // Step 2: calculate all CG coefficients for all ancillary reps of the same type as anc_rep
    const clebsch::coefficients C(config.anc_rep, dual_in_rep, config.out_rep);
    config.message_handler->message("Computed all Clebsch-Gordan coefficients for the requested ancilla irrep.");
    // Now the clebsch gordan coefficients are accessible via C(S_index, Sprime_index, alpha, Sdoubleprime_index).
    
    // Notice the following complication. The Kraus operators are labelled by vectors in the ancillary space (i.e. the irrep anc_rep).
    // Now for a fixed index l, the kraus operator K_l has matrix elements K_l(i,j) given by
    // K_l(i,j) = sqrt(d_out/d_in) * CG [vector dual to vector of index i in out_rep] x [vector of index j in in_rep] -> [vector of index l in anc_rep]
    // The complication is that the first index of C is in dual_in_rep, not in in_rep.
    // So we need to convert j to the corresponding index in dual_in_rep.
    // The GT patterns for dual_in_rep are obtained from those of in_rep by reversing the order of the rows, and then taking the complement to the maximum number of boxes (first el of first row in in_rep).

    // Now loop over input vector indices
    for (int l = 0; l < config.anc_rep.dimension(); ++l) {
        // Create Kraus operator K_l
        std::vector<std::complex<double>>* K_l = new std::vector<std::complex<double>>(config.out_rep.dimension() * config.in_rep.dimension(), 0.0);

        for (int in_vec_index = 0; in_vec_index < config.in_rep.dimension(); ++in_vec_index) {
            // Find the dual index. First, get the old pattern.
            clebsch::pattern p(config.in_rep, in_vec_index);
            // Initialize dual pattern to first indexed pattern in dual rep.
            clebsch::pattern p_dual(dual_in_rep);
            // Now set the dual pattern entries to the correct ones
            int max_boxes = p(1, 1);
            for (int l = 1; l <= config.N; ++l) {
                for (int k = 1; k <= l; ++k) {
                    p_dual(k, l) = max_boxes - p(config.N + 1 - l, k);
                }
            }
            // Find index of dual pattern
            int dual_in_vec_index = p_dual.index();

            // Now loop over output vector indices
            for (int out_vec_index = 0; out_vec_index < config.out_rep.dimension(); ++out_vec_index) {
                // The Kraus operator matrix element (out_vec_index, in_vec_index) is
                // sqrt(out_rep.dimension()/in_rep.dimension()) * C(dual_in_vec_index, out_vec_index, config.anc_alpha)
                double coeff = std::sqrt(static_cast<double>(config.out_rep.dimension()) / static_cast<double>(config.in_rep.dimension()))
                            * C(dual_in_vec_index, out_vec_index, config.anc_alpha, config.anc_rep.dimension() - 1);
                // If the coefficient is smaller than tolerance, set it to zero
                if (std::abs(coeff) < config.tol) {
                    coeff = 0.0;
                }

                // Set the matrix element in K_l. K_l has to be in col major format
                kraus->at(l * config.out_rep.dimension() * config.in_rep.dimension() + in_vec_index * config.out_rep.dimension() + out_vec_index) = std::complex<double>(coeff, 0.0);

            }


        }
        
    }
    config.message_handler->message("Generated " + std::to_string(config.anc_rep.dimension()) + " Kraus operators.");

    return 0;

}

