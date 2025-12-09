#ifndef ENTROPY_MANAGER_H
#define ENTROPY_MANAGER_H

#include <utility>
#include <cmath>
#include <stdexcept>

namespace entropy {

/**
 * @brief Manages entropy computations and transformations
 * 
 * Handles the mathematical relationship between epsilon-perturbed entropy
 * (computed from depolarized state) and the estimated true entropy.
 * 
 * Mathematical Background:
 * - Perturbed state: ρ_ε = (1-ε)ρ + ε·I/d
 *   where ρ is the true state, I is identity, d is system dimension
 * 
 * - Inequality: S(ρ) ≤ S(ρ_ε) ≤ S(ρ) + ε·log(d) + H_bin(ε)
 *   where S denotes von Neumann entropy
 * 
 * - Binary entropy: H_bin(ε) = -ε·log(ε) - (1-ε)·log(1-ε)
 * 
 * Estimated entropy:
 *   S_est(ρ) = [S(ρ_ε) - ε·log(d)] / (1-ε) - H_bin(ε) / [2(1-ε)]
 * 
 * Maximum error:
 *   error = H_bin(ε) / [2(1-ε)]
 * 
 * This class is lightweight and stateless except for cached values.
 */
class EntropyManager {
public:
    /**
     * @brief Construct entropy manager
     * 
     * @param epsilon Depolarization parameter (0 < ε < 1)
     * @param system_dimension Dimension d of the Hilbert space
     * 
     * @throws std::invalid_argument if epsilon not in (0,1) or dimension < 1
     * 
     * Precomputes constants:
     * - Binary entropy H_bin(ε)
     * - Epsilon correction term ε·log(d)
     * - Error estimate H_bin(ε) / [2(1-ε)]
     */
    EntropyManager(double epsilon, int system_dimension);

    /**
     * @brief Update with new epsilon entropy value
     * 
     * @param epsilon_entropy Perturbed entropy S(ρ_ε) from computation
     * 
     * Triggers recomputation of:
     * - estimated_entropy
     * - lower_bound = estimated_entropy - estimated_error
     * - upper_bound = estimated_entropy + estimated_error
     */
    void updateEpsilonEntropy(double epsilon_entropy);

    /**
     * @brief Get raw epsilon entropy (perturbed state)
     * @return S(ρ_ε) - last value set via updateEpsilonEntropy()
     */
    double getEpsilonEntropy() const;

    /**
     * @brief Get estimated true entropy (corrected)
     * @return Estimated S(ρ) after applying corrections
     */
    double getEstimatedEntropy() const;

    /**
     * @brief Get maximum estimation error
     * @return H_bin(ε) / [2(1-ε)]
     */
    double getEstimatedError() const;

    /**
     * @brief Get entropy bounds from estimation error
     * @return (lower_bound, upper_bound) where:
     *         lower_bound = estimated_entropy - estimated_error
     *         upper_bound = estimated_entropy + estimated_error
     */
    std::pair<double, double> getEntropyBounds() const;

    /**
     * @brief Get binary entropy H_bin(ε)
     * @return -ε·log(ε) - (1-ε)·log(1-ε)
     */
    double getBinaryEntropy() const;

    /**
     * @brief Get epsilon correction term
     * @return ε·log(d) where d is system dimension
     */
    double getEpsilonCorrection() const;

    /**
     * @brief Get depolarization parameter
     * @return ε value used in construction
     */
    double getEpsilon() const;

    /**
     * @brief Get system dimension
     * @return d (Hilbert space dimension)
     */
    int getSystemDimension() const;

    /**
     * @brief Reset entropy state (keeps configuration)
     * 
     * Clears epsilon_entropy, estimated_entropy, bounds.
     * Does NOT change epsilon or dimension (those are immutable).
     */
    void reset();

private:
    // Configuration (immutable after construction)
    double epsilon_;
    int system_dimension_;
    
    // Precomputed constants (computed once in constructor)
    double binary_entropy_;        // H_bin(ε)
    double epsilon_correction_;    // ε·log(d)
    double estimated_error_;       // error - H_bin(ε) / [2(1-ε)]
    double one_minus_epsilon_;     // 1 - ε (cached for efficiency)
    
    // State (updated via updateEpsilonEntropy)
    double epsilon_entropy_;       // S(ρ_ε) - raw perturbed entropy
    double estimated_entropy_;     // S_est(ρ) - estimated true entropy
    double lower_bound_;          // S_est(ρ) - error
    double upper_bound_;          // S_est(ρ) + error
    
    // Flag to track if entropy has been set
    bool entropy_initialized_;
    
    // Helper methods
    void computeEstimation();
    void computeBounds();
    
    // Static helper for binary entropy calculation
    static double computeBinaryEntropy(double epsilon);
};

} // namespace entropy

#endif // ENTROPY_MANAGER_H
