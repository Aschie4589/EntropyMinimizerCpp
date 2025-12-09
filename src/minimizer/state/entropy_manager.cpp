#include "minimizer/state/entropy_manager.h"
#include <limits>
#include <sstream>

namespace entropy {

EntropyManager::EntropyManager(double epsilon, int system_dimension)
    : epsilon_(epsilon),
      system_dimension_(system_dimension),
      binary_entropy_(0.0),
      epsilon_correction_(0.0),
      estimated_error_(0.0),
      one_minus_epsilon_(1.0 - epsilon),
      epsilon_entropy_(0.0),
      estimated_entropy_(0.0),
      lower_bound_(0.0),
      upper_bound_(0.0),
      entropy_initialized_(false) {
    
    // Validate parameters
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        std::ostringstream oss;
        oss << "Epsilon must be in (0, 1), got: " << epsilon;
        throw std::invalid_argument(oss.str());
    }
    
    if (system_dimension < 1) {
        std::ostringstream oss;
        oss << "System dimension must be positive, got: " << system_dimension;
        throw std::invalid_argument(oss.str());
    }
    
    // Precompute binary entropy H_bin(ε)
    binary_entropy_ = computeBinaryEntropy(epsilon);
    
    // Precompute epsilon correction: ε·log(d)
    epsilon_correction_ = epsilon * std::log(static_cast<double>(system_dimension));
    
    // Precompute estimated error: H_bin(ε) / [2(1-ε)]
    // Handle edge case where ε ≈ 1
    if (one_minus_epsilon_ > 1e-15) {
        estimated_error_ = binary_entropy_ / (2.0 * one_minus_epsilon_);
    } else {
        // If ε very close to 1, error estimate becomes large/undefined
        estimated_error_ = std::numeric_limits<double>::infinity();
    }
}

void EntropyManager::updateEpsilonEntropy(double epsilon_entropy) {
    epsilon_entropy_ = epsilon_entropy;
    entropy_initialized_ = true;
    
    computeEstimation();
    computeBounds();
}

double EntropyManager::getEpsilonEntropy() const {
    return epsilon_entropy_;
}

double EntropyManager::getEstimatedEntropy() const {
    return estimated_entropy_;
}

double EntropyManager::getEstimatedError() const {
    return estimated_error_;
}

std::pair<double, double> EntropyManager::getEntropyBounds() const {
    return {lower_bound_, upper_bound_};
}

double EntropyManager::getBinaryEntropy() const {
    return binary_entropy_;
}

double EntropyManager::getEpsilonCorrection() const {
    return epsilon_correction_;
}

double EntropyManager::getEpsilon() const {
    return epsilon_;
}

int EntropyManager::getSystemDimension() const {
    return system_dimension_;
}

void EntropyManager::reset() {
    epsilon_entropy_ = 0.0;
    estimated_entropy_ = 0.0;
    lower_bound_ = 0.0;
    upper_bound_ = 0.0;
    entropy_initialized_ = false;
}

void EntropyManager::computeEstimation() {
    // Formula: S_est(ρ) = [S(ρ_ε) - ε·log(d)] / (1-ε) - H_bin(ε) / [2(1-ε)]
    
    if (one_minus_epsilon_ > 1e-15) {
        // Normal case: ε not too close to 1
        double corrected_entropy = (epsilon_entropy_ - epsilon_correction_) / one_minus_epsilon_;
        estimated_entropy_ = corrected_entropy - estimated_error_;
    } else {
        // Edge case: ε ≈ 1
        // Estimation formula breaks down, fall back to epsilon_entropy
        estimated_entropy_ = epsilon_entropy_;
    }
}

void EntropyManager::computeBounds() {
    // Bounds represent uncertainty interval: estimated ± error
    lower_bound_ = estimated_entropy_ - estimated_error_;
    upper_bound_ = estimated_entropy_ + estimated_error_;
}

double EntropyManager::computeBinaryEntropy(double epsilon) {
    // H_bin(ε) = -ε·log(ε) - (1-ε)·log(1-ε)
    
    // Handle edge cases to avoid log(0)
    if (epsilon <= 1e-15) {
        // ε ≈ 0: H_bin(0) = 0
        return 0.0;
    }
    
    if (epsilon >= 1.0 - 1e-15) {
        // ε ≈ 1: H_bin(1) = 0
        return 0.0;
    }
    
    // Normal case
    double term1 = -epsilon * std::log(epsilon);
    double term2 = -(1.0 - epsilon) * std::log(1.0 - epsilon);
    
    return term1 + term2;
}

} // namespace entropy
