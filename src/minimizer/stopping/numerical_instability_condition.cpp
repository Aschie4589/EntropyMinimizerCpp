#include "minimizer/stopping/numerical_instability_condition.h"

namespace entropy {

NumericalInstabilityCondition::NumericalInstabilityCondition(size_t window_size)
    : window_size_(window_size),
      entropy_history_(std::make_unique<CircularBuffer<double>>(window_size)) {}

StopReason NumericalInstabilityCondition::check(size_t iteration, 
                                               double current_entropy, 
                                               double previous_entropy) {
    // Add current entropy to history
    entropy_history_->push(current_entropy);
    
    // Need at least 2 entries to check for negative improvement
    if (entropy_history_->size() < 2) {
        return StopReason::CONTINUE;
    }
    
    // Check all consecutive pairs in the buffer for negative improvements
    // Improvement = entropy[i-1] - entropy[i] should be >= 0
    // If < 0, entropy increased (numerical instability)
    for (size_t i = 1; i < entropy_history_->size(); ++i) {
        double older_entropy = entropy_history_->get(i - 1);
        double newer_entropy = entropy_history_->get(i);
        double improvement = older_entropy - newer_entropy;
        
        if (improvement < 0) {
            return StopReason::NUMERICAL_INSTABILITY;
        }
    }
    
    return StopReason::CONTINUE;
}

void NumericalInstabilityCondition::reset() {
    entropy_history_ = std::make_unique<CircularBuffer<double>>(window_size_);
}

std::string NumericalInstabilityCondition::description() const {
    return "NumericalInstability(window=" + std::to_string(window_size_) + ")";
}

} // namespace entropy
