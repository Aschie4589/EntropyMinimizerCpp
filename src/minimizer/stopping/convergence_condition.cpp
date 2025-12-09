#include "minimizer/stopping/convergence_condition.h"

namespace entropy {

ConvergenceCondition::ConvergenceCondition(size_t window_size, double tolerance)
    : window_size_(window_size),
      tolerance_(tolerance),
      entropy_history_(std::make_unique<CircularBuffer<double>>(window_size)) {}

StopReason ConvergenceCondition::check(size_t iteration, 
                                      double current_entropy, 
                                      double previous_entropy) {
    // Add current entropy to history
    entropy_history_->push(current_entropy);
    
    // Need full window to check convergence
    if (!entropy_history_->full()) {
        return StopReason::CONTINUE;
    }
    
    // Check if we have converged:
    // Average improvement = (oldest - newest) / window_size
    // This should be < tolerance
    double oldest = entropy_history_->oldest();
    double newest = entropy_history_->newest();
    double total_improvement = oldest - newest;
    double avg_improvement = total_improvement / static_cast<double>(window_size_);
    
    if (avg_improvement < tolerance_) {
        return StopReason::CONVERGED;
    }
    
    return StopReason::CONTINUE;
}

void ConvergenceCondition::reset() {
    entropy_history_ = std::make_unique<CircularBuffer<double>>(window_size_);
}

std::string ConvergenceCondition::description() const {
    return "Convergence(window=" + std::to_string(window_size_) + 
           ", tol=" + std::to_string(tolerance_) + ")";
}

} // namespace entropy
