#include "minimizer/stopping/max_iterations_condition.h"

namespace entropy {

MaxIterationsCondition::MaxIterationsCondition(size_t max_iterations)
    : max_iterations_(max_iterations) {}

StopReason MaxIterationsCondition::check(size_t iteration, 
                                        double current_entropy, 
                                        double previous_entropy) {
    if (iteration >= max_iterations_) {
        return StopReason::MAX_ITERATIONS;
    }
    return StopReason::CONTINUE;
}

void MaxIterationsCondition::reset() {
    // No internal state to reset
}

std::string MaxIterationsCondition::description() const {
    return "MaxIterations(limit=" + std::to_string(max_iterations_) + ")";
}

} // namespace entropy
