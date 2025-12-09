#include "minimizer/stopping/target_entropy_condition.h"

namespace entropy {

TargetEntropyCondition::TargetEntropyCondition(double target_entropy, 
                                               double tolerance)
    : target_entropy_(target_entropy),
      tolerance_(tolerance) {}

StopReason TargetEntropyCondition::check(size_t iteration, 
                                        double current_entropy, 
                                        double previous_entropy) {
    // Check if we've reached or gone below target
    // Use tolerance for floating-point comparison
    if (current_entropy <= target_entropy_ + tolerance_) {
        return StopReason::TARGET_REACHED;
    }
    
    return StopReason::CONTINUE;
}

void TargetEntropyCondition::reset() {
    // No internal state to reset
}

std::string TargetEntropyCondition::description() const {
    return "TargetEntropy(target=" + std::to_string(target_entropy_) + ")";
}

} // namespace entropy
