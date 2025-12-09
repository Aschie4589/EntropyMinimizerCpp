#ifndef TARGET_ENTROPY_CONDITION_H
#define TARGET_ENTROPY_CONDITION_H

#include "minimizer/stopping/condition.h"
#include <cmath>

namespace entropy {

/**
 * @brief Stops when entropy reaches or goes below a target value
 * 
 * Used in runMinimization(target_entropy) to stop optimization
 * once a specific entropy threshold is reached.
 */
class TargetEntropyCondition : public Condition {
public:
    /**
     * @brief Construct target entropy condition
     * 
     * @param target_entropy Target entropy value to reach
     * @param tolerance Small tolerance for floating-point comparison (default: 1e-12)
     */
    explicit TargetEntropyCondition(double target_entropy, 
                                    double tolerance = 1e-12);

    StopReason check(size_t iteration, 
                    double current_entropy, 
                    double previous_entropy) override;

    void reset() override;

    std::string description() const override;

private:
    double target_entropy_;
    double tolerance_;
};

} // namespace entropy

#endif // TARGET_ENTROPY_CONDITION_H
