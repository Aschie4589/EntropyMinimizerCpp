#ifndef MAX_ITERATIONS_CONDITION_H
#define MAX_ITERATIONS_CONDITION_H

#include "minimizer/stopping/condition.h"

namespace entropy {

/**
 * @brief Stops minimization when maximum iteration count is reached
 * 
 * Simple condition that tracks iteration count and stops when
 * the configured maximum is reached.
 */
class MaxIterationsCondition : public Condition {
public:
    /**
     * @brief Construct condition with iteration limit
     * @param max_iterations Maximum number of iterations allowed
     */
    explicit MaxIterationsCondition(size_t max_iterations);

    StopReason check(size_t iteration, 
                    double current_entropy, 
                    double previous_entropy) override;

    void reset() override;

    std::string description() const override;

private:
    size_t max_iterations_;
};

} // namespace entropy

#endif // MAX_ITERATIONS_CONDITION_H
