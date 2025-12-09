#ifndef CONDITION_H
#define CONDITION_H

#include "minimizer/stopping/stop_reason.h"
#include <string>

namespace entropy {

/**
 * @brief Abstract base class for stopping conditions
 * 
 * Each concrete condition implements a single stopping criterion.
 * Conditions are checked by ConditionsChecker on each iteration.
 * 
 * Design principles:
 * - Single responsibility: Each condition checks one thing
 * - Strategy pattern: Interchangeable implementations
 * - Testability: Can be tested in isolation without GPU
 */
class Condition {
public:
    virtual ~Condition() = default;

    /**
     * @brief Check if stopping condition is satisfied
     * 
     * @param iteration Current iteration number (0-based)
     * @param current_entropy Current entropy value
     * @param previous_entropy Previous entropy value (iteration - 1)
     * 
     * @return StopReason::CONTINUE if condition not satisfied, 
     *         otherwise specific reason for stopping
     * 
     * @note Called every iteration, so must be efficient (<1 µs)
     * @note Conditions should be stateful (track history internally)
     */
    virtual StopReason check(size_t iteration, 
                            double current_entropy, 
                            double previous_entropy) = 0;

    /**
     * @brief Reset condition state for a new minimization run
     * 
     * Called at the start of each run to clear any internal state.
     * This allows condition objects to be reused across multiple runs.
     */
    virtual void reset() = 0;

    /**
     * @brief Get human-readable description of condition
     * @return Description string for logging/debugging
     */
    virtual std::string description() const = 0;
};

} // namespace entropy

#endif // CONDITION_H
