#ifndef CONDITIONS_CHECKER_H
#define CONDITIONS_CHECKER_H

#include "minimizer/stopping/condition.h"
#include "minimizer/stopping/stop_reason.h"
#include <vector>
#include <memory>
#include <string>

namespace entropy {

/**
 * @brief Orchestrates multiple stopping conditions
 * 
 * Manages a collection of Condition objects and checks them in order.
 * Returns the first non-CONTINUE reason found (priority-based).
 * 
 * Typical usage:
 * 1. Create ConditionsChecker
 * 2. Add conditions in priority order (first added = highest priority)
 * 3. Call checkStoppingConditions() each iteration
 * 4. Reset conditions at start of each new run
 */
class ConditionsChecker {
public:
    /**
     * @brief Construct empty conditions checker
     */
    ConditionsChecker();

    /**
     * @brief Add a stopping condition
     * 
     * @param condition Condition to add (takes ownership)
     * 
     * Conditions are checked in the order they are added.
     * First condition to return non-CONTINUE wins.
     */
    void addCondition(std::unique_ptr<Condition> condition);

    /**
     * @brief Check all stopping conditions
     * 
     * @param iteration Current iteration number (0-based)
     * @param current_entropy Current entropy value
     * @param previous_entropy Previous entropy value
     * 
     * @return First non-CONTINUE StopReason, or CONTINUE if all pass
     * 
     * Conditions are evaluated in order until one returns non-CONTINUE.
     * This implements priority-based stopping (earlier conditions have priority).
     */
    StopReason checkStoppingConditions(size_t iteration,
                                      double current_entropy,
                                      double previous_entropy);

    /**
     * @brief Reset all conditions for a new run
     * 
     * Calls reset() on all registered conditions.
     * Should be called at the start of each minimization run.
     */
    void reset();

    /**
     * @brief Get number of registered conditions
     * @return Number of conditions
     */
    size_t size() const;

    /**
     * @brief Get description of all conditions
     * @return Multi-line string describing all conditions
     */
    std::string description() const;

private:
    std::vector<std::unique_ptr<Condition>> conditions_;
};

} // namespace entropy

#endif // CONDITIONS_CHECKER_H
