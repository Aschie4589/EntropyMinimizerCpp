#ifndef NUMERICAL_INSTABILITY_CONDITION_H
#define NUMERICAL_INSTABILITY_CONDITION_H

#include "minimizer/stopping/condition.h"
#include "minimizer/state/circular_buffer.h"
#include <memory>

namespace entropy {

/**
 * @brief Stops when entropy increases instead of decreases
 * 
 * Detects numerical instability by checking if any recent entropy
 * improvement is negative (entropy increased). This indicates the
 * algorithm has diverged or reached numerical precision limits.
 * 
 * This matches the logic from entropy_minimizer.cu lines 331-339.
 */
class NumericalInstabilityCondition : public Condition {
public:
    /**
     * @brief Construct numerical instability detector
     * 
     * @param window_size Number of recent iterations to check (default: 20)
     *                   If any improvement in this window is negative, stop.
     */
    explicit NumericalInstabilityCondition(size_t window_size = 20);

    StopReason check(size_t iteration, 
                    double current_entropy, 
                    double previous_entropy) override;

    void reset() override;

    std::string description() const override;

private:
    size_t window_size_;
    std::unique_ptr<CircularBuffer<double>> entropy_history_;
};

} // namespace entropy

#endif // NUMERICAL_INSTABILITY_CONDITION_H
