#ifndef CONVERGENCE_CONDITION_H
#define CONVERGENCE_CONDITION_H

#include "minimizer/stopping/condition.h"
#include "minimizer/state/circular_buffer.h"
#include <memory>

namespace entropy {

/**
 * @brief Stops when average entropy improvement falls below tolerance
 * 
 * Tracks entropy history in a circular buffer and computes average improvement.
 * Convergence is detected when:
 * 1. Buffer is full (enough history collected)
 * 2. All recent improvements are positive (no numerical instability)
 * 3. Average improvement over window < tolerance
 * 
 * This matches the logic from entropy_minimizer.cu lines 334-344.
 */
class ConvergenceCondition : public Condition {
public:
    /**
     * @brief Construct convergence condition
     * 
     * @param window_size Number of iterations to average over (default: 20)
     * @param tolerance Minimum average improvement to continue (default: 1e-15)
     */
    explicit ConvergenceCondition(size_t window_size = 20, 
                                  double tolerance = 1e-15);

    StopReason check(size_t iteration, 
                    double current_entropy, 
                    double previous_entropy) override;

    void reset() override;

    std::string description() const override;

private:
    size_t window_size_;
    double tolerance_;
    std::unique_ptr<CircularBuffer<double>> entropy_history_;
};

} // namespace entropy

#endif // CONVERGENCE_CONDITION_H
