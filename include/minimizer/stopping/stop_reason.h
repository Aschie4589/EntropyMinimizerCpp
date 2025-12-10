#ifndef STOP_REASON_H
#define STOP_REASON_H

#include <string>

namespace entropy {

/**
 * @brief Enumeration of reasons for stopping minimization
 * 
 * Used by Condition classes to indicate why minimization should stop.
 * CONTINUE means the condition is not satisfied and minimization should proceed.
 */
enum class StopReason {
    /// Condition not satisfied, continue minimization
    CONTINUE,
    
    /// Convergence criterion met (average improvement below tolerance)
    CONVERGED,
    
    /// Maximum iteration limit reached
    MAX_ITERATIONS,
    
    /// Numerical instability detected (entropy increased instead of decreased)
    NUMERICAL_INSTABILITY,
    
    /// Predictor forecasts entropy will exceed target by significant margin
    FORECASTED_DIVERGENCE,
    
    /// Predictor forecasts target entropy unreachable from current trajectory
    TARGET_UNREACHABLE
};

/**
 * @brief Convert StopReason to human-readable string
 * @param reason Stop reason to convert
 * @return String representation
 */
std::string to_string(StopReason reason);

} // namespace entropy

#endif // STOP_REASON_H
