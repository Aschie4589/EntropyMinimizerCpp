#include "minimizer/stopping/stop_reason.h"

namespace entropy {

std::string to_string(StopReason reason) {
    switch (reason) {
        case StopReason::CONTINUE:
            return "CONTINUE";
        case StopReason::CONVERGED:
            return "CONVERGED";
        case StopReason::MAX_ITERATIONS:
            return "MAX_ITERATIONS";
        case StopReason::NUMERICAL_INSTABILITY:
            return "NUMERICAL_INSTABILITY";
        case StopReason::FORECASTED_DIVERGENCE:
            return "FORECASTED_DIVERGENCE";
        case StopReason::TARGET_UNREACHABLE:
            return "TARGET_UNREACHABLE";
        default:
            return "UNKNOWN";
    }
}

} // namespace entropy
