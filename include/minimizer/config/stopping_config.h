#ifndef STOPPING_CONFIG_H_
#define STOPPING_CONFIG_H_

#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>

namespace entropy {

/**
 * @brief Configuration for stopping conditions
 * 
 * Defines when the minimization should terminate based on
 * iteration limits, convergence criteria, and target values.
 */
struct StoppingConfig {
    // Maximum number of iterations before forced termination
    int max_iterations = 500000;
    
    // Convergence tolerance for entropy stability detection
    double convergence_tolerance = 1e-15;
    
    // Window size for convergence detection (number of recent values to analyze)
    size_t convergence_window = 20;
    
    // Optional target entropy value (stop when reached)
    std::optional<double> target_entropy = std::nullopt;
    
    /**
     * @brief Construct with default values
     */
    StoppingConfig() = default;
    
    /**
     * @brief Construct with custom values
     */
    StoppingConfig(int max_iters, double conv_tol = 1e-15, size_t conv_window = 20)
        : max_iterations(max_iters), 
          convergence_tolerance(conv_tol), 
          convergence_window(conv_window) {
        validate();
    }
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (max_iterations <= 0) {
            throw std::invalid_argument(
                "StoppingConfig: max_iterations must be > 0, got " + std::to_string(max_iterations)
            );
        }
        if (convergence_tolerance <= 0.0) {
            throw std::invalid_argument(
                "StoppingConfig: convergence_tolerance must be > 0, got " + 
                std::to_string(convergence_tolerance)
            );
        }
        if (convergence_window < 2) {
            throw std::invalid_argument(
                "StoppingConfig: convergence_window must be >= 2, got " + 
                std::to_string(convergence_window)
            );
        }
        if (target_entropy.has_value() && target_entropy.value() < 0.0) {
            throw std::invalid_argument(
                "StoppingConfig: target_entropy must be >= 0, got " + 
                std::to_string(target_entropy.value())
            );
        }
    }
    
    /**
     * @brief Convert to string for debugging/logging
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "StoppingConfig{max_iterations=" << max_iterations 
            << ", convergence_tolerance=" << convergence_tolerance
            << ", convergence_window=" << convergence_window;
        if (target_entropy.has_value()) {
            oss << ", target_entropy=" << target_entropy.value();
        }
        oss << "}";
        return oss.str();
    }
    
    // Equality comparison for testing
    bool operator==(const StoppingConfig& other) const {
        return max_iterations == other.max_iterations && 
               convergence_tolerance == other.convergence_tolerance && 
               convergence_window == other.convergence_window &&
               target_entropy == other.target_entropy;
    }
    
    bool operator!=(const StoppingConfig& other) const {
        return !(*this == other);
    }
};

} // namespace entropy

#endif // STOPPING_CONFIG_H_
