#ifndef MULTI_RUN_CONFIG_H_
#define MULTI_RUN_CONFIG_H_

#include <stdexcept>
#include <string>
#include <sstream>

namespace entropy {

/**
 * @brief Configuration for multi-run orchestration
 * 
 * Controls behavior when running multiple minimization attempts
 * with different initial vectors to find global minimum.
 */
struct MultiRunConfig {
    // Number of independent minimization attempts
    int num_attempts = 100;
    
    // Track minimum-of-entropies (MOE) across all runs
    bool track_global_moe = true;
    
    // Restart failed runs with new random vectors
    bool restart_on_failure = false;
    
    // Maximum consecutive failures before giving up (when restart_on_failure=true)
    int max_failures = 10;
    
    /**
     * @brief Construct with default values
     */
    MultiRunConfig() = default;
    
    /**
     * @brief Construct with custom values
     */
    MultiRunConfig(int attempts, bool track_moe = true)
        : num_attempts(attempts), track_global_moe(track_moe) {
        validate();
    }
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (num_attempts <= 0) {
            throw std::invalid_argument(
                "MultiRunConfig: num_attempts must be > 0, got " + std::to_string(num_attempts)
            );
        }
        if (max_failures < 0) {
            throw std::invalid_argument(
                "MultiRunConfig: max_failures must be >= 0, got " + std::to_string(max_failures)
            );
        }
    }
    
    /**
     * @brief Convert to string for debugging/logging
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "MultiRunConfig{num_attempts=" << num_attempts
            << ", track_global_moe=" << (track_global_moe ? "true" : "false")
            << ", restart_on_failure=" << (restart_on_failure ? "true" : "false")
            << ", max_failures=" << max_failures << "}";
        return oss.str();
    }
    
    // Equality comparison for testing
    bool operator==(const MultiRunConfig& other) const {
        return num_attempts == other.num_attempts && 
               track_global_moe == other.track_global_moe && 
               restart_on_failure == other.restart_on_failure &&
               max_failures == other.max_failures;
    }
    
    bool operator!=(const MultiRunConfig& other) const {
        return !(*this == other);
    }
};

} // namespace entropy

#endif // MULTI_RUN_CONFIG_H_
