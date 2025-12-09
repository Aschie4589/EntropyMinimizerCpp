#ifndef CHECKPOINT_CONFIG_H_
#define CHECKPOINT_CONFIG_H_

#include <string>
#include <stdexcept>
#include <sstream>

namespace entropy {

/**
 * @brief Configuration for checkpoint management
 * 
 * Controls how and when checkpoints are saved, including
 * file naming, cleanup policies, and compression.
 */
struct CheckpointConfig {
    // Enable/disable checkpoint saving
    bool enabled = false;
    
    // Save checkpoint every N iterations
    int interval = 10000;
    
    // Directory for checkpoint files
    std::string directory = "checkpoints/";
    
    // Filename pattern with placeholders: {run_id}, {iteration}
    std::string filename_pattern = "{run_id}_iter{iteration}.dat";
    
    // Keep only the last N checkpoints (0 = keep all)
    int keep_last_n = 3;
    
    // Enable compression for checkpoint files
    bool compress = false;
    
    /**
     * @brief Construct with default values
     */
    CheckpointConfig() = default;
    
    /**
     * @brief Construct with custom values
     */
    CheckpointConfig(bool enable, int intv = 10000, const std::string& dir = "checkpoints/")
        : enabled(enable), interval(intv), directory(dir) {
        validate();
    }
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (interval <= 0) {
            throw std::invalid_argument(
                "CheckpointConfig: interval must be > 0, got " + std::to_string(interval)
            );
        }
        if (keep_last_n < 0) {
            throw std::invalid_argument(
                "CheckpointConfig: keep_last_n must be >= 0, got " + std::to_string(keep_last_n)
            );
        }
        if (directory.empty()) {
            throw std::invalid_argument(
                "CheckpointConfig: directory cannot be empty"
            );
        }
        if (filename_pattern.empty()) {
            throw std::invalid_argument(
                "CheckpointConfig: filename_pattern cannot be empty"
            );
        }
    }
    
    /**
     * @brief Convert to string for debugging/logging
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "CheckpointConfig{enabled=" << (enabled ? "true" : "false")
            << ", interval=" << interval
            << ", directory='" << directory << "'"
            << ", filename_pattern='" << filename_pattern << "'"
            << ", keep_last_n=" << keep_last_n
            << ", compress=" << (compress ? "true" : "false") << "}";
        return oss.str();
    }
    
    // Equality comparison for testing
    bool operator==(const CheckpointConfig& other) const {
        return enabled == other.enabled && 
               interval == other.interval && 
               directory == other.directory &&
               filename_pattern == other.filename_pattern &&
               keep_last_n == other.keep_last_n &&
               compress == other.compress;
    }
    
    bool operator!=(const CheckpointConfig& other) const {
        return !(*this == other);
    }
};

} // namespace entropy

#endif // CHECKPOINT_CONFIG_H_
