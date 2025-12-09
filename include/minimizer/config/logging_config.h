#ifndef LOGGING_CONFIG_H_
#define LOGGING_CONFIG_H_

#include <string>
#include <stdexcept>
#include <sstream>

namespace entropy {

/**
 * @brief Logging levels
 */
enum class LogLevel {
    DEBUG = -1,  // Verbose debug information
    INFO = 0,    // General information
    WARN = 1,    // Warnings
    ERROR = 2    // Errors only
};

/**
 * @brief Configuration for logging system
 * 
 * Controls message output including file logging,
 * console output, verbosity, and formatting.
 */
struct LoggingConfig {
    // Enable/disable logging system
    bool enabled = false;
    
    // Minimum log level to output
    LogLevel level = LogLevel::INFO;
    
    // Log file path (empty = no file logging)
    std::string file_path = "minimizer.log";
    
    // Log every N iterations (for periodic progress messages)
    int log_interval = 100;
    
    // Print logs to console
    bool print_to_console = true;
    
    // Enable colored output in console
    bool color_enabled = true;
    
    /**
     * @brief Construct with default values
     */
    LoggingConfig() = default;
    
    /**
     * @brief Construct with custom values
     */
    LoggingConfig(bool enable, LogLevel lvl = LogLevel::INFO)
        : enabled(enable), level(lvl) {
        validate();
    }
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (log_interval <= 0) {
            throw std::invalid_argument(
                "LoggingConfig: log_interval must be > 0, got " + std::to_string(log_interval)
            );
        }
    }
    
    /**
     * @brief Convert to string for debugging/logging
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "LoggingConfig{enabled=" << (enabled ? "true" : "false")
            << ", level=";
        switch(level) {
            case LogLevel::DEBUG: oss << "DEBUG"; break;
            case LogLevel::INFO: oss << "INFO"; break;
            case LogLevel::WARN: oss << "WARN"; break;
            case LogLevel::ERROR: oss << "ERROR"; break;
        }
        oss << ", file_path='" << file_path << "'"
            << ", log_interval=" << log_interval
            << ", print_to_console=" << (print_to_console ? "true" : "false")
            << ", color_enabled=" << (color_enabled ? "true" : "false") << "}";
        return oss.str();
    }
    
    // Equality comparison for testing
    bool operator==(const LoggingConfig& other) const {
        return enabled == other.enabled && 
               level == other.level && 
               file_path == other.file_path &&
               log_interval == other.log_interval &&
               print_to_console == other.print_to_console &&
               color_enabled == other.color_enabled;
    }
    
    bool operator!=(const LoggingConfig& other) const {
        return !(*this == other);
    }
};

} // namespace entropy

#endif // LOGGING_CONFIG_H_
