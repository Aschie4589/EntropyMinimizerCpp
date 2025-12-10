#ifndef PREDICTION_CONFIG_H_
#define PREDICTION_CONFIG_H_

#include <stdexcept>
#include <string>
#include <sstream>

namespace entropy {

/**
 * @brief Strategy types for entropy prediction
 */
enum class PredictionStrategyType {
    EXPONENTIAL,  // Exponential fitting with R² validation
    LINEAR,       // Linear extrapolation
    AUTO          // Automatic strategy selection based on data
};

/**
 * @brief Configuration for entropy prediction
 * 
 * Controls the prediction system that estimates convergence
 * and can trigger early stopping when convergence is predicted.
 */
struct PredictionConfig {
    // Enable/disable prediction system
    bool enabled = true;
    
    // Strategy to use for prediction
    PredictionStrategyType strategy_type = PredictionStrategyType::EXPONENTIAL;
    
    // Number of recent entropy values to use for prediction
    size_t window_size = 200;
    
    // R² threshold for exponential fit acceptance (0,1)
    double rsquared_threshold = 0.999;
    
    // Convergence tolerance for predicted entropy
    double convergence_tolerance = 1e-15;
    
    // Multiplier for target entropy divergence check (i.e. return stop if predicted > target × multiplier)
    double prediction_multiplier = 1.05;
 
    // Minimum data points required before making predictions
    size_t min_data_points = 100;
    
    /**
     * @brief Construct with default values
     */
    PredictionConfig() = default;
    
    /**
     * @brief Construct with custom values
     */
    PredictionConfig(bool enable, PredictionStrategyType strategy = PredictionStrategyType::AUTO)
        : enabled(enable), strategy_type(strategy) {
        validate();
    }
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (window_size < min_data_points) {
            throw std::invalid_argument(
                "PredictionConfig: window_size (" + std::to_string(window_size) + 
                ") must be >= min_data_points (" + std::to_string(min_data_points) + ")"
            );
        }
        if (rsquared_threshold <= 0.0 || rsquared_threshold >= 1.0) {
            throw std::invalid_argument(
                "PredictionConfig: rsquared_threshold must be in (0,1), got " + 
                std::to_string(rsquared_threshold)
            );
        }
        if (convergence_tolerance <= 0.0) {
            throw std::invalid_argument(
                "PredictionConfig: convergence_tolerance must be > 0, got " + 
                std::to_string(convergence_tolerance)
            );
        }
        if (min_data_points == 0) {
            throw std::invalid_argument(
                "PredictionConfig: min_data_points must be > 0"
            );
        }
    }
    
    /**
     * @brief Convert to string for debugging/logging
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "PredictionConfig{enabled=" << (enabled ? "true" : "false")
            << ", strategy=";
        switch(strategy_type) {
            case PredictionStrategyType::EXPONENTIAL: oss << "EXPONENTIAL"; break;
            case PredictionStrategyType::LINEAR: oss << "LINEAR"; break;
            case PredictionStrategyType::AUTO: oss << "AUTO"; break;
        }
        oss << ", window_size=" << window_size
            << ", rsquared_threshold=" << rsquared_threshold
            << ", convergence_tolerance=" << convergence_tolerance
            << ", min_data_points=" << min_data_points << "}";
        return oss.str();
    }
    
    // Equality comparison for testing
    bool operator==(const PredictionConfig& other) const {
        return enabled == other.enabled && 
               strategy_type == other.strategy_type && 
               window_size == other.window_size &&
               rsquared_threshold == other.rsquared_threshold &&
               convergence_tolerance == other.convergence_tolerance &&
               min_data_points == other.min_data_points;
    }
    
    bool operator!=(const PredictionConfig& other) const {
        return !(*this == other);
    }
};

} // namespace entropy

#endif // PREDICTION_CONFIG_H_
