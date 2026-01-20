#ifndef MINIMIZER_CONFIG_H_
#define MINIMIZER_CONFIG_H_

#include "minimizer/config/algorithm_config.h"
#include "minimizer/config/stopping_config.h"
#include "minimizer/config/prediction_config.h"
#include "minimizer/config/checkpoint_config.h"
#include "minimizer/config/logging_config.h"
#include "minimizer/config/resource_config.h"
#include "minimizer/config/multi_run_config.h"
#include <string>
#include <stdexcept>

namespace entropy {

/**
 * @brief Root configuration for the minimizer system
 * 
 * Composes all sub-configurations into a single validated
 * configuration object. Provides serialization support for
 * checkpoint metadata and configuration persistence.
 */
struct MinimizerConfig {
    // Algorithm execution parameters
    AlgorithmConfig algorithm;
    
    // Checkpoint management
    CheckpointConfig checkpoint;

    // Logging system
    LoggingConfig logging;

    // Multi-run orchestration
    MultiRunConfig multi_run;
    
    // Prediction system
    PredictionConfig prediction;        

    // Resource management (GPU/CPU allocation)
    ResourceConfig resource;

    // Stopping conditions
    StoppingConfig stopping;
    
    // Configuration version for compatibility tracking
    std::string version = "1.0.0";
    
    /**
     * @brief Construct with default sub-configurations
     */
    MinimizerConfig() = default;
    
    /**
     * @brief Validate all sub-configurations
     * @throws std::invalid_argument if any sub-config is invalid
     */
    void validate() const {
        try {
            algorithm.validate();
            stopping.validate();
            prediction.validate();
            checkpoint.validate();
            logging.validate();
            multi_run.validate();
            resource.validate();
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(
                std::string("MinimizerConfig validation failed: ") + e.what()
            );
        }
    }
    
    /**
     * @brief Convert to string for debugging/logging
     */
    std::string to_string() const {
        std::string result = "MinimizerConfig{\n";
        result += "  version: " + version + "\n";
        result += "  " + algorithm.to_string() + "\n";
        result += "  " + checkpoint.to_string() + "\n";
        result += "  " + logging.to_string() + "\n";
        result += "  " + multi_run.to_string() + "\n";
        result += "  " + prediction.to_string() + "\n";
        result += "  " + resource.to_string() + "\n";
        result += "  " + stopping.to_string() + "\n";
        result += "}";
        return result;
    }
    
    // Equality comparison for testing
    bool operator==(const MinimizerConfig& other) const {
        return algorithm == other.algorithm &&
               stopping == other.stopping &&
               prediction == other.prediction &&
               checkpoint == other.checkpoint &&
               logging == other.logging &&
               multi_run == other.multi_run &&
               resource == other.resource &&
               version == other.version;
    }
    
    bool operator!=(const MinimizerConfig& other) const {
        return !(*this == other);
    }
};

} // namespace entropy

#endif // MINIMIZER_CONFIG_H_
