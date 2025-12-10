#ifndef CONFIG_BUILDER_H_
#define CONFIG_BUILDER_H_

#include "minimizer/config/minimizer_config.h"
#include <stdexcept>

namespace entropy {

/**
 * @brief Fluent builder for MinimizerConfig
 * 
 * Provides a convenient fluent interface for constructing
 * MinimizerConfig objects with validation. Includes preset
 * configurations for common use cases.
 * 
 * Usage:
 *   auto config = ConfigBuilder()
 *       .setEpsilon(1e-3)
 *       .setMaxIterations(100000)
 *       .enablePrediction(true)
 *       .build();
 */
class ConfigBuilder {
public:
    ConfigBuilder() = default;
    
    // Algorithm configuration
    ConfigBuilder& setEpsilon(double epsilon) {
        config_.algorithm.epsilon = epsilon;
        return *this;
    }
    
    ConfigBuilder& setPrecision(PrecisionType precision) {
        config_.algorithm.precision = precision;
        return *this;
    }
    
    ConfigBuilder& setDeviceId(int device_id) {
        config_.algorithm.device_id = device_id;
        return *this;
    }
    
    // Stopping configuration
    ConfigBuilder& setMaxIterations(int max_iterations) {
        config_.stopping.max_iterations = max_iterations;
        return *this;
    }
    
    ConfigBuilder& setConvergenceTolerance(double tolerance) {
        config_.stopping.convergence_tolerance = tolerance;
        return *this;
    }
    
    ConfigBuilder& setConvergenceWindow(size_t window) {
        config_.stopping.convergence_window = window;
        return *this;
    }
    
    ConfigBuilder& setTargetEntropy(double target) {
        config_.stopping.target_entropy = target;
        return *this;
    }
    
    ConfigBuilder& clearTargetEntropy() {
        config_.stopping.target_entropy = std::nullopt;
        return *this;
    }
    
    // Prediction configuration
    ConfigBuilder& enablePrediction(bool enabled = true) {
        config_.prediction.enabled = enabled;
        return *this;
    }
    
    ConfigBuilder& setPredictionStrategy(PredictionStrategyType strategy) {
        config_.prediction.strategy_type = strategy;
        return *this;
    }
    
    ConfigBuilder& setPredictionWindowSize(size_t window_size) {
        config_.prediction.window_size = window_size;
        return *this;
    }
    
    ConfigBuilder& setRSquaredThreshold(double threshold) {
        config_.prediction.rsquared_threshold = threshold;
        return *this;
    }
    
    ConfigBuilder& setPredictionConvergenceTolerance(double tolerance) {
        config_.prediction.convergence_tolerance = tolerance;
        return *this;
    }

    ConfigBuilder& setPredictionMultiplier(double multiplier) {
        config_.prediction.prediction_multiplier = multiplier;
        return *this;
    }
    
    ConfigBuilder& setMinDataPoints(size_t min_points) {
        config_.prediction.min_data_points = min_points;
        return *this;
    }
    
    // Checkpoint configuration
    ConfigBuilder& enableCheckpoints(bool enabled = true) {
        config_.checkpoint.enabled = enabled;
        return *this;
    }
    
    ConfigBuilder& setCheckpointInterval(int interval) {
        config_.checkpoint.interval = interval;
        return *this;
    }
    
    ConfigBuilder& setCheckpointDirectory(const std::string& directory) {
        config_.checkpoint.directory = directory;
        return *this;
    }
    
    ConfigBuilder& setCheckpointFilenamePattern(const std::string& pattern) {
        config_.checkpoint.filename_pattern = pattern;
        return *this;
    }
    
    ConfigBuilder& setKeepLastN(int n) {
        config_.checkpoint.keep_last_n = n;
        return *this;
    }
    
    ConfigBuilder& enableCompression(bool enabled = true) {
        config_.checkpoint.compress = enabled;
        return *this;
    }
    
    // Logging configuration
    ConfigBuilder& enableLogging(bool enabled = true) {
        config_.logging.enabled = enabled;
        return *this;
    }
    
    ConfigBuilder& setLogLevel(LogLevel level) {
        config_.logging.level = level;
        return *this;
    }
    
    ConfigBuilder& setLogFilePath(const std::string& file_path) {
        config_.logging.file_path = file_path;
        return *this;
    }
    
    ConfigBuilder& setLogInterval(int interval) {
        config_.logging.log_interval = interval;
        return *this;
    }
    
    ConfigBuilder& setPrintToConsole(bool enabled = true) {
        config_.logging.print_to_console = enabled;
        return *this;
    }
    
    ConfigBuilder& setColorEnabled(bool enabled = true) {
        config_.logging.color_enabled = enabled;
        return *this;
    }
    
    // Multi-run configuration
    ConfigBuilder& setNumAttempts(int num_attempts) {
        config_.multi_run.num_attempts = num_attempts;
        return *this;
    }
    
    ConfigBuilder& setTrackGlobalMoe(bool track = true) {
        config_.multi_run.track_global_moe = track;
        return *this;
    }
    
    ConfigBuilder& setRestartOnFailure(bool restart = true) {
        config_.multi_run.restart_on_failure = restart;
        return *this;
    }
    
    ConfigBuilder& setMaxFailures(int max_failures) {
        config_.multi_run.max_failures = max_failures;
        return *this;
    }
    
    /**
     * @brief Build and validate the configuration
     * @return Validated MinimizerConfig
     * @throws std::invalid_argument if validation fails
     */
    MinimizerConfig build() const {
        config_.validate();
        return config_;
    }
    
    // ========== Preset Configurations ==========
    
    /**
     * @brief Default configuration for general use
     */
    static MinimizerConfig defaultConfig() {
        return ConfigBuilder()
            .setEpsilon(1e-3)
            .setMaxIterations(500000)
            .enablePrediction(true)
            .enableCheckpoints(false)
            .enableLogging(false)
            .setPredictionMultiplier(1.05)
            .build();
    }
    
    /**
     * @brief Fast configuration for testing (reduced iterations, minimal overhead)
     */
    static MinimizerConfig fastTestConfig() {
        return ConfigBuilder()
            .setEpsilon(1e-3)
            .setMaxIterations(1000)
            .enablePrediction(false)
            .enableCheckpoints(false)
            .enableLogging(false)
            .setNumAttempts(10)
            .build();
    }
    
    /**
     * @brief Production configuration (high accuracy, full features)
     */
    static MinimizerConfig productionConfig() {
        return ConfigBuilder()
            .setEpsilon(1e-4)
            .setPrecision(PrecisionType::DOUBLE)
            .setMaxIterations(1000000)
            .setConvergenceTolerance(1e-16)
            .enablePrediction(true)
            .setPredictionStrategy(PredictionStrategyType::AUTO)
            .enableCheckpoints(true)
            .setCheckpointInterval(50000)
            .setKeepLastN(5)
            .enableLogging(true)
            .setLogLevel(LogLevel::INFO)
            .setLogInterval(1000)
            .setNumAttempts(100)
            .build();
    }
    
    /**
     * @brief Debug configuration (verbose logging, frequent checkpoints)
     */
    static MinimizerConfig debugConfig() {
        return ConfigBuilder()
            .setEpsilon(1e-3)
            .setMaxIterations(10000)
            .enablePrediction(true)
            .enableCheckpoints(true)
            .setCheckpointInterval(10)
            .setKeepLastN(0)  // Keep all for debugging
            .enableLogging(true)
            .setLogLevel(LogLevel::DEBUG)
            .setLogInterval(1)
            .setPrintToConsole(true)
            .setColorEnabled(true)
            .build();
    }
    
private:
    mutable MinimizerConfig config_;
};

} // namespace entropy

#endif // CONFIG_BUILDER_H_
