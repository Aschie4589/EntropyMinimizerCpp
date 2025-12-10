#ifndef CONFIG_LOADER_H_
#define CONFIG_LOADER_H_

#include "minimizer/config/minimizer_config.h"
#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>
#include <regex>
#include <cstdlib>

namespace entropy {

/**
 * @brief Exception for configuration loading errors
 */
class ConfigLoadError : public std::runtime_error {
public:
    ConfigLoadError(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Loads MinimizerConfig from YAML files
 * 
 * Provides YAML parsing with:
 * - Environment variable expansion (${VAR} or $VAR)
 * - Default value injection for missing fields
 * - Type validation with helpful error messages
 * - Line number reporting for errors
 */
class ConfigLoader {
public:
    /**
     * @brief Load configuration from YAML file
     * @param filepath Path to YAML configuration file
     * @return Loaded and validated MinimizerConfig
     * @throws ConfigLoadError if file not found or parsing fails
     */
    static MinimizerConfig loadFromFile(const std::string& filepath) {
        try {
            YAML::Node root = YAML::LoadFile(filepath);
            return parseConfig(root, filepath);
        } catch (const YAML::Exception& e) {
            throw ConfigLoadError(
                "Failed to load config from '" + filepath + "': " + e.what()
            );
        }
    }
    
    /**
     * @brief Load configuration from YAML string
     * @param yaml_content YAML content as string
     * @return Loaded and validated MinimizerConfig
     * @throws ConfigLoadError if parsing fails
     */
    static MinimizerConfig loadFromString(const std::string& yaml_content) {
        try {
            YAML::Node root = YAML::Load(yaml_content);
            return parseConfig(root, "<string>");
        } catch (const YAML::Exception& e) {
            throw ConfigLoadError(
                "Failed to parse config from string: " + std::string(e.what())
            );
        }
    }
    
private:
    static MinimizerConfig parseConfig(const YAML::Node& root, const std::string& source) {
        MinimizerConfig config;
        
        // Check if config is nested under 'minimizer' key
        YAML::Node minimizer_node = root;
        if (root["minimizer"]) {
            minimizer_node = root["minimizer"];
        }
        
        // Parse each sub-configuration
        if (minimizer_node["algorithm"]) {
            parseAlgorithmConfig(minimizer_node["algorithm"], config.algorithm, source);
        }
        
        if (minimizer_node["stopping"]) {
            parseStoppingConfig(minimizer_node["stopping"], config.stopping, source);
        }
        
        if (minimizer_node["prediction"]) {
            parsePredictionConfig(minimizer_node["prediction"], config.prediction, source);
        }
        
        if (minimizer_node["checkpoint"]) {
            parseCheckpointConfig(minimizer_node["checkpoint"], config.checkpoint, source);
        }
        
        if (minimizer_node["logging"]) {
            parseLoggingConfig(minimizer_node["logging"], config.logging, source);
        }
        
        if (minimizer_node["multi_run"]) {
            parseMultiRunConfig(minimizer_node["multi_run"], config.multi_run, source);
        }
        
        // Validate complete configuration
        config.validate();
        return config;
    }
    
    static void parseAlgorithmConfig(const YAML::Node& node, AlgorithmConfig& config, const std::string& source) {
        if (node["epsilon"]) {
            config.epsilon = node["epsilon"].as<double>();
        }
        
        if (node["precision"]) {
            std::string prec_str = node["precision"].as<std::string>();
            if (prec_str == "DOUBLE" || prec_str == "double") {
                config.precision = PrecisionType::DOUBLE;
            } else if (prec_str == "FLOAT" || prec_str == "float") {
                config.precision = PrecisionType::FLOAT;
            } else {
                throw ConfigLoadError(
                    source + ": Invalid precision '" + prec_str + "', must be DOUBLE or FLOAT"
                );
            }
        }
        
        if (node["device_id"]) {
            config.device_id = node["device_id"].as<int>();
        }
    }
    
    static void parseStoppingConfig(const YAML::Node& node, StoppingConfig& config, const std::string& source) {
        if (node["max_iterations"]) {
            config.max_iterations = node["max_iterations"].as<int>();
        }
        
        if (node["convergence_tolerance"]) {
            config.convergence_tolerance = node["convergence_tolerance"].as<double>();
        }
        
        if (node["convergence_window"]) {
            config.convergence_window = node["convergence_window"].as<size_t>();
        }
        
        if (node["target_entropy"]) {
            config.target_entropy = node["target_entropy"].as<double>();
        }

    }
    
    static void parsePredictionConfig(const YAML::Node& node, PredictionConfig& config, const std::string& source) {
        if (node["enabled"]) {
            config.enabled = node["enabled"].as<bool>();
        }
        
        if (node["strategy"] || node["strategy_type"]) {
            const YAML::Node& strat_node = node["strategy"] ? node["strategy"] : node["strategy_type"];
            std::string strategy_str = strat_node.as<std::string>();
            
            if (strategy_str == "EXPONENTIAL" || strategy_str == "exponential") {
                config.strategy_type = PredictionStrategyType::EXPONENTIAL;
            } else if (strategy_str == "LINEAR" || strategy_str == "linear") {
                config.strategy_type = PredictionStrategyType::LINEAR;
            } else if (strategy_str == "AUTO" || strategy_str == "auto") {
                config.strategy_type = PredictionStrategyType::AUTO;
            } else {
                throw ConfigLoadError(
                    source + ": Invalid prediction strategy '" + strategy_str + 
                    "', must be EXPONENTIAL, LINEAR, or AUTO"
                );
            }
        }
        
        if (node["window_size"]) {
            config.window_size = node["window_size"].as<size_t>();
        }
        
        if (node["rsquared_threshold"]) {
            config.rsquared_threshold = node["rsquared_threshold"].as<double>();
        }
        
        if (node["convergence_tolerance"]) {
            config.convergence_tolerance = node["convergence_tolerance"].as<double>();
        }

        if (node["prediction_multiplier"]) {
            config.prediction_multiplier = node["prediction_multiplier"].as<double>();
        }
        
        if (node["min_data_points"]) {
            config.min_data_points = node["min_data_points"].as<size_t>();
        }
    }
    
    static void parseCheckpointConfig(const YAML::Node& node, CheckpointConfig& config, const std::string& source) {
        if (node["enabled"]) {
            config.enabled = node["enabled"].as<bool>();
        }
        
        if (node["interval"]) {
            config.interval = node["interval"].as<int>();
        }
        
        if (node["directory"]) {
            config.directory = expandEnvVars(node["directory"].as<std::string>());
        }
        
        if (node["filename_pattern"]) {
            config.filename_pattern = node["filename_pattern"].as<std::string>();
        }
        
        if (node["keep_last_n"]) {
            config.keep_last_n = node["keep_last_n"].as<int>();
        }
        
        if (node["compress"]) {
            config.compress = node["compress"].as<bool>();
        }
    }
    
    static void parseLoggingConfig(const YAML::Node& node, LoggingConfig& config, const std::string& source) {
        if (node["enabled"]) {
            config.enabled = node["enabled"].as<bool>();
        }
        
        if (node["level"]) {
            std::string level_str = node["level"].as<std::string>();
            
            if (level_str == "DEBUG" || level_str == "debug") {
                config.level = LogLevel::DEBUG;
            } else if (level_str == "INFO" || level_str == "info") {
                config.level = LogLevel::INFO;
            } else if (level_str == "WARN" || level_str == "warn" || level_str == "WARNING") {
                config.level = LogLevel::WARN;
            } else if (level_str == "ERROR" || level_str == "error") {
                config.level = LogLevel::ERROR;
            } else {
                throw ConfigLoadError(
                    source + ": Invalid log level '" + level_str + 
                    "', must be DEBUG, INFO, WARN, or ERROR"
                );
            }
        }
        
        if (node["file_path"] || node["file"]) {
            const YAML::Node& file_node = node["file_path"] ? node["file_path"] : node["file"];
            config.file_path = expandEnvVars(file_node.as<std::string>());
        }
        
        if (node["log_interval"] || node["interval"]) {
            const YAML::Node& interval_node = node["log_interval"] ? node["log_interval"] : node["interval"];
            config.log_interval = interval_node.as<int>();
        }
        
        if (node["print_to_console"] || node["console"]) {
            const YAML::Node& console_node = node["print_to_console"] ? node["print_to_console"] : node["console"];
            config.print_to_console = console_node.as<bool>();
        }
        
        if (node["color_enabled"] || node["color"]) {
            const YAML::Node& color_node = node["color_enabled"] ? node["color_enabled"] : node["color"];
            config.color_enabled = color_node.as<bool>();
        }
    }
    
    static void parseMultiRunConfig(const YAML::Node& node, MultiRunConfig& config, const std::string& source) {
        if (node["num_attempts"] || node["attempts"]) {
            const YAML::Node& attempts_node = node["num_attempts"] ? node["num_attempts"] : node["attempts"];
            config.num_attempts = attempts_node.as<int>();
        }
        
        if (node["track_global_moe"]) {
            config.track_global_moe = node["track_global_moe"].as<bool>();
        }
        
        if (node["restart_on_failure"]) {
            config.restart_on_failure = node["restart_on_failure"].as<bool>();
        }
        
        if (node["max_failures"]) {
            config.max_failures = node["max_failures"].as<int>();
        }
    }
    
    /**
     * @brief Expand environment variables in string
     * Supports ${VAR} and $VAR syntax
     */
    static std::string expandEnvVars(const std::string& input) {
        std::string result = input;
        
        // Pattern: ${VAR} or $VAR (word characters only)
        std::regex env_regex(R"(\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*))");
        std::smatch match;
        std::string::const_iterator search_start(result.cbegin());
        
        std::string expanded;
        size_t last_pos = 0;
        
        while (std::regex_search(search_start, result.cend(), match, env_regex)) {
            // Get the variable name (from either capture group)
            std::string var_name = match[1].matched ? match[1].str() : match[2].str();
            
            // Get environment variable value
            const char* env_value = std::getenv(var_name.c_str());
            std::string replacement = env_value ? env_value : "";
            
            // Build result string
            size_t match_pos = match.position(0) + (search_start - result.cbegin());
            expanded += result.substr(last_pos, match_pos - last_pos);
            expanded += replacement;
            
            last_pos = match_pos + match.length(0);
            search_start = result.cbegin() + last_pos;
        }
        
        // Append remaining string
        expanded += result.substr(last_pos);
        
        return expanded;
    }
};

} // namespace entropy

#endif // CONFIG_LOADER_H_
