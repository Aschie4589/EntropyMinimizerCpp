#include "minimizer/config/minimizer_config.h"
#include "minimizer/config/config_builder.h"
#include "minimizer/config/config_loader.h"
#include <gtest/gtest.h>
#include <fstream>

using namespace entropy;

// Test default config struct construction
TEST(ConfigTest, DefaultConstruction) {
    MinimizerConfig config;
    EXPECT_NO_THROW(config.validate());
    
    // Verify default values
    EXPECT_DOUBLE_EQ(config.algorithm.epsilon, 1e-3);
    EXPECT_EQ(config.algorithm.precision, PrecisionType::DOUBLE);
    EXPECT_EQ(config.algorithm.device_id, 0);
    
    EXPECT_EQ(config.stopping.max_iterations, 500000);
    EXPECT_DOUBLE_EQ(config.stopping.convergence_tolerance, 1e-15);
    EXPECT_EQ(config.stopping.convergence_window, 20);
    
    EXPECT_TRUE(config.prediction.enabled);
    EXPECT_EQ(config.prediction.strategy_type, PredictionStrategyType::AUTO);
    
    EXPECT_FALSE(config.checkpoint.enabled);
    EXPECT_FALSE(config.logging.enabled);
    
    EXPECT_EQ(config.multi_run.num_attempts, 100);
}

// Test config builder fluent interface
TEST(ConfigBuilderTest, FluentChaining) {
    auto config = ConfigBuilder()
        .setEpsilon(1e-4)
        .setMaxIterations(100000)
        .enablePrediction(false)
        .enableCheckpoints(true)
        .setCheckpointInterval(5000)
        .build();
    
    EXPECT_DOUBLE_EQ(config.algorithm.epsilon, 1e-4);
    EXPECT_EQ(config.stopping.max_iterations, 100000);
    EXPECT_FALSE(config.prediction.enabled);
    EXPECT_TRUE(config.checkpoint.enabled);
    EXPECT_EQ(config.checkpoint.interval, 5000);
}

// Test config validation
TEST(ConfigValidationTest, InvalidEpsilon) {
    AlgorithmConfig config;
    config.epsilon = -0.5;  // Invalid: must be in (0,1)
    EXPECT_THROW(config.validate(), std::invalid_argument);
    
    config.epsilon = 1.5;  // Invalid: must be in (0,1)
    EXPECT_THROW(config.validate(), std::invalid_argument);
    
    config.epsilon = 0.5;  // Valid
    EXPECT_NO_THROW(config.validate());
}

TEST(ConfigValidationTest, InvalidIterations) {
    StoppingConfig config;
    config.max_iterations = -100;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    
    config.max_iterations = 0;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    
    config.max_iterations = 1000;
    EXPECT_NO_THROW(config.validate());
}

TEST(ConfigValidationTest, InvalidConvergenceWindow) {
    StoppingConfig config;
    config.convergence_window = 1;  // Must be >= 2
    EXPECT_THROW(config.validate(), std::invalid_argument);
    
    config.convergence_window = 2;
    EXPECT_NO_THROW(config.validate());
}

TEST(ConfigValidationTest, PredictionWindowSize) {
    PredictionConfig config;
    config.window_size = 50;
    config.min_data_points = 100;  // window_size < min_data_points is invalid
    EXPECT_THROW(config.validate(), std::invalid_argument);
    
    config.window_size = 100;  // Equal is OK
    EXPECT_NO_THROW(config.validate());
}

// Test preset configs
TEST(ConfigBuilderTest, PresetConfigs) {
    auto default_cfg = ConfigBuilder::defaultConfig();
    EXPECT_NO_THROW(default_cfg.validate());
    EXPECT_EQ(default_cfg.stopping.max_iterations, 500000);
    
    auto fast_cfg = ConfigBuilder::fastTestConfig();
    EXPECT_NO_THROW(fast_cfg.validate());
    EXPECT_EQ(fast_cfg.stopping.max_iterations, 1000);
    EXPECT_FALSE(fast_cfg.prediction.enabled);
    
    auto prod_cfg = ConfigBuilder::productionConfig();
    EXPECT_NO_THROW(prod_cfg.validate());
    EXPECT_EQ(prod_cfg.stopping.max_iterations, 1000000);
    EXPECT_TRUE(prod_cfg.checkpoint.enabled);
    
    auto debug_cfg = ConfigBuilder::debugConfig();
    EXPECT_NO_THROW(debug_cfg.validate());
    EXPECT_EQ(debug_cfg.logging.level, LogLevel::DEBUG);
    EXPECT_EQ(debug_cfg.checkpoint.interval, 10);
}

// Test YAML loading
TEST(ConfigLoaderTest, LoadFromString) {
    std::string yaml_content = R"(
minimizer:
  algorithm:
    epsilon: 1.0e-2
    precision: FLOAT
    device_id: 1
  stopping:
    max_iterations: 50000
    convergence_tolerance: 1.0e-12
  prediction:
    enabled: false
)";
    
    auto config = ConfigLoader::loadFromString(yaml_content);
    
    EXPECT_DOUBLE_EQ(config.algorithm.epsilon, 1e-2);
    EXPECT_EQ(config.algorithm.precision, PrecisionType::FLOAT);
    EXPECT_EQ(config.algorithm.device_id, 1);
    EXPECT_EQ(config.stopping.max_iterations, 50000);
    EXPECT_DOUBLE_EQ(config.stopping.convergence_tolerance, 1e-12);
    EXPECT_FALSE(config.prediction.enabled);
    
    // Defaults should still apply for unspecified fields
    EXPECT_EQ(config.checkpoint.interval, 10000);
    EXPECT_EQ(config.multi_run.num_attempts, 100);
}

TEST(ConfigLoaderTest, LoadWithDefaults) {
    std::string yaml_content = R"(
minimizer:
  algorithm:
    epsilon: 5.0e-3
)";
    
    auto config = ConfigLoader::loadFromString(yaml_content);
    
    // Specified value
    EXPECT_DOUBLE_EQ(config.algorithm.epsilon, 5e-3);
    
    // Defaults should apply
    EXPECT_EQ(config.algorithm.precision, PrecisionType::DOUBLE);
    EXPECT_EQ(config.stopping.max_iterations, 500000);
    EXPECT_TRUE(config.prediction.enabled);
}

TEST(ConfigLoaderTest, InvalidPrecision) {
    std::string yaml_content = R"(
minimizer:
  algorithm:
    precision: INVALID
)";
    
    EXPECT_THROW(ConfigLoader::loadFromString(yaml_content), ConfigLoadError);
}

TEST(ConfigLoaderTest, InvalidStrategy) {
    std::string yaml_content = R"(
minimizer:
  prediction:
    strategy: UNKNOWN
)";
    
    EXPECT_THROW(ConfigLoader::loadFromString(yaml_content), ConfigLoadError);
}

TEST(ConfigLoaderTest, InvalidLogLevel) {
    std::string yaml_content = R"(
minimizer:
  logging:
    level: TRACE
)";
    
    EXPECT_THROW(ConfigLoader::loadFromString(yaml_content), ConfigLoadError);
}

// Test config equality
TEST(ConfigTest, Equality) {
    MinimizerConfig config1;
    MinimizerConfig config2;
    
    EXPECT_EQ(config1, config2);
    
    config2.algorithm.epsilon = 1e-4;
    EXPECT_NE(config1, config2);
}

// Test to_string methods
TEST(ConfigTest, ToString) {
    MinimizerConfig config;
    std::string str = config.to_string();
    
    EXPECT_TRUE(str.find("MinimizerConfig") != std::string::npos);
    EXPECT_TRUE(str.find("AlgorithmConfig") != std::string::npos);
    EXPECT_TRUE(str.find("epsilon") != std::string::npos);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
