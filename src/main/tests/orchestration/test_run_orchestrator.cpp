#include <gtest/gtest.h>
#include "minimizer/orchestration/run_orchestrator.h"
#include "minimizer/config/minimizer_config.h"
#include "compute/device/DeviceFactory.h"
#include <complex>
#include <vector>
#include <cmath>
#include <atomic>

using namespace entropy;

// ============================================================================
// Test Fixture
// ============================================================================

class RunOrchestratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create CPU device for testing
        cpu_device_ = DeviceFactory::create(DeviceFactory::DeviceType::CPU);
        ASSERT_NE(cpu_device_, nullptr);
        
        // Setup default config
        setupDefaultConfig();
    }
    
    void setupDefaultConfig() {
        // Algorithm config
        config_.algorithm.epsilon = 0.01;
        config_.algorithm.precision = PrecisionType::DOUBLE;
        
        // Create simple test Kraus operators (2×2 identity channel)
        kraus_count_ = 2;
        input_dim_ = 5;
        output_dim_ = 5;
        
        std::vector<std::complex<double>> kraus_data(
            kraus_count_ * input_dim_ * output_dim_,
            std::complex<double>(0.0, 0.0)
        );
        
        // K_0: Projector onto first basis state
        kraus_data[0] = 1.0;  // (0,0) element
        
        // K_1: Projector onto second basis state  
        kraus_data[input_dim_ * output_dim_ + input_dim_ + 1] = 1.0;  // (1,1) element
        
        kraus_ops_ = HostKrausOperators::fromDouble(
            kraus_data, kraus_count_, input_dim_, output_dim_
        );
        
        // Stopping config
        config_.stopping.max_iterations = 100;
        config_.stopping.convergence_window = 10;
        config_.stopping.convergence_tolerance = 1e-6;
        
        // Prediction and Checkpoint configs use defaults
    }
    
    HostVector createRandomVector(int dim) {
        std::vector<std::complex<double>> vec(dim);
        double norm = 0.0;
        
        for (int i = 0; i < dim; ++i) {
            vec[i] = std::complex<double>(
                static_cast<double>(rand()) / RAND_MAX,
                static_cast<double>(rand()) / RAND_MAX
            );
            norm += std::norm(vec[i]);
        }
        
        // Normalize
        norm = std::sqrt(norm);
        for (int i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
        
        return HostVector(vec);
    }
    
    std::unique_ptr<IComputeDevice> cpu_device_;
    MinimizerConfig config_;
    HostKrausOperators kraus_ops_;
    int kraus_count_;
    int input_dim_;
    int output_dim_;
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(RunOrchestratorTest, ConstructorValid) {
    EXPECT_NO_THROW(
        RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_)
    );
}

TEST_F(RunOrchestratorTest, ConstructorInvalidEpsilon) {
    config_.algorithm.epsilon = 0.0;  // Invalid
    
    EXPECT_THROW(
        RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_),
        std::invalid_argument
    );
}

TEST_F(RunOrchestratorTest, ConstructorWithCallback) {
    bool callback_invoked = false;
    
    auto callback = [&](int run_id, int iter, double entropy) {
        callback_invoked = true;
    };
    
    EXPECT_NO_THROW(
        RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_, callback)
    );
}

// ============================================================================
// Execute Tests
// ============================================================================

TEST_F(RunOrchestratorTest, ExecuteSimpleRun) {
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    HostVector initial = createRandomVector(input_dim_);
    RunResult result = orchestrator.execute(0, initial);
    
    EXPECT_TRUE(result.isSuccess());
    EXPECT_EQ(result.error_type, RunErrorType::NONE);
    EXPECT_EQ(result.run_id, 0);
    EXPECT_GT(result.iterations_taken, 0);
    EXPECT_LE(result.iterations_taken, config_.stopping.max_iterations);
    EXPECT_GE(result.runtime_seconds, 0.0);
    EXPECT_EQ(result.final_vector.dimension, input_dim_);
}

TEST_F(RunOrchestratorTest, ExecuteReachesMaxIterations) {
    config_.stopping.max_iterations = 10;
    config_.stopping.target_entropy = std::nullopt;  // Don't set target
    
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    HostVector initial = createRandomVector(input_dim_);
    RunResult result = orchestrator.execute(0, initial);
    
    EXPECT_TRUE(result.isSuccess());
    EXPECT_EQ(result.error_type, RunErrorType::NONE);
    // Max iterations is 10, but iterations are 0-indexed, so we get 11 iterations total (0,1,2,...,10)
    EXPECT_EQ(result.iterations_taken, 11);
}

TEST_F(RunOrchestratorTest, ExecuteWithConvergence) {
    config_.stopping.max_iterations = 1000;
    config_.stopping.convergence_window = 10;
    config_.stopping.convergence_tolerance = 1e-8;
    
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    HostVector initial = createRandomVector(input_dim_);
    RunResult result = orchestrator.execute(0, initial);
    
    EXPECT_TRUE(result.isSuccess());
    EXPECT_EQ(result.error_type, RunErrorType::NONE);
    // Should converge before max iterations
    EXPECT_LT(result.iterations_taken, 1000);
}

TEST_F(RunOrchestratorTest, ExecuteInvalidVectorDimension) {
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    HostVector invalid = createRandomVector(10);  // Wrong dimension
    RunResult result = orchestrator.execute(0, invalid);
    
    EXPECT_FALSE(result.isSuccess());
    EXPECT_EQ(result.error_type, RunErrorType::INVALID_INPUT);
    EXPECT_FALSE(result.error_message.empty());
    EXPECT_NE(result.error_message.find("dimension mismatch"), std::string::npos);
}

// ============================================================================
// Progress Callback Tests
// ============================================================================

TEST_F(RunOrchestratorTest, ProgressCallbackInvoked) {
    std::atomic<int> callback_count{0};
    std::atomic<int> last_iteration{-1};
    std::atomic<int> last_run_id{-1};
    
    auto callback = [&](int run_id, int iter, double entropy) {
        callback_count++;
        last_iteration = iter;
        last_run_id = run_id;
    };
    
    config_.stopping.max_iterations = 20;
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_, callback);
    
    HostVector initial = createRandomVector(input_dim_);
    RunResult result = orchestrator.execute(42, initial);
    
    EXPECT_TRUE(result.isSuccess());
    EXPECT_EQ(result.error_type, RunErrorType::NONE);
    EXPECT_GT(callback_count.load(), 0);
    EXPECT_EQ(last_run_id.load(), 42);
    EXPECT_GE(last_iteration.load(), 0);
}

TEST_F(RunOrchestratorTest, ProgressCallbackReceivesDecreasingEntropy) {
    std::vector<double> entropy_values;
    
    auto callback = [&](int run_id, int iter, double entropy) {
        entropy_values.push_back(entropy);
    };
    
    config_.stopping.max_iterations = 50;
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_, callback);
    
    HostVector initial = createRandomVector(input_dim_);
    RunResult result = orchestrator.execute(0, initial);
    
    EXPECT_TRUE(result.isSuccess());
    EXPECT_GT(entropy_values.size(), 0);
    
    // Check that entropy generally decreases (allowing for small fluctuations)
    if (entropy_values.size() > 10) {
        double first_avg = 0.0;
        double last_avg = 0.0;
        
        for (size_t i = 0; i < 5; ++i) {
            first_avg += entropy_values[i];
            last_avg += entropy_values[entropy_values.size() - 5 + i];
        }
        
        first_avg /= 5.0;
        last_avg /= 5.0;
        
        EXPECT_LT(last_avg, first_avg);
    }
}

// ============================================================================
// Multiple Run Tests
// ============================================================================

TEST_F(RunOrchestratorTest, MultipleSequentialRuns) {
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    for (int run = 0; run < 3; ++run) {
        HostVector initial = createRandomVector(input_dim_);
        RunResult result = orchestrator.execute(run, initial);
        
        EXPECT_TRUE(result.isSuccess());
        EXPECT_EQ(result.run_id, run);
    }
}

TEST_F(RunOrchestratorTest, DifferentInitialVectors) {
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    HostVector initial1 = createRandomVector(input_dim_);
    HostVector initial2 = createRandomVector(input_dim_);
    
    RunResult result1 = orchestrator.execute(0, initial1);
    RunResult result2 = orchestrator.execute(1, initial2);
    
    EXPECT_TRUE(result1.isSuccess());
    EXPECT_TRUE(result2.isSuccess());
    
    // Results might differ based on initial vectors
    // Just verify both completed successfully
}

// ============================================================================
// Target Entropy Tests
// ============================================================================

TEST_F(RunOrchestratorTest, TargetEntropyReached) {
    config_.stopping.target_entropy = 100.0;  // High target, easy to reach
    config_.stopping.max_iterations = 1000;
    
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    HostVector initial = createRandomVector(input_dim_);
    RunResult result = orchestrator.execute(0, initial);
    
    EXPECT_TRUE(result.isSuccess());
    // Should reach target quickly
    EXPECT_LT(result.iterations_taken, 1000);
}

// ============================================================================
// Result Validation Tests
// ============================================================================

TEST_F(RunOrchestratorTest, ResultContainsValidData) {
    config_.stopping.max_iterations = 50;
    
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    HostVector initial = createRandomVector(input_dim_);
    RunResult result = orchestrator.execute(123, initial);
    
    EXPECT_TRUE(result.isSuccess());
    EXPECT_EQ(result.run_id, 123);
    EXPECT_GT(result.iterations_taken, 0);
    EXPECT_LE(result.iterations_taken, 50);
    EXPECT_GT(result.runtime_seconds, 0.0);
    EXPECT_LT(result.runtime_seconds, 60.0);  // Should complete in under a minute
    EXPECT_EQ(result.final_vector.dimension, input_dim_);
    EXPECT_EQ(result.final_vector.data.size(), static_cast<size_t>(input_dim_));
    
    // Final vector should be normalized
    double norm = 0.0;
    for (const auto& val : result.final_vector.data) {
        norm += std::norm(val);
    }
    EXPECT_NEAR(norm, 1.0, 1e-6);
}

TEST_F(RunOrchestratorTest, FinalEntropyIsFinite) {
    RunOrchestrator orchestrator(config_, kraus_ops_, input_dim_, *cpu_device_);
    
    HostVector initial = createRandomVector(input_dim_);
    RunResult result = orchestrator.execute(0, initial);
    
    EXPECT_TRUE(result.isSuccess());
    EXPECT_TRUE(std::isfinite(result.final_entropy));
    EXPECT_FALSE(std::isnan(result.final_entropy));
    EXPECT_FALSE(std::isinf(result.final_entropy));
}
