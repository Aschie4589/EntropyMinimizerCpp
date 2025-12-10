#include <gtest/gtest.h>
#include "minimizer/algorithm/strategy_factory.h"
#include "compute/device/DeviceFactory.h"
#include <complex>
#include <limits>

using namespace entropy;

// ============================================================================
// Test Fixture
// ============================================================================

class StrategyFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create CPU device for testing
        cpu_device_ = DeviceFactory::create(DeviceFactory::DeviceType::CPU);
        ASSERT_NE(cpu_device_, nullptr);
        
        // Try to create CUDA device (may fail if no GPU available)
        try {
            cuda_device_ = DeviceFactory::create(DeviceFactory::DeviceType::CUDA, 0);
            has_cuda_ = (cuda_device_ != nullptr);
        } catch (...) {
            has_cuda_ = false;
            cuda_device_ = nullptr;
        }
        
        // Set standard test dimensions (ensure d² <= N constraint)
        kraus_count_ = 2;   // d=2, so d²=4
        input_dim_ = 8;     // N=8, satisfies d²=4 <= N=8
        output_dim_ = 8;    // M=8
    }
    
    std::unique_ptr<IComputeDevice> cpu_device_;
    std::unique_ptr<IComputeDevice> cuda_device_;
    bool has_cuda_ = false;
    
    int kraus_count_;
    int input_dim_;
    int output_dim_;
};

// ============================================================================
// Memory Estimation Tests
// ============================================================================

TEST_F(StrategyFactoryTest, estimateWorkspaceSize_float_vs_double) {
    // Double precision should use exactly 2x memory of float
    size_t float_size = StrategyFactory::estimateWorkspaceSize(
        *cpu_device_, kraus_count_, input_dim_, output_dim_, PrecisionType::FLOAT
    );
    
    size_t double_size = StrategyFactory::estimateWorkspaceSize(
        *cpu_device_, kraus_count_, input_dim_, output_dim_, PrecisionType::DOUBLE
    );
    
    // Double precision: complex<double> is 16 bytes, complex<float> is 8 bytes
    // Real: double is 8 bytes, float is 4 bytes
    // Both should scale by factor of 2
    EXPECT_EQ(double_size, 2 * float_size);
}

TEST_F(StrategyFactoryTest, estimateWorkspaceSize_scaling) {
    // Test that memory scales correctly with problem dimensions
    
    // Base case: d=2, N=8, M=4 (satisfies d²=4 <= N=8)
    size_t base = StrategyFactory::estimateWorkspaceSize(
        *cpu_device_, 2, 8, 4, PrecisionType::DOUBLE
    );
    
    // Double kraus_count (d → 2d = 4, so d²=16)
    // Need N >= 16 to satisfy constraint
    // Memory ≈ 3×d×M×c + 2×d²×N×c (d² term dominates)
    size_t double_d = StrategyFactory::estimateWorkspaceSize(
        *cpu_device_, 4, 16, 4, PrecisionType::DOUBLE
    );
    
    // Doubling d should approximately quadruple memory (d² scaling in vecs2)
    // But we also doubled N (from 8 to 16), so actual increase is higher
    // Allow tolerance for the combined effect
    EXPECT_GT(double_d, 3.0 * base);  // Should be at least 3x
    EXPECT_LT(double_d, 6.0 * base);  // But less than 6x
}

TEST_F(StrategyFactoryTest, estimateWorkspaceSize_dimensions_positive) {
    // All dimensions must be positive
    EXPECT_THROW(
        StrategyFactory::estimateWorkspaceSize(*cpu_device_, 0, input_dim_, output_dim_, PrecisionType::DOUBLE),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        StrategyFactory::estimateWorkspaceSize(*cpu_device_, kraus_count_, 0, output_dim_, PrecisionType::DOUBLE),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        StrategyFactory::estimateWorkspaceSize(*cpu_device_, kraus_count_, input_dim_, 0, PrecisionType::DOUBLE),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        StrategyFactory::estimateWorkspaceSize(*cpu_device_, -1, input_dim_, output_dim_, PrecisionType::DOUBLE),
        std::invalid_argument
    );
}

// ============================================================================
// Memory Query Tests
// ============================================================================

TEST_F(StrategyFactoryTest, getAvailableMemory_cpu) {
    // CPU backend should return SIZE_MAX (no practical limit)
    size_t available = StrategyFactory::getAvailableMemory(*cpu_device_);
    EXPECT_EQ(available, std::numeric_limits<size_t>::max());
}

TEST_F(StrategyFactoryTest, getAvailableMemory_cuda) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    // CUDA backend should return actual GPU memory (non-zero, less than SIZE_MAX)
    size_t available = StrategyFactory::getAvailableMemory(*cuda_device_);
    EXPECT_GT(available, 0);
    EXPECT_LT(available, std::numeric_limits<size_t>::max());
    
    // Should be reasonable GPU memory size (at least 100 MB, less than 1 TB)
    EXPECT_GT(available, 100 * 1024 * 1024);         // > 100 MB
    EXPECT_LT(available, 1024ULL * 1024 * 1024 * 1024); // < 1 TB
}

// ============================================================================
// Factory Creation Tests - AUTO Mode
// ============================================================================

TEST_F(StrategyFactoryTest, create_auto_cpu_returns_generic) {
    // AUTO on CPU should always return GenericMinimizationStrategy
    auto strategy = StrategyFactory::create(
        *cpu_device_,
        kraus_count_,
        input_dim_,
        output_dim_,
        PrecisionType::DOUBLE
    );
    
    ASSERT_NE(strategy, nullptr);
    
    // Verify it was initialized correctly (can get workspace size)
    // Note: Strategy needs initialize() before getWorkspaceSize()
    strategy->initialize(kraus_count_, input_dim_, output_dim_, 0.01, PrecisionType::DOUBLE);
    EXPECT_GT(strategy->getWorkspaceSize(), 0);
}

TEST_F(StrategyFactoryTest, create_auto_cuda_sufficient) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    // Use small dimensions to ensure memory fits
    int small_d = 2;
    int small_N = 4;
    int small_M = 4;
    
    // AUTO on CUDA with sufficient memory should work
    auto strategy = StrategyFactory::create(
        *cuda_device_,
        small_d,
        small_N,
        small_M,
        PrecisionType::FLOAT  // Use float to minimize memory
    );
    
    ASSERT_NE(strategy, nullptr);
    
    // Initialize and verify workspace
    strategy->initialize(small_d, small_N, small_M, 0.01, PrecisionType::FLOAT);
    EXPECT_GT(strategy->getWorkspaceSize(), 0);
}

TEST_F(StrategyFactoryTest, create_auto_cuda_insufficient) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    // Use dimensions that would require massive memory (exceeds GPU capacity)
    // d=500, so d²=250000, need N >= 250000
    // Memory dominated by d²×N×c ≈ 250000×250000×16 = 1 TB for DOUBLE
    int huge_d = 500;
    int huge_N = 250000;
    int huge_M = 100;
    
    // AUTO should fallback to Generic (no throw)
    auto strategy = StrategyFactory::create(
        *cuda_device_,
        huge_d,
        huge_N,
        huge_M,
        PrecisionType::DOUBLE
    );
    
    // Should successfully create Generic fallback
    ASSERT_NE(strategy, nullptr);
    
    // Note: We don't initialize with huge dimensions as it would actually allocate
    // Just verify strategy was created (factory logic tested)
}

// ============================================================================
// Factory Creation Tests - Explicit GENERIC
// ============================================================================

TEST_F(StrategyFactoryTest, create_explicit_generic_cpu) {
    // Explicit GENERIC should always work on CPU
    auto strategy = StrategyFactory::create(
        StrategyType::GENERIC,
        *cpu_device_,
        kraus_count_,
        input_dim_,
        output_dim_,
        PrecisionType::DOUBLE
    );
    
    ASSERT_NE(strategy, nullptr);
    strategy->initialize(kraus_count_, input_dim_, output_dim_, 0.01, PrecisionType::DOUBLE);
    EXPECT_GT(strategy->getWorkspaceSize(), 0);
}

TEST_F(StrategyFactoryTest, create_explicit_generic_cuda) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    // Explicit GENERIC should work on CUDA too
    auto strategy = StrategyFactory::create(
        StrategyType::GENERIC,
        *cuda_device_,
        kraus_count_,
        input_dim_,
        output_dim_,
        PrecisionType::DOUBLE
    );
    
    ASSERT_NE(strategy, nullptr);
    strategy->initialize(kraus_count_, input_dim_, output_dim_, 0.01, PrecisionType::DOUBLE);
    EXPECT_GT(strategy->getWorkspaceSize(), 0);
}

// ============================================================================
// Factory Creation Tests - Explicit CUDA
// ============================================================================

TEST_F(StrategyFactoryTest, create_explicit_cuda_on_cpu_throws) {
    // CUDA strategy on CPU device should throw
    EXPECT_THROW(
        StrategyFactory::create(
            StrategyType::CUDA,
            *cpu_device_,
            kraus_count_,
            input_dim_,
            output_dim_,
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
}

TEST_F(StrategyFactoryTest, create_explicit_cuda_sufficient) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    // Small dimensions to ensure memory fits
    int small_d = 2;
    int small_N = 4;
    int small_M = 4;
    
    // Explicit CUDA with sufficient memory should succeed
    auto strategy = StrategyFactory::create(
        StrategyType::CUDA,
        *cuda_device_,
        small_d,
        small_N,
        small_M,
        PrecisionType::FLOAT
    );
    
    ASSERT_NE(strategy, nullptr);
    strategy->initialize(small_d, small_N, small_M, 0.01, PrecisionType::FLOAT);
    EXPECT_GT(strategy->getWorkspaceSize(), 0);
}

TEST_F(StrategyFactoryTest, create_explicit_cuda_insufficient) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    // Huge dimensions that exceed GPU memory
    // d=500, d²=250000, N=250000, memory ≈ 1 TB
    int huge_d = 500;
    int huge_N = 250000;
    int huge_M = 100;
    
    // Explicit CUDA with insufficient memory should throw
    EXPECT_THROW(
        StrategyFactory::create(
            StrategyType::CUDA,
            *cuda_device_,
            huge_d,
            huge_N,
            huge_M,
            PrecisionType::DOUBLE
        ),
        std::runtime_error
    );
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(StrategyFactoryTest, create_invalid_dimensions) {
    // Negative dimensions should throw
    EXPECT_THROW(
        StrategyFactory::create(
            *cpu_device_,
            -1,
            input_dim_,
            output_dim_,
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        StrategyFactory::create(
            *cpu_device_,
            kraus_count_,
            -1,
            output_dim_,
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        StrategyFactory::create(
            *cpu_device_,
            kraus_count_,
            input_dim_,
            -1,
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
    
    // Zero dimensions should throw
    EXPECT_THROW(
        StrategyFactory::create(
            *cpu_device_,
            0,
            input_dim_,
            output_dim_,
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
}

// ============================================================================
// Integration Test - Strategy Initialization
// ============================================================================

TEST_F(StrategyFactoryTest, created_strategy_initializes_correctly) {
    // Create strategy via factory
    auto strategy = StrategyFactory::create(
        *cpu_device_,
        kraus_count_,
        input_dim_,
        output_dim_,
        PrecisionType::DOUBLE
    );
    
    ASSERT_NE(strategy, nullptr);
    
    // Should be able to initialize
    EXPECT_NO_THROW(
        strategy->initialize(
            kraus_count_,
            input_dim_,
            output_dim_,
            0.01,
            PrecisionType::DOUBLE
        )
    );
    
    // Verify workspace size matches estimate
    size_t actual_workspace = strategy->getWorkspaceSize();
    size_t estimated_workspace = StrategyFactory::estimateWorkspaceSize(
        *cpu_device_,
        kraus_count_,
        input_dim_,
        output_dim_,
        PrecisionType::DOUBLE
    );
    
    // Actual may be different from estimate:
    // - Strategy might allocate more efficiently (smaller actual)
    // - Estimate includes SVD workspace which may vary by implementation
    // Just verify both are reasonable and non-zero
    EXPECT_GT(actual_workspace, 0);
    EXPECT_GT(estimated_workspace, 0);
    
    // They should be in the same ballpark (within 2x)
    EXPECT_LT(actual_workspace, estimated_workspace * 2.0);
    EXPECT_GT(estimated_workspace, actual_workspace * 0.5);
}
