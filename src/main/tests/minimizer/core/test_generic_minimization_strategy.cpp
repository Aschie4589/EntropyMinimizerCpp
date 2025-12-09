#include <gtest/gtest.h>
#include "minimizer/algorithm/generic_minimization_strategy.h"
#include "minimizer/algorithm/cuda_minimization_strategy.h"
#include "compute/device/DeviceFactory.h"
#include <complex>
#include <vector>
#include <cmath>
#include <memory>

using namespace std::complex_literals;

// ============================================================================
// Test Fixtures
// ============================================================================

class GenericMinimizationStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create device (will auto-select CUDA if available, else CPU)
        device_ = DeviceFactory::create(DeviceFactory::DeviceType::AUTO);
        ASSERT_NE(device_, nullptr);
    }
    
    std::unique_ptr<IComputeDevice> device_;
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, ConstructorValidDevice) {
    EXPECT_NO_THROW(GenericMinimizationStrategy strategy(*device_));
}

TEST_F(GenericMinimizationStrategyTest, ConstructorCustomStreams) {
    EXPECT_NO_THROW(GenericMinimizationStrategy strategy(*device_, 8));
}

TEST_F(GenericMinimizationStrategyTest, ConstructorInvalidStreams) {
    EXPECT_THROW(
        GenericMinimizationStrategy strategy(*device_, 0),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        GenericMinimizationStrategy strategy(*device_, -1),
        std::invalid_argument
    );
}

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, InitializationDouble) {
    GenericMinimizationStrategy strategy(*device_);
    
    // Create simple 2×2 identity channel (d=1, N=2, M=2)
    int d = 1, N = 2, M = 2;
    std::vector<std::complex<double>> kraus_data{
        1.0 + 0.0i, 0.0 + 0.0i,
        0.0 + 0.0i, 1.0 + 0.0i
    };
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{1.0 + 0.0i, 0.0 + 0.0i};
    
    EXPECT_NO_THROW(strategy.initialize(kraus, 0.01, initial_vec));
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::DOUBLE);
}

TEST_F(GenericMinimizationStrategyTest, InitializationFloat) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 1, N = 2, M = 2;
    std::vector<std::complex<float>> kraus_data{
        {1.0f, 0.0f}, {0.0f, 0.0f},
        {0.0f, 0.0f}, {1.0f, 0.0f}
    };
    
    HostKrausOperators kraus = HostKrausOperators::fromFloat(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{1.0 + 0.0i, 0.0 + 0.0i};
    
    EXPECT_NO_THROW(strategy.initialize(kraus, 0.01, initial_vec));
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::FLOAT);
}

TEST_F(GenericMinimizationStrategyTest, InitializationInvalidDimensions) {
    GenericMinimizationStrategy strategy(*device_);
    
    // d > M violates SVD constraint
    int d = 5, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    EXPECT_THROW(
        strategy.initialize(kraus, 0.01, initial_vec),
        std::runtime_error
    );
}

TEST_F(GenericMinimizationStrategyTest, InitializationInvalidVectorSize) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> wrong_size(N + 1, 1.0 + 0.0i);
    
    EXPECT_THROW(
        strategy.initialize(kraus, 0.01, wrong_size),
        std::runtime_error
    );
}

TEST_F(GenericMinimizationStrategyTest, InitializationInvalidEpsilon) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 1, N = 2, M = 2;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.5 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    EXPECT_THROW(strategy.initialize(kraus, 0.0, initial_vec), std::runtime_error);
    EXPECT_THROW(strategy.initialize(kraus, 1.0, initial_vec), std::runtime_error);
    EXPECT_THROW(strategy.initialize(kraus, -0.1, initial_vec), std::runtime_error);
}

// ============================================================================
// Step Execution Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, StepOnceWithoutInitializeThrows) {
    GenericMinimizationStrategy strategy(*device_);
    
    EXPECT_THROW(strategy.stepOnce(), std::runtime_error);
}

TEST_F(GenericMinimizationStrategyTest, StepOnceIdentityChannel) {
    GenericMinimizationStrategy strategy(*device_);
    
    // Identity channel should preserve entropy
    int d = 1, N = 3, M = 3;
    std::vector<std::complex<double>> kraus_data{
        1.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i,
        0.0 + 0.0i, 1.0 + 0.0i, 0.0 + 0.0i,
        0.0 + 0.0i, 0.0 + 0.0i, 1.0 + 0.0i
    };
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{0.707 + 0.0i, 0.707 + 0.0i, 0.0 + 0.0i};
    
    strategy.initialize(kraus, 0.01, initial_vec);
    double entropy_before = strategy.getEntropy();
    
    EXPECT_NO_THROW(strategy.stepOnce());
    
    double entropy_after = strategy.getEntropy();
    
    // Entropy should be small (pure state preserved by identity)
    EXPECT_LE(entropy_after, 0.1);
}

TEST_F(GenericMinimizationStrategyTest, MultipleStepsDecreasesEntropy) {
    GenericMinimizationStrategy strategy(*device_);
    
    // Create a depolarizing channel (d=2, N=2, M=2)
    int d = 2, N = 2, M = 2;
    double alpha = 1.0 / std::sqrt(2.0);
    
    std::vector<std::complex<double>> kraus_data{
        // K_0 = alpha * I
        alpha, 0.0, 0.0, alpha,
        // K_1 = alpha * X
        0.0, alpha, alpha, 0.0
    };
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{0.6 + 0.0i, 0.8 + 0.0i};
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    std::vector<double> entropies;
    entropies.push_back(strategy.getEntropy());
    
    // Take 5 steps
    for (int i = 0; i < 5; ++i) {
        strategy.stepOnce();
        entropies.push_back(strategy.getEntropy());
    }
    
    // Check that entropy generally decreases or stabilizes
    EXPECT_LE(entropies.back(), entropies.front() + 0.1);
}

TEST_F(GenericMinimizationStrategyTest, EntropyMonotonicDecrease) {
    GenericMinimizationStrategy strategy(*device_);
    
    // 2-qubit depolarizing channel
    int d = 2, N = 4, M = 4;
    double p = 0.9;
    double alpha0 = std::sqrt(p);
    double alpha_other = std::sqrt((1.0 - p));
    
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.0);
    
    // K_0 = sqrt(p) * I
    for (int i = 0; i < N; ++i) {
        kraus_data[0 * M * N + i * M + i] = alpha0;
    }
    
    // K_1 = sqrt(1-p) * X⊗X (simplified as diagonal)
    for (int i = 0; i < N; ++i) {
        kraus_data[1 * M * N + i * M + i] = alpha_other;
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 0.0);
    initial_vec[0] = 0.5;
    initial_vec[1] = 0.5;
    initial_vec[2] = 0.5;
    initial_vec[3] = 0.5;
    
    strategy.initialize(kraus, 0.001, initial_vec);
    
    double prev_entropy = strategy.getEntropy();
    
    for (int i = 0; i < 10; ++i) {
        strategy.stepOnce();
        double curr_entropy = strategy.getEntropy();
        
        // Allow small numerical fluctuations but general decrease
        EXPECT_LE(curr_entropy, prev_entropy + 1e-6) 
            << "Entropy increased at iteration " << i;
        
        prev_entropy = curr_entropy;
    }
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, GetEntropyBeforeInitializeThrows) {
    GenericMinimizationStrategy strategy(*device_);
    
    EXPECT_THROW(strategy.getEntropy(), std::runtime_error);
}

TEST_F(GenericMinimizationStrategyTest, GetCurrentVectorWorks) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 1, N = 3, M = 3;
    std::vector<std::complex<double>> kraus_data(d * M * N);
    for (int i = 0; i < M * N; ++i) {
        kraus_data[i] = (i == 0 || i == 4 || i == 8) ? 1.0 + 0.0i : 0.0 + 0.0i;
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{1.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i};
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    std::vector<std::complex<double>> retrieved_vec(N);
    EXPECT_NO_THROW(strategy.getCurrentVector(retrieved_vec));
    
    // Vector should be approximately the initial vector
    EXPECT_NEAR(std::abs(retrieved_vec[0]), 1.0, 0.1);
}

TEST_F(GenericMinimizationStrategyTest, GetCurrentVectorWrongSizeThrows) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 1, N = 2, M = 2;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.5 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    std::vector<std::complex<double>> wrong_size(N + 1);
    EXPECT_THROW(strategy.getCurrentVector(wrong_size), std::runtime_error);
}

TEST_F(GenericMinimizationStrategyTest, GetCurrentVectorFloatConversion) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 1, N = 2, M = 2;
    std::vector<std::complex<float>> kraus_data{
        {1.0f, 0.0f}, {0.0f, 0.0f},
        {0.0f, 0.0f}, {1.0f, 0.0f}
    };
    
    HostKrausOperators kraus = HostKrausOperators::fromFloat(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{1.0 + 0.0i, 0.0 + 0.0i};
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    std::vector<std::complex<double>> retrieved_vec(N);
    EXPECT_NO_THROW(strategy.getCurrentVector(retrieved_vec));
    
    // Should correctly convert from float to double
    EXPECT_NEAR(std::abs(retrieved_vec[0]), 1.0, 0.01);
}

TEST_F(GenericMinimizationStrategyTest, VectorNormalizationPreserved) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 2, N = 3, M = 3;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    for (int k = 0; k < d; ++k) {
        for (int i = 0; i < M; ++i) {
            kraus_data[k * M * N + i * M + i] = 0.7;
        }
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{0.577 + 0.0i, 0.577 + 0.0i, 0.577 + 0.0i};
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    for (int step = 0; step < 5; ++step) {
        strategy.stepOnce();
        
        std::vector<std::complex<double>> vec(N);
        strategy.getCurrentVector(vec);
        
        // Check normalization
        double norm_sq = 0.0;
        for (const auto& v : vec) {
            norm_sq += std::norm(v);
        }
        
        EXPECT_NEAR(norm_sq, 1.0, 1e-6) 
            << "Norm not preserved at step " << step;
    }
}

// ============================================================================
// Precision Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, PrecisionPersistence) {
    GenericMinimizationStrategy strategy_float(*device_);
    GenericMinimizationStrategy strategy_double(*device_);
    
    int d = 1, N = 2, M = 2;
    
    std::vector<std::complex<float>> kraus_float(d * M * N, {0.5f, 0.0f});
    std::vector<std::complex<double>> kraus_double(d * M * N, 0.5 + 0.0i);
    
    HostKrausOperators kraus_f = HostKrausOperators::fromFloat(kraus_float, d, N, M);
    HostKrausOperators kraus_d = HostKrausOperators::fromDouble(kraus_double, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    strategy_float.initialize(kraus_f, 0.01, initial_vec);
    strategy_double.initialize(kraus_d, 0.01, initial_vec);
    
    EXPECT_EQ(strategy_float.getPrecision(), PrecisionType::FLOAT);
    EXPECT_EQ(strategy_double.getPrecision(), PrecisionType::DOUBLE);
}

TEST_F(GenericMinimizationStrategyTest, FloatVsDoubleConsistency) {
    GenericMinimizationStrategy strategy_float(*device_);
    GenericMinimizationStrategy strategy_double(*device_);
    
    int d = 2, N = 3, M = 3;
    
    std::vector<std::complex<float>> kraus_float(d * M * N);
    std::vector<std::complex<double>> kraus_double(d * M * N);
    
    // Same data in both precisions
    for (int i = 0; i < d * M * N; ++i) {
        double val = 0.3 + 0.1 * (i % 3);
        kraus_float[i] = {static_cast<float>(val), 0.0f};
        kraus_double[i] = {val, 0.0};
    }
    
    HostKrausOperators kraus_f = HostKrausOperators::fromFloat(kraus_float, d, N, M);
    HostKrausOperators kraus_d = HostKrausOperators::fromDouble(kraus_double, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N, 0.577 + 0.0i);
    
    strategy_float.initialize(kraus_f, 0.01, initial_vec);
    strategy_double.initialize(kraus_d, 0.01, initial_vec);
    
    // Run a few steps
    for (int i = 0; i < 3; ++i) {
        strategy_float.stepOnce();
        strategy_double.stepOnce();
    }
    
    double entropy_float = strategy_float.getEntropy();
    double entropy_double = strategy_double.getEntropy();
    
    // Should be close (within float precision)
    EXPECT_NEAR(entropy_float, entropy_double, 1e-4);
}

// ============================================================================
// Memory Requirement Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, GetMemoryRequiredScalesWithDimensions) {
    GenericMinimizationStrategy strategy(*device_);
    
    size_t mem_small = strategy.getMemoryRequired(2, 3, 4, PrecisionType::DOUBLE);
    size_t mem_large = strategy.getMemoryRequired(4, 6, 8, PrecisionType::DOUBLE);
    
    // Larger problem should require more memory
    EXPECT_GT(mem_large, mem_small);
}

TEST_F(GenericMinimizationStrategyTest, GetMemoryRequiredFloatVsDouble) {
    GenericMinimizationStrategy strategy(*device_);
    
    size_t mem_float = strategy.getMemoryRequired(2, 3, 4, PrecisionType::FLOAT);
    size_t mem_double = strategy.getMemoryRequired(2, 3, 4, PrecisionType::DOUBLE);
    
    // Double should use approximately 2x memory
    EXPECT_GT(mem_double, mem_float);
    EXPECT_LT(mem_double, 2.5 * mem_float);  // Within reasonable factor
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, LargeDimensionSystem) {
    GenericMinimizationStrategy strategy(*device_);
    
    // Larger system: 8-dimensional Hilbert space
    int d = 3, N = 8, M = 8;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.0);
    
    // Create valid channel
    double scale = 1.0 / std::sqrt(static_cast<double>(d));
    for (int k = 0; k < d; ++k) {
        for (int i = 0; i < M; ++i) {
            kraus_data[k * M * N + i * M + i] = scale;
        }
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> initial_vec(N);
    double norm = 0.0;
    for (int i = 0; i < N; ++i) {
        initial_vec[i] = 1.0 / std::sqrt(static_cast<double>(N));
        norm += std::norm(initial_vec[i]);
    }
    
    EXPECT_NO_THROW(strategy.initialize(kraus, 0.001, initial_vec));
    EXPECT_NO_THROW(strategy.stepOnce());
    
    double entropy = strategy.getEntropy();
    EXPECT_GE(entropy, 0.0);
    EXPECT_LT(entropy, std::log(N));
}

TEST_F(GenericMinimizationStrategyTest, SingleDimensionSystem) {
    GenericMinimizationStrategy strategy(*device_);
    
    // Minimal system: 1D
    int d = 1, N = 1, M = 1;
    std::vector<std::complex<double>> kraus_data{1.0 + 0.0i};
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{1.0 + 0.0i};
    
    EXPECT_NO_THROW(strategy.initialize(kraus, 0.01, initial_vec));
    EXPECT_NO_THROW(strategy.stepOnce());
    
    // 1D system should have zero entropy
    EXPECT_NEAR(strategy.getEntropy(), 0.0, 1e-10);
}

TEST_F(GenericMinimizationStrategyTest, VerySmallEpsilon) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 2, N = 3, M = 3;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.5 + 0.0i);
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec(N, 0.577 + 0.0i);
    
    // Very small epsilon (but still valid)
    EXPECT_NO_THROW(strategy.initialize(kraus, 1e-10, initial_vec));
    EXPECT_NO_THROW(strategy.stepOnce());
}

TEST_F(GenericMinimizationStrategyTest, HighEpsilon) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 2, N = 3, M = 3;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.5 + 0.0i);
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec(N, 0.577 + 0.0i);
    
    // High epsilon (close to 1)
    EXPECT_NO_THROW(strategy.initialize(kraus, 0.99, initial_vec));
    EXPECT_NO_THROW(strategy.stepOnce());
}

TEST_F(GenericMinimizationStrategyTest, ComplexInitialState) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 2, N = 4, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.5 + 0.0i);
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    // Complex initial state with phases
    std::vector<std::complex<double>> initial_vec{
        0.5 + 0.0i,
        0.0 + 0.5i,
        0.5 + 0.0i,
        0.0 - 0.5i
    };
    
    EXPECT_NO_THROW(strategy.initialize(kraus, 0.01, initial_vec));
    EXPECT_NO_THROW(strategy.stepOnce());
    
    std::vector<std::complex<double>> vec(N);
    strategy.getCurrentVector(vec);
    
    // Should still be normalized
    double norm_sq = 0.0;
    for (const auto& v : vec) {
        norm_sq += std::norm(v);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-6);
}

TEST_F(GenericMinimizationStrategyTest, ManyIterationsStability) {
    GenericMinimizationStrategy strategy(*device_);
    
    int d = 2, N = 3, M = 3;
    double scale = 1.0 / std::sqrt(2.0);
    
    std::vector<std::complex<double>> kraus_data{
        scale, 0.0, 0.0,  0.0, scale, 0.0,  0.0, 0.0, scale,
        0.0, scale, 0.0,  scale, 0.0, 0.0,  0.0, 0.0, scale
    };
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec{0.577 + 0.0i, 0.577 + 0.0i, 0.577 + 0.0i};
    
    strategy.initialize(kraus, 0.001, initial_vec);
    
    // Run many iterations - should remain stable
    for (int i = 0; i < 50; ++i) {
        EXPECT_NO_THROW(strategy.stepOnce()) 
            << "Failed at iteration " << i;
        
        double entropy = strategy.getEntropy();
        EXPECT_FALSE(std::isnan(entropy)) 
            << "NaN entropy at iteration " << i;
        EXPECT_FALSE(std::isinf(entropy)) 
            << "Inf entropy at iteration " << i;
    }
}

// ============================================================================
// Multi-Stream Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, DifferentStreamCountsWork) {
    for (int num_streams : {1, 2, 4, 8, 16}) {
        GenericMinimizationStrategy strategy(*device_, num_streams);
        
        int d = 4, N = 4, M = 4;
        std::vector<std::complex<double>> kraus_data(d * M * N, 0.5 + 0.0i);
        
        HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
        std::vector<std::complex<double>> initial_vec(N, 0.5 + 0.0i);
        
        EXPECT_NO_THROW(strategy.initialize(kraus, 0.01, initial_vec))
            << "Failed with " << num_streams << " streams";
        EXPECT_NO_THROW(strategy.stepOnce())
            << "Failed with " << num_streams << " streams";
    }
}

// ============================================================================
// Comparison with CUDA Strategy (if available)
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, ConsistencyWithCudaStrategy) {
    // Only run if CUDA is available
    if (device_->getBackend() != DeviceBackend::CUDA) {
        GTEST_SKIP() << "CUDA not available, skipping comparison test";
    }
    
    GenericMinimizationStrategy generic_strategy(*device_);
    CudaMinimizationStrategy cuda_strategy(*device_);
    
    int d = 2, N = 4, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N);
    
    for (int i = 0; i < d * M * N; ++i) {
        kraus_data[i] = 0.3 + 0.1 * std::sin(i);
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec(N, 0.5 + 0.0i);
    
    generic_strategy.initialize(kraus, 0.01, initial_vec);
    cuda_strategy.initialize(kraus, 0.01, initial_vec);
    
    // Run several steps
    for (int i = 0; i < 5; ++i) {
        generic_strategy.stepOnce();
        cuda_strategy.stepOnce();
    }
    
    double entropy_generic = generic_strategy.getEntropy();
    double entropy_cuda = cuda_strategy.getEntropy();
    
    // Should produce very similar results
    EXPECT_NEAR(entropy_generic, entropy_cuda, 1e-6);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
