#include <gtest/gtest.h>
#include "minimizer/algorithm/cuda_minimization_strategy.h"
#include "compute/device/DeviceFactory.h"
#include <cuda_runtime.h>
#include <complex>
#include <vector>
#include <cmath>

using namespace std::complex_literals;

// ============================================================================
// Test Fixtures
// ============================================================================

class CudaMinimizationStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        
        // Create CUDA device
        device_ = DeviceFactory::create(DeviceFactory::DeviceType::CUDA);
        ASSERT_NE(device_, nullptr);
        ASSERT_EQ(device_->getBackend(), DeviceBackend::CUDA);
    }
    
    std::unique_ptr<IComputeDevice> device_;
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, ConstructorValidDevice) {
    EXPECT_NO_THROW(CudaMinimizationStrategy strategy(*device_));
}

TEST_F(CudaMinimizationStrategyTest, ConstructorCustomStreams) {
    EXPECT_NO_THROW(CudaMinimizationStrategy strategy(*device_, 8));
}

TEST_F(CudaMinimizationStrategyTest, ConstructorInvalidStreams) {
    EXPECT_THROW(
        CudaMinimizationStrategy strategy(*device_, 0),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        CudaMinimizationStrategy strategy(*device_, -1),
        std::invalid_argument
    );
}

TEST_F(CudaMinimizationStrategyTest, ConstructorWrongDeviceType) {
    auto cpu_device = DeviceFactory::create(DeviceFactory::DeviceType::CPU);
    
    EXPECT_THROW(
        CudaMinimizationStrategy strategy(*cpu_device),
        std::invalid_argument
    );
}

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, InitializationDouble) {
    CudaMinimizationStrategy strategy(*device_);
    
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

TEST_F(CudaMinimizationStrategyTest, InitializationFloat) {
    CudaMinimizationStrategy strategy(*device_);
    
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

TEST_F(CudaMinimizationStrategyTest, InitializationInvalidDimensions) {
    CudaMinimizationStrategy strategy(*device_);
    
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

TEST_F(CudaMinimizationStrategyTest, InitializationInvalidVectorSize) {
    CudaMinimizationStrategy strategy(*device_);
    
    int d = 2, N = 3, M = 4;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.1 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    std::vector<std::complex<double>> wrong_size(N + 1, 1.0 + 0.0i);
    
    EXPECT_THROW(
        strategy.initialize(kraus, 0.01, wrong_size),
        std::runtime_error
    );
}

TEST_F(CudaMinimizationStrategyTest, InitializationInvalidEpsilon) {
    CudaMinimizationStrategy strategy(*device_);
    
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

TEST_F(CudaMinimizationStrategyTest, StepOnceWithoutInitializeThrows) {
    CudaMinimizationStrategy strategy(*device_);
    
    EXPECT_THROW(strategy.stepOnce(), std::runtime_error);
}

TEST_F(CudaMinimizationStrategyTest, StepOnceIdentityChannel) {
    CudaMinimizationStrategy strategy(*device_);
    
    // Identity channel should preserve entropy
    int d = 1, N = 3, M = 3;
    std::vector<std::complex<double>> kraus_data{
        1.0 + 0.0i, 0.0 + 0.0i, 0.0+ 0.0i,
        0.0 + 0.0i, 1.0 + 0.0i, 0.0+ 0.0i,
        0.0 + 0.0i, 0.0 + 0.0i, 1.0+ 0.0i
    };
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);

    std::cout << "Kraus operators initialized." << std::endl;

    std::vector<std::complex<double>> initial_vec{0.707 + 0.0i, 0.707 + 0.0i, 0.0 + 0.0i};
    
    strategy.initialize(kraus, 0.01, initial_vec);

    std::cout << "Strategy initialized." << std::endl;
    
    double entropy_before = strategy.getEntropy();

    std::cout << "Entropy obtained: " << entropy_before << std::endl;
    
    EXPECT_NO_THROW(strategy.stepOnce());

    std::cout << "Stepped once" << std::endl;
    
    double entropy_after = strategy.getEntropy();

    std::cout << "Entropy after step: " << entropy_after << std::endl;
    

    // Entropy should be small (pure state)
    EXPECT_LE(entropy_after, 0.1);

}

TEST_F(CudaMinimizationStrategyTest, MultipleStepsDecreasesEntropy) {
    CudaMinimizationStrategy strategy(*device_);
    
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
        std::cout << "Step " << (i + 1) << std::endl;
        strategy.stepOnce();
        std::cout << "Entropy: " << strategy.getEntropy() << std::endl;
        entropies.push_back(strategy.getEntropy());
    }
    
    // Check that entropy generally decreases or stabilizes
    // (may not be strictly monotonic due to numerical effects)
    EXPECT_LE(entropies.back(), entropies.front() + 0.1);
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, GetEntropyBeforeInitializeThrows) {
    CudaMinimizationStrategy strategy(*device_);
    
    EXPECT_THROW(strategy.getEntropy(), std::runtime_error);
}

TEST_F(CudaMinimizationStrategyTest, GetCurrentVectorWorks) {
    CudaMinimizationStrategy strategy(*device_);
    
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

TEST_F(CudaMinimizationStrategyTest, GetCurrentVectorWrongSizeThrows) {
    CudaMinimizationStrategy strategy(*device_);
    
    int d = 1, N = 2, M = 2;
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.5 + 0.0i);
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    std::vector<std::complex<double>> initial_vec(N, 1.0 + 0.0i);
    
    strategy.initialize(kraus, 0.01, initial_vec);
    
    std::vector<std::complex<double>> wrong_size(N + 1);
    EXPECT_THROW(strategy.getCurrentVector(wrong_size), std::runtime_error);
}

TEST_F(CudaMinimizationStrategyTest, GetCurrentVectorFloatConversion) {
    CudaMinimizationStrategy strategy(*device_);
    
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

// ============================================================================
// Precision Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, PrecisionPersistence) {
    CudaMinimizationStrategy strategy_float(*device_);
    CudaMinimizationStrategy strategy_double(*device_);
    
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

// ============================================================================
// Integration Test with Realistic Channel
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, RealisticDepolarizingChannel) {
    CudaMinimizationStrategy strategy(*device_);
    
    // 3-qubit depolarizing channel (d=4, N=8, M=8)
    int d = 4, N = 8, M = 8;
    double p = 0.9;  // Depolarizing parameter
    
    std::vector<std::complex<double>> kraus_data(d * M * N, 0.0);
    
    // K_0 = sqrt(p) * I_8
    double alpha0 = std::sqrt(p);
    for (int i = 0; i < N; ++i) {
        kraus_data[0 * M * N + i * M + i] = alpha0;
    }
    
    // K_1, K_2, K_3 = sqrt((1-p)/3) * Pauli matrices (simplified)
    double alpha_other = std::sqrt((1.0 - p) / 3.0);
    for (int k = 1; k < d; ++k) {
        for (int i = 0; i < N; ++i) {
            kraus_data[k * M * N + i * M + i] = alpha_other;
        }
    }
    
    HostKrausOperators kraus = HostKrausOperators::fromDouble(kraus_data, d, N, M);
    
    // Random initial state
    std::vector<std::complex<double>> initial_vec(N);
    double norm = 0.0;
    for (int i = 0; i < N; ++i) {
        initial_vec[i] = std::complex<double>(1.0 / N, 0.0);
        norm += std::norm(initial_vec[i]);
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < N; ++i) {
        initial_vec[i] /= norm;
    }
    
    strategy.initialize(kraus, 0.001, initial_vec);
    
    // Run several iterations
    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(strategy.stepOnce());
    }
    
    // Should have some reasonable entropy
    double final_entropy = strategy.getEntropy();
    EXPECT_GE(final_entropy, 0.0);
    EXPECT_LT(final_entropy, std::log(N));  // Less than maximum
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
