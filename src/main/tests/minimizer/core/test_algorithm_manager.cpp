#include <gtest/gtest.h>
#include "minimizer/algorithm/algorithm_manager.h"
#include "minimizer/algorithm/cuda_minimization_strategy.h"
#include "minimizer/algorithm/generic_minimization_strategy.h"
#include "compute/device/DeviceFactory.h"
#include <cuda_runtime.h>
#include <complex>
#include <vector>
#include <cmath>

using namespace entropy;
using namespace std::complex_literals;

// ============================================================================
// Test Fixtures
// ============================================================================

class AlgorithmManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create CUDA device
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        
        cuda_device_ = DeviceFactory::create(DeviceFactory::DeviceType::CUDA);
        ASSERT_NE(cuda_device_, nullptr);
        
        // Create CPU device
        cpu_device_ = DeviceFactory::create(DeviceFactory::DeviceType::CPU);
        ASSERT_NE(cpu_device_, nullptr);
        
        // Create test Kraus operators (2×2 identity channel for simplicity)
        setupTestKraus();
        
        // Create entropy managers
        epsilon_ = 0.01;
        int system_dim = 5;  // output_dim for entropy calculation
        cuda_entropy_manager_ = std::make_unique<EntropyManager>(epsilon_, system_dim);
        cpu_entropy_manager_ = std::make_unique<EntropyManager>(epsilon_, system_dim);
    }
    
    void setupTestKraus() {
        // Create identity channel: 2 Kraus operators, 5×5
        // This ensures kraus_count^2 (4) <= input_dim (5)
        kraus_count_ = 2;
        input_dim_ = 5;
        output_dim_ = 5;
        
        std::vector<std::complex<double>> kraus_data(kraus_count_ * input_dim_ * output_dim_, std::complex<double>(0.0, 0.0));
        
        // K_0: Projector onto first basis state (column-major)
        kraus_data[0] = 1.0;  // (0,0) element
        
        // K_1: Projector onto second basis state (column-major)
        kraus_data[input_dim_ * output_dim_ + input_dim_ + 1] = 1.0;  // (1,1) element
        
        kraus_ = HostKrausOperators::fromDouble(kraus_data, kraus_count_, input_dim_, output_dim_);
    }
    
    std::unique_ptr<IComputeDevice> cuda_device_;
    std::unique_ptr<IComputeDevice> cpu_device_;
    std::unique_ptr<EntropyManager> cuda_entropy_manager_;
    std::unique_ptr<EntropyManager> cpu_entropy_manager_;
    
    HostKrausOperators kraus_;
    int kraus_count_;
    int input_dim_;
    int output_dim_;
    double epsilon_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(AlgorithmManagerTest, ConstructorValid) {
    EXPECT_NO_THROW(
        AlgorithmManager manager(
            *cuda_device_,
            *cuda_entropy_manager_,
            kraus_,
            epsilon_,
            PrecisionType::DOUBLE
        )
    );
}

TEST_F(AlgorithmManagerTest, ConstructorInvalidEpsilon) {
    EXPECT_THROW(
        AlgorithmManager manager(
            *cuda_device_,
            *cuda_entropy_manager_,
            kraus_,
            0.0,  // Invalid: must be > 0
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        AlgorithmManager manager(
            *cuda_device_,
            *cuda_entropy_manager_,
            kraus_,
            1.0,  // Invalid: must be < 1
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        AlgorithmManager manager(
            *cuda_device_,
            *cuda_entropy_manager_,
            kraus_,
            -0.1,  // Invalid: negative
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
}

TEST_F(AlgorithmManagerTest, ConstructorInvalidDimensions) {
    auto bad_kraus = kraus_;
    bad_kraus.kraus_count = 0;
    
    EXPECT_THROW(
        AlgorithmManager manager(
            *cuda_device_,
            *cuda_entropy_manager_,
            bad_kraus,
            epsilon_,
            PrecisionType::DOUBLE
        ),
        std::invalid_argument
    );
}

TEST_F(AlgorithmManagerTest, InitialState) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    EXPECT_FALSE(manager.isInitialized());
    EXPECT_EQ(manager.getPrecision(), PrecisionType::DOUBLE);
    EXPECT_EQ(manager.getKrausCount(), kraus_count_);
    EXPECT_EQ(manager.getInputDim(), input_dim_);
    EXPECT_EQ(manager.getOutputDim(), output_dim_);
}

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(AlgorithmManagerTest, InitializeWithoutStrategy) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    std::vector<std::complex<double>> initial_vec = {1.0, 0.0};
    
    EXPECT_THROW(manager.initialize(initial_vec), std::runtime_error);
}

TEST_F(AlgorithmManagerTest, InitializeValidVector) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    std::vector<std::complex<double>> initial_vec(input_dim_, 0.0);
    initial_vec[0] = 1.0;  // Normalized basis state
    
    EXPECT_NO_THROW(manager.initialize(initial_vec));
    EXPECT_TRUE(manager.isInitialized());
}

TEST_F(AlgorithmManagerTest, InitializeWrongVectorSize) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    std::vector<std::complex<double>> wrong_size(input_dim_ + 1, 0.0);  // Wrong size
    
    EXPECT_THROW(manager.initialize(wrong_size), std::invalid_argument);
}

TEST_F(AlgorithmManagerTest, InitializeRandom) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    EXPECT_NO_THROW(manager.initializeRandom());
    EXPECT_TRUE(manager.isInitialized());
    
    // Check that vector is normalized
    auto vec = manager.getCurrentVector();
    double norm_sq = 0.0;
    for (const auto& v : vec) {
        norm_sq += std::norm(v);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-10);
}

TEST_F(AlgorithmManagerTest, InitializeMultipleTimes) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    std::vector<std::complex<double>> vec1(input_dim_, 0.0);
    vec1[0] = 1.0;
    std::vector<std::complex<double>> vec2(input_dim_, 0.0);
    vec2[1] = 1.0;
    
    EXPECT_NO_THROW(manager.initialize(vec1));
    EXPECT_NO_THROW(manager.initialize(vec2));  // Re-initialization should work
    
    auto retrieved = manager.getCurrentVector();
    EXPECT_NEAR(std::abs(retrieved[0] - vec2[0]), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(retrieved[1] - vec2[1]), 0.0, 1e-10);
}

// ============================================================================
// StepOnce Tests
// ============================================================================

TEST_F(AlgorithmManagerTest, StepOnceBeforeInitialize) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    EXPECT_THROW(manager.stepOnce(), std::runtime_error);
}

TEST_F(AlgorithmManagerTest, StepOnceUpdatesEntropy) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    std::vector<std::complex<double>> initial_vec(input_dim_, 1.0 / std::sqrt(static_cast<double>(input_dim_)));
    manager.initialize(initial_vec);
    
    // Entropy should be updated after stepOnce
    EXPECT_NO_THROW(manager.stepOnce());
    
    double entropy = manager.getEpsilonEntropy();
    EXPECT_GE(entropy, 0.0);  // Entropy should be non-negative
}

TEST_F(AlgorithmManagerTest, MultipleSteps) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    manager.initializeRandom();
    
    // Execute multiple steps - should not throw
    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(manager.stepOnce());
    }
}

// ============================================================================
// State Access Tests (Getters/Setters)
// ============================================================================

TEST_F(AlgorithmManagerTest, GetCurrentVectorBeforeInitialize) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    EXPECT_THROW(manager.getCurrentVector(), std::runtime_error);
}

TEST_F(AlgorithmManagerTest, GetCurrentVectorAfterInitialize) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    std::vector<std::complex<double>> initial_vec(input_dim_, 0.0);
    initial_vec[0] = 1.0;
    manager.initialize(initial_vec);
    
    auto retrieved = manager.getCurrentVector();
    ASSERT_EQ(retrieved.size(), initial_vec.size());
    
    for (size_t i = 0; i < retrieved.size(); ++i) {
        EXPECT_NEAR(std::abs(retrieved[i] - initial_vec[i]), 0.0, 1e-10);
    }
}

TEST_F(AlgorithmManagerTest, SetCurrentVectorBeforeInitialize) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    std::vector<std::complex<double>> vec(input_dim_, 0.0);
    vec[0] = 1.0;
    EXPECT_THROW(manager.setCurrentVector(vec), std::runtime_error);
}

TEST_F(AlgorithmManagerTest, SetCurrentVectorValid) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    manager.initializeRandom();
    
    std::vector<std::complex<double>> new_vec(input_dim_, 0.0);
    new_vec[1] = 1.0;
    EXPECT_NO_THROW(manager.setCurrentVector(new_vec));
    
    auto retrieved = manager.getCurrentVector();
    for (size_t i = 0; i < retrieved.size(); ++i) {
        EXPECT_NEAR(std::abs(retrieved[i] - new_vec[i]), 0.0, 1e-10);
    }
}

TEST_F(AlgorithmManagerTest, SetCurrentVectorWrongSize) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    manager.initializeRandom();
    
    std::vector<std::complex<double>> wrong_size(input_dim_ + 1, 0.0);
    EXPECT_THROW(manager.setCurrentVector(wrong_size), std::invalid_argument);
}

// ============================================================================
// Entropy Accessor Tests
// ============================================================================

TEST_F(AlgorithmManagerTest, EntropyAccessorsDelegate) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    manager.initializeRandom();
    manager.stepOnce();
    
    // These should not throw
    EXPECT_NO_THROW(manager.getEpsilonEntropy());
    EXPECT_NO_THROW(manager.getEstimatedEntropy());
    EXPECT_NO_THROW(manager.getEntropyBounds());
    
    // Bounds should be consistent
    auto bounds = manager.getEntropyBounds();
    EXPECT_LE(bounds.first, bounds.second);
}

// ============================================================================
// Strategy Management Tests
// ============================================================================

TEST_F(AlgorithmManagerTest, SetStrategyNull) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    EXPECT_THROW(manager.setStrategy(nullptr), std::invalid_argument);
}

TEST_F(AlgorithmManagerTest, SetStrategyValid) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    EXPECT_NO_THROW(manager.setStrategy(std::move(strategy)));
}

TEST_F(AlgorithmManagerTest, SwitchStrategy) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    // Set CUDA strategy
    auto cuda_strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(cuda_strategy));
    
    manager.initializeRandom();
    manager.stepOnce();
    
    // Switch to generic strategy (after re-initialization)
    auto generic_strategy = std::make_unique<GenericMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(generic_strategy));
    
    manager.initializeRandom();
    EXPECT_NO_THROW(manager.stepOnce());
}

// ============================================================================
// Precision Tests
// ============================================================================

TEST_F(AlgorithmManagerTest, FloatPrecision) {
    // Create float precision Kraus operators
    std::vector<std::complex<float>> kraus_data_float(kraus_count_ * input_dim_ * output_dim_);
    
    // K_0: column-major [1, 0, 0, 0]
    kraus_data_float[0] = 1.0f;
    kraus_data_float[1] = 0.0f;
    kraus_data_float[2] = 0.0f;
    kraus_data_float[3] = 0.0f;
    
    // K_1: column-major [0, 0, 0, 1]
    kraus_data_float[4] = 0.0f;
    kraus_data_float[5] = 0.0f;
    kraus_data_float[6] = 0.0f;
    kraus_data_float[7] = 1.0f;
    
    auto kraus_float = HostKrausOperators::fromFloat(kraus_data_float, kraus_count_, input_dim_, output_dim_);
    
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_float,
        epsilon_,
        PrecisionType::FLOAT
    );
    
    EXPECT_EQ(manager.getPrecision(), PrecisionType::FLOAT);
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    manager.initializeRandom();
    EXPECT_NO_THROW(manager.stepOnce());
}

TEST_F(AlgorithmManagerTest, DoublePrecision) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    EXPECT_EQ(manager.getPrecision(), PrecisionType::DOUBLE);
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    manager.initializeRandom();
    EXPECT_NO_THROW(manager.stepOnce());
}

// ============================================================================
// Device Validation Tests
// ============================================================================

TEST_F(AlgorithmManagerTest, DeviceValidationCUDA) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    manager.initializeRandom();
    
    // stepOnce should validate that device memory belongs to CUDA device
    EXPECT_NO_THROW(manager.stepOnce());
}

TEST_F(AlgorithmManagerTest, DeviceValidationCPU) {
    std::cout << "In CPU Device Validation Test" << std::endl;
    AlgorithmManager manager(
        *cpu_device_,
        *cpu_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    std::cout << "Initialized AlgorithmManager with CPU device." << std::endl;
    auto strategy = std::make_unique<GenericMinimizationStrategy>(*cpu_device_);
    std::cout << "Created GenericMinimizationStrategy." << std::endl;
    manager.setStrategy(std::move(strategy));
    std::cout << "Set strategy in AlgorithmManager." << std::endl;
    manager.initializeRandom();
    std::cout << "Initialized random state." << std::endl;
    // stepOnce should validate that device memory belongs to CPU device
    EXPECT_NO_THROW(manager.stepOnce());
    std::cout << "stepOnce completed successfully." << std::endl;
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(AlgorithmManagerTest, CompleteWorkflow) {
    AlgorithmManager manager(
        *cuda_device_,
        *cuda_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    // 1. Set strategy
    auto strategy = std::make_unique<CudaMinimizationStrategy>(*cuda_device_);
    manager.setStrategy(std::move(strategy));
    
    // 2. Initialize with random vector
    manager.initializeRandom();
    EXPECT_TRUE(manager.isInitialized());
    
    // 3. Run several steps
    for (int i = 0; i < 5; ++i) {
        manager.stepOnce();
    }
    
    // 4. Get final vector
    auto final_vec = manager.getCurrentVector();
    EXPECT_EQ(final_vec.size(), static_cast<size_t>(input_dim_));
}

TEST_F(AlgorithmManagerTest, GenericStrategyWorkflow) {
    AlgorithmManager manager(
        *cpu_device_,
        *cpu_entropy_manager_,
        kraus_,
        epsilon_,
        PrecisionType::DOUBLE
    );
    
    auto strategy = std::make_unique<GenericMinimizationStrategy>(*cpu_device_);
    manager.setStrategy(std::move(strategy));
    
    manager.initializeRandom();
    
    // Run a few steps with generic strategy
    for (int i = 0; i < 3; ++i) {
        EXPECT_NO_THROW(manager.stepOnce());
    }
    
    auto final_vec = manager.getCurrentVector();
    EXPECT_EQ(final_vec.size(), static_cast<size_t>(input_dim_));
}
