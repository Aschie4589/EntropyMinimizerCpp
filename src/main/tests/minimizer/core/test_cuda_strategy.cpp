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
        
        std::cout << "Created CUDA device for testing: " << device_->getName() << std::endl;

        // Create test Kraus operators (matching generic strategy tests)
        setupTestKraus();
        std::cout << "Setup Kraus operators (double precision) for testing..." << std::endl;
        
        // Setup epsilon
        epsilon_ = 0.01;
        std::cout << "Using epsilon = " << epsilon_ << std::endl;
    }
    
    void setupTestKraus() {
        // Create identity channel: 2 Kraus operators, 5×5 (matching generic tests)
        kraus_count_ = 2;
        input_dim_ = 5;
        output_dim_ = 5;
        
        std::vector<std::complex<double>> kraus_data(kraus_count_ * input_dim_ * output_dim_, std::complex<double>(0.0, 0.0));

        // Setup as projector (matching generic tests)
        for (int i = 0; i < kraus_count_; ++i) {
            kraus_data[i * input_dim_ * output_dim_ + i * input_dim_ + i] = std::complex<double>(1.0, 0.0);
        }
        
        kraus_ = HostKrausOperators::fromDouble(kraus_data, kraus_count_, input_dim_, output_dim_);
    }
    
    std::unique_ptr<IComputeDevice> device_;
    HostKrausOperators kraus_;
    int kraus_count_;
    int input_dim_;
    int output_dim_;
    double epsilon_;
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
}

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, InitializeValidDimensions) {
    CudaMinimizationStrategy strategy(*device_);
    
    EXPECT_NO_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE)
    );
}

TEST_F(CudaMinimizationStrategyTest, InitializeInvalidEpsilon) {
    CudaMinimizationStrategy strategy(*device_);
    
    EXPECT_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, 0.0, PrecisionType::DOUBLE),
        std::runtime_error
    );
}

TEST_F(CudaMinimizationStrategyTest, InitializeInvalidOutputDimension) {
    CudaMinimizationStrategy strategy(*device_);

    // Setup more kraus operators than output dimension (this should give error)
    kraus_count_ = 6; // (6>5)
    EXPECT_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE),
        std::runtime_error
    );
}


TEST_F(CudaMinimizationStrategyTest, GetWorkspaceSize) {
    CudaMinimizationStrategy strategy(*device_);
    
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE);
    
    size_t workspace_size = strategy.getWorkspaceSize();
    EXPECT_GT(workspace_size, 0u);
}

// ============================================================================
// GetPrecision Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, GetPrecisionAfterInitialization) {
    CudaMinimizationStrategy strategy(*device_);
    
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::FLOAT);
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::FLOAT);
    
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE);
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::DOUBLE);
}

TEST_F(CudaMinimizationStrategyTest, GetPrecisionBeforeInitialization) {
    CudaMinimizationStrategy strategy(*device_);
    
    // Default precision before initialization
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::DOUBLE);
}

// ============================================================================
// stepOnce Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, stepOnceImplTestNoThrow) {
    CudaMinimizationStrategy strategy(*device_);
    
    EXPECT_NO_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE)
    );
    
    // Start with simple non-zero vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(1.0/std::sqrt(input_dim_), 0.0));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));

    // Allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    d_kraus->copyFromHost(kraus_.data.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));

    EXPECT_NO_THROW(
        strategy.stepOnce(d_kraus.get(), d_vec.get())
    );
}

TEST_F(CudaMinimizationStrategyTest, StepOnceWithoutInitialization) {
    CudaMinimizationStrategy strategy(*device_);
    
    // Start with simple non-zero vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(1.0/std::sqrt(input_dim_), 0.0));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));

    // Allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    d_kraus->copyFromHost(kraus_.data.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));

    EXPECT_THROW(
        strategy.stepOnce(d_kraus.get(), d_vec.get()),
        std::runtime_error
    );
}

TEST_F(CudaMinimizationStrategyTest, StepOnceExpectDecreaseEntropy) {
    CudaMinimizationStrategy strategy(*device_);
    
    EXPECT_NO_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE)
    );
    
    // Start with simple non-zero vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(0.0, 0.0));
    host_vec[0] = 1.0; // Start from mixed state to see entropy decrease

    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));

    // Create slightly more complicated Kraus operators
    kraus_count_ = 2;
    input_dim_ = 5;
    output_dim_ = 5;
 
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> new_kraus(kraus_count_ * input_dim_ * output_dim_, std::complex<double>(0.0,0.0));        
    // First kraus is identity, second is a permutation
    for (int i=0; i<input_dim_; i++){
        new_kraus[input_dim_*i+i] = std::complex<double>(1.0/std::sqrt(kraus_count_),0.0);
        new_kraus[input_dim_*output_dim_+input_dim_*(output_dim_-i-1)+i] = std::complex<double>(1.0/std::sqrt(kraus_count_),0.0);
    }
    d_kraus->copyFromHost(new_kraus.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));

    double entropies[5];
    for (int i = 0; i < 5; ++i) {
        entropies[i] = strategy.stepOnce(d_kraus.get(), d_vec.get());
    }
    
    // Check that entropy decreased at every step
    for (int i = 1; i < 5; ++i) {
        EXPECT_LE(entropies[i], entropies[i-1]);
    }
}

// ============================================================================
// Precision Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, FloatPrecisionWorkflow) {
    // Create float precision Kraus
    std::vector<std::complex<float>> kraus_data_float(kraus_count_ * input_dim_ * output_dim_, std::complex<float>(0.0f, 0.0f));
    
    // Setup as projector (matching double precision setup)
    for (int i = 0; i < kraus_count_; ++i) {
        kraus_data_float[i * input_dim_ * output_dim_ + i * input_dim_ + i] = std::complex<float>(1.0f, 0.0f);
    }
    
    auto kraus_float = HostKrausOperators::fromFloat(kraus_data_float, kraus_count_, input_dim_, output_dim_);
    
    CudaMinimizationStrategy strategy(*device_);
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::FLOAT);
    
    // Allocate and initialize vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<float>));
    std::vector<std::complex<float>> host_vec(input_dim_, std::complex<float>(1.0f/std::sqrt(float(input_dim_)), 0.0f));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<float>));
    
    // Allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<float>));
    d_kraus->copyFromHost(kraus_float.data.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<float>));
    
    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(strategy.stepOnce(d_kraus.get(), d_vec.get()));
    }
}

TEST_F(CudaMinimizationStrategyTest, DoublePrecisionWorkflow) {
    CudaMinimizationStrategy strategy(*device_);
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Allocate and initialize vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(1.0/std::sqrt(double(input_dim_)), 0.0));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));
    
    // Allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    d_kraus->copyFromHost(kraus_.data.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    
    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(strategy.stepOnce(d_kraus.get(), d_vec.get()));
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, LongRunStability) {
    CudaMinimizationStrategy strategy(*device_);
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Allocate and initialize vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(1.0/std::sqrt(double(input_dim_)), 0.0));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));
    
    // Allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    d_kraus->copyFromHost(kraus_.data.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    
    // Run many steps to check stability
    for (int i = 0; i < 50; ++i) {
        double entropy = 0.0;
        EXPECT_NO_THROW(entropy = strategy.stepOnce(d_kraus.get(), d_vec.get()));
        
        EXPECT_GE(entropy, 0.0);
        EXPECT_FALSE(std::isnan(entropy));
        EXPECT_FALSE(std::isinf(entropy));
    }
}

TEST_F(CudaMinimizationStrategyTest, StateVectorNormalization) {
    CudaMinimizationStrategy strategy(*device_);
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Allocate and initialize vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(1.0/std::sqrt(double(input_dim_)), 0.0));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));
    
    // Allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    d_kraus->copyFromHost(kraus_.data.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    
    for (int i = 0; i < 10; ++i) {
        strategy.stepOnce(d_kraus.get(), d_vec.get());
    }
    
    // Check normalization
    std::vector<std::complex<double>> vec_host(input_dim_);
    d_vec->copyToHost(vec_host.data(), input_dim_ * sizeof(std::complex<double>));
    
    double norm_sq = 0.0;
    for (const auto& v : vec_host) {
        norm_sq += std::norm(v);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-6);
}

// ============================================================================
// Device Memory Validation Tests
// ============================================================================

TEST_F(CudaMinimizationStrategyTest, DeviceMemoryValidation) {
    CudaMinimizationStrategy strategy(*device_);
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Allocate and initialize vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(1.0/std::sqrt(double(input_dim_)), 0.0));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));
    
    // Allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    d_kraus->copyFromHost(kraus_.data.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    
    // stepOnce should validate device memory belongs to CUDA device
    EXPECT_NO_THROW(strategy.stepOnce(d_kraus.get(), d_vec.get()));
}
