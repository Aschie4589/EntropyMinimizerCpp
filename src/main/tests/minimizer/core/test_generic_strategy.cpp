#include <gtest/gtest.h>
#include "minimizer/algorithm/generic_minimization_strategy.h"
#include "compute/device/DeviceFactory.h"
#include <complex>
#include <vector>
#include <cmath>

using namespace std::complex_literals;

// ============================================================================
// Test Fixtures
// ============================================================================

class GenericMinimizationStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create CPU device
        device_ = DeviceFactory::create(DeviceFactory::DeviceType::CPU);
        ASSERT_NE(device_, nullptr);
        ASSERT_EQ(device_->getBackend(), DeviceBackend::CPU);
        
        std::cout << "Created CPU device for testing: " << device_->getName() << std::endl;

        // Create test Kraus operators
        setupTestKraus();
        std::cout << "Setup Kraus operators (double precision) for testing..." << std::endl;
        
        // Create entropy manager
        epsilon_ = 0.01;
        std::cout << "Using epsilon = " << epsilon_ << std::endl;
    }
    
    void setupTestKraus() {
        // Create identity channel: 2 Kraus operators, 5×5
        kraus_count_ = 2;
        input_dim_ = 5;
        output_dim_ = 5;
        
        kraus_num_ = kraus_count_;
        kraus_size_ = input_dim_ * output_dim_;
        vec_dim_ = input_dim_;
        
        kraus_data_.resize(kraus_num_ * kraus_size_, std::complex<double>(0.0, 0.0));
        
        // Setup as projector: each Kraus operator is a diagonal element
        for (int i = 0; i < kraus_num_; ++i) {
            kraus_data_[i * kraus_size_ + i * input_dim_ + i] = std::complex<double>(1.0, 0.0);
        }
        
        // Setup initial state vector
        vec_.resize(vec_dim_);
        for (int i = 0; i < vec_dim_; ++i) {
            vec_[i] = 1.0 / std::sqrt(static_cast<double>(vec_dim_));
        }
    }
    
    std::unique_ptr<IComputeDevice> device_;
    std::vector<std::complex<double>> kraus_data_;
    std::vector<std::complex<double>> vec_;
    int kraus_count_;
    int kraus_num_;
    int kraus_size_;
    int vec_dim_;
    int input_dim_;
    int output_dim_;
    double epsilon_;
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, ConstructorValidDevice) {
    EXPECT_NO_THROW(GenericMinimizationStrategy strategy(*device_));
}

TEST_F(GenericMinimizationStrategyTest, ConstructorInvalidStreamNumber) {
    EXPECT_THROW(GenericMinimizationStrategy strategy(*device_, 0), std::invalid_argument);
}

TEST_F(GenericMinimizationStrategyTest, ConstructorValidStreamNumber) {
    EXPECT_NO_THROW(GenericMinimizationStrategy strategy(*device_, 8));
}


// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, InitializeValidDimensions) {
    GenericMinimizationStrategy strategy(*device_);
    
    EXPECT_NO_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE)
    );
}

TEST_F(GenericMinimizationStrategyTest, InitializeInvalidEpsilon) {
    GenericMinimizationStrategy strategy(*device_);
    
    EXPECT_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, 0.0, PrecisionType::DOUBLE),
        std::runtime_error
    );
}

TEST_F(GenericMinimizationStrategyTest, InitializeInvalidOutputDimension) {
    GenericMinimizationStrategy strategy(*device_);

    // Setup more kraus operators than output dimension (this should give error hopefully)
    kraus_count_ = 6; // (6>5)
    EXPECT_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE),
        std::runtime_error
    );
}

TEST_F(GenericMinimizationStrategyTest, InitializeInvalidInputDimension) {
    GenericMinimizationStrategy strategy(*device_);

    // Setup more kraus operators than sqrt of input dimension (this should give error hopefully)
    kraus_count_ = 4; // (16>5)
    EXPECT_THROW(
        strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE),
        std::runtime_error
    );
}

TEST_F(GenericMinimizationStrategyTest, GetWorkspaceSize) {
    GenericMinimizationStrategy strategy(*device_);
    
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE);
    
    size_t workspace_size = strategy.getWorkspaceSize();
    EXPECT_GT(workspace_size, 0u);
}


// ============================================================================
// GetPrecision Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, GetPrecisionAfterInitialization) {
    GenericMinimizationStrategy strategy(*device_);
    
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::FLOAT);
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::FLOAT);
    
    strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE);
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::DOUBLE);
}

TEST_F(GenericMinimizationStrategyTest, GetPrecisionBeforeInitialization) {
    GenericMinimizationStrategy strategy(*device_);
    
    // Default precision before initialization
    EXPECT_EQ(strategy.getPrecision(), PrecisionType::DOUBLE);
}

// ============================================================================
// stepOnce Tests
// ============================================================================

// First test: no throw
TEST_F(GenericMinimizationStrategyTest, stepOnceImplTestNoThrow) {
    GenericMinimizationStrategy strategy(*device_);
    
    EXPECT_NO_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE)
    );
    // Start with simple non-zero vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(1.0/std::sqrt(input_dim_), 0.0));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));

    // Also allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    d_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));


    EXPECT_NO_THROW(
        strategy.stepOnce(d_kraus.get(), d_vec.get()) // Passing device pointers
    );
}

TEST_F(GenericMinimizationStrategyTest, StepOnceWithoutInitialization) {
    GenericMinimizationStrategy strategy(*device_);
    
    // Start with simple non-zero vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(1.0/std::sqrt(input_dim_), 0.0));
    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));

    // Also allocate Kraus on device
    std::unique_ptr<IDeviceMemory> d_kraus = device_->allocate(kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));
    d_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * input_dim_ * output_dim_ * sizeof(std::complex<double>));

    EXPECT_THROW(
        strategy.stepOnce(d_kraus.get(), d_vec.get()),
        std::runtime_error
    );
}

TEST_F(GenericMinimizationStrategyTest, StepOnceExpectDecreaseEntropy) {
    GenericMinimizationStrategy strategy(*device_);
    
    EXPECT_NO_THROW(
        strategy.initialize(kraus_count_, input_dim_, output_dim_, epsilon_, PrecisionType::DOUBLE)
    );
    // Start with simple non-zero vector
    std::unique_ptr<IDeviceMemory> d_vec = device_->allocate(input_dim_ * sizeof(std::complex<double>));
    std::vector<std::complex<double>> host_vec(input_dim_, std::complex<double>(0.0, 0.0));
    host_vec[0] = 1.0; // Start from mixed state to see entropy decrease

    d_vec->copyFromHost(host_vec.data(), input_dim_ * sizeof(std::complex<double>));

    // Also allocate Kraus on device. Make them slightly more complicated Kraus operators
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

TEST_F(GenericMinimizationStrategyTest, FloatPrecisionWorkflow) {
    using Complex = std::complex<float>;
    GenericMinimizationStrategy strategy(*device_);
    
    // Allocate device memory for float precision
    auto d_kraus = device_->allocate(kraus_num_ * kraus_size_ * sizeof(Complex));
    auto d_vec = device_->allocate(vec_dim_ * sizeof(Complex));
    
    // Initialize with float data
    std::vector<Complex> kraus_float(kraus_num_ * kraus_size_);
    for (size_t i = 0; i < kraus_num_ * kraus_size_; ++i) {
        kraus_float[i] = Complex(static_cast<float>(kraus_data_[i].real()), 
                                 static_cast<float>(kraus_data_[i].imag()));
    }
    
    std::vector<Complex> vec_float(vec_dim_);
    for (int i = 0; i < vec_dim_; ++i) {
        vec_float[i] = Complex(1.0f / std::sqrt(static_cast<float>(vec_dim_)));
    }
    
    d_kraus->copyFromHost(kraus_float.data(), kraus_num_ * kraus_size_ * sizeof(Complex));
    d_vec->copyFromHost(vec_float.data(), vec_dim_ * sizeof(Complex));
    
    strategy.initialize(kraus_num_, output_dim_, input_dim_, epsilon_, PrecisionType::FLOAT);
    
    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(strategy.stepOnce(d_kraus.get(), d_vec.get()));
    }
}

TEST_F(GenericMinimizationStrategyTest, DoublePrecisionWorkflow) {
    using Complex = std::complex<double>;
    GenericMinimizationStrategy strategy(*device_);
    
    auto d_kraus = device_->allocate(kraus_num_ * kraus_size_ * sizeof(Complex));
    auto d_vec = device_->allocate(vec_dim_ * sizeof(Complex));
    
    d_kraus->copyFromHost(kraus_data_.data(), kraus_num_ * kraus_size_ * sizeof(Complex));
    d_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    
    strategy.initialize(kraus_num_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(strategy.stepOnce(d_kraus.get(), d_vec.get()));
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, LongRunStability) {
    using Complex = std::complex<double>;
    GenericMinimizationStrategy strategy(*device_);
    
    auto d_kraus = device_->allocate(kraus_num_ * kraus_size_ * sizeof(Complex));
    auto d_vec = device_->allocate(vec_dim_ * sizeof(Complex));
    
    d_kraus->copyFromHost(kraus_data_.data(), kraus_num_ * kraus_size_ * sizeof(Complex));
    d_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    
    strategy.initialize(kraus_num_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Run many steps to check stability
    for (int i = 0; i < 50; ++i) {
        double entropy = strategy.stepOnce(d_kraus.get(), d_vec.get());
        
        EXPECT_GE(entropy, 0.0);
        EXPECT_FALSE(std::isnan(entropy));
        EXPECT_FALSE(std::isinf(entropy));
    }
}

TEST_F(GenericMinimizationStrategyTest, StateVectorNormalization) {
    using Complex = std::complex<double>;
    GenericMinimizationStrategy strategy(*device_);
    
    auto d_kraus = device_->allocate(kraus_num_ * kraus_size_ * sizeof(Complex));
    auto d_vec = device_->allocate(vec_dim_ * sizeof(Complex));
    
    d_kraus->copyFromHost(kraus_data_.data(), kraus_num_ * kraus_size_ * sizeof(Complex));
    d_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    
    strategy.initialize(kraus_num_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    for (int i = 0; i < 10; ++i) {
        strategy.stepOnce(d_kraus.get(), d_vec.get());
    }
    
    // Copy back and check normalization
    std::vector<Complex> vec_result(vec_dim_);
    d_vec->copyToHost(vec_result.data(), vec_dim_ * sizeof(Complex));
    
    double norm_sq = 0.0;
    for (const auto& v : vec_result) {
        norm_sq += std::norm(v);
    }
    EXPECT_NEAR(norm_sq, 1.0, 1e-6);
}

// ============================================================================
// CPU-Specific Tests
// ============================================================================

TEST_F(GenericMinimizationStrategyTest, DeviceMemoryValidation) {
    using Complex = std::complex<double>;
    GenericMinimizationStrategy strategy(*device_);
    
    auto d_kraus = device_->allocate(kraus_num_ * kraus_size_ * sizeof(Complex));
    auto d_vec = device_->allocate(vec_dim_ * sizeof(Complex));
    
    d_kraus->copyFromHost(kraus_data_.data(), kraus_num_ * kraus_size_ * sizeof(Complex));
    d_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    
    strategy.initialize(kraus_num_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // stepOnce should validate device memory belongs to CPU device
    EXPECT_NO_THROW(strategy.stepOnce(d_kraus.get(), d_vec.get()));
}
