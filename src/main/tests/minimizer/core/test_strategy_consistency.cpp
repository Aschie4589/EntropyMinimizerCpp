#include <gtest/gtest.h>
#include "minimizer/algorithm/generic_minimization_strategy.h"
#include "minimizer/algorithm/cuda_minimization_strategy.h"
#include "compute/device/DeviceFactory.h"
#include <complex>
#include <vector>
#include <cmath>

using namespace std::complex_literals;

// ============================================================================
// Strategy Consistency Tests
// ============================================================================
// These tests verify that CUDA and CPU implementations produce consistent results

class StrategyConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create CPU device
        cpu_device_ = DeviceFactory::create(DeviceFactory::DeviceType::CPU);
        ASSERT_NE(cpu_device_, nullptr);
        ASSERT_EQ(cpu_device_->getBackend(), DeviceBackend::CPU);
        
        // Create CUDA device (if available)
        try {
            cuda_device_ = DeviceFactory::create(DeviceFactory::DeviceType::CUDA);
            cuda_available_ = (cuda_device_ != nullptr);
        } catch (...) {
            cuda_available_ = false;
        }
        
        if (!cuda_available_) {
            GTEST_SKIP() << "CUDA device not available, skipping consistency tests";
        }
        
        // Setup test parameters
        setupTestData();
    }
    
    void setupTestData() {
        // Create identity channel: 2 Kraus operators, 5×5
        kraus_count_ = 2;
        input_dim_ = 5;
        output_dim_ = 5;
        epsilon_ = 0.01;
        
        kraus_size_ = input_dim_ * output_dim_;
        vec_dim_ = input_dim_;
        
        // Setup Kraus operators as projectors
        kraus_data_.resize(kraus_count_ * kraus_size_, std::complex<double>(0.0, 0.0));
        for (int i = 0; i < kraus_count_; ++i) {
            kraus_data_[i * kraus_size_ + i * input_dim_ + i] = std::complex<double>(1.0, 0.0);
        }
        
        // Setup initial state vector (normalized)
        vec_.resize(vec_dim_);
        for (int i = 0; i < vec_dim_; ++i) {
            vec_[i] = 1.0 / std::sqrt(static_cast<double>(vec_dim_));
        }
    }
    
    std::unique_ptr<IComputeDevice> cpu_device_;
    std::unique_ptr<IComputeDevice> cuda_device_;
    bool cuda_available_ = false;
    
    std::vector<std::complex<double>> kraus_data_;
    std::vector<std::complex<double>> vec_;
    int kraus_count_;
    int kraus_size_;
    int vec_dim_;
    int input_dim_;
    int output_dim_;
    double epsilon_;
};

// ============================================================================
// Initialization Consistency
// ============================================================================


TEST_F(StrategyConsistencyTest, GetPrecisionConsistency) {
    GenericMinimizationStrategy cpu_strategy(*cpu_device_);
    CudaMinimizationStrategy cuda_strategy(*cuda_device_);
    
    cpu_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    cuda_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    EXPECT_EQ(cpu_strategy.getPrecision(), cuda_strategy.getPrecision());
    
    // Test with float precision
    GenericMinimizationStrategy cpu_strategy_f(*cpu_device_);
    CudaMinimizationStrategy cuda_strategy_f(*cuda_device_);
    
    cpu_strategy_f.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::FLOAT);
    cuda_strategy_f.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::FLOAT);
    
    EXPECT_EQ(cpu_strategy_f.getPrecision(), cuda_strategy_f.getPrecision());
}

// ============================================================================
// Entropy Computation Consistency
// ============================================================================

TEST_F(StrategyConsistencyTest, SingleStepEntropyConsistency) {
    using Complex = std::complex<double>;
    
    // Setup CPU strategy
    GenericMinimizationStrategy cpu_strategy(*cpu_device_);
    auto cpu_kraus = cpu_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cpu_vec = cpu_device_->allocate(vec_dim_ * sizeof(Complex));
    cpu_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cpu_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    cpu_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Setup CUDA strategy
    CudaMinimizationStrategy cuda_strategy(*cuda_device_);
    auto cuda_kraus = cuda_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cuda_vec = cuda_device_->allocate(vec_dim_ * sizeof(Complex));
    cuda_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cuda_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    cuda_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Run one step on both
    double cpu_entropy = cpu_strategy.stepOnce(cpu_kraus.get(), cpu_vec.get());
    double cuda_entropy = cuda_strategy.stepOnce(cuda_kraus.get(), cuda_vec.get());
    
    // Entropies should match within numerical tolerance
    EXPECT_NEAR(cpu_entropy, cuda_entropy, 1e-6) 
        << "CPU entropy: " << cpu_entropy << ", CUDA entropy: " << cuda_entropy;
}

TEST_F(StrategyConsistencyTest, MultiStepEntropyConsistency) {
    using Complex = std::complex<double>;
    
    // Setup CPU strategy
    GenericMinimizationStrategy cpu_strategy(*cpu_device_);
    auto cpu_kraus = cpu_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cpu_vec = cpu_device_->allocate(vec_dim_ * sizeof(Complex));
    cpu_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cpu_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    cpu_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Setup CUDA strategy
    CudaMinimizationStrategy cuda_strategy(*cuda_device_);
    auto cuda_kraus = cuda_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cuda_vec = cuda_device_->allocate(vec_dim_ * sizeof(Complex));
    cuda_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cuda_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    cuda_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Run multiple steps and compare at each iteration
    const int num_steps = 10;
    for (int i = 0; i < num_steps; ++i) {
        double cpu_entropy = cpu_strategy.stepOnce(cpu_kraus.get(), cpu_vec.get());
        double cuda_entropy = cuda_strategy.stepOnce(cuda_kraus.get(), cuda_vec.get());
        
        EXPECT_NEAR(cpu_entropy, cuda_entropy, 1e-6) 
            << "Step " << i << ": CPU entropy: " << cpu_entropy 
            << ", CUDA entropy: " << cuda_entropy;
    }
}

// ============================================================================
// State Vector Consistency
// ============================================================================

TEST_F(StrategyConsistencyTest, FinalStateVectorConsistency) {
    using Complex = std::complex<double>;
    
    // Setup CPU strategy
    GenericMinimizationStrategy cpu_strategy(*cpu_device_);
    auto cpu_kraus = cpu_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cpu_vec = cpu_device_->allocate(vec_dim_ * sizeof(Complex));
    cpu_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cpu_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    cpu_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Setup CUDA strategy
    CudaMinimizationStrategy cuda_strategy(*cuda_device_);
    auto cuda_kraus = cuda_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cuda_vec = cuda_device_->allocate(vec_dim_ * sizeof(Complex));
    cuda_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cuda_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    cuda_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Run several steps
    const int num_steps = 5;
    for (int i = 0; i < num_steps; ++i) {
        cpu_strategy.stepOnce(cpu_kraus.get(), cpu_vec.get());
        cuda_strategy.stepOnce(cuda_kraus.get(), cuda_vec.get());
    }
    
    // Copy back final state vectors
    std::vector<Complex> cpu_result(vec_dim_);
    std::vector<Complex> cuda_result(vec_dim_);
    cpu_vec->copyToHost(cpu_result.data(), vec_dim_ * sizeof(Complex));
    cuda_vec->copyToHost(cuda_result.data(), vec_dim_ * sizeof(Complex));
    
    // SVD can have phase ambiguity (global phase doesn't matter for quantum states)
    // Check that magnitudes match, or that vectors match up to a global phase
    // First check if they match element-wise
    bool match_directly = true;
    for (int i = 0; i < vec_dim_; ++i) {
        if (std::abs(cpu_result[i] - cuda_result[i]) > 1e-6) {
            match_directly = false;
            break;
        }
    }
    
    if (!match_directly) {
        // Check if they differ by a global phase (e.g., all signs flipped)
        // Find the phase difference from first non-zero element
        Complex phase_factor = cuda_result[0] / cpu_result[0];
        bool match_with_phase = true;
        for (int i = 0; i < vec_dim_; ++i) {
            Complex expected = cpu_result[i] * phase_factor;
            if (std::abs(cuda_result[i] - expected) > 1e-6) {
                match_with_phase = false;
                break;
            }
        }
        EXPECT_TRUE(match_with_phase) 
            << "Vectors don't match even accounting for global phase. "
            << "Phase factor: " << phase_factor;
    }
}

TEST_F(StrategyConsistencyTest, NormalizationConsistency) {
    using Complex = std::complex<double>;
    
    // Setup CPU strategy
    GenericMinimizationStrategy cpu_strategy(*cpu_device_);
    auto cpu_kraus = cpu_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cpu_vec = cpu_device_->allocate(vec_dim_ * sizeof(Complex));
    cpu_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cpu_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    cpu_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Setup CUDA strategy
    CudaMinimizationStrategy cuda_strategy(*cuda_device_);
    auto cuda_kraus = cuda_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cuda_vec = cuda_device_->allocate(vec_dim_ * sizeof(Complex));
    cuda_kraus->copyFromHost(kraus_data_.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cuda_vec->copyFromHost(vec_.data(), vec_dim_ * sizeof(Complex));
    cuda_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::DOUBLE);
    
    // Run several steps
    for (int i = 0; i < 10; ++i) {
        cpu_strategy.stepOnce(cpu_kraus.get(), cpu_vec.get());
        cuda_strategy.stepOnce(cuda_kraus.get(), cuda_vec.get());
    }
    
    // Copy back and check both are normalized
    std::vector<Complex> cpu_result(vec_dim_);
    std::vector<Complex> cuda_result(vec_dim_);
    cpu_vec->copyToHost(cpu_result.data(), vec_dim_ * sizeof(Complex));
    cuda_vec->copyToHost(cuda_result.data(), vec_dim_ * sizeof(Complex));
    
    double cpu_norm = 0.0;
    double cuda_norm = 0.0;
    for (int i = 0; i < vec_dim_; ++i) {
        cpu_norm += std::norm(cpu_result[i]);
        cuda_norm += std::norm(cuda_result[i]);
    }
    
    EXPECT_NEAR(cpu_norm, 1.0, 1e-6);
    EXPECT_NEAR(cuda_norm, 1.0, 1e-6);
    EXPECT_NEAR(cpu_norm, cuda_norm, 1e-6);
}

// ============================================================================
// Float Precision Consistency
// ============================================================================

TEST_F(StrategyConsistencyTest, FloatPrecisionConsistency) {
    using Complex = std::complex<float>;
    
    // Convert test data to float
    std::vector<Complex> kraus_float(kraus_count_ * kraus_size_);
    std::vector<Complex> vec_float(vec_dim_);
    for (size_t i = 0; i < kraus_data_.size(); ++i) {
        kraus_float[i] = Complex(static_cast<float>(kraus_data_[i].real()),
                                 static_cast<float>(kraus_data_[i].imag()));
    }
    for (size_t i = 0; i < vec_.size(); ++i) {
        vec_float[i] = Complex(static_cast<float>(vec_[i].real()),
                              static_cast<float>(vec_[i].imag()));
    }
    
    // Setup CPU strategy
    GenericMinimizationStrategy cpu_strategy(*cpu_device_);
    auto cpu_kraus = cpu_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cpu_vec = cpu_device_->allocate(vec_dim_ * sizeof(Complex));
    cpu_kraus->copyFromHost(kraus_float.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cpu_vec->copyFromHost(vec_float.data(), vec_dim_ * sizeof(Complex));
    cpu_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::FLOAT);
    
    // Setup CUDA strategy
    CudaMinimizationStrategy cuda_strategy(*cuda_device_);
    auto cuda_kraus = cuda_device_->allocate(kraus_count_ * kraus_size_ * sizeof(Complex));
    auto cuda_vec = cuda_device_->allocate(vec_dim_ * sizeof(Complex));
    cuda_kraus->copyFromHost(kraus_float.data(), kraus_count_ * kraus_size_ * sizeof(Complex));
    cuda_vec->copyFromHost(vec_float.data(), vec_dim_ * sizeof(Complex));
    cuda_strategy.initialize(kraus_count_, output_dim_, input_dim_, epsilon_, PrecisionType::FLOAT);
    
    // Run one step on both
    double cpu_entropy = cpu_strategy.stepOnce(cpu_kraus.get(), cpu_vec.get());
    double cuda_entropy = cuda_strategy.stepOnce(cuda_kraus.get(), cuda_vec.get());
    
    // Use larger tolerance for float precision
    EXPECT_NEAR(cpu_entropy, cuda_entropy, 1e-4)
        << "CPU entropy: " << cpu_entropy << ", CUDA entropy: " << cuda_entropy;
}
