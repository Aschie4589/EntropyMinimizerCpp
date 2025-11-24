#include <gtest/gtest.h>
#include "fixtures/ComputeTestFixture.h"
#include "compute/backends/cuda/CudaRandom.h"
#include "compute/backends/cpu/CpuRandom.h"
#include "compute/backends/cuda/CudaMemory.h"
#include "compute/backends/cpu/CpuMemory.h"
#include <vector>
#include <complex>
#include <numeric>
#include <cmath>

// ====================
// CUDA Random Tests
// ====================

TEST_F(ComputeTest, CudaRandom_Creation) {
    auto rng = std::make_unique<CudaRandom>();
    EXPECT_NE(rng->getGenerator(), nullptr);
}

TEST_F(ComputeTest, CudaRandom_SetSeed) {
    auto rng = std::make_unique<CudaRandom>();
    
    // Should not throw
    EXPECT_NO_THROW(rng->setSeed(42));
    EXPECT_NO_THROW(rng->setSeed(123456789ULL));
}

TEST_F(ComputeTest, CudaRandom_GenerateUniform_Double) {
    auto rng = std::make_unique<CudaRandom>();
    auto mem = std::make_unique<CudaMemory>(100 * sizeof(double));
    
    rng->setSeed(42);
    EXPECT_NO_THROW(rng->generateUniform(mem->data(), 100, PrecisionType::DOUBLE));
    
    // Verify values are in [0, 1)
    std::vector<double> host_data(100);
    mem->copyToHost(host_data.data(), 100 * sizeof(double));
    
    for (double val : host_data) {
        EXPECT_GE(val, 0.0);
        EXPECT_LT(val, 1.0);
    }
}

TEST_F(ComputeTest, CudaRandom_GenerateUniform_Float) {
    auto rng = std::make_unique<CudaRandom>();
    auto mem = std::make_unique<CudaMemory>(100 * sizeof(float));
    
    rng->setSeed(42);
    EXPECT_NO_THROW(rng->generateUniform(mem->data(), 100, PrecisionType::FLOAT));
    
    // Verify values are in [0, 1)
    std::vector<float> host_data(100);
    mem->copyToHost(host_data.data(), 100 * sizeof(float));
    
    for (float val : host_data) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LT(val, 1.0f);
    }
}

TEST_F(ComputeTest, CudaRandom_GenerateNormal_Double) {
    auto rng = std::make_unique<CudaRandom>();
    auto mem = std::make_unique<CudaMemory>(1000 * sizeof(double));
    
    rng->setSeed(42);
    EXPECT_NO_THROW(rng->generateNormal(mem->data(), 1000, PrecisionType::DOUBLE));
    
    // Verify statistical properties (mean ≈ 0, stddev ≈ 1)
    std::vector<double> host_data(1000);
    mem->copyToHost(host_data.data(), 1000 * sizeof(double));
    
    double mean = std::accumulate(host_data.begin(), host_data.end(), 0.0) / host_data.size();
    
    double variance = 0.0;
    for (double val : host_data) {
        variance += (val - mean) * (val - mean);
    }
    variance /= host_data.size();
    double stddev = std::sqrt(variance);
    
    // With 1000 samples, mean should be close to 0, stddev close to 1
    EXPECT_NEAR(mean, 0.0, 0.1);     // Mean within ±0.1
    EXPECT_NEAR(stddev, 1.0, 0.1);   // Stddev within ±0.1
}

TEST_F(ComputeTest, CudaRandom_GenerateComplexNormal_Double) {
    auto rng = std::make_unique<CudaRandom>();
    auto mem = std::make_unique<CudaMemory>(100 * sizeof(std::complex<double>));
    
    rng->setSeed(42);
    EXPECT_NO_THROW(rng->generateComplexNormal(mem->data(), 100, PrecisionType::DOUBLE));
    
    // Verify data was generated (not all zeros)
    std::vector<std::complex<double>> host_data(100);
    mem->copyToHost(host_data.data(), 100 * sizeof(std::complex<double>));
    
    bool has_nonzero = false;
    for (const auto& val : host_data) {
        if (std::abs(val) > 1e-10) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(ComputeTest, CudaRandom_Reproducibility) {
    auto rng1 = std::make_unique<CudaRandom>();
    auto rng2 = std::make_unique<CudaRandom>();
    
    auto mem1 = std::make_unique<CudaMemory>(100 * sizeof(double));
    auto mem2 = std::make_unique<CudaMemory>(100 * sizeof(double));
    
    // Same seed should produce same results
    rng1->setSeed(12345);
    rng2->setSeed(12345);
    
    rng1->generateUniform(mem1->data(), 100, PrecisionType::DOUBLE);
    rng2->generateUniform(mem2->data(), 100, PrecisionType::DOUBLE);
    
    std::vector<double> host_data1(100);
    std::vector<double> host_data2(100);
    mem1->copyToHost(host_data1.data(), 100 * sizeof(double));
    mem2->copyToHost(host_data2.data(), 100 * sizeof(double));
    
    EXPECT_TRUE(vectorsAlmostEqual(host_data1, host_data2, 1e-15));
}

// ====================
// CPU Random Tests
// ====================

TEST_F(ComputeTest, CpuRandom_Creation) {
    auto rng = std::make_unique<CpuRandom>();
    // Just verify it creates without throwing
    EXPECT_NO_THROW(rng->setSeed(42));
}

TEST_F(ComputeTest, CpuRandom_GenerateUniform_Double) {
    auto rng = std::make_unique<CpuRandom>();
    auto mem = std::make_unique<CpuMemory>(100 * sizeof(double));
    
    rng->setSeed(42);
    EXPECT_NO_THROW(rng->generateUniform(mem->data(), 100, PrecisionType::DOUBLE));
    
    // Verify values are in [0, 1)
    double* data = static_cast<double*>(mem->data());
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_GE(data[i], 0.0);
        EXPECT_LT(data[i], 1.0);
    }
}

TEST_F(ComputeTest, CpuRandom_GenerateNormal_Double) {
    auto rng = std::make_unique<CpuRandom>();
    auto mem = std::make_unique<CpuMemory>(1000 * sizeof(double));
    
    rng->setSeed(42);
    EXPECT_NO_THROW(rng->generateNormal(mem->data(), 1000, PrecisionType::DOUBLE));
    
    // Verify statistical properties
    double* data = static_cast<double*>(mem->data());
    double mean = 0.0;
    for (size_t i = 0; i < 1000; ++i) {
        mean += data[i];
    }
    mean /= 1000.0;
    
    double variance = 0.0;
    for (size_t i = 0; i < 1000; ++i) {
        variance += (data[i] - mean) * (data[i] - mean);
    }
    variance /= 1000.0;
    double stddev = std::sqrt(variance);
    
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(stddev, 1.0, 0.1);
}

TEST_F(ComputeTest, CpuRandom_GenerateComplexNormal_Double) {
    auto rng = std::make_unique<CpuRandom>();
    auto mem = std::make_unique<CpuMemory>(100 * sizeof(std::complex<double>));
    
    rng->setSeed(42);
    EXPECT_NO_THROW(rng->generateComplexNormal(mem->data(), 100, PrecisionType::DOUBLE));
    
    // Verify data was generated
    std::complex<double>* data = static_cast<std::complex<double>*>(mem->data());
    bool has_nonzero = false;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(data[i]) > 1e-10) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(ComputeTest, CpuRandom_Reproducibility) {
    auto rng1 = std::make_unique<CpuRandom>();
    auto rng2 = std::make_unique<CpuRandom>();
    
    auto mem1 = std::make_unique<CpuMemory>(100 * sizeof(double));
    auto mem2 = std::make_unique<CpuMemory>(100 * sizeof(double));
    
    // Same seed should produce same results
    rng1->setSeed(12345);
    rng2->setSeed(12345);
    
    rng1->generateUniform(mem1->data(), 100, PrecisionType::DOUBLE);
    rng2->generateUniform(mem2->data(), 100, PrecisionType::DOUBLE);
    
    double* data1 = static_cast<double*>(mem1->data());
    double* data2 = static_cast<double*>(mem2->data());
    
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(data1[i], data2[i]);
    }
}

// ====================
// Cross-Backend Comparison
// ====================

TEST_F(ComputeTest, CrossBackend_DifferentSeedsProduceDifferentResults) {
    auto cuda_rng = std::make_unique<CudaRandom>();
    auto cpu_rng = std::make_unique<CpuRandom>();
    
    auto cuda_mem = std::make_unique<CudaMemory>(100 * sizeof(double));
    auto cpu_mem = std::make_unique<CpuMemory>(100 * sizeof(double));
    
    // Different seeds
    cuda_rng->setSeed(111);
    cpu_rng->setSeed(222);
    
    cuda_rng->generateUniform(cuda_mem->data(), 100, PrecisionType::DOUBLE);
    cpu_rng->generateUniform(cpu_mem->data(), 100, PrecisionType::DOUBLE);
    
    std::vector<double> cuda_data(100);
    cuda_mem->copyToHost(cuda_data.data(), 100 * sizeof(double));
    double* cpu_data = static_cast<double*>(cpu_mem->data());
    
    // Should be different (at least some values)
    bool has_difference = false;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(cuda_data[i] - cpu_data[i]) > 1e-10) {
            has_difference = true;
            break;
        }
    }
    EXPECT_TRUE(has_difference);
}

// ====================
// Interface Polymorphism Tests
// ====================

TEST_F(ComputeTest, IRandomGenerator_Polymorphism) {
    // Test that we can use IRandomGenerator* for both backends
    std::unique_ptr<IRandomGenerator> cuda_rng = std::make_unique<CudaRandom>();
    std::unique_ptr<IRandomGenerator> cpu_rng = std::make_unique<CpuRandom>();
    
    auto cuda_mem = std::make_unique<CudaMemory>(50 * sizeof(double));
    auto cpu_mem = std::make_unique<CpuMemory>(50 * sizeof(double));
    
    // Both should work through interface
    EXPECT_NO_THROW(cuda_rng->setSeed(42));
    EXPECT_NO_THROW(cpu_rng->setSeed(42));
    
    EXPECT_NO_THROW(cuda_rng->generateUniform(cuda_mem->data(), 50, PrecisionType::DOUBLE));
    EXPECT_NO_THROW(cpu_rng->generateUniform(cpu_mem->data(), 50, PrecisionType::DOUBLE));
}
