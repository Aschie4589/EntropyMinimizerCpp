#include <gtest/gtest.h>
#include "fixtures/ComputeTestFixture.h"
#include "compute/backends/cuda/CudaMemory.h"
#include "compute/backends/cpu/CpuMemory.h"
#include <vector>
#include <complex>

// ====================
// CUDA Memory Tests
// ====================

TEST_F(ComputeTest, CudaMemory_Allocation) {
    // Test basic allocation
    size_t size = 1024;
    auto mem = std::make_unique<CudaMemory>(size, 0);
    
    EXPECT_NE(mem->data(), nullptr);
    EXPECT_EQ(mem->size(), size);
    EXPECT_EQ(mem->getBackend(), DeviceBackend::CUDA);
}

TEST_F(ComputeTest, CudaMemory_CopyFromHost) {
    size_t count = 100;
    auto mem = std::make_unique<CudaMemory>(count * sizeof(double), 0);
    
    // Create test data
    auto host_data = generateRandomDoubles(count);
    
    // Copy to device
    EXPECT_NO_THROW(mem->copyFromHost(host_data.data(), count * sizeof(double)));
    
    // Copy back and verify
    std::vector<double> result(count);
    EXPECT_NO_THROW(mem->copyToHost(result.data(), count * sizeof(double)));
    
    EXPECT_TRUE(vectorsAlmostEqual(host_data, result));
}

TEST_F(ComputeTest, CudaMemory_CopyToHost) {
    size_t count = 50;
    auto mem = std::make_unique<CudaMemory>(count * sizeof(std::complex<double>), 0);
    
    // Create complex test data
    auto host_data = generateRandomComplex(count);
    
    // Copy to device
    mem->copyFromHost(host_data.data(), count * sizeof(std::complex<double>));
    
    // Copy back
    std::vector<std::complex<double>> result(count);
    mem->copyToHost(result.data(), count * sizeof(std::complex<double>));
    
    EXPECT_TRUE(vectorsAlmostEqual(host_data, result, 1e-15));
}

TEST_F(ComputeTest, CudaMemory_Fill) {
    size_t size = 256;
    auto mem = std::make_unique<CudaMemory>(size, 0);
    
    // Fill with pattern
    mem->fill(0xAB, size);
    
    // Verify
    std::vector<unsigned char> result(size);
    mem->copyToHost(result.data(), size);
    
    for (auto byte : result) {
        EXPECT_EQ(byte, 0xAB);
    }
}

TEST_F(ComputeTest, CudaMemory_MoveSemantics) {
    size_t size = 512;
    auto mem1 = std::make_unique<CudaMemory>(size, 0);
    void* original_ptr = mem1->data();
    
    // Move construct
    CudaMemory mem2(std::move(*mem1));
    EXPECT_EQ(mem2.data(), original_ptr);
    EXPECT_EQ(mem2.size(), size);
    EXPECT_EQ(mem1->data(), nullptr);  // Moved-from object is empty
    EXPECT_EQ(mem1->size(), 0);
}

TEST_F(ComputeTest, CudaMemory_DeviceToDeviceCopy) {
    size_t count = 128;
    auto mem1 = std::make_unique<CudaMemory>(count * sizeof(double), 0);
    auto mem2 = std::make_unique<CudaMemory>(count * sizeof(double), 0);
    
    // Fill mem1 with data
    auto host_data = generateRandomDoubles(count);
    mem1->copyFromHost(host_data.data(), count * sizeof(double));
    
    // Copy from mem1 to mem2 (device-to-device)
    EXPECT_NO_THROW(mem2->copyFrom(mem1.get(), count * sizeof(double)));
    
    // Verify
    std::vector<double> result(count);
    mem2->copyToHost(result.data(), count * sizeof(double));
    
    EXPECT_TRUE(vectorsAlmostEqual(host_data, result));
}

// ====================
// CPU Memory Tests
// ====================

TEST_F(ComputeTest, CpuMemory_Allocation) {
    size_t size = 2048;
    auto mem = std::make_unique<CpuMemory>(size);
    
    EXPECT_NE(mem->data(), nullptr);
    EXPECT_EQ(mem->size(), size);
    EXPECT_EQ(mem->getBackend(), DeviceBackend::CPU);
}

TEST_F(ComputeTest, CpuMemory_CopyFromHost) {
    size_t count = 200;
    auto mem = std::make_unique<CpuMemory>(count * sizeof(double));
    
    auto host_data = generateRandomDoubles(count);
    
    // Copy to "device" (CPU memory)
    EXPECT_NO_THROW(mem->copyFromHost(host_data.data(), count * sizeof(double)));
    
    // Verify by direct access
    auto* cpu_ptr = static_cast<const double*>(mem->data());
    for (size_t i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(cpu_ptr[i], host_data[i]);
    }
}

TEST_F(ComputeTest, CpuMemory_Fill) {
    size_t size = 1024;
    auto mem = std::make_unique<CpuMemory>(size);
    
    mem->fill(0x42, size);
    
    auto* ptr = static_cast<const unsigned char*>(mem->data());
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(ptr[i], 0x42);
    }
}

TEST_F(ComputeTest, CpuMemory_MoveSemantics) {
    size_t size = 256;
    auto mem1 = std::make_unique<CpuMemory>(size);
    void* original_ptr = mem1->data();
    
    // Move construct
    CpuMemory mem2(std::move(*mem1));
    EXPECT_EQ(mem2.data(), original_ptr);
    EXPECT_EQ(mem2.size(), size);
    EXPECT_EQ(mem1->data(), nullptr);
    EXPECT_EQ(mem1->size(), 0);
}

// ====================
// Cross-Backend Tests
// ====================

TEST_F(ComputeTest, CrossBackend_CpuToCuda) {
    size_t count = 75;
    auto cpu_mem = std::make_unique<CpuMemory>(count * sizeof(double));
    auto cuda_mem = std::make_unique<CudaMemory>(count * sizeof(double), 0);
    
    // Fill CPU memory
    auto host_data = generateRandomDoubles(count);
    cpu_mem->copyFromHost(host_data.data(), count * sizeof(double));
    
    // Copy from CPU to CUDA
    EXPECT_NO_THROW(cuda_mem->copyFrom(cpu_mem.get(), count * sizeof(double)));
    
    // Verify by copying back from CUDA
    std::vector<double> result(count);
    cuda_mem->copyToHost(result.data(), count * sizeof(double));
    
    EXPECT_TRUE(vectorsAlmostEqual(host_data, result));
}

TEST_F(ComputeTest, CrossBackend_CudaToCpu) {
    size_t count = 60;
    auto cuda_mem = std::make_unique<CudaMemory>(count * sizeof(std::complex<double>), 0);
    auto cpu_mem = std::make_unique<CpuMemory>(count * sizeof(std::complex<double>));
    
    // Fill CUDA memory
    auto host_data = generateRandomComplex(count);
    cuda_mem->copyFromHost(host_data.data(), count * sizeof(std::complex<double>));
    
    // Copy from CUDA to CPU
    EXPECT_NO_THROW(cpu_mem->copyFrom(cuda_mem.get(), count * sizeof(std::complex<double>)));
    
    // Verify by direct access to CPU memory
    auto* cpu_ptr = static_cast<const std::complex<double>*>(cpu_mem->data());
    for (size_t i = 0; i < count; ++i) {
        EXPECT_TRUE(almostEqual(cpu_ptr[i], host_data[i]));
    }
}

// ====================
// Error Handling Tests
// ====================

TEST_F(ComputeTest, CudaMemory_ZeroAllocationThrows) {
    EXPECT_THROW(CudaMemory(0, 0), std::invalid_argument);
}

TEST_F(ComputeTest, CpuMemory_ZeroAllocationThrows) {
    EXPECT_THROW(CpuMemory(0), std::invalid_argument);
}

TEST_F(ComputeTest, CudaMemory_CopyExceedsSizeThrows) {
    auto mem = std::make_unique<CudaMemory>(100, 0);
    std::vector<unsigned char> data(200);
    
    EXPECT_THROW(mem->copyFromHost(data.data(), 200), std::invalid_argument);
}

TEST_F(ComputeTest, CpuMemory_CopyExceedsSizeThrows) {
    auto mem = std::make_unique<CpuMemory>(50);
    std::vector<unsigned char> data(100);
    
    EXPECT_THROW(mem->copyFromHost(data.data(), 100), std::invalid_argument);
}
