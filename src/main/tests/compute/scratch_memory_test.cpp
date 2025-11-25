#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include <vector>

#include "compute/memory/IScratchMemory.h"
#include "compute/memory/CudaScratchMemory.h"
#include "compute/memory/CpuScratchMemory.h"

// =============================================================================
// CUDA Scratch Memory Tests
// =============================================================================

class CudaScratchMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(CudaScratchMemoryTest, DefaultConstructor) {
    CudaScratchMemory scratch(0, 0);
    EXPECT_EQ(scratch.capacity(), 0);
    EXPECT_EQ(scratch.getBackend(), DeviceBackend::CUDA);
}

TEST_F(CudaScratchMemoryTest, ConstructorWithInitialCapacity) {
    const size_t initial_size = 1024;
    CudaScratchMemory scratch(initial_size, 0);
    
    EXPECT_GE(scratch.capacity(), initial_size);
    EXPECT_EQ(scratch.getBackend(), DeviceBackend::CUDA);
}

TEST_F(CudaScratchMemoryTest, RequestZeroBytesThrows) {
    CudaScratchMemory scratch(0, 0);
    EXPECT_THROW(scratch.request(0), std::invalid_argument);
}

TEST_F(CudaScratchMemoryTest, FirstRequest) {
    CudaScratchMemory scratch(0, 0);
    const size_t size = 2048;
    
    void* ptr = scratch.request(size);
    
    EXPECT_NE(ptr, nullptr);
    EXPECT_GE(scratch.capacity(), size);
    
    // Verify it's valid device memory by attempting a memset
    cudaError_t err = cudaMemset(ptr, 0, size);
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(CudaScratchMemoryTest, SubsequentSmallerRequestReusesMemory) {
    CudaScratchMemory scratch(0, 0);
    
    // First request
    const size_t size1 = 4096;
    void* ptr1 = scratch.request(size1);
    size_t capacity1 = scratch.capacity();
    
    // Second smaller request should return same pointer
    const size_t size2 = 2048;
    void* ptr2 = scratch.request(size2);
    size_t capacity2 = scratch.capacity();
    
    EXPECT_EQ(ptr1, ptr2);  // Same pointer
    EXPECT_EQ(capacity1, capacity2);  // Capacity unchanged
}

TEST_F(CudaScratchMemoryTest, GrowthStrategy) {
    CudaScratchMemory scratch(0, 0);
    
    // First request
    const size_t size1 = 1000;
    scratch.request(size1);
    size_t capacity1 = scratch.capacity();
    EXPECT_GE(capacity1, size1);
    
    // Request slightly larger (should trigger growth with 1.5x strategy)
    const size_t size2 = capacity1 + 100;
    scratch.request(size2);
    size_t capacity2 = scratch.capacity();
    
    // New capacity should be at least requested size
    EXPECT_GE(capacity2, size2);
    // Growth strategy should allocate more than just requested
    // (either 1.5x old capacity or requested size, whichever is larger)
    size_t expected_min = std::max(size2, capacity1 + capacity1 / 2);
    EXPECT_GE(capacity2, expected_min);
}

TEST_F(CudaScratchMemoryTest, LargeAllocation) {
    CudaScratchMemory scratch(0, 0);
    
    // Request 100 MB
    const size_t large_size = 100 * 1024 * 1024;
    
    // Check if we have enough free memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    if (free_mem < large_size * 2) {
        GTEST_SKIP() << "Insufficient GPU memory for large allocation test";
    }
    
    void* ptr = scratch.request(large_size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_GE(scratch.capacity(), large_size);
    
    // Verify memory is accessible
    cudaError_t err = cudaMemset(ptr, 0, large_size);
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(CudaScratchMemoryTest, Release) {
    CudaScratchMemory scratch(0, 0);
    
    // Allocate some memory
    const size_t size = 4096;
    void* ptr = scratch.request(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_GT(scratch.capacity(), 0);
    
    // Release
    scratch.release();
    EXPECT_EQ(scratch.capacity(), 0);
    
    // Can request again after release
    void* new_ptr = scratch.request(size);
    EXPECT_NE(new_ptr, nullptr);
    EXPECT_GE(scratch.capacity(), size);
}

TEST_F(CudaScratchMemoryTest, MultipleReleases) {
    CudaScratchMemory scratch(0, 0);
    scratch.request(1024);
    
    // Multiple releases should be safe
    scratch.release();
    scratch.release();
    scratch.release();
    
    EXPECT_EQ(scratch.capacity(), 0);
}

TEST_F(CudaScratchMemoryTest, MoveConstructor) {
    CudaScratchMemory scratch1(0, 0);
    const size_t size = 2048;
    void* ptr1 = scratch1.request(size);
    size_t capacity1 = scratch1.capacity();
    
    // Move construct
    CudaScratchMemory scratch2(std::move(scratch1));
    
    // scratch2 should have the resources
    EXPECT_EQ(scratch2.capacity(), capacity1);
    void* ptr2 = scratch2.request(size);
    EXPECT_EQ(ptr2, ptr1);  // Same underlying allocation
    
    // scratch1 should be empty
    EXPECT_EQ(scratch1.capacity(), 0);
}

TEST_F(CudaScratchMemoryTest, MoveAssignment) {
    CudaScratchMemory scratch1(0, 0);
    const size_t size = 2048;
    void* ptr1 = scratch1.request(size);
    size_t capacity1 = scratch1.capacity();
    
    CudaScratchMemory scratch2(0, 0);
    scratch2.request(1024);  // Pre-allocate scratch2
    
    // Move assign
    scratch2 = std::move(scratch1);
    
    // scratch2 should have scratch1's resources
    EXPECT_EQ(scratch2.capacity(), capacity1);
    void* ptr2 = scratch2.request(size);
    EXPECT_EQ(ptr2, ptr1);
    
    // scratch1 should be empty
    EXPECT_EQ(scratch1.capacity(), 0);
}

TEST_F(CudaScratchMemoryTest, RepeatedRequestsSimulateRealUsage) {
    CudaScratchMemory scratch(0, 0);
    
    // Simulate repeated SVD workspace requests of varying sizes
    std::vector<size_t> request_sizes = {
        1024, 2048, 1500, 3000, 2500, 5000, 4000, 10000, 8000
    };
    
    void* last_ptr = nullptr;
    size_t reallocation_count = 0;
    
    for (size_t size : request_sizes) {
        void* ptr = scratch.request(size);
        EXPECT_NE(ptr, nullptr);
        EXPECT_GE(scratch.capacity(), size);
        
        // Count reallocations (pointer changes)
        if (last_ptr != nullptr && ptr != last_ptr) {
            reallocation_count++;
        }
        last_ptr = ptr;
    }
    
    // With growth strategy, should have fewer reallocations than requests
    EXPECT_LT(reallocation_count, request_sizes.size());
}

// =============================================================================
// CPU Scratch Memory Tests
// =============================================================================

class CpuScratchMemoryTest : public ::testing::Test {
};

TEST_F(CpuScratchMemoryTest, DefaultConstructor) {
    CpuScratchMemory scratch;
    EXPECT_EQ(scratch.capacity(), 0);
    EXPECT_EQ(scratch.getBackend(), DeviceBackend::CPU);
}

TEST_F(CpuScratchMemoryTest, ConstructorWithInitialCapacity) {
    const size_t initial_size = 1024;
    CpuScratchMemory scratch(initial_size);
    
    EXPECT_GE(scratch.capacity(), initial_size);
    EXPECT_EQ(scratch.getBackend(), DeviceBackend::CPU);
}

TEST_F(CpuScratchMemoryTest, RequestZeroBytesThrows) {
    CpuScratchMemory scratch;
    EXPECT_THROW(scratch.request(0), std::invalid_argument);
}

TEST_F(CpuScratchMemoryTest, FirstRequest) {
    CpuScratchMemory scratch;
    const size_t size = 2048;
    
    void* ptr = scratch.request(size);
    
    EXPECT_NE(ptr, nullptr);
    EXPECT_GE(scratch.capacity(), size);
    
    // Verify memory is accessible
    memset(ptr, 0, size);
    
    // Check alignment (should be 64-byte aligned)
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 64, 0) << "Memory should be 64-byte aligned";
}

TEST_F(CpuScratchMemoryTest, MemoryAlignment) {
    CpuScratchMemory scratch;
    
    // Test various sizes
    std::vector<size_t> sizes = {1, 64, 128, 1024, 4096, 10000};
    
    for (size_t size : sizes) {
        void* ptr = scratch.request(size);
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % 64, 0) << "Memory should be 64-byte aligned for size " << size;
    }
}

TEST_F(CpuScratchMemoryTest, SubsequentSmallerRequestReusesMemory) {
    CpuScratchMemory scratch;
    
    // First request
    const size_t size1 = 4096;
    void* ptr1 = scratch.request(size1);
    size_t capacity1 = scratch.capacity();
    
    // Second smaller request should return same pointer
    const size_t size2 = 2048;
    void* ptr2 = scratch.request(size2);
    size_t capacity2 = scratch.capacity();
    
    EXPECT_EQ(ptr1, ptr2);
    EXPECT_EQ(capacity1, capacity2);
}

TEST_F(CpuScratchMemoryTest, GrowthStrategy) {
    CpuScratchMemory scratch;
    
    // First request
    const size_t size1 = 1000;
    scratch.request(size1);
    size_t capacity1 = scratch.capacity();
    EXPECT_GE(capacity1, size1);
    
    // Request larger
    const size_t size2 = capacity1 + 100;
    scratch.request(size2);
    size_t capacity2 = scratch.capacity();
    
    EXPECT_GE(capacity2, size2);
    size_t expected_min = std::max(size2, capacity1 + capacity1 / 2);
    EXPECT_GE(capacity2, expected_min);
}

TEST_F(CpuScratchMemoryTest, LargeAllocation) {
    CpuScratchMemory scratch;
    
    // Request 100 MB
    const size_t large_size = 100 * 1024 * 1024;
    
    void* ptr = scratch.request(large_size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_GE(scratch.capacity(), large_size);
    
    // Verify memory is accessible (write pattern)
    char* bytes = static_cast<char*>(ptr);
    bytes[0] = 'A';
    bytes[large_size - 1] = 'Z';
    EXPECT_EQ(bytes[0], 'A');
    EXPECT_EQ(bytes[large_size - 1], 'Z');
}

TEST_F(CpuScratchMemoryTest, Release) {
    CpuScratchMemory scratch;
    
    const size_t size = 4096;
    void* ptr = scratch.request(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_GT(scratch.capacity(), 0);
    
    scratch.release();
    EXPECT_EQ(scratch.capacity(), 0);
    
    // Can request again
    void* new_ptr = scratch.request(size);
    EXPECT_NE(new_ptr, nullptr);
    EXPECT_GE(scratch.capacity(), size);
}

TEST_F(CpuScratchMemoryTest, MultipleReleases) {
    CpuScratchMemory scratch;
    scratch.request(1024);
    
    scratch.release();
    scratch.release();
    scratch.release();
    
    EXPECT_EQ(scratch.capacity(), 0);
}

TEST_F(CpuScratchMemoryTest, MoveConstructor) {
    CpuScratchMemory scratch1;
    const size_t size = 2048;
    void* ptr1 = scratch1.request(size);
    size_t capacity1 = scratch1.capacity();
    
    CpuScratchMemory scratch2(std::move(scratch1));
    
    EXPECT_EQ(scratch2.capacity(), capacity1);
    void* ptr2 = scratch2.request(size);
    EXPECT_EQ(ptr2, ptr1);
    
    EXPECT_EQ(scratch1.capacity(), 0);
}

TEST_F(CpuScratchMemoryTest, MoveAssignment) {
    CpuScratchMemory scratch1;
    const size_t size = 2048;
    void* ptr1 = scratch1.request(size);
    size_t capacity1 = scratch1.capacity();
    
    CpuScratchMemory scratch2;
    scratch2.request(1024);
    
    scratch2 = std::move(scratch1);
    
    EXPECT_EQ(scratch2.capacity(), capacity1);
    void* ptr2 = scratch2.request(size);
    EXPECT_EQ(ptr2, ptr1);
    
    EXPECT_EQ(scratch1.capacity(), 0);
}

TEST_F(CpuScratchMemoryTest, ReadWriteVerification) {
    CpuScratchMemory scratch;
    const size_t size = 1024;
    
    void* ptr = scratch.request(size);
    double* data = static_cast<double*>(ptr);
    
    // Write pattern
    const size_t num_doubles = size / sizeof(double);
    for (size_t i = 0; i < num_doubles; i++) {
        data[i] = static_cast<double>(i) * 3.14;
    }
    
    // Read back and verify
    for (size_t i = 0; i < num_doubles; i++) {
        EXPECT_DOUBLE_EQ(data[i], static_cast<double>(i) * 3.14);
    }
}

// =============================================================================
// Interface Polymorphism Tests
// =============================================================================

TEST(ScratchMemoryInterfaceTest, CudaPolymorphism) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    std::unique_ptr<IScratchMemory> scratch = std::make_unique<CudaScratchMemory>(0, 0);
    
    EXPECT_EQ(scratch->capacity(), 0);
    EXPECT_EQ(scratch->getBackend(), DeviceBackend::CUDA);
    
    void* ptr = scratch->request(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_GE(scratch->capacity(), 1024);
}

TEST(ScratchMemoryInterfaceTest, CpuPolymorphism) {
    std::unique_ptr<IScratchMemory> scratch = std::make_unique<CpuScratchMemory>();
    
    EXPECT_EQ(scratch->capacity(), 0);
    EXPECT_EQ(scratch->getBackend(), DeviceBackend::CPU);
    
    void* ptr = scratch->request(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_GE(scratch->capacity(), 1024);
}
