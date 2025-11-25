#include <gtest/gtest.h>
#include "fixtures/ComputeTestFixture.h"
#include "compute/backends/cuda/CudaStream.h"
#include "compute/backends/cpu/CpuStream.h"
#include "compute/backends/cuda/CudaMemory.h"
#include <cuda_runtime.h>

// Simple CUDA kernel for testing async execution
__global__ void delayKernel(int* flag, int delay_ms) {
    // Busy wait to simulate work
    clock_t start = clock();
    clock_t target = start + delay_ms * 1000;  // Rough approximation
    while (clock() < target) {
        // Spin
    }
    *flag = 1;
}

// ====================
// CUDA Stream Tests
// ====================

TEST_F(ComputeTest, CudaStream_Creation) {
    auto stream = std::make_unique<CudaStream>(0);
    
    EXPECT_NE(stream->getNativeHandle(), nullptr);
}

TEST_F(ComputeTest, CudaStream_Synchronize) {
    auto stream = std::make_unique<CudaStream>(0);
    auto mem = std::make_unique<CudaMemory>(sizeof(int), 0);
    
    int* device_flag = static_cast<int*>(mem->data());
    int host_flag = 0;
    
    // Initialize to 0
    mem->copyFromHost(&host_flag, sizeof(int));
    
    // Launch kernel on stream
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream->getNativeHandle());
    delayKernel<<<1, 1, 0, cuda_stream>>>(device_flag, 10);
    
    // Synchronize
    EXPECT_NO_THROW(stream->synchronize());
    
    // Verify kernel completed
    mem->copyToHost(&host_flag, sizeof(int));
    EXPECT_EQ(host_flag, 1);
}

TEST_F(ComputeTest, CudaStream_IsComplete) {
    auto stream = std::make_unique<CudaStream>(0);
    
    // Stream should be idle initially
    EXPECT_TRUE(stream->isComplete());
    
    auto mem = std::make_unique<CudaMemory>(sizeof(int), 0);
    int* device_flag = static_cast<int*>(mem->data());
    
    // Launch long-running kernel
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream->getNativeHandle());
    delayKernel<<<1, 1, 0, cuda_stream>>>(device_flag, 100);
    
    // Immediately check - should be running
    bool initially_complete = stream->isComplete();
    
    // Wait for completion
    stream->synchronize();
    
    // Now should be complete
    EXPECT_TRUE(stream->isComplete());
    
    // Note: Depending on GPU speed, initially_complete might be true or false
    // So we don't assert on it, just verify final state is correct
}

TEST_F(ComputeTest, CudaStream_MoveSemantics) {
    auto stream1 = std::make_unique<CudaStream>(0);
    void* original_handle = stream1->getNativeHandle();
    
    // Move construct
    CudaStream stream2(std::move(*stream1));
    EXPECT_EQ(stream2.getNativeHandle(), original_handle);
    EXPECT_EQ(stream1->getNativeHandle(), nullptr);
}

TEST_F(ComputeTest, CudaStream_MultipleStreams) {
    // Create multiple streams
    auto stream1 = std::make_unique<CudaStream>(0);
    auto stream2 = std::make_unique<CudaStream>(0);
    auto stream3 = std::make_unique<CudaStream>(0);
    
    // All should have different handles
    EXPECT_NE(stream1->getNativeHandle(), stream2->getNativeHandle());
    EXPECT_NE(stream2->getNativeHandle(), stream3->getNativeHandle());
    EXPECT_NE(stream1->getNativeHandle(), stream3->getNativeHandle());
}

// ====================
// CPU Stream Tests
// ====================

TEST_F(ComputeTest, CpuStream_Creation) {
    auto stream = std::make_unique<CpuStream>();
    
    // CPU streams are no-ops, so handle can be null
    EXPECT_NO_THROW(stream->getNativeHandle());
}

TEST_F(ComputeTest, CpuStream_Synchronize) {
    auto stream = std::make_unique<CpuStream>();
    
    // Should be a no-op
    EXPECT_NO_THROW(stream->synchronize());
}

TEST_F(ComputeTest, CpuStream_IsComplete) {
    auto stream = std::make_unique<CpuStream>();
    
    // CPU operations are synchronous, always complete
    EXPECT_TRUE(stream->isComplete());
}

TEST_F(ComputeTest, CpuStream_MoveSemantics) {
    auto stream1 = std::make_unique<CpuStream>();
    
    // CPU streams are stateless, just verify we can create and destroy them
    EXPECT_TRUE(stream1->isComplete());
}

// ====================
// Interface Polymorphism Tests
// ====================

TEST_F(ComputeTest, IStream_Polymorphism) {
    // Test that we can use IStream* for both backends
    std::unique_ptr<IStream> cuda_stream = std::make_unique<CudaStream>(0);
    std::unique_ptr<IStream> cpu_stream = std::make_unique<CpuStream>();
    
    // Both should work through interface
    EXPECT_NO_THROW(cuda_stream->synchronize());
    EXPECT_NO_THROW(cpu_stream->synchronize());
    
    EXPECT_NO_THROW(cuda_stream->isComplete());
    EXPECT_NO_THROW(cpu_stream->isComplete());
}
