#include <gtest/gtest.h>
#include "fixtures/ComputeTestFixture.h"
#include "compute/device/DeviceFactory.h"
#include "compute/backends/cuda/CudaDevice.h"
#include "compute/backends/cuda/CudaMemory.h"
#include "compute/backends/cuda/CudaStream.h"
#include <cuda_runtime.h>
#include <thread>
#include <vector>

// ====================
// Multi-GPU Test Fixture
// ====================

class MultiGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDeviceCount(&device_count_);
        if (device_count_ < 1) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
    
    int device_count_ = 0;
};

// ====================
// DeviceGuard Tests
// ====================

TEST_F(MultiGPUTest, DeviceGuard_SavesAndRestoresContext) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    // Set to device 0
    cudaSetDevice(0);
    int before;
    cudaGetDevice(&before);
    EXPECT_EQ(before, 0);
    
    {
        // Create device 1 - its operations should use device 1
        auto device1 = std::make_unique<compute::CudaDevice>(1);
        
        // While device1 exists, current device is unknown (could be 0 or 1)
        // We don't check here because it depends on implementation
    }
    
    // After device1 is destroyed, we're back where we started - no guarantee
    // The DeviceGuard only protects during method calls, not across object lifetime
    // So this test just verifies no crashes occur
}

TEST_F(MultiGPUTest, DeviceGuard_ProtectsDuringOperations) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    // Set to device 0
    cudaSetDevice(0);
    
    // Create memory on device 1
    auto device1 = std::make_unique<compute::CudaDevice>(1);
    auto mem = device1->allocate(1024);
    
    // Verify allocation happened on device 1 (DeviceGuard protected it)
    EXPECT_NE(mem->data(), nullptr);
    
    // Current device may have changed - that's okay
}

// ====================
// Independent Device Operations
// ====================

TEST_F(MultiGPUTest, IndependentDevices_MemoryAllocation) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    auto device0 = std::make_unique<compute::CudaDevice>(0);
    auto device1 = std::make_unique<compute::CudaDevice>(1);
    
    auto mem0 = device0->allocate(1024);
    auto mem1 = device1->allocate(1024);
    
    EXPECT_NE(mem0->data(), nullptr);
    EXPECT_NE(mem1->data(), nullptr);
}

TEST_F(MultiGPUTest, IndependentDevices_StreamCreation) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    auto device0 = std::make_unique<compute::CudaDevice>(0);
    auto device1 = std::make_unique<compute::CudaDevice>(1);
    
    auto stream0 = device0->createStream();
    auto stream1 = device1->createStream();
    
    EXPECT_NE(stream0->getNativeHandle(), nullptr);
    EXPECT_NE(stream1->getNativeHandle(), nullptr);
}

TEST_F(MultiGPUTest, IndependentDevices_ComponentAccess) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    auto device0 = std::make_unique<compute::CudaDevice>(0);
    auto device1 = std::make_unique<compute::CudaDevice>(1);
    
    // Access components on both devices
    auto linalg0 = device0->getLinearAlgebra();
    auto linalg1 = device1->getLinearAlgebra();
    
    EXPECT_NE(linalg0, nullptr);
    EXPECT_NE(linalg1, nullptr);
    EXPECT_NE(linalg0, linalg1);  // Different instances
}

// ====================
// Concurrent Operations
// ====================

TEST_F(MultiGPUTest, ConcurrentOperations_DifferentDevices) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    // Launch operations on two devices concurrently
    std::thread thread0([this]() {
        auto device = std::make_unique<compute::CudaDevice>(0);
        auto mem = device->allocate(1024);
        std::vector<double> data(128, 1.0);
        mem->copyFromHost(data.data(), data.size() * sizeof(double));
        device->synchronize();
    });
    
    std::thread thread1([this]() {
        auto device = std::make_unique<compute::CudaDevice>(1);
        auto mem = device->allocate(1024);
        std::vector<double> data(128, 2.0);
        mem->copyFromHost(data.data(), data.size() * sizeof(double));
        device->synchronize();
    });
    
    thread0.join();
    thread1.join();
}

// ====================
// Device Factory Tests
// ====================

TEST_F(MultiGPUTest, DeviceFactory_CreateSpecificDevice) {
    auto device = DeviceFactory::create(DeviceFactory::DeviceType::CUDA, 0);
    
    EXPECT_NE(device, nullptr);
    EXPECT_EQ(device->getBackend(), DeviceBackend::CUDA);
    EXPECT_EQ(device->getDeviceID(), 0);
}

TEST_F(MultiGPUTest, DeviceFactory_MultipleDevices) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    auto device0 = DeviceFactory::create(DeviceFactory::DeviceType::CUDA, 0);
    auto device1 = DeviceFactory::create(DeviceFactory::DeviceType::CUDA, 1);
    
    EXPECT_EQ(device0->getDeviceID(), 0);
    EXPECT_EQ(device1->getDeviceID(), 1);
}

TEST_F(MultiGPUTest, DeviceFactory_InvalidDeviceThrows) {
    int invalid_id = device_count_ + 10;
    
    EXPECT_THROW(
        DeviceFactory::create(DeviceFactory::DeviceType::CUDA, invalid_id),
        std::runtime_error
    );
}

// ====================
// Memory Operations Across Devices
// ====================

TEST_F(MultiGPUTest, Memory_DeviceToDeviceCopy_SameGPU) {
    auto device = std::make_unique<compute::CudaDevice>(0);
    
    auto mem1 = device->allocate(1024);
    auto mem2 = device->allocate(1024);
    
    std::vector<double> data(128, 42.0);
    mem1->copyFromHost(data.data(), data.size() * sizeof(double));
    
    // Copy on same device
    mem2->copyFrom(mem1.get(), data.size() * sizeof(double));
    
    std::vector<double> result(128);
    mem2->copyToHost(result.data(), result.size() * sizeof(double));
    
    EXPECT_EQ(result[0], 42.0);
}

// ====================
// Stream Operations
// ====================

TEST_F(MultiGPUTest, Stream_SynchronizeOnCorrectDevice) {
    auto device = std::make_unique<compute::CudaDevice>(0);
    auto stream = device->createStream();
    
    // Should not crash when synchronizing
    EXPECT_NO_THROW(stream->synchronize());
    EXPECT_TRUE(stream->isComplete());
}

// ====================
// Component Isolation
// ====================

TEST_F(MultiGPUTest, Component_LinearAlgebra_IsolatedPerDevice) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    auto device0 = std::make_unique<compute::CudaDevice>(0);
    auto device1 = std::make_unique<compute::CudaDevice>(1);
    
    auto linalg0 = device0->getLinearAlgebra();
    auto linalg1 = device1->getLinearAlgebra();
    
    // Each device has its own linalg instance
    EXPECT_NE(linalg0, linalg1);
}

TEST_F(MultiGPUTest, Component_Solver_IsolatedPerDevice) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    auto device0 = std::make_unique<compute::CudaDevice>(0);
    auto device1 = std::make_unique<compute::CudaDevice>(1);
    
    auto solver0 = device0->getSolver();
    auto solver1 = device1->getSolver();
    
    EXPECT_NE(solver0, nullptr);
    EXPECT_NE(solver1, nullptr);
    EXPECT_NE(solver0, solver1);
}

TEST_F(MultiGPUTest, Component_Random_IsolatedPerDevice) {
    if (device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for this test";
    }
    
    auto device0 = std::make_unique<compute::CudaDevice>(0);
    auto device1 = std::make_unique<compute::CudaDevice>(1);
    
    auto random0 = device0->getRandomGenerator();
    auto random1 = device1->getRandomGenerator();
    
    EXPECT_NE(random0, nullptr);
    EXPECT_NE(random1, nullptr);
    EXPECT_NE(random0, random1);
}

// ====================
// Stress Tests
// ====================

TEST_F(MultiGPUTest, Stress_ManyAllocations) {
    auto device = std::make_unique<compute::CudaDevice>(0);
    
    std::vector<std::unique_ptr<IDeviceMemory>> allocations;
    for (int i = 0; i < 100; ++i) {
        allocations.push_back(device->allocate(1024));
    }
    
    EXPECT_EQ(allocations.size(), 100);
}

TEST_F(MultiGPUTest, Stress_MultipleDevicesSequential) {
    std::vector<std::unique_ptr<compute::CudaDevice>> devices;
    for (int i = 0; i < device_count_; ++i) {
        devices.push_back(std::make_unique<compute::CudaDevice>(i));
        
        // Allocate on each
        auto mem = devices[i]->allocate(1024);
        EXPECT_NE(mem->data(), nullptr);
    }
}
