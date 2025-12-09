#include <gtest/gtest.h>
#include "minimizer/orchestration/device_pool.h"
#include "minimizer/orchestration/resource_monitor.h"
#include <thread>
#include <vector>
#include <set>

using namespace entropy;

// ============================================================================
// Test Fixture
// ============================================================================

class DevicePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability for conditional tests
        cuda_available_ = DeviceFactory::isCudaAvailable();
        if (cuda_available_) {
            cuda_device_count_ = DeviceFactory::getCudaDeviceCount();
        }
    }
    
    bool cuda_available_;
    int cuda_device_count_;
};

// ============================================================================
// Constructor and Validation Tests
// ============================================================================

TEST_F(DevicePoolTest, CPUOnlyConfiguration) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 4;
    
    DevicePool pool(limits);
    
    EXPECT_EQ(pool.numDevices(), 4);
    EXPECT_EQ(pool.numGPUs(), 0);
    EXPECT_EQ(pool.numCPUs(), 4);
}

TEST_F(DevicePoolTest, InvalidNegativeGPUs) {
    ResourceLimits limits;
    limits.max_gpus = -2;
    limits.max_cpus = 4;
    
    EXPECT_THROW(DevicePool pool(limits), std::invalid_argument);
}

TEST_F(DevicePoolTest, InvalidNegativeCPUs) {
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = -1;
    
    EXPECT_THROW(DevicePool pool(limits), std::invalid_argument);
}

TEST_F(DevicePoolTest, InvalidZeroDevices) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 0;
    
    EXPECT_THROW(DevicePool pool(limits), std::invalid_argument);
}

TEST_F(DevicePoolTest, GPUOnlyConfiguration) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    ResourceLimits limits;
    limits.max_gpus = std::min(2, cuda_device_count_);
    limits.max_cpus = 0;
    
    DevicePool pool(limits);
    
    EXPECT_EQ(pool.numDevices(), limits.max_gpus);
    EXPECT_EQ(pool.numGPUs(), limits.max_gpus);
    EXPECT_EQ(pool.numCPUs(), 0);
}

TEST_F(DevicePoolTest, MixedConfiguration) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    ResourceLimits limits;
    limits.max_gpus = std::min(2, cuda_device_count_);
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    
    EXPECT_EQ(pool.numDevices(), limits.max_gpus + limits.max_cpus);
    EXPECT_EQ(pool.numGPUs(), limits.max_gpus);
    EXPECT_EQ(pool.numCPUs(), 2);
}

TEST_F(DevicePoolTest, RequestMoreGPUsThanAvailable) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    ResourceLimits limits;
    limits.max_gpus = cuda_device_count_ + 10;  // Request more than available
    limits.max_cpus = 1;
    
    EXPECT_THROW(DevicePool pool(limits), std::runtime_error);
}

TEST_F(DevicePoolTest, GPURequestWhenCUDAUnavailable) {
    if (cuda_available_) {
        GTEST_SKIP() << "Test requires CUDA to be unavailable";
    }
    
    ResourceLimits limits;
    limits.max_gpus = 1;
    limits.max_cpus = 1;
    
    EXPECT_THROW(DevicePool pool(limits), std::runtime_error);
}

// ============================================================================
// Device Assignment Tests
// ============================================================================

TEST_F(DevicePoolTest, GetDeviceForWorkerReturnsDevice) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    
    // Should not throw
    IComputeDevice& device0 = pool.getDeviceForWorker(0);
    IComputeDevice& device1 = pool.getDeviceForWorker(1);
    
    (void)device0;
    (void)device1;
}

TEST_F(DevicePoolTest, RoundRobinAssignment) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 3;
    
    DevicePool pool(limits);
    
    // Workers 0, 1, 2 should get devices 0, 1, 2
    IComputeDevice& dev0 = pool.getDeviceForWorker(0);
    IComputeDevice& dev1 = pool.getDeviceForWorker(1);
    IComputeDevice& dev2 = pool.getDeviceForWorker(2);
    
    // Worker 3 should wrap around to device 0
    IComputeDevice& dev3 = pool.getDeviceForWorker(3);
    
    // Verify round-robin (same worker ID gets same device)
    EXPECT_EQ(&dev0, &dev3);
    
    // Different workers get different devices (until wraparound)
    EXPECT_NE(&dev0, &dev1);
    EXPECT_NE(&dev1, &dev2);
}

TEST_F(DevicePoolTest, SameWorkerGetsSameDevice) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 4;
    
    DevicePool pool(limits);
    
    IComputeDevice& dev1 = pool.getDeviceForWorker(5);
    IComputeDevice& dev2 = pool.getDeviceForWorker(5);
    
    EXPECT_EQ(&dev1, &dev2);
}

TEST_F(DevicePoolTest, NegativeWorkerIdThrows) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    
    EXPECT_THROW(pool.getDeviceForWorker(-1), std::runtime_error);
    EXPECT_THROW(pool.getDeviceForWorker(-100), std::runtime_error);
}

TEST_F(DevicePoolTest, LargeWorkerIdHandled) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 4;
    
    DevicePool pool(limits);
    
    // Should not throw even with large worker IDs
    EXPECT_NO_THROW({
        pool.getDeviceForWorker(100);
        pool.getDeviceForWorker(1000);
        pool.getDeviceForWorker(10000);
    });
}

// ============================================================================
// Device Info Tests
// ============================================================================

TEST_F(DevicePoolTest, GetDeviceInfoCPUOnly) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 3;
    
    DevicePool pool(limits);
    
    auto info = pool.getDeviceInfo();
    EXPECT_EQ(info.size(), 3);
    
    for (const auto& device_info : info) {
        EXPECT_EQ(device_info.type, DeviceType::CPU);
        EXPECT_EQ(device_info.device_id, -1);
        EXPECT_FALSE(device_info.in_use);  // Initially not in use
    }
}

TEST_F(DevicePoolTest, GetDeviceInfoGPUOnly) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    ResourceLimits limits;
    limits.max_gpus = std::min(2, cuda_device_count_);
    limits.max_cpus = 0;
    
    DevicePool pool(limits);
    
    auto info = pool.getDeviceInfo();
    EXPECT_EQ(info.size(), limits.max_gpus);
    
    for (size_t i = 0; i < info.size(); ++i) {
        EXPECT_EQ(info[i].type, DeviceType::GPU);
        EXPECT_EQ(info[i].device_id, static_cast<int>(i));
        EXPECT_FALSE(info[i].in_use);
    }
}

TEST_F(DevicePoolTest, GetDeviceInfoMixed) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    ResourceLimits limits;
    limits.max_gpus = std::min(2, cuda_device_count_);
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    
    auto info = pool.getDeviceInfo();
    EXPECT_EQ(info.size(), limits.max_gpus + limits.max_cpus);
    
    // First entries should be GPUs
    for (int i = 0; i < limits.max_gpus; ++i) {
        EXPECT_EQ(info[i].type, DeviceType::GPU);
        EXPECT_EQ(info[i].device_id, i);
    }
    
    // Remaining entries should be CPUs
    for (size_t i = limits.max_gpus; i < info.size(); ++i) {
        EXPECT_EQ(info[i].type, DeviceType::CPU);
        EXPECT_EQ(info[i].device_id, -1);
    }
}

TEST_F(DevicePoolTest, DeviceMarkedInUseAfterAssignment) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 3;
    
    DevicePool pool(limits);
    
    // Initially not in use
    auto info_before = pool.getDeviceInfo();
    for (const auto& device_info : info_before) {
        EXPECT_FALSE(device_info.in_use);
    }
    
    // Assign to workers
    pool.getDeviceForWorker(0);
    pool.getDeviceForWorker(1);
    
    // Check in_use status
    auto info_after = pool.getDeviceInfo();
    EXPECT_TRUE(info_after[0].in_use);
    EXPECT_TRUE(info_after[1].in_use);
    EXPECT_FALSE(info_after[2].in_use);  // Not yet assigned
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(DevicePoolTest, ConcurrentDeviceAssignment) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 4;
    
    DevicePool pool(limits);
    
    const int num_workers = 16;
    std::vector<std::thread> workers;
    std::vector<IComputeDevice*> assigned_devices(num_workers, nullptr);
    
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([&pool, &assigned_devices, i]() {
            assigned_devices[i] = &pool.getDeviceForWorker(i);
        });
    }
    
    for (auto& worker : workers) {
        worker.join();
    }
    
    // Verify all got devices
    for (int i = 0; i < num_workers; ++i) {
        EXPECT_NE(assigned_devices[i], nullptr);
    }
    
    // Verify round-robin pattern
    for (int i = 0; i < num_workers; ++i) {
        int expected_same = (i + 4) % num_workers;  // 4 CPUs, so +4 should be same device
        if (expected_same != i) {
            EXPECT_EQ(assigned_devices[i], assigned_devices[expected_same]);
        }
    }
}

TEST_F(DevicePoolTest, ConcurrentGetDeviceInfo) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 4;
    
    DevicePool pool(limits);
    
    std::atomic<bool> stop{false};
    std::vector<std::thread> readers;
    
    // Start reader threads
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&pool, &stop]() {
            while (!stop) {
                auto info = pool.getDeviceInfo();
                EXPECT_EQ(info.size(), 4);
            }
        });
    }
    
    // Let them read concurrently
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    stop = true;
    for (auto& reader : readers) {
        reader.join();
    }
}

// ============================================================================
// Device Type Tests
// ============================================================================

TEST(DeviceTypeTest, EnumValues) {
    DeviceType gpu = DeviceType::GPU;
    DeviceType cpu = DeviceType::CPU;
    
    EXPECT_NE(gpu, cpu);
}

TEST(DeviceInfoTest, DefaultConstructor) {
    DeviceInfo info;
    
    EXPECT_EQ(info.device_id, -1);
    EXPECT_EQ(info.type, DeviceType::CPU);
    EXPECT_FALSE(info.in_use);
}

TEST(DeviceInfoTest, ParameterizedConstructor) {
    DeviceInfo gpu_info(2, DeviceType::GPU, true);
    
    EXPECT_EQ(gpu_info.device_id, 2);
    EXPECT_EQ(gpu_info.type, DeviceType::GPU);
    EXPECT_TRUE(gpu_info.in_use);
    
    DeviceInfo cpu_info(-1, DeviceType::CPU, false);
    
    EXPECT_EQ(cpu_info.device_id, -1);
    EXPECT_EQ(cpu_info.type, DeviceType::CPU);
    EXPECT_FALSE(cpu_info.in_use);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(DevicePoolTest, SingleDevicePool) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 1;
    
    DevicePool pool(limits);
    
    // All workers should get the same device
    IComputeDevice& dev0 = pool.getDeviceForWorker(0);
    IComputeDevice& dev1 = pool.getDeviceForWorker(1);
    IComputeDevice& dev2 = pool.getDeviceForWorker(2);
    
    EXPECT_EQ(&dev0, &dev1);
    EXPECT_EQ(&dev1, &dev2);
}

TEST_F(DevicePoolTest, ManyWorkersFewDevices) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    
    // 100 workers, 2 devices
    std::set<IComputeDevice*> unique_devices;
    for (int i = 0; i < 100; ++i) {
        unique_devices.insert(&pool.getDeviceForWorker(i));
    }
    
    // Should only see 2 unique device pointers
    EXPECT_EQ(unique_devices.size(), 2);
}

// ============================================================================
// Dynamic Resizing Tests (Phase 2 Step 2.2)
// ============================================================================

TEST_F(DevicePoolTest, ResizeScaleUpCPUs) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.numCPUs(), 2);
    EXPECT_EQ(pool.numDevices(), 2);
    
    // Scale up to 4 CPUs
    pool.resize(0, 4);
    
    EXPECT_EQ(pool.numCPUs(), 4);
    EXPECT_EQ(pool.numDevices(), 4);
    EXPECT_EQ(pool.activeDeviceCount(), 4);
}

TEST_F(DevicePoolTest, ResizeScaleDownCPUs) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 4;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.numCPUs(), 4);
    
    // Scale down to 2 CPUs
    pool.resize(0, 2);
    
    // Devices still exist but marked for removal
    EXPECT_EQ(pool.numDevices(), 4);  // Not removed yet
    EXPECT_EQ(pool.activeDeviceCount(), 2);  // Only 2 active
    
    // Workers 2 and 3 should be marked for shutdown
    EXPECT_TRUE(pool.shouldShutdown(2));
    EXPECT_TRUE(pool.shouldShutdown(3));
    EXPECT_FALSE(pool.shouldShutdown(0));
    EXPECT_FALSE(pool.shouldShutdown(1));
}

TEST_F(DevicePoolTest, ResizeScaleUpGPUs) {
    if (!cuda_available_ || cuda_device_count_ < 2) {
        GTEST_SKIP() << "Test requires 2+ CUDA GPUs";
    }
    
    ResourceLimits limits;
    limits.max_gpus = 1;
    limits.max_cpus = 0;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.numGPUs(), 1);
    
    // Scale up to 2 GPUs
    pool.resize(2, 0);
    
    EXPECT_EQ(pool.numGPUs(), 2);
    EXPECT_EQ(pool.numDevices(), 2);
    EXPECT_EQ(pool.activeDeviceCount(), 2);
}

TEST_F(DevicePoolTest, ResizeScaleDownGPUs) {
    if (!cuda_available_ || cuda_device_count_ < 2) {
        GTEST_SKIP() << "Test requires 2+ CUDA GPUs";
    }
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 0;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.numGPUs(), 2);
    
    // Scale down to 1 GPU
    pool.resize(1, 0);
    
    // Device still exists but marked
    EXPECT_EQ(pool.numDevices(), 2);
    EXPECT_EQ(pool.activeDeviceCount(), 1);
    
    // Worker 1 should be marked for shutdown
    EXPECT_TRUE(pool.shouldShutdown(1));
    EXPECT_FALSE(pool.shouldShutdown(0));
}

TEST_F(DevicePoolTest, ResizeMixedScaleUp) {
    if (!cuda_available_ || cuda_device_count_ < 2) {
        GTEST_SKIP() << "Test requires 2+ CUDA GPUs";
    }
    
    ResourceLimits limits;
    limits.max_gpus = 1;
    limits.max_cpus = 1;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.numGPUs(), 1);
    EXPECT_EQ(pool.numCPUs(), 1);
    EXPECT_EQ(pool.numDevices(), 2);
    
    // Scale up both
    pool.resize(2, 3);
    
    EXPECT_EQ(pool.numGPUs(), 2);
    EXPECT_EQ(pool.numCPUs(), 3);
    EXPECT_EQ(pool.numDevices(), 5);
    EXPECT_EQ(pool.activeDeviceCount(), 5);
}

TEST_F(DevicePoolTest, ResizeMixedScaleDown) {
    if (!cuda_available_ || cuda_device_count_ < 2) {
        GTEST_SKIP() << "Test requires 2+ CUDA GPUs";
    }
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 3;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.numDevices(), 5);
    
    // Scale down both
    pool.resize(1, 1);
    
    EXPECT_EQ(pool.numDevices(), 5);  // Not removed yet
    EXPECT_EQ(pool.activeDeviceCount(), 2);  // Only 2 active (1 GPU + 1 CPU)
}

TEST_F(DevicePoolTest, FinalizeRemovalsActuallyRemovesDevices) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 4;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.numDevices(), 4);
    
    // Scale down
    pool.resize(0, 2);
    EXPECT_EQ(pool.numDevices(), 4);  // Still there
    EXPECT_EQ(pool.activeDeviceCount(), 2);
    
    // Finalize removals
    pool.finalizeRemovals();
    
    EXPECT_EQ(pool.numDevices(), 2);  // Actually removed
    EXPECT_EQ(pool.numCPUs(), 2);
    EXPECT_EQ(pool.activeDeviceCount(), 2);
}

TEST_F(DevicePoolTest, MarkDeviceForRemoval) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 3;
    
    DevicePool pool(limits);
    
    // Manually mark worker 1's device
    pool.markDeviceForRemoval(1);
    
    EXPECT_TRUE(pool.shouldShutdown(1));
    EXPECT_FALSE(pool.shouldShutdown(0));
    EXPECT_FALSE(pool.shouldShutdown(2));
    
    // After finalize, device should be removed
    pool.finalizeRemovals();
    EXPECT_EQ(pool.numDevices(), 2);
}

TEST_F(DevicePoolTest, ShouldShutdownAfterResize) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 5;
    
    DevicePool pool(limits);
    
    // No shutdowns initially
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(pool.shouldShutdown(i));
    }
    
    // Scale down to 2 CPUs
    pool.resize(0, 2);
    
    // Workers 2, 3, 4 should shutdown (devices at indices 2, 3, 4)
    EXPECT_FALSE(pool.shouldShutdown(0));
    EXPECT_FALSE(pool.shouldShutdown(1));
    EXPECT_TRUE(pool.shouldShutdown(2));
    EXPECT_TRUE(pool.shouldShutdown(3));
    EXPECT_TRUE(pool.shouldShutdown(4));
}

TEST_F(DevicePoolTest, ResizeInvalidNegativeGPUs) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    
    EXPECT_THROW(pool.resize(-1, 2), std::invalid_argument);
}

TEST_F(DevicePoolTest, ResizeInvalidNegativeCPUs) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    
    EXPECT_THROW(pool.resize(0, -1), std::invalid_argument);
}

TEST_F(DevicePoolTest, ResizeInvalidZeroDevices) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    
    // Cannot resize to zero devices
    EXPECT_THROW(pool.resize(0, 0), std::invalid_argument);
}

TEST_F(DevicePoolTest, ConcurrentResizeAndAccess) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 4;
    
    DevicePool pool(limits);
    
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    
    // Workers accessing devices
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back([&pool, i, &stop]() {
            while (!stop.load()) {
                try {
                    auto& dev = pool.getDeviceForWorker(i);
                    (void)dev;  // Use the device
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } catch (...) {
                    // May throw if pool becomes invalid
                }
            }
        });
    }
    
    // Resize thread
    std::thread resizer([&pool, &stop]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        pool.resize(0, 6);  // Scale up
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        pool.resize(0, 4);  // Scale back down
        stop.store(true);
    });
    
    resizer.join();
    for (auto& w : workers) {
        w.join();
    }
    
    // Should not crash
    EXPECT_EQ(pool.activeDeviceCount(), 4);
}

TEST_F(DevicePoolTest, MultipleResizesInSequence) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 2;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.numDevices(), 2);
    
    // Resize up
    pool.resize(0, 4);
    EXPECT_EQ(pool.numCPUs(), 4);
    
    // Resize down
    pool.resize(0, 2);
    EXPECT_EQ(pool.activeDeviceCount(), 2);
    
    // Finalize
    pool.finalizeRemovals();
    EXPECT_EQ(pool.numDevices(), 2);
    
    // Resize up again
    pool.resize(0, 5);
    EXPECT_EQ(pool.numCPUs(), 5);
    EXPECT_EQ(pool.numDevices(), 5);
}

TEST_F(DevicePoolTest, ActiveDeviceCountUpdatesCorrectly) {
    ResourceLimits limits;
    limits.max_gpus = 0;
    limits.max_cpus = 5;
    
    DevicePool pool(limits);
    EXPECT_EQ(pool.activeDeviceCount(), 5);
    
    pool.resize(0, 3);
    EXPECT_EQ(pool.activeDeviceCount(), 3);
    
    pool.markDeviceForRemoval(0);
    EXPECT_EQ(pool.activeDeviceCount(), 2);
    
    pool.finalizeRemovals();
    EXPECT_EQ(pool.activeDeviceCount(), 2);
    EXPECT_EQ(pool.numDevices(), 2);
}
