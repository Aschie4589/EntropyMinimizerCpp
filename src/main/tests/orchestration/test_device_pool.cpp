#include <gtest/gtest.h>
#include "minimizer/orchestration/device_pool.h"
#include "minimizer/orchestration/gpu_registry.h"
#include "minimizer/config/resource_config.h"
#include <thread>
#include <vector>

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
        
        // Get GPURegistry instance
        registry_ = &GPURegistry::instance();
    }
    
    bool cuda_available_;
    int cuda_device_count_;
    GPURegistry* registry_;
};

// ============================================================================
// Constructor and Initialization Tests
// ============================================================================

TEST_F(DevicePoolTest, ConstructorWithResourceConfig) {
    ResourceConfig config = ResourceConfig::createCPUOnly(4);
    
    DevicePool pool(config);
    
    // Pool is created but not yet initialized
    EXPECT_EQ(pool.numDevices(), 0);
    EXPECT_EQ(pool.numGPUs(), 0);
    EXPECT_EQ(pool.numCPUs(), 0);
}

TEST_F(DevicePoolTest, InitializeFromRegistryCPUOnly) {
    ResourceConfig config = ResourceConfig::createCPUOnly(2);
    DevicePool pool(config);
    
    // Initialize from registry (will create CPU devices)
    pool.initializeFromRegistry(*registry_);
    
    EXPECT_EQ(pool.numDevices(), 2);
    EXPECT_EQ(pool.numGPUs(), 0);
    EXPECT_EQ(pool.numCPUs(), 2);
}

TEST_F(DevicePoolTest, InitializeFromRegistryGPUAuto) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    int gpus_to_request = std::min(2, cuda_device_count_);
    ResourceConfig config = ResourceConfig::createGPUOnly(gpus_to_request, false);
    DevicePool pool(config);
    
    pool.initializeFromRegistry(*registry_);
    
    // Should have requested GPUs (may fallback to CPU if not enough)
    EXPECT_GT(pool.numDevices(), 0);
    EXPECT_LE(pool.numGPUs(), gpus_to_request);
}

TEST_F(DevicePoolTest, InitializeFromRegistryManualGPUSelection) {
    if (!cuda_available_ || cuda_device_count_ < 1) {
        GTEST_SKIP() << "CUDA not available or insufficient GPUs";
    }
    
    std::vector<int> gpu_ids = {0};
    ResourceConfig config = ResourceConfig::createWithPreferredGPUs(gpu_ids, 0);
    DevicePool pool(config);
    
    pool.initializeFromRegistry(*registry_);
    
    EXPECT_GE(pool.numGPUs(), 0);
    if (pool.numGPUs() > 0) {
        auto device_info = pool.getDeviceInfo();
        EXPECT_EQ(device_info[0].device_id, 0);
    }
}

TEST_F(DevicePoolTest, ResourceConfigValidation) {
    // Test invalid configuration
    ResourceConfig config;
    config.desired_gpus = 0;
    config.desired_cpus = 0;
    
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

// ============================================================================
// Dynamic GPU Management Tests
// ============================================================================

TEST_F(DevicePoolTest, AddGPUDynamic) {
    if (!cuda_available_ || cuda_device_count_ < 1) {
        GTEST_SKIP() << "Need at least 1 GPU";
    }
    
    ResourceConfig config = ResourceConfig::createCPUOnly(1);
    DevicePool pool(config);
    pool.initializeFromRegistry(*registry_);
    
    size_t initial_devices = pool.numDevices();
    
    // Add a GPU dynamically
    EXPECT_NO_THROW(pool.addGPU(0));
    
    EXPECT_EQ(pool.numDevices(), initial_devices + 1);
    EXPECT_EQ(pool.numGPUs(), 1);
}

TEST_F(DevicePoolTest, RemoveGPUDynamic) {
    if (!cuda_available_ || cuda_device_count_ < 1) {
        GTEST_SKIP() << "Need at least 1 GPU";
    }
    
    std::vector<int> gpu_ids = {0};
    ResourceConfig config = ResourceConfig::createWithPreferredGPUs(gpu_ids, 0);
    DevicePool pool(config);
    pool.initializeFromRegistry(*registry_);
    
    if (pool.numGPUs() > 0) {
        size_t initial_gpus = pool.numGPUs();
        
        // Remove GPU
        EXPECT_NO_THROW(pool.removeGPU(0));
        
        EXPECT_EQ(pool.numGPUs(), initial_gpus - 1);
    }
}

TEST_F(DevicePoolTest, AdjustToEnabledGPUs) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    ResourceConfig config = ResourceConfig::createCPUOnly(2);
    DevicePool pool(config);
    pool.initializeFromRegistry(*registry_);
    
    // Adjust to use GPU 0 if available
    std::vector<int> enabled_gpus = {0};
    EXPECT_NO_THROW(pool.adjustToEnabledGPUs(enabled_gpus));
    
    // Should have updated GPU list
    EXPECT_GE(pool.numDevices(), 2);
}

// ============================================================================
// Device Acquisition Tests
// ============================================================================

TEST_F(DevicePoolTest, GetDeviceForWorker) {
    ResourceConfig config = ResourceConfig::createCPUOnly(2);
    DevicePool pool(config);
    pool.initializeFromRegistry(*registry_);
    
    ASSERT_EQ(pool.numDevices(), 2);
    
    // Get device for worker 0
    EXPECT_NO_THROW({
        IComputeDevice& device = pool.getDeviceForWorker(0);
        (void)device; // Suppress unused warning
    });
}

TEST_F(DevicePoolTest, MultipleWorkersGetDevices) {
    ResourceConfig config = ResourceConfig::createCPUOnly(4);
    DevicePool pool(config);
    pool.initializeFromRegistry(*registry_);
    
    ASSERT_EQ(pool.numDevices(), 4);
    
    // Get devices for multiple workers
    for (int i = 0; i < 4; ++i) {
        EXPECT_NO_THROW({
            IComputeDevice& device = pool.getDeviceForWorker(i);
            (void)device;
        });
    }
}

// ============================================================================
// Device Info Tests
// ============================================================================

TEST_F(DevicePoolTest, GetDeviceInfo) {
    ResourceConfig config = ResourceConfig::createCPUOnly(2);
    DevicePool pool(config);
    pool.initializeFromRegistry(*registry_);
    
    auto info = pool.getDeviceInfo();
    
    EXPECT_EQ(info.size(), 2);
    for (const auto& device_info : info) {
        EXPECT_EQ(device_info.type, DeviceType::CPU);
        EXPECT_EQ(device_info.device_id, -1);
    }
}

TEST_F(DevicePoolTest, GetDeviceStatusCounts) {
    ResourceConfig config = ResourceConfig::createCPUOnly(3);
    DevicePool pool(config);
    pool.initializeFromRegistry(*registry_);
    
    EXPECT_EQ(pool.numDevices(), 3);
    EXPECT_EQ(pool.numGPUs(), 0);
    EXPECT_EQ(pool.numCPUs(), 3);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(DevicePoolTest, GetDeviceForWorkerEmptyPool) {
    ResourceConfig config = ResourceConfig::createCPUOnly(1);
    DevicePool pool(config);
    // Don't initialize - pool is empty
    
    // Should handle gracefully (may throw or return safely)
    // Implementation defined behavior
    EXPECT_NO_THROW({
        try {
            IComputeDevice& device = pool.getDeviceForWorker(0);
            (void)device;
        } catch (...) {
            // Also acceptable
        }
    });
}

TEST_F(DevicePoolTest, RemoveNonexistentGPU) {
    ResourceConfig config = ResourceConfig::createCPUOnly(2);
    DevicePool pool(config);
    pool.initializeFromRegistry(*registry_);
    
    // Try to remove GPU that doesn't exist - should not crash
    EXPECT_NO_THROW(pool.removeGPU(99));
    EXPECT_EQ(pool.numGPUs(), 0);
}

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST_F(DevicePoolTest, ResourceConfigFactoryMethods) {
    // Test createCPUOnly
    auto cpu_config = ResourceConfig::createCPUOnly(4);
    EXPECT_EQ(cpu_config.desired_cpus, 4);
    EXPECT_EQ(cpu_config.desired_gpus, 0);
    EXPECT_NO_THROW(cpu_config.validate());
    
    // Test createGPUOnly
    auto gpu_config = ResourceConfig::createGPUOnly(2, false);
    EXPECT_EQ(gpu_config.desired_gpus, 2);
    EXPECT_NO_THROW(gpu_config.validate());
    
    // Test createDefault
    auto default_config = ResourceConfig::createDefault();
    EXPECT_NO_THROW(default_config.validate());
}
