#ifndef DEVICE_POOL_H_
#define DEVICE_POOL_H_

#include "compute/device/IComputeDevice.h"
#include "compute/device/DeviceFactory.h"
#include "minimizer/config/resource_config.h"
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace entropy {

// Forward declarations
class GPURegistry;

/**
 * @brief Device type classification
 */
enum class DeviceType {
    GPU,  ///< CUDA GPU device
    CPU   ///< CPU device
};

/**
 * @brief Information about a device in the pool
 */
struct DeviceInfo {
    int device_id;       ///< CUDA device ID (0-N for GPU, -1 for CPU)
    DeviceType type;     ///< GPU or CPU
    bool in_use;         ///< Currently assigned to a worker
    
    DeviceInfo() : device_id(-1), type(DeviceType::CPU), in_use(false) {}
    
    DeviceInfo(int id, DeviceType t, bool used = false)
        : device_id(id), type(t), in_use(used) {}
};

/**
 * @brief Manages a pool of compute devices for work distribution
 * 
 * Owns and manages IComputeDevice instances (GPU and CPU backends).
 * Provides assignment of devices to worker threads.
 * Initialized with static resource limits from ResourceConfig (i.e. how
 * many GPUs and CPUs can be used at most, other devices if implemented).
 * 
 * Design:
 * - Each worker thread gets assigned one device via getDeviceForWorker()
 * - Assignment is deterministic: worker N gets device N.
 * - If more workers than devices, devices are reused round-robin.
 * - Devices are owned by the pool (RAII)
 * - Thread-safe for concurrent worker initialization
 * 
 * Initialization order:
 * 1. GPUs first (device_id 0 to max_gpus-1)
 * 2. CPUs second (device_id -1 for all)
 * 
 * Usage:
 * @code
 * ResourceConfig config;
 * config.max_gpus = 2;
 * config.max_cpus = 4;
 * 
 * DevicePool pool(config);
 * 
 * // Worker threads
 * for (int i = 0; i < 6; ++i) {
 *     auto& device = pool.getDeviceForWorker(i);
 *     // Use device for computation
 * }
 * @endcode
 */
class DevicePool {
public:
    /**
     * @brief Construct device pool from resource configuration
     * 
     * Initializes device pool but does NOT create devices yet.
     * Call initializeFromRegistry() to create devices dynamically.
     * 
     * @param config Resource configuration
     * @throws std::invalid_argument if config invalid
     */
    explicit DevicePool(const ResourceConfig& config);
    
    /**
     * @brief Destructor - releases all devices
     */
    ~DevicePool() = default;
    
    // Non-copyable (owns unique_ptr resources)
    DevicePool(const DevicePool&) = delete;
    DevicePool& operator=(const DevicePool&) = delete;
    
    // Non-movable (workers hold references to devices)
    DevicePool(DevicePool&&) = delete;
    DevicePool& operator=(DevicePool&&) = delete;
    
    /**
     * @brief Acquire a device from the pool
     * 
     * Selects the least-loaded available device (not marked for removal).
     * Increments reference count to track usage.
     * 
     * @return Pointer to acquired device
     * @throws std::runtime_error if no devices available
     */
    IComputeDevice* acquireDevice();
    
    /**
     * @brief Release a previously acquired device
     * 
     * Decrements reference count. When count reaches 0 and device is
     * marked for removal, it becomes eligible for cleanup.
     * 
     * @param device Pointer to device to release
     */
    void releaseDevice(IComputeDevice* device);
    
    /**
     * @brief Get device for worker thread (round-robin assignment)
     * 
     * @deprecated Use acquireDevice/releaseDevice pattern instead
     * 
     * Returns reference to device at index (worker_id % num_devices).
     * Same worker_id always gets same device.
     * Thread-safe for concurrent calls with different worker_ids.
     * 
     * @param worker_id Worker thread identifier (0-based)
     * @return Reference to assigned compute device
     * @throws std::runtime_error if pool is empty or worker_id negative
     */
    IComputeDevice& getDeviceForWorker(int worker_id);
    
    /**
     * @brief Get total number of devices in pool
     * 
     * @return Number of initialized devices (GPUs + CPUs)
     */
    size_t numDevices() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return devices_.size();
    }
    
    /**
     * @brief Get information about all devices
     * 
     * @return Vector of DeviceInfo for each device in pool
     */
    std::vector<DeviceInfo> getDeviceInfo() const;
    
    /**
     * @brief Get number of GPU devices in pool
     * 
     * @return Count of GPU devices
     */
    size_t numGPUs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_gpus_;
    }
    
    /**
     * @brief Get number of CPU devices in pool
     * 
     * @return Count of CPU devices
     */
    size_t numCPUs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_cpus_;
    }
    
    // ========================================================================
    // New Dynamic Device Management (Phase 2)
    // ========================================================================
    
    /**
     * @brief Initialize devices from GPURegistry based on ResourceConfig
     * 
     * Uses GPURegistry to select best available GPUs.
     * Creates CPU devices as specified in config.
     * 
     * @param registry GPURegistry instance to query for GPUs
     * @throws std::runtime_error if no devices available and no fallback
     */
    void initializeFromRegistry(GPURegistry& registry);
    
    /**
     * @brief Add specific GPU to the pool
     * 
     * Creates a GPU device for the given CUDA device ID.
     * 
     * @param device_id CUDA device ID
     * @throws std::runtime_error if device creation fails
     */
    void addGPU(int device_id);
    
    /**
     * @brief Remove specific GPU from the pool
     * 
     * Marks the GPU device for removal. Workers using this device
     * should be terminated by WorkerThreadPool.
     * 
     * @param device_id CUDA device ID
     */
    void removeGPU(int device_id);
    
    /**
     * @brief Adjust pool to match enabled GPUs from registry
     * 
     * Compares current GPUs with enabled list:
     * - Removes GPUs that are no longer enabled
     * - Does NOT automatically add new GPUs (requires explicit request)
     * 
     * @param enabled_gpu_ids List of enabled GPU device IDs
     */
    void adjustToEnabledGPUs(const std::vector<int>& enabled_gpu_ids);
    
    /**
     * @brief Mark device for removal
     * 
     * Prevents new acquisitions but allows current workers to finish.
     * Device will be cleaned up when reference count reaches 0.
     * 
     * @param physical_device_id CUDA device ID
     */
    void markDeviceForRemoval(int physical_device_id);
    
    /**
     * @brief Clean up devices marked for removal with zero references
     * 
     * Removes devices from pool and frees resources.
     * Only removes devices where reference_count == 0.
     */
    void cleanupMarkedDevices();
    
    /**
     * @brief Get physical device ID from device pointer
     * 
     * @param device Device pointer
     * @return CUDA device ID (GPU) or -1 (CPU)
     * @throws std::runtime_error if device not found in pool
     */
    int getPhysicalDeviceID(const IComputeDevice* device) const;
    
    /**
     * @brief Get list of device IDs marked for removal
     * 
     * @return Vector of physical device IDs pending cleanup
     */
    std::vector<int> getDevicesMarkedForRemoval() const;
    
    /**
     * @brief Get current reference count for a device
     * 
     * @param device Device pointer
     * @return Number of workers currently using this device
     */
    int getReferenceCount(const IComputeDevice* device) const;
    
    /**
     * @brief Device status for monitoring
     */
    struct DeviceStatus {
        int pool_index;              ///< Index in pool
        int physical_device_id;      ///< CUDA device ID (GPU) or -1 (CPU)
        DeviceType type;             ///< GPU or CPU
        bool available;              ///< Not marked for removal
        size_t free_memory;          ///< Free memory if queryable, else 0
    };
    
    /**
     * @brief Get status of all devices in pool
     * 
     * @return Vector of device status information
     */
    std::vector<DeviceStatus> getDeviceStatus() const;
    
    /**
     * @brief Get number of active GPU devices
     * 
     * @return Count of GPU devices not marked for removal
     */
    size_t numActiveGPUs() const;
    
    /**
     * @brief Get number of active CPU devices
     * 
     * @return Count of CPU devices not marked for removal
     */
    size_t numActiveCPUs() const;
    


private:
    /**
     * @brief Internal device storage
     */
    struct DeviceSlot {
        std::unique_ptr<IComputeDevice> device;  ///< Owned device instance
        DeviceInfo info;                          ///< Device metadata
        bool marked_for_removal;                  ///< Pending removal flag
        int reference_count;                      ///< Number of workers using this device
        
        DeviceSlot(std::unique_ptr<IComputeDevice> dev, DeviceInfo inf)
            : device(std::move(dev)), info(inf), marked_for_removal(false), reference_count(0) {}
    };
    

    
    /**
     * @brief Create a single GPU device
     * 
     * @param device_id CUDA device ID
     */
    void createGPU(int device_id);
    
    /**
     * @brief Create a single CPU device
     */
    void createCPU();
    

    
    std::vector<DeviceSlot> devices_;  ///< Owned devices in assignment order
    size_t num_gpus_;                  ///< Count of GPU devices
    size_t num_cpus_;                  ///< Count of CPU devices
    mutable std::mutex mutex_;         ///< Protects device access
    
    // Configuration (for ResourceConfig constructor)
    bool use_resource_config_ = false;  ///< Using new ResourceConfig path
    int desired_gpus_ = 0;              ///< Desired GPU count
    int desired_cpus_ = 0;              ///< Desired CPU count
    size_t min_gpu_memory_ = 0;         ///< Minimum GPU memory requirement
    bool fallback_to_cpu_ = true;       ///< Fallback to CPU if no GPUs
};

} // namespace entropy

#endif // DEVICE_POOL_H_
