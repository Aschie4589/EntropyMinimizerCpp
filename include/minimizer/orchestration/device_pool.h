#ifndef DEVICE_POOL_H_
#define DEVICE_POOL_H_

#include "compute/device/IComputeDevice.h"
#include "compute/device/DeviceFactory.h"
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace entropy {

// Forward declaration
struct ResourceLimits;

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
 * Provides round-robin assignment of devices to worker threads.
 * Initialized with static resource limits from ResourceLimits.
 * 
 * Design:
 * - Each worker thread gets assigned one device via getDeviceForWorker()
 * - Assignment is deterministic: worker N gets device (N % pool_size)
 * - Devices are owned by the pool (RAII)
 * - Thread-safe for concurrent worker initialization
 * 
 * Initialization order:
 * 1. GPUs first (device_id 0 to max_gpus-1)
 * 2. CPUs second (device_id -1 for all)
 * 
 * Usage:
 * @code
 * ResourceLimits limits;
 * limits.max_gpus = 2;
 * limits.max_cpus = 4;
 * 
 * DevicePool pool(limits);
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
     * @brief Construct device pool from resource limits
     * 
     * Initializes max_gpus GPU devices and max_cpus CPU devices.
     * GPUs are initialized first, then CPUs.
     * 
     * @param limits Resource limits specifying device counts
     * @throws std::invalid_argument if limits invalid (negative counts)
     * @throws std::runtime_error if GPU initialization fails when requested
     */
    explicit DevicePool(const ResourceLimits& limits);
    
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
     * @brief Get device for worker thread (round-robin assignment)
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
    // Dynamic Resizing (Phase 2 Step 2.2)
    // ========================================================================
    
    /**
     * @brief Resize the device pool to new GPU/CPU counts
     * 
     * Handles both scaling up (adding devices) and scaling down (marking
     * devices for removal). When scaling down, devices are marked but not
     * immediately removed - workers must check shouldShutdown() and exit,
     * then finalizeRemovals() cleans up.
     * 
     * Scale-up: New devices are appended to the pool immediately.
     * Scale-down: Excess devices are marked for removal, workers notified.
     * 
     * @param new_max_gpus New GPU count (>= 0)
     * @param new_max_cpus New CPU count (>= 0)
     * @throws std::invalid_argument if new counts invalid (negative or both zero)
     * @throws std::runtime_error if CUDA unavailable but GPUs requested
     */
    void resize(int new_max_gpus, int new_max_cpus);
    
    /**
     * @brief Check if worker should shutdown due to device removal
     * 
     * Workers should call this periodically. Returns true if the worker's
     * assigned device has been marked for removal.
     * 
     * @param worker_id Worker thread identifier
     * @return true if worker should exit, false otherwise
     */
    bool shouldShutdown(int worker_id) const;
    
    /**
     * @brief Mark worker's device for removal
     * 
     * Called by workers during graceful shutdown to signal they've
     * released the device.
     * 
     * @param worker_id Worker thread identifier
     */
    void markDeviceForRemoval(int worker_id);
    
    /**
     * @brief Finalize removal of marked devices
     * 
     * Should be called after workers have exited. Removes all devices
     * marked for removal from the pool.
     */
    void finalizeRemovals();
    
    /**
     * @brief Get count of active (not marked for removal) devices
     * 
     * @return Number of devices currently active
     */
    size_t activeDeviceCount() const;

private:
    /**
     * @brief Internal device storage
     */
    struct DeviceSlot {
        std::unique_ptr<IComputeDevice> device;  ///< Owned device instance
        DeviceInfo info;                          ///< Device metadata
        bool marked_for_removal;                  ///< Pending removal flag
        
        DeviceSlot(std::unique_ptr<IComputeDevice> dev, DeviceInfo inf)
            : device(std::move(dev)), info(inf), marked_for_removal(false) {}
    };
    
    /**
     * @brief Initialize GPU devices
     * 
     * @param count Number of GPUs to initialize (0 to CUDA device count)
     * @throws std::runtime_error if CUDA unavailable but count > 0
     */
    void initializeGPUs(int count);
    
    /**
     * @brief Initialize CPU devices
     * 
     * @param count Number of CPU devices to initialize
     */
    void initializeCPUs(int count);
    
    /**
     * @brief Add GPU devices to pool (for scale-up)
     * 
     * @param count Number of GPUs to add
     * @throws std::runtime_error if CUDA unavailable or initialization fails
     */
    void addGPUs(int count);
    
    /**
     * @brief Add CPU devices to pool (for scale-up)
     * 
     * @param count Number of CPUs to add
     */
    void addCPUs(int count);
    
    /**
     * @brief Mark GPU devices for removal (for scale-down)
     * 
     * Marks the last 'count' GPU devices for removal. Workers using these
     * devices will be notified via shouldShutdown().
     * 
     * @param count Number of GPUs to mark for removal
     */
    void removeGPUs(int count);
    
    /**
     * @brief Mark CPU devices for removal (for scale-down)
     * 
     * Marks the last 'count' CPU devices for removal.
     * 
     * @param count Number of CPUs to mark for removal
     */
    void removeCPUs(int count);
    
    std::vector<DeviceSlot> devices_;  ///< Owned devices in assignment order
    size_t num_gpus_;                  ///< Count of GPU devices
    size_t num_cpus_;                  ///< Count of CPU devices
    mutable std::mutex mutex_;         ///< Protects device access
};

} // namespace entropy

#endif // DEVICE_POOL_H_
