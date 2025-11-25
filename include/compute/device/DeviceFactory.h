#ifndef DEVICE_FACTORY_H_
#define DEVICE_FACTORY_H_

#include "compute/device/IComputeDevice.h"
#include <memory>
#include <string>

/**
 * @brief Factory for creating compute devices with automatic backend detection
 * 
 * Provides centralized device creation with fallback logic:
 * 1. AUTO mode: Try CUDA first, fall back to CPU if unavailable
 * 2. Explicit mode: Create specific backend or throw if unavailable
 * 
 * Example usage:
 * @code
 * // Auto-detect best available device
 * auto device = DeviceFactory::create();
 * 
 * // Request specific CUDA device
 * auto cuda_dev = DeviceFactory::create(DeviceType::CUDA, 1);
 * 
 * // Force CPU (for testing)
 * auto cpu_dev = DeviceFactory::create(DeviceType::CPU);
 * @endcode
 */
class DeviceFactory {
public:
    /**
     * @brief Device type selection
     */
    enum class DeviceType {
        AUTO,   ///< Auto-detect: prefer CUDA, fallback to CPU
        CUDA,   ///< Require CUDA (throw if unavailable)
        CPU     ///< Use CPU backend
    };
    
    /**
     * @brief Create compute device with specified backend
     * 
     * @param type Device type (AUTO, CUDA, or CPU)
     * @param device_id Device ID for CUDA (ignored for CPU)
     * @param device_scratch_size Device scratch pool size in bytes (CUDA only)
     * @param host_scratch_size Host scratch pool size in bytes
     * @return Unique pointer to created device
     * 
     * @throws std::runtime_error if requested backend unavailable
     */
    static std::unique_ptr<IComputeDevice> create(
        DeviceType type = DeviceType::AUTO,
        int device_id = 0,
        size_t device_scratch_size = 64 * 1024 * 1024,  // 64 MB
        size_t host_scratch_size = 16 * 1024 * 1024      // 16 MB
    );
    
    /**
     * @brief Check if CUDA is available on this system
     * 
     * @return true if at least one CUDA device is available
     */
    static bool isCudaAvailable();
    
    /**
     * @brief Get number of available CUDA devices
     * 
     * @return Number of CUDA devices (0 if CUDA unavailable)
     */
    static int getCudaDeviceCount();
    
    /**
     * @brief Get device properties as human-readable string
     * 
     * @param device_id CUDA device ID
     * @return Device name and compute capability
     * @throws std::runtime_error if device_id invalid
     */
    static std::string getCudaDeviceInfo(int device_id = 0);

private:
    DeviceFactory() = delete;  // Static class, no instances
};

#endif // DEVICE_FACTORY_H_
