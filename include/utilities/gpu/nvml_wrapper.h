#ifndef NVML_WRAPPER_H_
#define NVML_WRAPPER_H_

#include <string>
#include <nvml.h>

namespace utils {

/**
 * @brief RAII wrapper for NVIDIA Management Library (NVML)
 * 
 * Provides safe access to GPU information without initializing CUDA contexts.
 * Singleton pattern ensures nvmlInit() is called once and nvmlShutdown() on exit.
 * 
 * Key advantages over CUDA runtime queries:
 * - No CUDA context initialization (no memory allocation)
 * - More accurate free memory reporting
 * - Access to utilization, temperature, and other metrics
 * 
 * Thread-safe for queries (NVML is thread-safe).
 */
class NVMLWrapper {
public:
    /**
     * @brief Device information structure
     */
    struct DeviceInfo {
        size_t total_memory;      ///< Total memory in bytes
        size_t free_memory;       ///< Free memory in bytes
        size_t used_memory;       ///< Used memory in bytes
        unsigned int utilization; ///< GPU utilization percentage (0-100)
        unsigned int temperature; ///< GPU temperature in Celsius
        std::string name;         ///< Device name
        
        DeviceInfo() 
            : total_memory(0), free_memory(0), used_memory(0),
              utilization(0), temperature(0) {}
    };
    
    /**
     * @brief Get singleton instance
     * 
     * Initializes NVML on first call.
     */
    static NVMLWrapper& instance();
    
    /**
     * @brief Get information about a specific GPU
     * 
     * @param cuda_device_id CUDA device ID (0-based index)
     * @param info Output structure to fill with device information
     * @return true if successful, false on error
     */
    bool getDeviceInfo(int cuda_device_id, DeviceInfo& info);
    
    /**
     * @brief Get number of available GPUs
     * 
     * @return Number of GPUs, or 0 if NVML not initialized
     */
    unsigned int getDeviceCount();
    
    /**
     * @brief Check if NVML is initialized and working
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Get last error message
     */
    std::string getLastError() const { return last_error_; }
    
private:
    NVMLWrapper();
    ~NVMLWrapper();
    
    // Non-copyable, non-movable
    NVMLWrapper(const NVMLWrapper&) = delete;
    NVMLWrapper& operator=(const NVMLWrapper&) = delete;
    NVMLWrapper(NVMLWrapper&&) = delete;
    NVMLWrapper& operator=(NVMLWrapper&&) = delete;
    
    bool initialized_;
    std::string last_error_;
};

} // namespace utils

#endif // NVML_WRAPPER_H_
