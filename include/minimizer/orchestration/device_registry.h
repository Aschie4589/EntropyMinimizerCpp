#ifndef DEVICE_REGISTRY_H_
#define DEVICE_REGISTRY_H_

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <sstream>

namespace entropy {

/**
 * @brief Singleton registry for GPU device configuration and availability
 * 
 * Central authority for:
 * - Loading GPU configuration from YAML files
 * - Querying GPU availability and properties via CUDA APIs
 * - Selecting best GPUs based on memory and utilization
 * - Notifying subscribers of configuration changes
 * - Tracking which GPUs are in use by this process
 * 
 * Design:
 * - Singleton pattern for global access
 * - Thread-safe for concurrent queries
 * - File watcher for dynamic configuration updates
 * - Extensible for future NVML integration
 * 
 * Configuration File Format (gpu_config.yaml):
 * @code{.yaml}
 * gpus:
 *   - device_id: 0
 *     enabled: true
 *   - device_id: 1
 *     enabled: false
 * poll_interval_ms: 1000
 * memory_buffer_percent: 20
 * @endcode
 * 
 * Usage:
 * @code
 * DeviceRegistry& registry = DeviceRegistry::instance();
 * registry.loadConfig("configs/gpu_config.yaml");
 * 
 * // Get enabled GPUs
 * auto enabled_gpus = registry.getEnabledGPUs();
 * 
 * // Select best GPU
 * auto gpu_id = registry.selectBestGPU(1024 * 1024 * 1024);  // 1GB required
 * 
 * // Subscribe to changes
 * registry.subscribeToChanges([](const std::vector<int>& gpus) {
 *     std::cout << "Enabled GPUs changed: " << gpus.size() << std::endl;
 * });
 * registry.startWatching();
 * @endcode
 */
class DeviceRegistry {
public:
    /**
     * @brief Information about a GPU device
     */
    struct GPUInfo {
        int device_id;                      ///< CUDA device ID
        bool enabled;                       ///< Enabled in configuration
        size_t total_memory;                ///< Total memory (bytes) from cudaMemGetInfo
        size_t free_memory;                 ///< Free memory (bytes) from cudaMemGetInfo
        float utilization;                  ///< GPU utilization 0-100% (future: NVML)
        int compute_capability_major;       ///< Compute capability major version
        int compute_capability_minor;       ///< Compute capability minor version
        std::string name;                   ///< Device name
        bool in_use_by_us;                  ///< Tracked by this process
        
        GPUInfo() 
            : device_id(-1), enabled(false), total_memory(0), 
              free_memory(0), utilization(0.0f), 
              compute_capability_major(0), compute_capability_minor(0),
              in_use_by_us(false) {}
        
        /**
         * @brief Convert to string for debugging
         */
        std::string to_string() const {
            std::ostringstream oss;
            oss << "GPUInfo{device_id=" << device_id
                << ", enabled=" << enabled
                << ", total_memory=" << (total_memory / (1024*1024)) << "MB"
                << ", free_memory=" << (free_memory / (1024*1024)) << "MB"
                << ", utilization=" << utilization << "%"
                << ", compute_capability=" << compute_capability_major << "." << compute_capability_minor
                << ", name='" << name << "'"
                << ", in_use_by_us=" << in_use_by_us << "}";
            return oss.str();
        }
    };
    
    /**
     * @brief Get singleton instance
     */
    static DeviceRegistry& instance();
    
    /**
     * @brief Load configuration from YAML file
     * 
     * Parses gpu_config.yaml and validates device IDs against available CUDA devices.
     * Updates internal GPU info from CUDA runtime.
     * 
     * @param config_path Path to configuration file
     * @throws std::runtime_error if file not found or invalid format
     * @throws std::invalid_argument if device IDs exceed available devices
     */
    void loadConfig(const std::string& config_path);
    
    /**
     * @brief Reload configuration from previously loaded file
     * 
     * Used by file watcher to detect changes.
     * Thread-safe with read operations.
     */
    void reloadConfig();
    
    /**
     * @brief Get list of enabled GPU device IDs
     * 
     * @return Vector of device IDs marked as enabled in config
     */
    std::vector<int> getEnabledGPUs() const;
    
    /**
     * @brief Check if specific GPU is enabled
     * 
     * @param device_id CUDA device ID
     * @return true if GPU exists and is enabled
     */
    bool isGPUEnabled(int device_id) const;
    
    /**
     * @brief Select best available GPU based on criteria
     * 
     * Selection criteria:
     * 1. Must be enabled in config
     * 2. Must not be in use by us
     * 3. Must have at least required_memory free
     * 4. Prefer GPU with most free memory
     * 
     * @param required_memory Minimum free memory in bytes (0 = no requirement)
     * @return Device ID of best GPU, or std::nullopt if none available
     */
    std::optional<int> selectBestGPU(size_t required_memory = 0) const;
    
    /**
     * @brief Select N best available GPUs
     * 
     * Uses same criteria as selectBestGPU(), returns top N by free memory.
     * 
     * @param count Number of GPUs to select
     * @param required_memory Minimum free memory per GPU (bytes)
     * @return Vector of device IDs (may be fewer than count if not enough available)
     */
    std::vector<int> selectBestGPUs(int count, size_t required_memory = 0) const;
    
    /**
     * @brief Get information about specific GPU
     * 
     * @param device_id CUDA device ID
     * @return GPUInfo structure (default-constructed if device_id invalid)
     */
    GPUInfo getGPUInfo(int device_id) const;
    
    /**
     * @brief Get information about all GPUs
     * 
     * @return Vector of GPUInfo for all detected CUDA devices
     */
    std::vector<GPUInfo> getAllGPUInfo() const;
    
    /**
     * @brief Mark GPU as in use by this process
     * 
     * Used by DevicePool to track allocation.
     * 
     * @param device_id CUDA device ID
     * @param in_use true to mark in use, false to release
     */
    void markGPUInUse(int device_id, bool in_use);
    
    /**
     * @brief Refresh GPU info from CUDA runtime
     * 
     * Updates memory and utilization for all GPUs.
     * Called automatically by loadConfig() and reloadConfig().
     * Can be called manually for up-to-date info.
     */
    void refreshGPUInfo();
    
    /**
     * @brief Callback type for configuration changes
     * 
     * Called when enabled GPU list changes.
     * 
     * @param enabled_gpus Vector of currently enabled GPU device IDs
     */
    using ConfigChangeCallback = std::function<void(const std::vector<int>&)>;
    
    /**
     * @brief Subscribe to configuration change notifications
     * 
     * @param callback Function to call when enabled GPUs change
     */
    void subscribeToChanges(ConfigChangeCallback callback);
    
    /**
     * @brief Start watching configuration file for changes
     * 
     * Spawns background thread that polls file modification time.
     * Calls reloadConfig() and notifies subscribers on change.
     * 
     * @param poll_interval How often to check file (default: 1000ms)
     */
    void startWatching(std::chrono::milliseconds poll_interval = std::chrono::milliseconds(1000));
    
    /**
     * @brief Stop watching configuration file
     * 
     * Joins watcher thread if running.
     * Called automatically by destructor.
     */
    void stopWatching();
    
    /**
     * @brief Get memory buffer percentage
     * 
     * Safety margin for memory calculations (e.g., 20 = use only 80% of reported free memory)
     * 
     * @return Buffer percentage (0-100)
     */
    int getMemoryBufferPercent() const { return memory_buffer_percent_; }
    
private:
    // Singleton: private constructor/destructor
    DeviceRegistry() = default;
    ~DeviceRegistry();
    
    // Non-copyable, non-movable
    DeviceRegistry(const DeviceRegistry&) = delete;
    DeviceRegistry& operator=(const DeviceRegistry&) = delete;
    DeviceRegistry(DeviceRegistry&&) = delete;
    DeviceRegistry& operator=(DeviceRegistry&&) = delete;
    
    /**
     * @brief Background thread function for file watching
     */
    void watchConfigFile();
    
    /**
     * @brief Notify all subscribers of configuration change
     */
    void notifySubscribers();
    
    /**
     * @brief Query GPU properties from CUDA runtime
     * 
     * @param device_id CUDA device ID
     * @return GPUInfo with properties populated
     */
    GPUInfo queryGPUProperties(int device_id) const;
    
    /**
     * @brief Get file modification time
     * 
     * @param path File path
     * @return Modification time, or 0 if file doesn't exist
     */
    std::time_t getFileModTime(const std::string& path) const;
    
    // Thread synchronization
    mutable std::shared_mutex mutex_;           ///< Protects gpus_, config_path_, subscribers_
    
    // GPU state
    std::map<int, GPUInfo> gpus_;               ///< Map device_id -> GPUInfo
    
    // Configuration
    std::string config_path_;                   ///< Path to loaded config file
    int memory_buffer_percent_ = 20;            ///< Memory safety buffer (percent)
    
    // Change notifications
    std::vector<ConfigChangeCallback> subscribers_;  ///< Registered callbacks
    std::thread config_watcher_;                     ///< File watcher thread
    std::atomic<bool> watching_{false};              ///< File watcher active flag
    std::chrono::milliseconds watch_interval_{1000}; ///< File poll interval
    std::time_t last_mod_time_{0};                   ///< Last known file modification time
};

} // namespace entropy

#endif // DEVICE_REGISTRY_H_
