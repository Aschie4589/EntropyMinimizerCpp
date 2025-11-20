#ifndef GPU_RESOURCE_MANAGER_H
#define GPU_RESOURCE_MANAGER_H

#include <vector>
#include <mutex>
#include <atomic>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

/**
 * GPUResourceManager - Dynamic GPU allocation with runtime configuration
 * 
 * Features:
 * - Runtime enable/disable of GPUs via config file
 * - Memory-aware allocation (checks free memory before acquiring)
 * - Thread-safe acquisition/release
 * - Automatic config file monitoring
 * 
 * Usage:
 *   GPUResourceManager mgr("gpu_config.txt");
 *   int gpu = mgr.acquireGPU();
 *   // ... do work on GPU ...
 *   mgr.releaseGPU(gpu);
 * 
 * Config file format (gpu_config.txt):
 *   # Lines starting with # are comments
 *   GPU 0: enabled
 *   GPU 1: disabled    # Reserved for other user
 *   GPU 2: enabled
 *   GPU 3: enabled
 */
class GPUResourceManager {
public:
    /**
     * Constructor
     * @param config_file Path to GPU configuration file (auto-reloaded)
     * @param min_free_memory Minimum free GPU memory required (bytes), default 5GB
     * @param check_interval How often to reload config file (seconds), default 10s
     */
    GPUResourceManager(
        const std::string& config_file = "",
        size_t min_free_memory = 5ULL * 1024 * 1024 * 1024,  // 5GB default
        int check_interval = 10
    );
    
    ~GPUResourceManager();
    
    /**
     * Acquire an available GPU
     * @param required_memory Optional: specific memory requirement (0 = use default)
     * @return GPU ID (0-based), or -1 if no GPU available
     */
    int acquireGPU(size_t required_memory = 0);
    
    /**
     * Release a previously acquired GPU
     * @param gpu_id The GPU ID to release
     */
    void releaseGPU(int gpu_id);
    
    /**
     * Manually enable a GPU at runtime
     * @param gpu_id GPU to enable
     */
    void enableGPU(int gpu_id);
    
    /**
     * Manually disable a GPU at runtime (waits for in-use GPU to be released)
     * @param gpu_id GPU to disable
     * @param force If true, marks disabled immediately (jobs still running)
     */
    void disableGPU(int gpu_id, bool force = false);
    
    /**
     * Get current number of available (enabled and not in-use) GPUs
     */
    int getAvailableGPUCount();
    
    /**
     * Get total number of GPUs in system
     */
    int getTotalGPUCount() const { return num_gpus; }
    
    /**
     * Check if a specific GPU is currently available
     */
    bool isGPUAvailable(int gpu_id);
    
    /**
     * Get info about all GPUs (for monitoring)
     */
    std::string getStatusString();
    
    /**
     * Force reload of configuration file
     */
    void reloadConfig();
    
    /**
     * Wait until a GPU becomes available (blocking)
     * @param timeout_seconds Maximum time to wait (0 = wait forever)
     * @return GPU ID, or -1 on timeout
     */
    int waitForGPU(int timeout_seconds = 0);

private:
    struct GPUState {
        bool enabled;           // Is GPU enabled in config?
        bool in_use;           // Is GPU currently acquired?
        size_t free_memory;    // Last known free memory
        int jobs_processed;    // Counter for load balancing
        std::chrono::system_clock::time_point last_check;
    };
    
    int num_gpus;
    size_t min_free_memory_bytes;
    std::string config_file_path;
    int config_check_interval_sec;
    
    std::vector<GPUState> gpu_states;
    std::mutex gpu_mutex;
    std::atomic<bool> should_stop{false};
    
    std::chrono::system_clock::time_point last_config_check;
    
    // Internal methods
    void initializeGPUs();
    void loadConfigFile();
    bool checkGPUMemory(int gpu_id, size_t required_memory);
    void updateGPUMemoryInfo(int gpu_id);
    int selectBestGPU(size_t required_memory);
};

#endif // GPU_RESOURCE_MANAGER_H
