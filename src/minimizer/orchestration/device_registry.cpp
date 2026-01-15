#include "minimizer/orchestration/device_registry.h"
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <mutex>
#include <sys/stat.h>

namespace entropy {

// =============================================================================
// Singleton Access
// =============================================================================

DeviceRegistry& DeviceRegistry::instance() {
    static DeviceRegistry instance;
    return instance;
}

DeviceRegistry::~DeviceRegistry() {
    stopWatching();
}

// =============================================================================
// Configuration Management
// =============================================================================

void DeviceRegistry::loadConfig(const std::string& config_path) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // Check file exists
    std::ifstream file(config_path);
    if (!file.good()) {
        throw std::runtime_error("DeviceRegistry::loadConfig: Config file not found: " + config_path);
    }
    
    // Parse YAML
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("DeviceRegistry::loadConfig: YAML parse error: " + std::string(e.what()));
    }
    
    // Get available CUDA device count
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        // No CUDA devices available - not necessarily an error
        device_count = 0;
    }
    
    // Clear existing GPU info
    gpus_.clear();
    
    // Parse GPU list
    if (config["gpus"]) {
        for (const auto& gpu_node : config["gpus"]) {
            int device_id = gpu_node["device_id"].as<int>();
            bool enabled = gpu_node["enabled"].as<bool>(true);
            
            // Validate device ID
            if (device_id < 0 || device_id >= device_count) {
                throw std::invalid_argument(
                    "DeviceRegistry::loadConfig: Invalid device_id " + std::to_string(device_id) +
                    " (available: 0-" + std::to_string(device_count - 1) + ")"
                );
            }
            
            // Query GPU properties
            GPUInfo info = queryGPUProperties(device_id);
            info.enabled = enabled;
            
            gpus_[device_id] = info;
        }
    }
    
    // Parse optional settings
    if (config["memory_buffer_percent"]) {
        memory_buffer_percent_ = config["memory_buffer_percent"].as<int>();
        if (memory_buffer_percent_ < 0 || memory_buffer_percent_ > 100) {
            throw std::invalid_argument(
                "DeviceRegistry::loadConfig: memory_buffer_percent must be 0-100, got " + 
                std::to_string(memory_buffer_percent_)
            );
        }
    }
    
    // Store config path for reloading
    config_path_ = config_path;
    last_mod_time_ = getFileModTime(config_path);
}

void DeviceRegistry::reloadConfig() {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    
    if (config_path_.empty()) {
        return;  // No config loaded yet
    }
    
    std::string path = config_path_;  // Copy under read lock
    read_lock.unlock();
    
    // Check if file was modified
    std::time_t current_mod_time = getFileModTime(path);
    if (current_mod_time == last_mod_time_) {
        return;  // No change
    }
    
    // Get current enabled GPUs for change detection
    std::vector<int> old_enabled = getEnabledGPUs();
    
    // Reload (will acquire write lock)
    try {
        loadConfig(path);
    } catch (const std::exception& e) {
        // Log error but don't crash - keep using old config
        // In production, this should use proper logging
        return;
    }
    
    // Check if enabled GPUs changed
    std::vector<int> new_enabled = getEnabledGPUs();
    if (old_enabled != new_enabled) {
        notifySubscribers();
    }
}

std::vector<int> DeviceRegistry::getEnabledGPUs() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<int> enabled;
    for (const auto& [device_id, info] : gpus_) {
        if (info.enabled) {
            enabled.push_back(device_id);
        }
    }
    
    std::sort(enabled.begin(), enabled.end());
    return enabled;
}

bool DeviceRegistry::isGPUEnabled(int device_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = gpus_.find(device_id);
    return (it != gpus_.end()) && it->second.enabled;
}

// =============================================================================
// Device Selection
// =============================================================================

std::optional<int> DeviceRegistry::selectBestGPU(size_t required_memory) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    // Find eligible GPUs
    std::vector<std::pair<int, size_t>> candidates;  // (device_id, free_memory)
    
    for (const auto& [device_id, info] : gpus_) {
        // Check eligibility
        if (!info.enabled) continue;
        if (info.in_use_by_us) continue;
        
        // Apply memory buffer
        size_t usable_memory = info.free_memory * (100 - memory_buffer_percent_) / 100;
        
        if (usable_memory >= required_memory) {
            candidates.emplace_back(device_id, usable_memory);
        }
    }
    
    if (candidates.empty()) {
        return std::nullopt;
    }
    
    // Sort by free memory (descending)
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return candidates[0].first;
}

std::vector<int> DeviceRegistry::selectBestGPUs(int count, size_t required_memory) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    // Find eligible GPUs
    std::vector<std::pair<int, size_t>> candidates;  // (device_id, free_memory)
    
    for (const auto& [device_id, info] : gpus_) {
        // Check eligibility
        if (!info.enabled) continue;
        if (info.in_use_by_us) continue;
        
        // Apply memory buffer
        size_t usable_memory = info.free_memory * (100 - memory_buffer_percent_) / 100;
        
        if (usable_memory >= required_memory) {
            candidates.emplace_back(device_id, usable_memory);
        }
    }
    
    // Sort by free memory (descending)
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top N
    std::vector<int> result;
    int n = std::min(count, static_cast<int>(candidates.size()));
    for (int i = 0; i < n; ++i) {
        result.push_back(candidates[i].first);
    }
    
    return result;
}

// =============================================================================
// Resource Tracking
// =============================================================================

DeviceRegistry::GPUInfo DeviceRegistry::getGPUInfo(int device_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = gpus_.find(device_id);
    if (it != gpus_.end()) {
        return it->second;
    }
    
    return GPUInfo();  // Default constructed
}

std::vector<DeviceRegistry::GPUInfo> DeviceRegistry::getAllGPUInfo() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<GPUInfo> result;
    for (const auto& [device_id, info] : gpus_) {
        result.push_back(info);
    }
    
    // Sort by device_id
    std::sort(result.begin(), result.end(),
        [](const GPUInfo& a, const GPUInfo& b) { return a.device_id < b.device_id; });
    
    return result;
}

void DeviceRegistry::markGPUInUse(int device_id, bool in_use) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = gpus_.find(device_id);
    if (it != gpus_.end()) {
        it->second.in_use_by_us = in_use;
    }
}

void DeviceRegistry::refreshGPUInfo() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    for (auto& [device_id, info] : gpus_) {
        // Query memory
        size_t free_mem = 0;
        size_t total_mem = 0;
        
        cudaError_t err = cudaSetDevice(device_id);
        if (err == cudaSuccess) {
            err = cudaMemGetInfo(&free_mem, &total_mem);
            if (err == cudaSuccess) {
                info.free_memory = free_mem;
                info.total_memory = total_mem;
            }
        }
    }
}

// =============================================================================
// Change Notifications
// =============================================================================

void DeviceRegistry::subscribeToChanges(ConfigChangeCallback callback) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    subscribers_.push_back(callback);
}

void DeviceRegistry::startWatching(std::chrono::milliseconds poll_interval) {
    if (watching_.exchange(true)) {
        return;  // Already watching
    }
    
    watch_interval_ = poll_interval;
    config_watcher_ = std::thread(&DeviceRegistry::watchConfigFile, this);
}

void DeviceRegistry::stopWatching() {
    if (!watching_.exchange(false)) {
        return;  // Not watching
    }
    
    if (config_watcher_.joinable()) {
        config_watcher_.join();
    }
}

// =============================================================================
// Private Methods
// =============================================================================

void DeviceRegistry::watchConfigFile() {
    while (watching_) {
        reloadConfig();
        std::this_thread::sleep_for(watch_interval_);
    }
}

void DeviceRegistry::notifySubscribers() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<int> enabled_gpus = getEnabledGPUs();
    
    // Call subscribers without holding lock (callbacks might lock)
    std::vector<ConfigChangeCallback> callbacks = subscribers_;
    lock.unlock();
    
    for (const auto& callback : callbacks) {
        try {
            callback(enabled_gpus);
        } catch (...) {
            // Swallow exceptions from callbacks to avoid crashing watcher thread
        }
    }
}

DeviceRegistry::GPUInfo DeviceRegistry::queryGPUProperties(int device_id) const {
    GPUInfo info;
    info.device_id = device_id;
    info.enabled = false;  // Will be set by caller
    
    // Set device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return info;
    }
    
    // Get device properties
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err == cudaSuccess) {
        info.name = props.name;
        info.compute_capability_major = props.major;
        info.compute_capability_minor = props.minor;
    }
    
    // Get memory info
    size_t free_mem = 0;
    size_t total_mem = 0;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err == cudaSuccess) {
        info.free_memory = free_mem;
        info.total_memory = total_mem;
    }
    
    // Utilization will be 0.0 for now (requires NVML)
    info.utilization = 0.0f;
    
    info.in_use_by_us = false;
    
    return info;
}

std::time_t DeviceRegistry::getFileModTime(const std::string& path) const {
    struct stat file_stat;
    if (stat(path.c_str(), &file_stat) == 0) {
        return file_stat.st_mtime;
    }
    return 0;
}

} // namespace entropy
