#include "minimizer/orchestration/gpu_registry.h"
#include "utilities/gpu/nvml_wrapper.h"
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <mutex>
#include <sys/stat.h>

// Debug 
#include <iostream>
namespace entropy {

// =============================================================================
// Singleton Access
// =============================================================================

GPURegistry& GPURegistry::instance() {
    static GPURegistry instance;
    return instance;
}

GPURegistry::~GPURegistry() {
    stopWatching();
}

// =============================================================================
// Configuration Management
// =============================================================================

void GPURegistry::loadConfig(const std::string& config_path) {

    std::cout << "GPURegistry: Loading config from " << config_path << std::endl;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // Check file exists
    std::ifstream file(config_path);
    if (!file.good()) {
        throw std::runtime_error("GPURegistry::loadConfig: Config file not found: " + config_path);
    }
    
    std::cout << "GPURegistry: File found. Parsing YAML config." << std::endl;
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("GPURegistry::loadConfig: YAML parse error: " + std::string(e.what()));
    }
    
    std::cout << "GPURegistry: Querying available CUDA devices." << std::endl;
    // Get available CUDA device count
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        // No CUDA devices available - not necessarily an error
        device_count = 0;
    }
    
    std::cout << "GPURegistry: Found " << device_count << " CUDA devices." << std::endl;
    // Clear existing GPU info
    gpus_.clear();
    
    // Parse GPU list
    if (config["gpus"]) {
        std::cout << "GPURegistry: Parsing GPU entries from config." << std::endl;
        for (const auto& gpu_node : config["gpus"]) {
            int device_id = gpu_node["device_id"].as<int>();
            bool enabled = gpu_node["enabled"].as<bool>(true);
            
            std::cout << "GPURegistry: Processing GPU device_id=" 
                      << device_id << ", enabled=" << enabled << std::endl;
            // Validate device ID
            if (device_id < 0 || device_id >= device_count) {
                throw std::invalid_argument(
                    "GPURegistry::loadConfig: Invalid device_id " + std::to_string(device_id) +
                    " (available: 0-" + std::to_string(device_count - 1) + ")"
                );
            }
            
            GPUInfo info;
            info.device_id = device_id;
            info.enabled = enabled;
            
            // Only query properties (which initializes CUDA runtime) for enabled GPUs
            if (enabled) {
                std::cout << "GPURegistry: Querying properties for enabled device " << device_id << std::endl;
                GPUInfo detailed_info = queryGPUProperties(device_id);
                info = detailed_info;
                info.enabled = enabled;  // Preserve enabled flag
            } else {
                std::cout << "GPURegistry: Skipping property query for disabled device " << device_id << std::endl;
                // For disabled GPUs, only get basic properties without initializing runtime
                cudaDeviceProp props;
                cudaError_t err = cudaGetDeviceProperties(&props, device_id);
                if (err == cudaSuccess) {
                    info.name = props.name;
                    info.compute_capability_major = props.major;
                    info.compute_capability_minor = props.minor;
                    info.total_memory = props.totalGlobalMem;
                    info.free_memory = 0;  // Unknown without initializing runtime
                }
                info.utilization = 0.0f;
                info.in_use_by_us = false;
            }
            
            gpus_[device_id] = info;

            std::cout << "GPURegistry: Registered GPU: " << info.to_string() << std::endl;
        }
    }
    
    // Parse optional settings
    if (config["memory_buffer_percent"]) {
        memory_buffer_percent_ = config["memory_buffer_percent"].as<int>();
        if (memory_buffer_percent_ < 0 || memory_buffer_percent_ > 100) {
            throw std::invalid_argument(
                "GPURegistry::loadConfig: memory_buffer_percent must be 0-100, got " + 
                std::to_string(memory_buffer_percent_)
            );
        }
    }
    
    // Store config path for reloading
    config_path_ = config_path;
    last_mod_time_ = getFileModTime(config_path);
}

void GPURegistry::reloadConfig() {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    std::cout << "GPURegistry: Reloading config from " << config_path_ << std::endl;
    if (config_path_.empty()) {
        return;  // No config loaded yet
    }
    
    std::string path = config_path_;  // Copy under read lock
    read_lock.unlock();
    
    std::cout << "GPURegistry: Checking if config file was modified." << std::endl;
    // Check if file was modified
    std::time_t current_mod_time = getFileModTime(path);
    if (current_mod_time == last_mod_time_) {
        std::cout << "GPURegistry: No changes detected in config file." << std::endl;
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
    std::cout << "GPURegistry: Config reloaded successfully." << std::endl;   
    // Check if enabled GPUs changed
    std::vector<int> new_enabled = getEnabledGPUs();
    if (old_enabled != new_enabled) {
        std::cout << "GPURegistry: Enabled GPUs changed. Notifying subscribers." << std::endl;
        notifySubscribers();
    }
}

std::vector<int> GPURegistry::getEnabledGPUs() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::cout << "GPURegistry: Getting list of enabled GPUs." << std::endl;    
    std::vector<int> enabled;
    for (const auto& [device_id, info] : gpus_) {
        if (info.enabled) {
            enabled.push_back(device_id);
        }
    }
    std::cout << "GPURegistry: Enabled GPUs: "; 
    for (int id : enabled) {
        std::cout << id << " ";
    }
    std::cout << std::endl;   
    std::sort(enabled.begin(), enabled.end());
    return enabled;
}

bool GPURegistry::isGPUEnabled(int device_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = gpus_.find(device_id);
    return (it != gpus_.end()) && it->second.enabled;
}

// =============================================================================
// Device Selection
// =============================================================================

std::optional<int> GPURegistry::selectBestGPU(size_t required_memory) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    // Find eligible GPUs
    std::vector<std::pair<int, size_t>> candidates;  // (device_id, free_memory)
    
    std::cout << "GPURegistry: Selecting best GPU for required memory " << required_memory << std::endl;
    for (const auto& [device_id, info] : gpus_) {
        // Check eligibility
        if (!info.enabled) continue;
        if (info.in_use_by_us) continue;
        std::cout << "GPURegistry: GPU " << device_id << " is eligible." << std::endl;
        // Apply memory buffer
        size_t usable_memory = info.free_memory * (100 - memory_buffer_percent_) / 100;
        std::cout << "GPURegistry: GPU " << device_id << " usable memory after buffer: " << usable_memory << std::endl;
        
        if (usable_memory >= required_memory) {
            candidates.emplace_back(device_id, usable_memory);
            std::cout << "GPURegistry: GPU " << device_id << " added to candidates." << std::endl;
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

std::vector<int> GPURegistry::selectBestGPUs(int count, size_t required_memory) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    // Find eligible GPUs
    std::vector<std::pair<int, size_t>> candidates;  // (device_id, free_memory)
    
    std::cout << "GPURegistry: Selecting best " << count 
              << " GPUs for required memory " << required_memory << std::endl;
    for (const auto& [device_id, info] : gpus_) {
        // Check eligibility
        if (!info.enabled) continue;
        if (info.in_use_by_us) continue;
        
        std::cout << "GPURegistry: GPU " << device_id << " is eligible." << std::endl;
        // Apply memory buffer
        size_t usable_memory = info.free_memory * (100 - memory_buffer_percent_) / 100;
        std::cout << "GPURegistry: GPU " << device_id << " usable memory after buffer: " << usable_memory << std::endl;
        if (usable_memory >= required_memory) {
            candidates.emplace_back(device_id, usable_memory);
        }
    }
    
    // Sort by free memory (descending)
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    std::cout << "GPURegistry: Sorted candidate GPUs by usable memory." << std::endl;
    // Return top N
    std::vector<int> result;
    int n = std::min(count, static_cast<int>(candidates.size()));
    for (int i = 0; i < n; ++i) {
        result.push_back(candidates[i].first);
    }
    std::cout << "GPURegistry: Selected GPUs: ";
    for (int id : result) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    return result;
}

// =============================================================================
// Resource Tracking
// =============================================================================

GPURegistry::GPUInfo GPURegistry::getGPUInfo(int device_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::cout << "GPURegistry: Getting info for GPU " << device_id << std::endl;
    auto it = gpus_.find(device_id);
    if (it != gpus_.end()) {
        return it->second;
    }
    
    return GPUInfo();  // Default constructed
}

std::vector<GPURegistry::GPUInfo> GPURegistry::getAllGPUInfo() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<GPUInfo> result;
    std::cout << "GPURegistry: Getting info for all GPUs." << std::endl;
    for (const auto& [device_id, info] : gpus_) {
        std::cout << "GPURegistry: GPU " << device_id << " info: " << info.to_string() << std::endl;
        result.push_back(info);
    }
    
    // Sort by device_id
    std::sort(result.begin(), result.end(),
        [](const GPUInfo& a, const GPUInfo& b) { return a.device_id < b.device_id; });
    
    return result;
}

void GPURegistry::markGPUInUse(int device_id, bool in_use) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = gpus_.find(device_id);
    if (it != gpus_.end()) {
        it->second.in_use_by_us = in_use;
    }
}

void GPURegistry::refreshGPUInfo() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    utils::NVMLWrapper& nvml = utils::NVMLWrapper::instance();
    
    for (auto& [device_id, info] : gpus_) {
        // Use NVML to query memory without initializing CUDA context
        utils::NVMLWrapper::DeviceInfo nvml_info;
        if (nvml.getDeviceInfo(device_id, nvml_info)) {
            info.free_memory = nvml_info.free_memory;
            info.total_memory = nvml_info.total_memory;
            info.utilization = static_cast<float>(nvml_info.utilization);
        } else {
            // Fallback: keep existing values or mark as unavailable
            std::cerr << "Warning: Failed to refresh GPU " << device_id 
                      << " info: " << nvml.getLastError() << std::endl;
        }
    }
}

// =============================================================================
// Change Notifications
// =============================================================================

void GPURegistry::subscribeToChanges(ConfigChangeCallback callback) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    subscribers_.push_back(callback);
}

void GPURegistry::startWatching(std::chrono::milliseconds poll_interval) {
    if (watching_.exchange(true)) {
        return;  // Already watching
    }
    
    watch_interval_ = poll_interval;
    config_watcher_ = std::thread(&GPURegistry::watchConfigFile, this);
}

void GPURegistry::stopWatching() {
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

void GPURegistry::watchConfigFile() {
    std::cout << "GPURegistry: Starting config file watcher thread." << std::endl;
    while (watching_) {
        std::cout << "GPURegistry: Watching config file for changes." << std::endl;
        reloadConfig();
        std::cout << "GPURegistry: Sleeping for " 
                  << watch_interval_.count() << " ms before next check." << std::endl;
        std::this_thread::sleep_for(watch_interval_);
    }
}

void GPURegistry::notifySubscribers() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::cout << "GPURegistry: Notifying subscribers of config change." << std::endl;
    std::vector<int> enabled_gpus = getEnabledGPUs();
    
    // Call subscribers without holding lock (callbacks might lock)
    std::vector<ConfigChangeCallback> callbacks = subscribers_;
    lock.unlock();
    std::cout << "GPURegistry: Invoking " 
              << callbacks.size() << " subscriber callbacks." << std::endl;
    for (const auto& callback : callbacks) {
        try {
            callback(enabled_gpus);
        } catch (...) {
            std::cout << "GPURegistry: Exception caught in subscriber callback." << std::endl;
            // Swallow exceptions from callbacks to avoid crashing watcher thread
        }
    }
}

GPURegistry::GPUInfo GPURegistry::queryGPUProperties(int device_id) const {
    GPUInfo info;
    info.device_id = device_id;
    info.enabled = false;  // Will be set by caller
    
    // Get device properties (doesn't initialize CUDA context)
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err == cudaSuccess) {
        info.name = props.name;
        info.compute_capability_major = props.major;
        info.compute_capability_minor = props.minor;
    }
    
    // Use NVML to get memory and utilization (doesn't initialize CUDA context)
    utils::NVMLWrapper& nvml = utils::NVMLWrapper::instance();
    utils::NVMLWrapper::DeviceInfo nvml_info;
    if (nvml.getDeviceInfo(device_id, nvml_info)) {
        info.free_memory = nvml_info.free_memory;
        info.total_memory = nvml_info.total_memory;
        info.utilization = static_cast<float>(nvml_info.utilization);
    } else {
        // Fallback: use device properties for total memory, set free to 0
        info.total_memory = props.totalGlobalMem;
        info.free_memory = 0;
        info.utilization = 0.0f;
        std::cerr << "Warning: NVML query failed for GPU " << device_id 
                  << ": " << nvml.getLastError() << std::endl;
    }
    
    info.in_use_by_us = false;
    
    return info;
}

std::time_t GPURegistry::getFileModTime(const std::string& path) const {
    std::cout << "GPURegistry: Getting modification time for file " << path << std::endl;
    struct stat file_stat;
    if (stat(path.c_str(), &file_stat) == 0) {
        std::cout << "GPURegistry: Modification time for file " << path << " is " << file_stat.st_mtime << std::endl;
        return file_stat.st_mtime;
    }
    return 0;
}

} // namespace entropy
