#include "utilities/gpu/gpu_resource_manager.h"

#include <iostream>
#include <sstream>
#include <thread>
#include <iomanip>
#include <cstring>

GPUResourceManager::GPUResourceManager(
    const std::string& config_file,
    size_t min_free_memory,
    int check_interval
) : config_file_path(config_file),
    min_free_memory_bytes(min_free_memory),
    config_check_interval_sec(check_interval)
{
    initializeGPUs();
    
    if (!config_file_path.empty()) {
        loadConfigFile();
    }
    
    last_config_check = std::chrono::system_clock::now();
}

GPUResourceManager::~GPUResourceManager() {
    should_stop.store(true);
}

void GPUResourceManager::initializeGPUs() {
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess) {
        std::cerr << "Error getting GPU count: " << cudaGetErrorString(err) << std::endl;
        num_gpus = 0;
        return;
    }
    
    std::cout << "Detected " << num_gpus << " GPUs" << std::endl;
    
    gpu_states.resize(num_gpus);
    
    // Initialize all GPUs as enabled by default
    for (int i = 0; i < num_gpus; i++) {
        gpu_states[i].enabled = true;
        gpu_states[i].in_use = false;
        gpu_states[i].free_memory = 0;
        gpu_states[i].jobs_processed = 0;
        gpu_states[i].last_check = std::chrono::system_clock::now();
        
        // Query GPU properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (" << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) 
                  << " GB)" << std::endl;
        
        // Update memory info
        updateGPUMemoryInfo(i);
    }
}

void GPUResourceManager::loadConfigFile() {
    if (config_file_path.empty()) return;
    
    std::ifstream config_file(config_file_path);
    if (!config_file.is_open()) {
        std::cout << "Config file not found: " << config_file_path 
                  << " (using defaults: all GPUs enabled)" << std::endl;
        return;
    }
    
    std::string line;
    int line_num = 0;
    while (std::getline(config_file, line)) {
        line_num++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Parse format: "GPU 0: enabled" or "GPU 1: disabled"
        std::istringstream iss(line);
        std::string gpu_label;
        int gpu_id;
        char colon;
        std::string status;
        
        if (!(iss >> gpu_label >> gpu_id >> colon >> status)) {
            std::cerr << "Warning: Invalid config line " << line_num 
                      << ": " << line << std::endl;
            continue;
        }
        
        if (gpu_label != "GPU" && gpu_label != "gpu") {
            std::cerr << "Warning: Expected 'GPU' keyword at line " << line_num << std::endl;
            continue;
        }
        
        if (gpu_id < 0 || gpu_id >= num_gpus) {
            std::cerr << "Warning: GPU " << gpu_id << " out of range (0-" 
                      << (num_gpus - 1) << ")" << std::endl;
            continue;
        }
        
        // Convert status to lowercase for comparison
        std::transform(status.begin(), status.end(), status.begin(), ::tolower);
        
        if (status == "enabled" || status == "enable" || status == "on") {
            gpu_states[gpu_id].enabled = true;
            std::cout << "  GPU " << gpu_id << " enabled via config" << std::endl;
        } else if (status == "disabled" || status == "disable" || status == "off") {
            gpu_states[gpu_id].enabled = false;
            std::cout << "  GPU " << gpu_id << " disabled via config" << std::endl;
        } else {
            std::cerr << "Warning: Unknown status '" << status 
                      << "' for GPU " << gpu_id << std::endl;
        }
    }
    
    config_file.close();
}

void GPUResourceManager::reloadConfig() {
    std::lock_guard<std::mutex> lock(gpu_mutex);
    std::cout << "Reloading GPU configuration from " << config_file_path << std::endl;
    loadConfigFile();
    last_config_check = std::chrono::system_clock::now();
}

void GPUResourceManager::updateGPUMemoryInfo(int gpu_id) {
    if (gpu_id < 0 || gpu_id >= num_gpus) return;
    
    int current_device;
    cudaGetDevice(&current_device);
    
    cudaSetDevice(gpu_id);
    
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (err == cudaSuccess) {
        gpu_states[gpu_id].free_memory = free_mem;
        gpu_states[gpu_id].last_check = std::chrono::system_clock::now();
    } else {
        std::cerr << "Error getting memory info for GPU " << gpu_id 
                  << ": " << cudaGetErrorString(err) << std::endl;
    }
    
    // Restore original device
    cudaSetDevice(current_device);
}

bool GPUResourceManager::checkGPUMemory(int gpu_id, size_t required_memory) {
    updateGPUMemoryInfo(gpu_id);
    return gpu_states[gpu_id].free_memory >= required_memory;
}

int GPUResourceManager::selectBestGPU(size_t required_memory) {
    // Strategy: Select GPU with:
    // 1. Most free memory (if all have enough)
    // 2. Least number of jobs processed (load balancing)
    
    int best_gpu = -1;
    size_t max_free_memory = 0;
    int min_jobs = INT_MAX;
    
    for (int i = 0; i < num_gpus; i++) {
        if (!gpu_states[i].enabled || gpu_states[i].in_use) continue;
        
        updateGPUMemoryInfo(i);
        
        if (gpu_states[i].free_memory < required_memory) continue;
        
        // Prefer GPU with most free memory, or least jobs if tied
        if (best_gpu == -1 || 
            gpu_states[i].free_memory > max_free_memory ||
            (gpu_states[i].free_memory == max_free_memory && 
             gpu_states[i].jobs_processed < min_jobs)) {
            best_gpu = i;
            max_free_memory = gpu_states[i].free_memory;
            min_jobs = gpu_states[i].jobs_processed;
        }
    }
    
    return best_gpu;
}

int GPUResourceManager::acquireGPU(size_t required_memory) {
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    // Check if we should reload config
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_config_check).count();
    
    if (!config_file_path.empty() && elapsed >= config_check_interval_sec) {
        loadConfigFile();
        last_config_check = now;
    }
    
    // Use default minimum if not specified
    if (required_memory == 0) {
        required_memory = min_free_memory_bytes;
    }
    
    int gpu_id = selectBestGPU(required_memory);
    
    if (gpu_id != -1) {
        gpu_states[gpu_id].in_use = true;
        gpu_states[gpu_id].jobs_processed++;
        
        std::cout << "Acquired GPU " << gpu_id 
                  << " (Free: " << (gpu_states[gpu_id].free_memory / (1024.0 * 1024.0 * 1024.0))
                  << " GB, Jobs: " << gpu_states[gpu_id].jobs_processed << ")" << std::endl;
    }
    
    return gpu_id;
}

void GPUResourceManager::releaseGPU(int gpu_id) {
    if (gpu_id < 0 || gpu_id >= num_gpus) {
        std::cerr << "Warning: Attempted to release invalid GPU " << gpu_id << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    if (!gpu_states[gpu_id].in_use) {
        std::cerr << "Warning: GPU " << gpu_id << " was not in use" << std::endl;
        return;
    }
    
    gpu_states[gpu_id].in_use = false;
    updateGPUMemoryInfo(gpu_id);
    
    std::cout << "Released GPU " << gpu_id << std::endl;
}

void GPUResourceManager::enableGPU(int gpu_id) {
    if (gpu_id < 0 || gpu_id >= num_gpus) {
        std::cerr << "Error: GPU " << gpu_id << " out of range" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    if (gpu_states[gpu_id].enabled) {
        std::cout << "GPU " << gpu_id << " is already enabled" << std::endl;
        return;
    }
    
    gpu_states[gpu_id].enabled = true;
    std::cout << "Enabled GPU " << gpu_id << std::endl;
}

void GPUResourceManager::disableGPU(int gpu_id, bool force) {
    if (gpu_id < 0 || gpu_id >= num_gpus) {
        std::cerr << "Error: GPU " << gpu_id << " out of range" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    if (!gpu_states[gpu_id].enabled) {
        std::cout << "GPU " << gpu_id << " is already disabled" << std::endl;
        return;
    }
    
    if (gpu_states[gpu_id].in_use && !force) {
        std::cerr << "Warning: GPU " << gpu_id << " is currently in use. "
                  << "Use force=true to disable anyway, or wait for release." << std::endl;
        return;
    }
    
    gpu_states[gpu_id].enabled = false;
    
    if (gpu_states[gpu_id].in_use) {
        std::cout << "Force-disabled GPU " << gpu_id << " (job still running)" << std::endl;
    } else {
        std::cout << "Disabled GPU " << gpu_id << std::endl;
    }
}

int GPUResourceManager::getAvailableGPUCount() {
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    int count = 0;
    for (int i = 0; i < num_gpus; i++) {
        if (gpu_states[i].enabled && !gpu_states[i].in_use) {
            updateGPUMemoryInfo(i);
            if (gpu_states[i].free_memory >= min_free_memory_bytes) {
                count++;
            }
        }
    }
    return count;
}

bool GPUResourceManager::isGPUAvailable(int gpu_id) {
    if (gpu_id < 0 || gpu_id >= num_gpus) return false;
    
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    if (!gpu_states[gpu_id].enabled || gpu_states[gpu_id].in_use) {
        return false;
    }
    
    updateGPUMemoryInfo(gpu_id);
    return gpu_states[gpu_id].free_memory >= min_free_memory_bytes;
}

std::string GPUResourceManager::getStatusString() {
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    std::ostringstream oss;
    oss << "\n=== GPU Resource Manager Status ===\n";
    oss << "Total GPUs: " << num_gpus << "\n";
    oss << "Minimum required memory: " 
        << (min_free_memory_bytes / (1024.0 * 1024.0 * 1024.0)) 
        << " GB\n\n";
    
    for (int i = 0; i < num_gpus; i++) {
        updateGPUMemoryInfo(i);
        
        oss << "GPU " << i << ": ";
        
        if (!gpu_states[i].enabled) {
            oss << "[DISABLED]";
        } else if (gpu_states[i].in_use) {
            oss << "[IN USE]";
        } else {
            oss << "[AVAILABLE]";
        }
        
        oss << " Free: " << std::fixed << std::setprecision(2)
            << (gpu_states[i].free_memory / (1024.0 * 1024.0 * 1024.0))
            << " GB, Jobs: " << gpu_states[i].jobs_processed << "\n";
    }
    
    oss << "===================================\n";
    
    return oss.str();
}

int GPUResourceManager::waitForGPU(int timeout_seconds) {
    auto start_time = std::chrono::system_clock::now();
    
    while (true) {
        int gpu_id = acquireGPU();
        
        if (gpu_id != -1) {
            return gpu_id;
        }
        
        // Check timeout
        if (timeout_seconds > 0) {
            auto now = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time).count();
            
            if (elapsed >= timeout_seconds) {
                std::cerr << "Timeout waiting for GPU after " 
                          << timeout_seconds << " seconds" << std::endl;
                return -1;
            }
        }
        
        // Wait a bit before trying again
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Print status periodically
        static int wait_count = 0;
        if (++wait_count % 5 == 0) {
            std::cout << "Still waiting for available GPU..." << std::endl;
            std::cout << getStatusString();
        }
    }
}
