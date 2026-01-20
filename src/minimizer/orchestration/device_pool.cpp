#include "minimizer/orchestration/device_pool.h"
#include "minimizer/orchestration/gpu_registry.h"
#include <sstream>
#include <algorithm>
#include <set>
#include <limits>


// Debug
#include <iostream>
namespace entropy {

// New constructor for ResourceConfig (Phase 2)
DevicePool::DevicePool(const ResourceConfig& config)
    : num_gpus_(0), num_cpus_(0),
      use_resource_config_(true),
      desired_gpus_(config.desired_gpus),
      desired_cpus_(config.desired_cpus),
      min_gpu_memory_(config.min_gpu_memory),
      fallback_to_cpu_(config.fallback_to_cpu) {
    
    config.validate();
    
    // Don't create devices here - wait for initializeFromRegistry()
}

std::vector<DeviceInfo> DevicePool::getDeviceInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<DeviceInfo> info;
    info.reserve(devices_.size());
    
    for (const auto& slot : devices_) {
        info.push_back(slot.info);
    }
    
    return info;
}

// ============================================================================
// New Dynamic Device Management Implementation (Phase 2)
// ============================================================================

void DevicePool::initializeFromRegistry(GPURegistry& registry) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "DevicePool: Initializing devices from GPURegistry..." << std::endl;
    // Get enabled GPUs from registry based on config
    std::vector<int> selected_gpus;
    
    if (desired_gpus_ > 0) {

        std::cout << "DevicePool: Selecting up to " 
                  << desired_gpus_ << " GPUs with minimum "
                  << min_gpu_memory_ / (1024*1024) << " MB free memory." 
                  << std::endl;

        selected_gpus = registry.selectBestGPUs(desired_gpus_, min_gpu_memory_);
        std::cout << "DevicePool: Selected GPUs: ";
        for (int gpu_id : selected_gpus) {
            std::cout << gpu_id << " ";
        }
        std::cout << std::endl;      
        // Mark GPUs as in use and create devices
        for (int gpu_id : selected_gpus) {
            std::cout << "DevicePool: Marking GPU " << gpu_id << " as in use." << std::endl;
            registry.markGPUInUse(gpu_id, true);
            std::cout << "DevicePool: Creating GPU device " << gpu_id << std::endl;
            createGPU(gpu_id);

        }
    }
    
    // Create CPU workers
    for (int i = 0; i < desired_cpus_; ++i) {
        createCPU();
    }
    
    // Fallback if no devices created
    if (devices_.empty() && fallback_to_cpu_) {
        createCPU();
    }
    
    if (devices_.empty()) {
        throw std::runtime_error(
            "DevicePool::initializeFromRegistry: No devices available and fallback disabled"
        );
    }
}

IComputeDevice& DevicePool::getDeviceForWorker(int worker_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "DevicePool: Assigning device for worker " << worker_id << std::endl;
    
    if (devices_.empty()) {
        throw std::runtime_error("DevicePool: No devices available");
    }

    std::cout << "DevicePool: Total devices available: " << devices_.size() << std::endl;
    
    // Round-robin assignment: worker_id % num_devices
    size_t device_index = worker_id % devices_.size();

    std::cout << "DevicePool: Assigned device index " << device_index 
              << " to worker " << worker_id << std::endl;

    std::cout << "DevicePool: Device info: "
              << devices_[device_index].info.device_id << ", type: "
              << (devices_[device_index].info.type == DeviceType::GPU ? "GPU" : "CPU")
              << std::endl;
    return *devices_[device_index].device;
}

IComputeDevice* DevicePool::acquireDevice() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (devices_.empty()) {
        throw std::runtime_error("DevicePool: No devices available");
    }
    
    // Find available device with lowest reference count
    DeviceSlot* best_device = nullptr;
    int min_ref_count = std::numeric_limits<int>::max();
    
    for (auto& slot : devices_) {
        if (slot.marked_for_removal) {
            continue;  // Skip devices pending removal
        }
        
        if (slot.reference_count < min_ref_count) {
            best_device = &slot;
            min_ref_count = slot.reference_count;
        }
    }
    
    if (!best_device) {
        throw std::runtime_error("DevicePool: No available devices (all marked for removal)");
    }
    
    // Increment reference count
    best_device->reference_count++;
    
    std::cout << "DevicePool: Acquired device " << best_device->info.device_id 
              << " (type=" << (best_device->info.type == DeviceType::GPU ? "GPU" : "CPU")
              << ", ref_count=" << best_device->reference_count << ")" << std::endl;
    
    return best_device->device.get();
}

void DevicePool::releaseDevice(IComputeDevice* device) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& slot : devices_) {
        if (slot.device.get() == device) {
            if (slot.reference_count > 0) {
                slot.reference_count--;
                std::cout << "DevicePool: Released device " << slot.info.device_id 
                          << " (ref_count=" << slot.reference_count << ")" << std::endl;
            } else {
                std::cerr << "DevicePool: Warning - releasing device with ref_count already 0" << std::endl;
            }
            return;
        }
    }
    
    throw std::runtime_error("DevicePool: Attempted to release unknown device");
}

void DevicePool::addGPU(int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    createGPU(device_id);
}

void DevicePool::removeGPU(int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find and mark the GPU for removal
    for (auto& slot : devices_) {
        if (slot.info.type == DeviceType::GPU && 
            slot.info.device_id == device_id &&
            !slot.marked_for_removal) {
            slot.marked_for_removal = true;
            return;
        }
    }
}

void DevicePool::adjustToEnabledGPUs(const std::vector<int>& enabled_gpu_ids) {

    std::cout << "DevicePool: Adjusting to enabled GPUs..." << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find current GPU device_ids
    std::vector<int> current_gpus;

    for (const auto& slot : devices_) {
        if (slot.info.type == DeviceType::GPU) {
            current_gpus.push_back(slot.info.device_id);
        }
    }
    std::cout << "DevicePool: Current GPUs in pool: ";
    for (int gpu_id : current_gpus) {
        std::cout << gpu_id << " ";
    }
    std::cout << std::endl;    
    // Convert enabled list to set for fast lookup
    std::set<int> enabled_set(enabled_gpu_ids.begin(), enabled_gpu_ids.end());
    
    // Remove GPUs that are no longer enabled
    for (int gpu_id : current_gpus) {
        // Check if the gpu is still enabled
        if (enabled_set.find(gpu_id) == enabled_set.end()) {
            // Mark for removal
            std::cout << "DevicePool: Removing GPU " << gpu_id << std::endl;
            removeGPU(gpu_id);
            std::cout << "DevicePool: Removed GPU " << gpu_id << "." << std::endl;
        }
    }

    // Add newly enabled GPUs
    for (int gpu_id : enabled_gpu_ids) {
        if (std::find(current_gpus.begin(), current_gpus.end(), gpu_id) == current_gpus.end()) {
            // Create new GPU device
            std::cout << "DevicePool: Adding GPU " << gpu_id << std::endl;
            createGPU(gpu_id);

            std::cout << "DevicePool: Added GPU " << gpu_id << "." << std::endl;

        }
    }
    std::cout << "DevicePool: Adjusted GPUs to enabled list." << std::endl;
}

std::vector<DevicePool::DeviceStatus> DevicePool::getDeviceStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<DeviceStatus> status_list;
    status_list.reserve(devices_.size());
    
    for (size_t i = 0; i < devices_.size(); ++i) {
        const auto& slot = devices_[i];
        
        DeviceStatus status;
        status.pool_index = static_cast<int>(i);
        status.physical_device_id = slot.info.device_id;
        status.type = slot.info.type;
        status.available = !slot.marked_for_removal;
        status.free_memory = 0;  // TODO: Query from device if possible
        
        status_list.push_back(status);
    }
    
    return status_list;
}

size_t DevicePool::numActiveGPUs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t count = 0;
    for (const auto& slot : devices_) {
        if (slot.info.type == DeviceType::GPU && !slot.marked_for_removal) {
            count++;
        }
    }
    return count;
}

size_t DevicePool::numActiveCPUs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t count = 0;
    for (const auto& slot : devices_) {
        if (slot.info.type == DeviceType::CPU && !slot.marked_for_removal) {
            count++;
        }
    }
    return count;
}

void DevicePool::createGPU(int device_id) {
    // Check CUDA availability
    if (!DeviceFactory::isCudaAvailable()) {
        throw std::runtime_error(
            "CUDA not available, cannot create GPU " + std::to_string(device_id)
        );
    }
    
    try {
        auto device = DeviceFactory::create(
            DeviceFactory::DeviceType::CUDA,
            device_id,
            64 * 1024 * 1024,  // 64 MB device scratch
            16 * 1024 * 1024   // 16 MB host scratch
        );
        
        DeviceInfo info(device_id, DeviceType::GPU, false);
        
        // Insert after existing GPUs, before CPUs
        size_t insert_pos = num_gpus_;
        devices_.insert(
            devices_.begin() + insert_pos,
            DeviceSlot(std::move(device), info)
        );
        num_gpus_++;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to create GPU " + std::to_string(device_id) + ": " + e.what()
        );
    }
}

void DevicePool::createCPU() {
    try {
        auto device = DeviceFactory::create(
            DeviceFactory::DeviceType::CPU,
            0,
            0,
            16 * 1024 * 1024
        );
        
        DeviceInfo info(-1, DeviceType::CPU, false);
        devices_.emplace_back(std::move(device), info);
        num_cpus_++;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to create CPU device: " + std::string(e.what())
        );
    }
}

int DevicePool::getPhysicalDeviceID(const IComputeDevice* device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& slot : devices_) {
        if (slot.device.get() == device) {
            return slot.info.device_id;
        }
    }
    
    throw std::runtime_error("DevicePool: Device not found in pool");
}

std::vector<int> DevicePool::getDevicesMarkedForRemoval() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<int> marked;
    for (const auto& slot : devices_) {
        if (slot.marked_for_removal && slot.info.type == DeviceType::GPU) {
            marked.push_back(slot.info.device_id);
        }
    }
    return marked;
}

int DevicePool::getReferenceCount(const IComputeDevice* device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& slot : devices_) {
        if (slot.device.get() == device) {
            return slot.reference_count;
        }
    }
    return 0;
}

void DevicePool::markDeviceForRemoval(int physical_device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& slot : devices_) {
        if (slot.info.type == DeviceType::GPU && 
            slot.info.device_id == physical_device_id &&
            !slot.marked_for_removal) {
            slot.marked_for_removal = true;
            std::cout << "DevicePool: Marked device " << physical_device_id 
                      << " for removal (ref_count=" << slot.reference_count << ")" << std::endl;
            return;
        }
    }
    
    std::cout << "DevicePool: Warning - device " << physical_device_id 
              << " not found or already marked" << std::endl;
}

void DevicePool::cleanupMarkedDevices() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t initial_size = devices_.size();
    
    auto it = std::remove_if(devices_.begin(), devices_.end(),
        [](const DeviceSlot& slot) {
            if (slot.marked_for_removal && slot.reference_count == 0) {
                std::cout << "DevicePool: Cleaning up device " << slot.info.device_id 
                          << " (type=" << (slot.info.type == DeviceType::GPU ? "GPU" : "CPU") 
                          << ")" << std::endl;
                return true;
            }
            if (slot.marked_for_removal && slot.reference_count > 0) {
                std::cout << "DevicePool: Cannot cleanup device " << slot.info.device_id 
                          << " yet (ref_count=" << slot.reference_count << ")" << std::endl;
            }
            return false;
        });
    
    size_t removed_count = std::distance(it, devices_.end());
    
    // Count GPU removals for updating num_gpus_
    size_t removed_gpus = 0;
    for (auto iter = it; iter != devices_.end(); ++iter) {
        if (iter->info.type == DeviceType::GPU) {
            removed_gpus++;
        }
    }
    
    devices_.erase(it, devices_.end());
    num_gpus_ -= removed_gpus;
    
    if (removed_count > 0) {
        std::cout << "DevicePool: Cleaned up " << removed_count << " device(s) "
                  << "(" << removed_gpus << " GPU(s))" << std::endl;
    } else {
        std::cout << "DevicePool: No devices cleaned up (" 
                  << (initial_size - devices_.size()) << " still have references)" << std::endl;
    }
}
} // namespace entropy
