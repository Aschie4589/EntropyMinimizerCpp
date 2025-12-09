#include "minimizer/orchestration/device_pool.h"
#include "minimizer/orchestration/resource_monitor.h"
#include <sstream>

namespace entropy {

DevicePool::DevicePool(const ResourceLimits& limits)
    : num_gpus_(0), num_cpus_(0) {
    
    // Validate limits
    if (limits.max_gpus < 0) {
        throw std::invalid_argument(
            "Invalid max_gpus: " + std::to_string(limits.max_gpus) + " (must be >= 0)"
        );
    }
    if (limits.max_cpus < 0) {
        throw std::invalid_argument(
            "Invalid max_cpus: " + std::to_string(limits.max_cpus) + " (must be >= 0)"
        );
    }
    if (limits.max_gpus == 0 && limits.max_cpus == 0) {
        throw std::invalid_argument(
            "Invalid configuration: at least one GPU or CPU device required"
        );
    }
    
    // Initialize devices in order: GPUs first, then CPUs
    initializeGPUs(limits.max_gpus);
    initializeCPUs(limits.max_cpus);
}

void DevicePool::initializeGPUs(int count) {
    if (count == 0) {
        return;
    }
    
    // Check CUDA availability
    if (!DeviceFactory::isCudaAvailable()) {
        throw std::runtime_error(
            "CUDA not available, but " + std::to_string(count) + " GPUs requested"
        );
    }
    
    int available_gpus = DeviceFactory::getCudaDeviceCount();
    if (count > available_gpus) {
        throw std::runtime_error(
            "Requested " + std::to_string(count) + " GPUs, but only " +
            std::to_string(available_gpus) + " available"
        );
    }
    
    // Create GPU devices
    for (int i = 0; i < count; ++i) {
        try {
            auto device = DeviceFactory::create(
                DeviceFactory::DeviceType::CUDA,
                i,  // device_id
                64 * 1024 * 1024,  // 64 MB device scratch
                16 * 1024 * 1024   // 16 MB host scratch
            );
            
            DeviceInfo info(i, DeviceType::GPU, false);
            devices_.emplace_back(std::move(device), info);
            num_gpus_++;
            
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to initialize GPU " + std::to_string(i) + ": " + e.what()
            );
        }
    }
}

void DevicePool::initializeCPUs(int count) {
    if (count == 0) {
        return;
    }
    
    // Create CPU devices
    for (int i = 0; i < count; ++i) {
        try {
            auto device = DeviceFactory::create(
                DeviceFactory::DeviceType::CPU,
                0,   // device_id (ignored for CPU)
                0,   // no device scratch for CPU
                16 * 1024 * 1024   // 16 MB host scratch
            );
            
            DeviceInfo info(-1, DeviceType::CPU, false);
            devices_.emplace_back(std::move(device), info);
            num_cpus_++;
            
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to initialize CPU device " + std::to_string(i) + ": " + e.what()
            );
        }
    }
}

IComputeDevice& DevicePool::getDeviceForWorker(int worker_id) {
    if (worker_id < 0) {
        throw std::runtime_error(
            "Invalid worker_id: " + std::to_string(worker_id) + " (must be >= 0)"
        );
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (devices_.empty()) {
        throw std::runtime_error("DevicePool is empty, cannot assign device");
    }
    
    // Round-robin assignment
    size_t device_index = worker_id % devices_.size();
    
    // Mark as in use
    devices_[device_index].info.in_use = true;
    
    return *devices_[device_index].device;
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
// Dynamic Resizing Implementation (Phase 2 Step 2.2)
// ============================================================================

void DevicePool::resize(int new_max_gpus, int new_max_cpus) {
    // Validate new limits
    if (new_max_gpus < 0) {
        throw std::invalid_argument(
            "Invalid new_max_gpus: " + std::to_string(new_max_gpus) + " (must be >= 0)"
        );
    }
    if (new_max_cpus < 0) {
        throw std::invalid_argument(
            "Invalid new_max_cpus: " + std::to_string(new_max_cpus) + " (must be >= 0)"
        );
    }
    if (new_max_gpus == 0 && new_max_cpus == 0) {
        throw std::invalid_argument(
            "Invalid resize: at least one GPU or CPU device required"
        );
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Handle GPU resize
    int current_gpus = static_cast<int>(num_gpus_);
    if (new_max_gpus > current_gpus) {
        // Scale up: add GPUs
        addGPUs(new_max_gpus - current_gpus);
    } else if (new_max_gpus < current_gpus) {
        // Scale down: mark GPUs for removal
        removeGPUs(current_gpus - new_max_gpus);
    }
    
    // Handle CPU resize
    int current_cpus = static_cast<int>(num_cpus_);
    if (new_max_cpus > current_cpus) {
        // Scale up: add CPUs
        addCPUs(new_max_cpus - current_cpus);
    } else if (new_max_cpus < current_cpus) {
        // Scale down: mark CPUs for removal
        removeCPUs(current_cpus - new_max_cpus);
    }
}

void DevicePool::addGPUs(int count) {
    if (count == 0) {
        return;
    }
    
    // Check CUDA availability
    if (!DeviceFactory::isCudaAvailable()) {
        throw std::runtime_error(
            "CUDA not available, cannot add " + std::to_string(count) + " GPUs"
        );
    }
    
    int available_gpus = DeviceFactory::getCudaDeviceCount();
    int next_gpu_id = static_cast<int>(num_gpus_);
    
    if (next_gpu_id + count > available_gpus) {
        throw std::runtime_error(
            "Cannot add " + std::to_string(count) + " GPUs: only " +
            std::to_string(available_gpus - next_gpu_id) + " available"
        );
    }
    
    // Find insertion point (after existing GPUs, before CPUs)
    size_t insert_pos = num_gpus_;
    
    for (int i = 0; i < count; ++i) {
        int gpu_id = next_gpu_id + i;
        try {
            auto device = DeviceFactory::create(
                DeviceFactory::DeviceType::CUDA,
                gpu_id,
                64 * 1024 * 1024,  // 64 MB device scratch
                16 * 1024 * 1024   // 16 MB host scratch
            );
            
            DeviceInfo info(gpu_id, DeviceType::GPU, false);
            devices_.insert(
                devices_.begin() + insert_pos + i,
                DeviceSlot(std::move(device), info)
            );
            num_gpus_++;
            
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to add GPU " + std::to_string(gpu_id) + ": " + e.what()
            );
        }
    }
}

void DevicePool::addCPUs(int count) {
    if (count == 0) {
        return;
    }
    
    // CPUs are appended to the end
    for (int i = 0; i < count; ++i) {
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
                "Failed to add CPU device: " + std::string(e.what())
            );
        }
    }
}

void DevicePool::removeGPUs(int count) {
    if (count == 0) {
        return;
    }
    
    // Mark last 'count' GPUs for removal
    size_t gpus_to_mark = std::min(static_cast<size_t>(count), num_gpus_);
    size_t marked = 0;
    
    for (size_t i = 0; i < devices_.size() && marked < gpus_to_mark; ++i) {
        if (devices_[i].info.type == DeviceType::GPU && !devices_[i].marked_for_removal) {
            // Mark GPUs from the end (highest device_id first)
            size_t target_idx = num_gpus_ - 1 - marked;
            for (size_t j = 0; j < devices_.size(); ++j) {
                if (devices_[j].info.type == DeviceType::GPU && 
                    !devices_[j].marked_for_removal &&
                    devices_[j].info.device_id == static_cast<int>(target_idx)) {
                    devices_[j].marked_for_removal = true;
                    marked++;
                    break;
                }
            }
        }
    }
}

void DevicePool::removeCPUs(int count) {
    if (count == 0) {
        return;
    }
    
    // Mark last 'count' CPUs for removal (from the end of the devices_ vector)
    size_t cpus_to_mark = std::min(static_cast<size_t>(count), num_cpus_);
    size_t marked = 0;
    
    for (auto it = devices_.rbegin(); it != devices_.rend() && marked < cpus_to_mark; ++it) {
        if (it->info.type == DeviceType::CPU && !it->marked_for_removal) {
            it->marked_for_removal = true;
            marked++;
        }
    }
}

bool DevicePool::shouldShutdown(int worker_id) const {
    if (worker_id < 0) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (devices_.empty()) {
        return false;
    }
    
    size_t device_index = worker_id % devices_.size();
    return devices_[device_index].marked_for_removal;
}

void DevicePool::markDeviceForRemoval(int worker_id) {
    if (worker_id < 0) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (devices_.empty()) {
        return;
    }
    
    size_t device_index = worker_id % devices_.size();
    devices_[device_index].marked_for_removal = true;
}

void DevicePool::finalizeRemovals() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Remove devices marked for removal
    auto new_end = std::remove_if(
        devices_.begin(),
        devices_.end(),
        [](const DeviceSlot& slot) { return slot.marked_for_removal; }
    );
    
    // Update counts
    size_t new_gpu_count = 0;
    size_t new_cpu_count = 0;
    for (auto it = devices_.begin(); it != new_end; ++it) {
        if (it->info.type == DeviceType::GPU) {
            new_gpu_count++;
        } else {
            new_cpu_count++;
        }
    }
    
    devices_.erase(new_end, devices_.end());
    num_gpus_ = new_gpu_count;
    num_cpus_ = new_cpu_count;
}

size_t DevicePool::activeDeviceCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t active = 0;
    for (const auto& slot : devices_) {
        if (!slot.marked_for_removal) {
            active++;
        }
    }
    return active;
}

} // namespace entropy
