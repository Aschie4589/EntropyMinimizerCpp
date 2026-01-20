#include "utilities/gpu/nvml_wrapper.h"
#include <iostream>
#include <cstring>

namespace utils {

NVMLWrapper::NVMLWrapper() : initialized_(false), last_error_("") {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        last_error_ = std::string("Failed to initialize NVML: ") + nvmlErrorString(result);
        std::cerr << "WARNING: " << last_error_ << std::endl;
        initialized_ = false;
    } else {
        initialized_ = true;
    }
}

NVMLWrapper::~NVMLWrapper() {
    if (initialized_) {
        nvmlShutdown();
    }
}

NVMLWrapper& NVMLWrapper::instance() {
    static NVMLWrapper instance;
    return instance;
}

bool NVMLWrapper::getDeviceInfo(int cuda_device_id, DeviceInfo& info) {
    if (!initialized_) {
        last_error_ = "NVML not initialized";
        return false;
    }
    
    nvmlDevice_t device;
    nvmlReturn_t result;
    
    // Get device handle from CUDA device index
    result = nvmlDeviceGetHandleByIndex(cuda_device_id, &device);
    if (result != NVML_SUCCESS) {
        last_error_ = std::string("Failed to get device handle: ") + nvmlErrorString(result);
        return false;
    }
    
    // Get memory info
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result == NVML_SUCCESS) {
        info.total_memory = memory.total;
        info.free_memory = memory.free;
        info.used_memory = memory.used;
    } else {
        last_error_ = std::string("Failed to get memory info: ") + nvmlErrorString(result);
        // Continue to get other info even if memory query fails
    }
    
    // Get utilization rates
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result == NVML_SUCCESS) {
        info.utilization = utilization.gpu;
    } else {
        info.utilization = 0;
        // Non-critical, don't fail
    }
    
    // Get temperature
    unsigned int temp;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result == NVML_SUCCESS) {
        info.temperature = temp;
    } else {
        info.temperature = 0;
        // Non-critical, don't fail
    }
    
    // Get device name
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetName(device, name, sizeof(name));
    if (result == NVML_SUCCESS) {
        info.name = name;
    } else {
        info.name = "Unknown";
        // Non-critical, don't fail
    }
    
    return true;
}

unsigned int NVMLWrapper::getDeviceCount() {
    if (!initialized_) {
        return 0;
    }
    
    unsigned int count;
    nvmlReturn_t result = nvmlDeviceGetCount(&count);
    if (result != NVML_SUCCESS) {
        last_error_ = std::string("Failed to get device count: ") + nvmlErrorString(result);
        return 0;
    }
    
    return count;
}

} // namespace utils
