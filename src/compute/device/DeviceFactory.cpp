#include "compute/device/DeviceFactory.h"
#include "compute/backends/cpu/CpuDevice.h"

#ifdef __CUDACC__
#include "compute/backends/cuda/CudaDevice.h"
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <sstream>
#include <iostream>

using namespace compute;

std::unique_ptr<IComputeDevice> DeviceFactory::create(
    DeviceType type,
    int device_id,
    size_t device_scratch_size,
    size_t host_scratch_size
) {
    if (type == DeviceType::AUTO) {
        // Try CUDA first
        if (isCudaAvailable()) {
            std::cout << "DeviceFactory: Auto-detected CUDA, creating CudaDevice" << std::endl;
            #ifdef __CUDACC__
            return std::make_unique<CudaDevice>(device_id, device_scratch_size, host_scratch_size);
            #endif
        }
        
        // Fallback to CPU
        std::cout << "DeviceFactory: CUDA unavailable, falling back to CpuDevice" << std::endl;
        return std::make_unique<CpuDevice>(device_id, host_scratch_size);
    }
    
    if (type == DeviceType::CUDA) {
        #ifdef __CUDACC__
        if (!isCudaAvailable()) {
            throw std::runtime_error(
                "CUDA device requested but CUDA is not available on this system"
            );
        }
        
        int device_count = getCudaDeviceCount();
        if (device_id >= device_count) {
            throw std::runtime_error(
                "CUDA device " + std::to_string(device_id) + " requested but only " +
                std::to_string(device_count) + " devices available"
            );
        }
        
        return std::make_unique<CudaDevice>(device_id, device_scratch_size, host_scratch_size);
        #else
        throw std::runtime_error(
            "CUDA device requested but this binary was not compiled with CUDA support"
        );
        #endif
    }
    
    // DeviceType::CPU
    return std::make_unique<CpuDevice>(device_id, host_scratch_size);
}

bool DeviceFactory::isCudaAvailable() {
    #ifdef __CUDACC__
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    // cudaGetDeviceCount returns cudaSuccess even if no devices
    // So we need to check both the error and the count
    if (error != cudaSuccess) {
        return false;
    }
    
    return device_count > 0;
    #else
    return false;
    #endif
}

int DeviceFactory::getCudaDeviceCount() {
    #ifdef __CUDACC__
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        return 0;
    }
    
    return device_count;
    #else
    return 0;
    #endif
}

std::string DeviceFactory::getCudaDeviceInfo(int device_id) {
    #ifdef __CUDACC__
    int device_count = getCudaDeviceCount();
    
    if (device_id >= device_count) {
        throw std::runtime_error(
            "Device " + std::to_string(device_id) + " does not exist (only " +
            std::to_string(device_count) + " devices available)"
        );
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    
    if (error != cudaSuccess) {
        throw std::runtime_error(
            "Failed to get properties for device " + std::to_string(device_id) +
            ": " + cudaGetErrorString(error)
        );
    }
    
    std::ostringstream oss;
    oss << "Device " << device_id << ": " << prop.name
        << " (Compute Capability " << prop.major << "." << prop.minor << ")"
        << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB";
    
    return oss.str();
    #else
    throw std::runtime_error("CUDA not available - binary not compiled with CUDA support");
    #endif
}
