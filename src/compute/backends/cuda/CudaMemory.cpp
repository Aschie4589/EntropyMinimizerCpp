#include "compute/backends/cuda/CudaMemory.h"
#include "utilities/cuda/error_handling.h"  // Your existing CUDA_CHECK macro
#include <stdexcept>

CudaMemory::CudaMemory(size_t bytes, int device_id) : size_(bytes), device_id_(device_id) {
    if (bytes == 0) {
        throw std::invalid_argument("Cannot allocate 0 bytes");
    }
    
    DeviceGuard guard(device_id_);
    CUDA_CHECK(cudaMalloc(&device_ptr_, bytes));
}

CudaMemory::~CudaMemory() {
    DeviceGuard guard(device_id_);
    free();
}

CudaMemory::CudaMemory(CudaMemory&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_), device_id_(other.device_id_) {
    other.device_ptr_ = nullptr;
    other.size_ = 0;
}

CudaMemory& CudaMemory::operator=(CudaMemory&& other) noexcept {
    if (this != &other) {
        // Free existing memory
        DeviceGuard guard(device_id_);
        free();
        
        // Move from other
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        device_id_ = other.device_id_;
        
        // Nullify other
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void CudaMemory::copyFromHost(const void* host_ptr, size_t bytes) {
    DeviceGuard guard(device_id_);
    
    if (bytes > size_) {
        throw std::invalid_argument("Copy size exceeds allocated memory");
    }
    
    CUDA_CHECK(cudaMemcpy(device_ptr_, host_ptr, bytes, cudaMemcpyHostToDevice));
}

void CudaMemory::copyToHost(void* host_ptr, size_t bytes) const {
    DeviceGuard guard(device_id_);
    
    if (bytes > size_) {
        throw std::invalid_argument("Copy size exceeds allocated memory");
    }
    
    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr_, bytes, cudaMemcpyDeviceToHost));
}

void CudaMemory::copyFrom(const IDeviceMemory* other, size_t bytes) {
    DeviceGuard guard(device_id_);
    
    if (bytes > size_) {
        throw std::invalid_argument("Copy size exceeds allocated memory");
    }
    
    if (other->getBackend() == DeviceBackend::CUDA) {
        // Device-to-device copy (same GPU or peer GPUs)
        CUDA_CHECK(cudaMemcpy(device_ptr_, other->data(), bytes, cudaMemcpyDeviceToDevice));
    } else {
        // Other backend (e.g., CPU) - go through host
        void* temp_host = operator new(bytes);
        other->copyToHost(temp_host, bytes);
        copyFromHost(temp_host, bytes);
        operator delete(temp_host);
    }
}

void CudaMemory::fill(int value, size_t bytes) {
    DeviceGuard guard(device_id_);
    
    if (bytes > size_) {
        throw std::invalid_argument("Fill size exceeds allocated memory");
    }
    
    CUDA_CHECK(cudaMemset(device_ptr_, value, bytes));
}

void CudaMemory::free() {
    if (device_ptr_ != nullptr) {
        cudaFree(device_ptr_);  // Don't check errors in cleanup
        device_ptr_ = nullptr;
        size_ = 0;
    }
}
