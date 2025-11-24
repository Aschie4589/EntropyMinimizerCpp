#include "compute/backends/cpu/CpuMemory.h"
#include <stdexcept>
#include <cstring>

CpuMemory::CpuMemory(size_t bytes) : size_(bytes) {
    if (bytes == 0) {
        throw std::invalid_argument("Cannot allocate 0 bytes");
    }
    
    host_ptr_ = operator new(bytes);
    
    if (host_ptr_ == nullptr) {
        throw std::bad_alloc();
    }
}

CpuMemory::~CpuMemory() {
    free();
}

CpuMemory::CpuMemory(CpuMemory&& other) noexcept
    : host_ptr_(other.host_ptr_), size_(other.size_) {
    other.host_ptr_ = nullptr;
    other.size_ = 0;
}

CpuMemory& CpuMemory::operator=(CpuMemory&& other) noexcept {
    if (this != &other) {
        // Free existing memory
        free();
        
        // Move from other
        host_ptr_ = other.host_ptr_;
        size_ = other.size_;
        
        // Nullify other
        other.host_ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void CpuMemory::copyFromHost(const void* host_ptr, size_t bytes) {
    if (bytes > size_) {
        throw std::invalid_argument("Copy size exceeds allocated memory");
    }
    
    std::memcpy(host_ptr_, host_ptr, bytes);
}

void CpuMemory::copyToHost(void* host_ptr, size_t bytes) const {
    if (bytes > size_) {
        throw std::invalid_argument("Copy size exceeds allocated memory");
    }
    
    std::memcpy(host_ptr, host_ptr_, bytes);
}

void CpuMemory::copyFrom(const IDeviceMemory* other, size_t bytes) {
    if (bytes > size_) {
        throw std::invalid_argument("Copy size exceeds allocated memory");
    }
    
    if (other->getBackend() == DeviceBackend::CPU) {
        // Direct memory copy (both on CPU)
        std::memcpy(host_ptr_, other->data(), bytes);
    } else {
        // Other backend (e.g., CUDA) - use its copyToHost
        other->copyToHost(host_ptr_, bytes);
    }
}

void CpuMemory::fill(int value, size_t bytes) {
    if (bytes > size_) {
        throw std::invalid_argument("Fill size exceeds allocated memory");
    }
    
    std::memset(host_ptr_, value, bytes);
}

void CpuMemory::free() {
    if (host_ptr_ != nullptr) {
        operator delete(host_ptr_);
        host_ptr_ = nullptr;
        size_ = 0;
    }
}
