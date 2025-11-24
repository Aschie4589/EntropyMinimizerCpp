#ifndef CUDA_MEMORY_H_
#define CUDA_MEMORY_H_

#include "compute/memory/IDeviceMemory.h"
#include <cuda_runtime.h>

class CudaMemory : public IDeviceMemory {
public:
    // Allocate device memory
    explicit CudaMemory(size_t bytes);
    
    ~CudaMemory() override;
    
    // IDeviceMemory interface
    void* data() override { return device_ptr_; }
    const void* data() const override { return device_ptr_; }
    size_t size() const override { return size_; }
    
    void copyFromHost(const void* host_ptr, size_t bytes) override;
    void copyToHost(void* host_ptr, size_t bytes) const override;
    void copyFrom(const IDeviceMemory* other, size_t bytes) override;
    void fill(int value, size_t bytes) override;
    
    DeviceBackend getBackend() const override { return DeviceBackend::CUDA; }
    
    // Move-only semantics
    CudaMemory(CudaMemory&& other) noexcept;
    CudaMemory& operator=(CudaMemory&& other) noexcept;
    
    // CUDA-specific: get raw device pointer
    template<typename T = void>
    T* devicePtr() { return static_cast<T*>(device_ptr_); }
    
    template<typename T = void>
    const T* devicePtr() const { return static_cast<const T*>(device_ptr_); }
    
private:
    void* device_ptr_ = nullptr;
    size_t size_ = 0;
    
    void free();
};

#endif // CUDA_MEMORY_H_
