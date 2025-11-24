#ifndef CPU_MEMORY_H_
#define CPU_MEMORY_H_

#include "compute/memory/IDeviceMemory.h"
#include <cstring>

class CpuMemory : public IDeviceMemory {
public:
    // Allocate CPU memory
    explicit CpuMemory(size_t bytes);
    
    ~CpuMemory() override;
    
    // IDeviceMemory interface
    void* data() override { return host_ptr_; }
    const void* data() const override { return host_ptr_; }
    size_t size() const override { return size_; }
    
    void copyFromHost(const void* host_ptr, size_t bytes) override;
    void copyToHost(void* host_ptr, size_t bytes) const override;
    void copyFrom(const IDeviceMemory* other, size_t bytes) override;
    void fill(int value, size_t bytes) override;
    
    DeviceBackend getBackend() const override { return DeviceBackend::CPU; }
    
    // Move-only semantics
    CpuMemory(CpuMemory&& other) noexcept;
    CpuMemory& operator=(CpuMemory&& other) noexcept;
    
    // CPU-specific: direct pointer access (same as data())
    template<typename T = void>
    T* hostPtr() { return static_cast<T*>(host_ptr_); }
    
    template<typename T = void>
    const T* hostPtr() const { return static_cast<const T*>(host_ptr_); }
    
private:
    void* host_ptr_ = nullptr;
    size_t size_ = 0;
    
    void free();
};

#endif // CPU_MEMORY_H_
