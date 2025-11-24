
#ifndef DEVICEMEMORY_H_
#define DEVICEMEMORY_H_

#include <cstddef>
#include "compute/core/ComputeTypes.h"

class IDeviceMemory {
public:
    virtual ~IDeviceMemory() = default;
    
    // Memory operations
    virtual void* data() = 0;
    virtual const void* data() const = 0;
    virtual size_t size() const = 0;
    
    // Host <-> Device transfers
    virtual void copyFromHost(const void* host_ptr, size_t bytes) = 0;
    virtual void copyToHost(void* host_ptr, size_t bytes) const = 0;
    
    // Device <-> Device transfers
    virtual void copyFrom(const IDeviceMemory* other, size_t bytes) = 0;
    
    // Fill operations
    virtual void fill(int value, size_t bytes) = 0;
    
    // Metadata
    virtual DeviceBackend getBackend() const = 0;
    
    // Non-copyable (move-only)
    IDeviceMemory(const IDeviceMemory&) = delete;
    IDeviceMemory& operator=(const IDeviceMemory&) = delete;
    
protected:
    IDeviceMemory() = default;
};

#endif // DEVICEMEMORY_H_