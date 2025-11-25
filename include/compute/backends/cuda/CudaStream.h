#ifndef CUDA_STREAM_H_
#define CUDA_STREAM_H_

#include "compute/stream/IComputeStream.h"
#include <cuda_runtime.h>

class CudaStream : public IStream {
public:
    // RAII helper to ensure operations happen on correct device
    class DeviceGuard {
    public:
        explicit DeviceGuard(int device_id) {
            cudaGetDevice(&previous_device_);
            if (previous_device_ != device_id) {
                cudaSetDevice(device_id);
            }
        }
        
        ~DeviceGuard() {
            cudaSetDevice(previous_device_);
        }
        
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;
        
    private:
        int previous_device_;
    };
    
    explicit CudaStream(int device_id);
    ~CudaStream() override;
    
    void synchronize() override;
    bool isComplete() const override;
    void* getNativeHandle() override;
    
    // Move-only
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    
    // Get CUDA-specific handle (for internal use)
    cudaStream_t getCudaStream() const { return stream_; }
    
private:
    cudaStream_t stream_ = nullptr;
    int device_id_;
};

#endif // CUDA_STREAM_H_