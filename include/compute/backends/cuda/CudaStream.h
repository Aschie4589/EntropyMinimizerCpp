#ifndef CUDA_STREAM_H_
#define CUDA_STREAM_H_

#include "compute/stream/IComputeStream.h"
#include <cuda_runtime.h>

class CudaStream : public IStream {
public:
    CudaStream();
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
};

#endif // CUDA_STREAM_H_