#ifndef UTILITIES_CUDA_CUDA_STREAM_H_
#define UTILITIES_CUDA_CUDA_STREAM_H_

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

class CudaStream {
private:
    cudaStream_t stream_;
    bool owns_stream_;
    
public:
    // Constructor: create stream
    CudaStream() : stream_(nullptr), owns_stream_(true) {
        cudaError_t err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "cudaStreamCreate failed: " + std::string(cudaGetErrorString(err))
            );
        }
    }
    
    // Destructor: automatically destroy stream
    ~CudaStream() {
        if (owns_stream_ && stream_ != nullptr) {
            cudaStreamDestroy(stream_);  // Automatic cleanup!
        }
    }
    
    // Delete copy
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // Move semantics
    CudaStream(CudaStream&& other) noexcept
        : stream_(other.stream_), owns_stream_(other.owns_stream_) {
        other.owns_stream_ = false;
    }
    
    // Get raw handle (for CUDA functions)
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
    // Synchronize
    void synchronize() {
        cudaError_t err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaStreamSynchronize failed");
        }
    }
    
    // Query if stream is done
    bool isComplete() const {
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess) {
            return true;
        } else if (err == cudaErrorNotReady) {
            return false;
        } else {
            throw std::runtime_error("cudaStreamQuery failed");
        }
    }
};

#endif // UTILITIES_CUDA_CUDA_STREAM_H_