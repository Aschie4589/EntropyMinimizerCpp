#include "compute/backends/cuda/CudaStream.h"
#include "utilities/cuda/error_handling.h"  // Your existing CUDA_CHECK macro

CudaStream::CudaStream() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaStream::~CudaStream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);  // Don't check errors in destructor
        stream_ = nullptr;
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        // Clean up existing stream
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        // Move from other
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool CudaStream::isComplete() const {
    cudaError_t status = cudaStreamQuery(stream_);
    if (status == cudaSuccess) {
        return true;  // Stream is idle
    } else if (status == cudaErrorNotReady) {
        return false;  // Stream is still working
    } else {
        CUDA_CHECK(status);  // Actual error, throw
        return false;  // Unreachable
    }
}

void* CudaStream::getNativeHandle() {
    return static_cast<void*>(stream_);
}