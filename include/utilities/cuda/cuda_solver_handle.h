#ifndef UTILITIES_CUDA_CUDA_SOLVER_HANDLE_H_
#define UTILITIES_CUDA_CUDA_SOLVER_HANDLE_H_

#include <cusolverDn.h>
#include <stdexcept>

class CudaSolverHandle {
private:
    cusolverDnHandle_t handle_;
    bool owns_handle_;
    
public:
    CudaSolverHandle() : handle_(nullptr), owns_handle_(true) {
        cusolverStatus_t status = cusolverDnCreate(&handle_);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnCreate failed");
        } 
    }
    
    ~CudaSolverHandle() {
        if (owns_handle_ && handle_ != nullptr) {
            cusolverDnDestroy(handle_);
        }
    }
    
    // Delete copy
    CudaSolverHandle(const CudaSolverHandle&) = delete;
    CudaSolverHandle& operator=(const CudaSolverHandle&) = delete;
    
    // Move semantics
    CudaSolverHandle(CudaSolverHandle&& other) noexcept
        : handle_(other.handle_), owns_handle_(other.owns_handle_) {
        other.owns_handle_ = false;
    }
    
    cusolverDnHandle_t get() const { return handle_; }
    operator cusolverDnHandle_t() const { return handle_; }
    
    // Set stream
    void setStream(cudaStream_t stream) {
        cusolverStatus_t status = cusolverDnSetStream(handle_, stream);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnSetStream failed");
        }
    }
};

#endif // UTILITIES_CUDA_CUDA_SOLVER_HANDLE_H_