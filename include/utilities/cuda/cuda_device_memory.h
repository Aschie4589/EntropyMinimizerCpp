#ifndef UTILITIES_CUDA_CUDA_DEVICE_MEMORY_H_
#define UTILITIES_CUDA_CUDA_DEVICE_MEMORY_H_

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

template<typename T>
class CudaDeviceMemory {
private:
    T* ptr_;
    size_t size_;
    bool owns_memory_;
    
public:
    // Constructor: allocate memory (allocate size of T times count)
    explicit CudaDeviceMemory(size_t count) 
        : ptr_(nullptr), size_(count), owns_memory_(true) {
        cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "cudaMalloc failed: " + std::string(cudaGetErrorString(err))
            );
        }
    }
    
    // Destructor: automatically free memory
    ~CudaDeviceMemory() {
        if (owns_memory_ && ptr_ != nullptr) {
            cudaFree(ptr_);  // Automatically called when object destroyed!
        }
    }
    
    // Delete copy constructor/assignment (can't copy GPU memory easily)
    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;
    
    // Move constructor (transfer ownership)
    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), owns_memory_(other.owns_memory_) {
        other.owns_memory_ = false;  // Don't let moved-from object free memory
    }
    
    // Move assignment
    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
        if (this != &other) {
            // Free our current memory
            if (owns_memory_ && ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            // Take ownership from other
            ptr_ = other.ptr_;
            size_ = other.size_;
            owns_memory_ = other.owns_memory_;
            other.owns_memory_ = false;
        }
        return *this;
    }
    
    // Accessors
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    // Implicit conversion to raw pointer (for CUDA functions)
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }
    
    // Copy to/from host
    void copyFromHost(const T* host_ptr, size_t count) {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        cudaError_t err = cudaMemcpy(ptr_, host_ptr, 
            count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy H2D failed");
        }
    }
    
    void copyToHost(T* host_ptr, size_t count) const {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds allocated size");
        }
        cudaError_t err = cudaMemcpy(host_ptr, ptr_, 
            count * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy D2H failed");
        }
    }
    
    // Zero memory
    void zero() {
        cudaError_t err = cudaMemset(ptr_, 0, size_ * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed");
        }
    }
};

#endif // UTILITIES_CUDA_CUDA_DEVICE_MEMORY_H_