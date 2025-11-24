#ifndef UTILITIES_CUDA_CUDA_BLAS_HANDLE_H_
#define UTILITIES_CUDA_CUDA_BLAS_HANDLE_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

/**
 * @brief RAII wrapper for cuBLAS handle
 * 
 * Automatically creates cuBLAS handle on construction and destroys it on destruction.
 * Prevents memory leaks and ensures exception safety.
 * 
 * Example usage:
 *   CudaBlasHandle handle;
 *   cublasZdscal(handle.get(), n, &alpha, x, incx);
 *   // Automatically destroyed when out of scope
 */
class CudaBlasHandle {
private:
    cublasHandle_t handle_;
    bool owns_handle_;
    
public:
    /**
     * @brief Construct a new CudaBlasHandle and create cuBLAS handle
     * @throws std::runtime_error if cuBLAS handle creation fails
     */
    CudaBlasHandle() : handle_(nullptr), owns_handle_(true) {
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "cublasCreate failed with status code: " << status;
            throw std::runtime_error(oss.str());
        }
    }
    
    /**
     * @brief Destroy the CudaBlasHandle and release cuBLAS resources
     */
    ~CudaBlasHandle() {
        if (owns_handle_ && handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }
    
    // Delete copy constructor and copy assignment (cuBLAS handles cannot be copied)
    CudaBlasHandle(const CudaBlasHandle&) = delete;
    CudaBlasHandle& operator=(const CudaBlasHandle&) = delete;
    
    /**
     * @brief Move constructor - transfers ownership
     */
    CudaBlasHandle(CudaBlasHandle&& other) noexcept
        : handle_(other.handle_), owns_handle_(other.owns_handle_) {
        other.owns_handle_ = false;
    }
    
    /**
     * @brief Move assignment - transfers ownership
     */
    CudaBlasHandle& operator=(CudaBlasHandle&& other) noexcept {
        if (this != &other) {
            // Destroy our current handle if we own it
            if (owns_handle_ && handle_ != nullptr) {
                cublasDestroy(handle_);
            }
            
            // Take ownership from other
            handle_ = other.handle_;
            owns_handle_ = other.owns_handle_;
            other.owns_handle_ = false;
        }
        return *this;
    }
    
    /**
     * @brief Get the raw cuBLAS handle
     * @return cublasHandle_t 
     */
    cublasHandle_t get() const { 
        return handle_; 
    }
    
    /**
     * @brief Implicit conversion to cublasHandle_t for use in cuBLAS functions
     */
    operator cublasHandle_t() const { 
        return handle_; 
    }
    
    /**
     * @brief Set the CUDA stream for cuBLAS operations
     * @param stream CUDA stream to associate with this handle
     * @throws std::runtime_error if cublasSetStream fails
     */
    void setStream(cudaStream_t stream) {
        cublasStatus_t status = cublasSetStream(handle_, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "cublasSetStream failed with status code: " << status;
            throw std::runtime_error(oss.str());
        }
    }
    
    /**
     * @brief Set the pointer mode for cuBLAS operations
     * @param mode Pointer mode (CUBLAS_POINTER_MODE_HOST or CUBLAS_POINTER_MODE_DEVICE)
     * @throws std::runtime_error if cublasSetPointerMode fails
     */
    void setPointerMode(cublasPointerMode_t mode) {
        cublasStatus_t status = cublasSetPointerMode(handle_, mode);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "cublasSetPointerMode failed with status code: " << status;
            throw std::runtime_error(oss.str());
        }
    }
    
    /**
     * @brief Get the current pointer mode
     * @return cublasPointerMode_t Current pointer mode
     * @throws std::runtime_error if cublasGetPointerMode fails
     */
    cublasPointerMode_t getPointerMode() const {
        cublasPointerMode_t mode;
        cublasStatus_t status = cublasGetPointerMode(handle_, &mode);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "cublasGetPointerMode failed with status code: " << status;
            throw std::runtime_error(oss.str());
        }
        return mode;
    }
    
    /**
     * @brief Set the math mode for cuBLAS operations (e.g., tensor core usage)
     * @param mode Math mode to set
     * @throws std::runtime_error if cublasSetMathMode fails
     */
    void setMathMode(cublasMath_t mode) {
        cublasStatus_t status = cublasSetMathMode(handle_, mode);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "cublasSetMathMode failed with status code: " << status;
            throw std::runtime_error(oss.str());
        }
    }
};

#endif // UTILITIES_CUDA_CUDA_BLAS_HANDLE_H_