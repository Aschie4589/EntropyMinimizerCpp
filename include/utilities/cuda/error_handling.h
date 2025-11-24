#ifndef UTILITIES_CUDA_ERROR_HANDLING_H_
#define UTILITIES_CUDA_ERROR_HANDLING_H_

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <sstream>

// Macro for CUDA runtime errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::ostringstream oss; \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(err); \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

// Macro for cuSOLVER errors
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::ostringstream oss; \
            oss << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ \
                << " - status code: " << status; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

// Macro for cuBLAS errors
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::ostringstream oss; \
            oss << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                << " - status code: " << status; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#endif // UTILITIES_CUDA_ERROR_HANDLING_H_