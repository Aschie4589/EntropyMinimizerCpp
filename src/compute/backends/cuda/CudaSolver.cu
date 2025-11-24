#include "compute/backends/cuda/CudaSolver.h"
#include "compute/backends/cuda/CudaStream.h"
#include "compute/backends/cuda/CudaMemory.h"
#include "utilities/cuda/error_handling.h"
#include <cuComplex.h>
#include <stdexcept>
#include <algorithm>

CudaSolver::CudaSolver() : handle_(nullptr), params_(nullptr) {
    CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params_));
}

CudaSolver::~CudaSolver() {
    if (params_) {
        cusolverDnDestroyParams(params_);
    }
    if (handle_) {
        cusolverDnDestroy(handle_);
    }
}

CudaSolver::CudaSolver(CudaSolver&& other) noexcept 
    : handle_(other.handle_), params_(other.params_) {
    other.handle_ = nullptr;
    other.params_ = nullptr;
}

CudaSolver& CudaSolver::operator=(CudaSolver&& other) noexcept {
    if (this != &other) {
        if (params_) {
            cusolverDnDestroyParams(params_);
        }
        if (handle_) {
            cusolverDnDestroy(handle_);
        }
        handle_ = other.handle_;
        params_ = other.params_;
        other.handle_ = nullptr;
        other.params_ = nullptr;
    }
    return *this;
}

// ============================================================================
// QR FACTORIZATION
// ============================================================================

void CudaSolver::geqrf(
    int m, int n,
    void* A,
    void* tau,
    void* workspace,
    int workspace_size,
    PrecisionType precision,
    IStream* stream
) {
    if (stream) {
        auto* cudaStream = dynamic_cast<CudaStream*>(stream);
        if (cudaStream) {
            CUSOLVER_CHECK(cusolverDnSetStream(handle_, cudaStream->getCudaStream()));
        }
    }
    
    int lda = m;  // Always m for column-major m×n matrix
    
    // Device info for error checking
    auto devInfo = std::make_unique<CudaMemory>(sizeof(int));
    
    if (precision == PrecisionType::DOUBLE) {
        CUSOLVER_CHECK(cusolverDnZgeqrf( // This is the legacy cusolver api (2.4.2.8)
            handle_, m, n,
            static_cast<cuDoubleComplex*>(A), lda,
            static_cast<cuDoubleComplex*>(tau),
            static_cast<cuDoubleComplex*>(workspace), workspace_size,
            static_cast<int*>(devInfo->data())
        ));
    } else {
        CUSOLVER_CHECK(cusolverDnCgeqrf(
            handle_, m, n, // handle (host), m (host), n (host)
            static_cast<cuComplex*>(A), lda, // A (device), lda (host)
            static_cast<cuComplex*>(tau), // tau (device)
            static_cast<cuComplex*>(workspace), workspace_size, // workspace (device), workspace_size (host)
            static_cast<int*>(devInfo->data()) // devInfo (device)
        ));
    }
    
    // Check devInfo
    int info_host;
    devInfo->copyToHost(&info_host, sizeof(int));
    if (info_host != 0) {
        throw std::runtime_error("cuSOLVER geqrf failed with info = " + std::to_string(info_host));
    }
    
    if (stream) {
        CUSOLVER_CHECK(cusolverDnSetStream(handle_, nullptr));
    }
}

void CudaSolver::orgqr(
    int m, int n, int k,
    void* A,
    const void* tau,
    void* workspace,
    int workspace_size,
    PrecisionType precision,
    IStream* stream
) {
    if (stream) {
        auto* cudaStream = dynamic_cast<CudaStream*>(stream);
        if (cudaStream) {
            CUSOLVER_CHECK(cusolverDnSetStream(handle_, cudaStream->getCudaStream()));
        }
    }
    
    int lda = m;
    
    auto devInfo = std::make_unique<CudaMemory>(sizeof(int));
    
    if (precision == PrecisionType::DOUBLE) {
        CUSOLVER_CHECK(cusolverDnZungqr(
            handle_, m, n, k,
            static_cast<cuDoubleComplex*>(A), lda,
            static_cast<const cuDoubleComplex*>(tau),
            static_cast<cuDoubleComplex*>(workspace), workspace_size,
            static_cast<int*>(devInfo->data())
        ));
    } else {
        CUSOLVER_CHECK(cusolverDnCungqr(
            handle_, m, n, k,
            static_cast<cuComplex*>(A), lda,
            static_cast<const cuComplex*>(tau),
            static_cast<cuComplex*>(workspace), workspace_size,
            static_cast<int*>(devInfo->data())
        ));
    }
    
    int info_host;
    devInfo->copyToHost(&info_host, sizeof(int));
    if (info_host != 0) {
        throw std::runtime_error("cuSOLVER orgqr failed with info = " + std::to_string(info_host));
    }
    
    if (stream) {
        CUSOLVER_CHECK(cusolverDnSetStream(handle_, nullptr));
    }
}

// ============================================================================
// SVD - Standard (uses new 64-bit generic API)
// ============================================================================

void CudaSolver::gesvd(
    char jobu, char jobvt,
    int m, int n,
    void* A,
    void* S,
    void* U,
    void* VT,
    void* workspace,
    int workspace_size,
    PrecisionType precision,
    IStream* stream
) {
    if (stream) {
        auto* cudaStream = dynamic_cast<CudaStream*>(stream);
        if (cudaStream) {
            CUSOLVER_CHECK(cusolverDnSetStream(handle_, cudaStream->getCudaStream()));
        }
    }
    
    int64_t lda = m;
    int64_t ldu = (jobu == 'A' || jobu == 'S') ? m : 1;
    int64_t ldvt = 0;
    if (jobvt == 'A') ldvt = n;
    else if (jobvt == 'S') ldvt = std::min(m, n);
    else ldvt = 1;
    
    auto devInfo = std::make_unique<CudaMemory>(sizeof(int));
    
    signed char jobz_u = jobu;
    signed char jobz_vt = jobvt;
    
    if (precision == PrecisionType::DOUBLE) {
        size_t workspace_bytes = static_cast<size_t>(workspace_size) * sizeof(cuDoubleComplex);
        CUSOLVER_CHECK(cusolverDnXgesvd(
            handle_, params_,
            jobz_u, jobz_vt,
            m, n,
            CUDA_C_64F, A, lda,
            CUDA_R_64F, S,
            CUDA_C_64F, U, ldu,
            CUDA_C_64F, VT, ldvt,
            CUDA_C_64F,
            workspace, workspace_bytes,
            nullptr, 0,  // No host workspace
            static_cast<int*>(devInfo->data())
        ));
    } else {
        size_t workspace_bytes = static_cast<size_t>(workspace_size) * sizeof(cuComplex);
        CUSOLVER_CHECK(cusolverDnXgesvd(
            handle_, params_,
            jobz_u, jobz_vt,
            m, n,
            CUDA_C_32F, A, lda,
            CUDA_R_32F, S,
            CUDA_C_32F, U, ldu,
            CUDA_C_32F, VT, ldvt,
            CUDA_C_32F,
            workspace, workspace_bytes,
            nullptr, 0,  // No host workspace
            static_cast<int*>(devInfo->data())
        ));
    }
    
    int info_host;
    devInfo->copyToHost(&info_host, sizeof(int));
    if (info_host != 0) {
        throw std::runtime_error("cuSOLVER gesvd failed with info = " + std::to_string(info_host));
    }
    
    if (stream) {
        CUSOLVER_CHECK(cusolverDnSetStream(handle_, nullptr));
    }
}

// ============================================================================
// SVD - Polar decomposition based (more accurate)
// ============================================================================

void CudaSolver::gesvdp(
    int jobz,
    int econ,
    int m, int n,
    void* A,
    void* S,
    void* U,
    void* V,
    void* workspace,
    int workspace_size,
    PrecisionType precision,
    IStream* stream
) {
    if (stream) {
        auto* cudaStream = dynamic_cast<CudaStream*>(stream);
        if (cudaStream) {
            CUSOLVER_CHECK(cusolverDnSetStream(handle_, cudaStream->getCudaStream()));
        }
    }
    
    int64_t lda = m;
    // Leading dimensions must always be valid even if vectors not computed
    int64_t ldu = m;
    int64_t ldv = n;
    
    // Allocate devInfo on device and initialize
    int* devInfo_ptr;
    CUDA_CHECK(cudaMalloc(&devInfo_ptr, sizeof(int)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(devInfo_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    cusolverEigMode_t eigmode = (jobz == 1) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    
    // Allocate host error sigma (required even if not used)
    double h_err_sigma = 0.0;
    
    if (precision == PrecisionType::DOUBLE) {
        // Convert workspace_size from element count to bytes
        size_t workspace_bytes = static_cast<size_t>(workspace_size) * sizeof(cuDoubleComplex);
        
        cusolverStatus_t status = cusolverDnXgesvdp(
            handle_, params_,
            eigmode, econ,
            m, n,
            CUDA_C_64F, A, lda,
            CUDA_R_64F, S,
            CUDA_C_64F, U, ldu,
            CUDA_C_64F, V, ldv,
            CUDA_C_64F,
            workspace,
            workspace_bytes,
            nullptr,  // host workspace
            0,        // host workspace size
            devInfo_ptr,
            &h_err_sigma
        );
        
        if (status != CUSOLVER_STATUS_SUCCESS) {
            cudaFree(devInfo_ptr);
            throw std::runtime_error("cusolverDnXgesvdp returned status = " + std::to_string(status));
        }
    } else {
        // Convert workspace_size from element count to bytes
        size_t workspace_bytes = static_cast<size_t>(workspace_size) * sizeof(cuComplex);
        
        cusolverStatus_t status = cusolverDnXgesvdp(
            handle_, params_,
            eigmode, econ,
            m, n,
            CUDA_C_32F, A, lda,
            CUDA_R_32F, S,
            CUDA_C_32F, U, ldu,
            CUDA_C_32F, V, ldv,
            CUDA_C_32F,
            workspace,
            workspace_bytes,
            nullptr,  // host workspace
            0,        // host workspace size
            devInfo_ptr,
            &h_err_sigma
        );
        
        if (status != CUSOLVER_STATUS_SUCCESS) {
            cudaFree(devInfo_ptr);
            throw std::runtime_error("cusolverDnXgesvdp returned status = " + std::to_string(status));
        }
    }
    
    // Synchronize to ensure the operation is complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int info_host;
    CUDA_CHECK(cudaMemcpy(&info_host, devInfo_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(devInfo_ptr);
    
    if (info_host != 0) {
        throw std::runtime_error("cuSOLVER gesvdp failed with info = " + std::to_string(info_host));
    }
    
    if (stream) {
        CUSOLVER_CHECK(cusolverDnSetStream(handle_, nullptr));
    }
}

// ============================================================================
// SVD - Randomized for low-rank approximation
// ============================================================================

void CudaSolver::gesvdr(
    char jobu, char jobv,
    int m, int n, int k,
    void* A,
    void* S,
    void* U,
    void* V,
    void* workspace,
    int workspace_size,
    PrecisionType precision,
    int oversampling,
    int niters,
    IStream* stream
) {
    if (stream) {
        auto* cudaStream = dynamic_cast<CudaStream*>(stream);
        if (cudaStream) {
            CUSOLVER_CHECK(cusolverDnSetStream(handle_, cudaStream->getCudaStream()));
        }
    }
    
    // Default oversampling if not specified
    if (oversampling < 0) {
        oversampling = std::min(2 * k, std::min(m, n) - k);
    }
    
    int64_t lda = m;
    int64_t ldu = (jobu == 'S') ? m : 1;
    int64_t ldv = (jobv == 'S') ? n : 1;
    
    auto devInfo = std::make_unique<CudaMemory>(sizeof(int));
    
    signed char jobz_u = jobu;
    signed char jobz_v = jobv;
    
    if (precision == PrecisionType::DOUBLE) {
        CUSOLVER_CHECK(cusolverDnXgesvdr(
            handle_, params_,
            jobz_u, jobz_v,
            m, n,
            k,                // rank
            oversampling,     // oversampling parameter p
            niters,           // number of iterations
            CUDA_C_64F, A, lda,
            CUDA_R_64F, S,
            CUDA_C_64F, U, ldu,
            CUDA_C_64F, V, ldv,
            CUDA_C_64F,
            workspace,
            workspace_size,
            nullptr,  // host workspace
            0,        // host workspace size
            static_cast<int*>(devInfo->data())
        ));
    } else {
        CUSOLVER_CHECK(cusolverDnXgesvdr(
            handle_, params_,
            jobz_u, jobz_v,
            m, n,
            k,                // rank
            oversampling,     // oversampling parameter p
            niters,           // number of iterations
            CUDA_C_32F, A, lda,
            CUDA_R_32F, S,
            CUDA_C_32F, U, ldu,
            CUDA_C_32F, V, ldv,
            CUDA_C_32F,
            workspace,
            workspace_size,
            nullptr,  // host workspace
            0,        // host workspace size
            static_cast<int*>(devInfo->data())
        ));
    }
    
    int info_host;
    devInfo->copyToHost(&info_host, sizeof(int));
    if (info_host != 0) {
        throw std::runtime_error("cuSOLVER gesvdr failed with info = " + std::to_string(info_host));
    }
    
    if (stream) {
        CUSOLVER_CHECK(cusolverDnSetStream(handle_, nullptr));
    }
}

// ============================================================================
// WORKSPACE QUERIES
// ============================================================================

int CudaSolver::geqrf_workspace_size(
    int m, int n,
    PrecisionType precision
) {
    int lwork = 0;
    int lda = m;
    
    if (precision == PrecisionType::DOUBLE) {
        // Create dummy array for query
        cuDoubleComplex dummy;
        CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(
            handle_, m, n,
            &dummy, lda,
            &lwork
        ));
    } else {
        cuComplex dummy;
        CUSOLVER_CHECK(cusolverDnCgeqrf_bufferSize(
            handle_, m, n,
            &dummy, lda,
            &lwork
        ));
    }
    
    return lwork;
}

int CudaSolver::orgqr_workspace_size(
    int m, int n, int k,
    PrecisionType precision
) {
    int lwork = 0;
    int lda = m;
    
    if (precision == PrecisionType::DOUBLE) {
        cuDoubleComplex dummy_A;
        cuDoubleComplex dummy_tau;
        CUSOLVER_CHECK(cusolverDnZungqr_bufferSize(
            handle_, m, n, k,
            &dummy_A, lda,
            &dummy_tau,
            &lwork
        ));
    } else {
        cuComplex dummy_A;
        cuComplex dummy_tau;
        CUSOLVER_CHECK(cusolverDnCungqr_bufferSize(
            handle_, m, n, k,
            &dummy_A, lda,
            &dummy_tau,
            &lwork
        ));
    }
    
    return lwork;
}

int CudaSolver::gesvd_workspace_size(
    int m, int n,
    PrecisionType precision
) {
    size_t workspace_bytes_device = 0;
    size_t workspace_bytes_host = 0;
    int64_t lda = m;
    int64_t ldu = m;  // Assume 'A' mode for workspace query
    int64_t ldvt = n;
    
    if (precision == PrecisionType::DOUBLE) {
        CUSOLVER_CHECK(cusolverDnXgesvd_bufferSize(
            handle_, params_,
            'A', 'A',  // worst case
            m, n,
            CUDA_C_64F, nullptr, lda,
            CUDA_R_64F, nullptr,
            CUDA_C_64F, nullptr, ldu,
            CUDA_C_64F, nullptr, ldvt,
            CUDA_C_64F,
            &workspace_bytes_device,
            &workspace_bytes_host
        ));
        // Only return device workspace size (host workspace not used)
        return static_cast<int>(workspace_bytes_device / sizeof(cuDoubleComplex));
    } else {
        CUSOLVER_CHECK(cusolverDnXgesvd_bufferSize(
            handle_, params_,
            'A', 'A',  // worst case
            m, n,
            CUDA_C_32F, nullptr, lda,
            CUDA_R_32F, nullptr,
            CUDA_C_32F, nullptr, ldu,
            CUDA_C_32F, nullptr, ldvt,
            CUDA_C_32F,
            &workspace_bytes_device,
            &workspace_bytes_host
        ));
        // Only return device workspace size (host workspace not used)
        return static_cast<int>(workspace_bytes_device / sizeof(cuComplex));
    }
}

int CudaSolver::gesvdp_workspace_size(
    int m, int n,
    int econ,
    PrecisionType precision
) {
    size_t workspace_bytes = 0;
    size_t host_workspace_bytes = 0;  // Not used but required by API
    int64_t lda = m;
    int64_t ldu = m;
    int64_t ldv = n;
    
    if (precision == PrecisionType::DOUBLE) {
        CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
            handle_, params_,
            CUSOLVER_EIG_MODE_VECTOR,
            econ,
            m, n,
            CUDA_C_64F, nullptr, lda,
            CUDA_R_64F, nullptr,
            CUDA_C_64F, nullptr, ldu,
            CUDA_C_64F, nullptr, ldv,
            CUDA_C_64F,
            &workspace_bytes,
            &host_workspace_bytes
        ));
        return static_cast<int>(workspace_bytes / sizeof(cuDoubleComplex));
    } else {
        CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
            handle_, params_,
            CUSOLVER_EIG_MODE_VECTOR,
            econ,
            m, n,
            CUDA_C_32F, nullptr, lda,
            CUDA_R_32F, nullptr,
            CUDA_C_32F, nullptr, ldu,
            CUDA_C_32F, nullptr, ldv,
            CUDA_C_32F,
            &workspace_bytes,
            &host_workspace_bytes
        ));
        return static_cast<int>(workspace_bytes / sizeof(cuComplex));
    }
}

int CudaSolver::gesvdr_workspace_size(
    int m, int n, int k,
    int oversampling,
    int niters,
    PrecisionType precision
) {
    // Default oversampling if not specified
    if (oversampling < 0) {
        oversampling = std::min(2 * k, std::min(m, n) - k);
    }
    
    size_t workspace_bytes = 0;
    size_t host_workspace_bytes = 0;  // Not used but required by API
    int64_t lda = m;
    int64_t ldu = m;
    int64_t ldv = n;
    
    if (precision == PrecisionType::DOUBLE) {
        CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
            handle_, params_,
            'S', 'S',  // Both U and V
            m, n,
            k, oversampling, niters,
            CUDA_C_64F, nullptr, lda,
            CUDA_R_64F, nullptr,
            CUDA_C_64F, nullptr, ldu,
            CUDA_C_64F, nullptr, ldv,
            CUDA_C_64F,
            &workspace_bytes,
            &host_workspace_bytes
        ));
        return static_cast<int>(workspace_bytes / sizeof(cuDoubleComplex));
    } else {
        CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
            handle_, params_,
            'S', 'S',  // Both U and V
            m, n,
            k, oversampling, niters,
            CUDA_C_32F, nullptr, lda,
            CUDA_R_32F, nullptr,
            CUDA_C_32F, nullptr, ldu,
            CUDA_C_32F, nullptr, ldv,
            CUDA_C_32F,
            &workspace_bytes,
            &host_workspace_bytes
        ));
        return static_cast<int>(workspace_bytes / sizeof(cuComplex));
    }
}
