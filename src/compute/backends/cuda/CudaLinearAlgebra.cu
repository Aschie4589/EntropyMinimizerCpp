#include "compute/backends/cuda/CudaLinearAlgebra.h"
#include "compute/backends/cuda/CudaStream.h"
#include "utilities/cuda/error_handling.h"
#include <cuComplex.h>
#include <stdexcept>

// ============================================================================
// DeviceGuard Implementation
// ============================================================================

CudaLinearAlgebra::DeviceGuard::DeviceGuard(int device_id, cublasHandle_t handle, IStream* stream)
    : handle_(handle), stream_was_set_(false) {
    // Set correct device
    cudaGetDevice(&previous_device_);
    if (previous_device_ != device_id) {
        cudaSetDevice(device_id);
    }
    
    // Set stream if provided
    if (stream) {
        auto* cudaStream = dynamic_cast<CudaStream*>(stream);
        if (cudaStream) {
            CUBLAS_CHECK(cublasSetStream(handle_, cudaStream->getCudaStream()));
            stream_was_set_ = true;
        }
    }
}

CudaLinearAlgebra::DeviceGuard::~DeviceGuard() {
    if (stream_was_set_) {
        cublasSetStream(handle_, nullptr);  // Don't check errors in destructor
    }
    cudaSetDevice(previous_device_);
}

// ============================================================================
// CudaLinearAlgebra Implementation
// ============================================================================

CudaLinearAlgebra::CudaLinearAlgebra(int device_id) : handle_(nullptr), device_id_(device_id) {
    // Ensure we create the handle on the correct device
    int previous_device;
    cudaGetDevice(&previous_device);
    if (previous_device != device_id_) {
        cudaSetDevice(device_id_);
    }
    
    // Create the cuBLAS handle. Note: handle is device-specific.
    CUBLAS_CHECK(cublasCreate(&handle_));
    
    // Restore previous device
    if (previous_device != device_id_) {
        cudaSetDevice(previous_device);
    }
}

CudaLinearAlgebra::~CudaLinearAlgebra() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

CudaLinearAlgebra::CudaLinearAlgebra(CudaLinearAlgebra&& other) noexcept 
    : handle_(other.handle_), device_id_(other.device_id_) {
    other.handle_ = nullptr;
}

CudaLinearAlgebra& CudaLinearAlgebra::operator=(CudaLinearAlgebra&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            cublasDestroy(handle_);
        }
        handle_ = other.handle_;
        device_id_ = other.device_id_;
        other.handle_ = nullptr;
    }
    return *this;
}

cublasOperation_t CudaLinearAlgebra::toCublasOp(Transpose trans) {
    switch (trans) {
        case Transpose::NO_TRANS:   return CUBLAS_OP_N;
        case Transpose::TRANS:      return CUBLAS_OP_T;
        case Transpose::CONJ_TRANS: return CUBLAS_OP_C;
        default: throw std::invalid_argument("Invalid transpose operation");
    }
}

// ============================================================================
// LEVEL 1 BLAS
// ============================================================================

void CudaLinearAlgebra::axpy(
    int n,
    const void* alpha,
    const void* x,
    void* y,
    PrecisionType precision,
    IStream* stream
) { 
    DeviceGuard guard(device_id_, handle_, stream);
    
    if (precision == PrecisionType::DOUBLE) {
        CUBLAS_CHECK(cublasZaxpy(
            handle_, n,
            static_cast<const cuDoubleComplex*>(alpha),
            static_cast<const cuDoubleComplex*>(x), 1, // incx and incy are stride parameters, 1 by convention
            static_cast<cuDoubleComplex*>(y), 1
        ));
    } else {
        CUBLAS_CHECK(cublasCaxpy(
            handle_, n,
            static_cast<const cuComplex*>(alpha),
            static_cast<const cuComplex*>(x), 1, // same here
            static_cast<cuComplex*>(y), 1
        ));
    }
}

void CudaLinearAlgebra::dotc(
    int n,
    const void* x,
    const void* y,
    void* result,
    PrecisionType precision,
    IStream* stream
) {
    DeviceGuard guard(device_id_, handle_, stream);
    
    if (precision == PrecisionType::DOUBLE) {
        CUBLAS_CHECK(cublasZdotc(
            handle_, n,
            static_cast<const cuDoubleComplex*>(x), 1,
            static_cast<const cuDoubleComplex*>(y), 1,
            static_cast<cuDoubleComplex*>(result)
        ));
    } else {
        CUBLAS_CHECK(cublasCdotc(
            handle_, n,
            static_cast<const cuComplex*>(x), 1,
            static_cast<const cuComplex*>(y), 1,
            static_cast<cuComplex*>(result)
        ));
    }
}

void CudaLinearAlgebra::norm2(
    int n,
    const void* x,
    void* result,
    PrecisionType precision,
    IStream* stream
) {
    DeviceGuard guard(device_id_, handle_, stream);
    
    if (precision == PrecisionType::DOUBLE) {
        double result_d;
        CUBLAS_CHECK(cublasDznrm2(
            handle_, n,
            static_cast<const cuDoubleComplex*>(x), 1,
            &result_d
        ));
        *static_cast<double*>(result) = result_d;
    } else {
        float result_f;
        CUBLAS_CHECK(cublasScnrm2(
            handle_, n,
            static_cast<const cuComplex*>(x), 1,
            &result_f
        ));
        *static_cast<float*>(result) = result_f;
    }
}

void CudaLinearAlgebra::scal(
    int n,
    const void* alpha, // host or device
    void* x, // device
    PrecisionType precision,
    IStream* stream
) {
    DeviceGuard guard(device_id_, handle_, stream);
    
    if (precision == PrecisionType::DOUBLE) {
        CUBLAS_CHECK(cublasZscal(
            handle_, n,
            static_cast<const cuDoubleComplex*>(alpha),
            static_cast<cuDoubleComplex*>(x), 1
        ));
    } else {
        CUBLAS_CHECK(cublasCscal(
            handle_, n,
            static_cast<const cuComplex*>(alpha),
            static_cast<cuComplex*>(x), 1
        ));
    }
}

// ============================================================================
// LEVEL 2 BLAS
// ============================================================================

void CudaLinearAlgebra::gemv(
    Transpose trans,
    int m, int n,
    const void* alpha,
    const void* A,
    const void* x,
    const void* beta,
    void* y,
    PrecisionType precision,
    IStream* stream
) {
    DeviceGuard guard(device_id_, handle_, stream);
    
    cublasOperation_t op = toCublasOp(trans);
    
    // m and n are STORAGE dimensions (same as cuBLAS convention)
    // lda = m (leading dimension = number of rows in storage)
    int lda = m;
    
    if (precision == PrecisionType::DOUBLE) {
        CUBLAS_CHECK(cublasZgemv(
            handle_, op, // op: CUBLAS operation (no transpose, transpose, conjugate transpose)
            m, n, // m: rows of A, n: columns of A
            static_cast<const cuDoubleComplex*>(alpha), // alpha: scalar multiplier (host or device)
            static_cast<const cuDoubleComplex*>(A), lda, // A: matrix pointer (device), lda: leading dimension
            static_cast<const cuDoubleComplex*>(x), 1, // x: vector pointer (device), incx: stride
            static_cast<const cuDoubleComplex*>(beta), // beta: scalar multiplier (host or device)
            static_cast<cuDoubleComplex*>(y), 1 // y: vector pointer (device), incy: stride
        ));
    } else {
        CUBLAS_CHECK(cublasCgemv(
            handle_, op, // op: CUBLAS operation (no transpose, transpose, conjugate transpose)
            m, n, // m: rows of A, n: columns of A
            static_cast<const cuComplex*>(alpha), // alpha: scalar multiplier (host or device)
            static_cast<const cuComplex*>(A), lda, // A: matrix pointer (device), lda: leading dimension
            static_cast<const cuComplex*>(x), 1, // x: vector pointer (device), incx: stride
            static_cast<const cuComplex*>(beta), // beta: scalar multiplier (host or device)
            static_cast<cuComplex*>(y), 1 // y: vector pointer (device), incy: stride
        ));
    }
}

// ============================================================================
// LEVEL 3 BLAS
// ============================================================================

/**
 * @brief CUDA matrix-matrix multiply: C := alpha*op(A)*op(B) + beta*C
 * 
 * Performs: C := alpha * op(A) * op(B) + beta * C
 * 
 * [mxn] = scal * [mxk] * [kxn] + beta * [mxn]
 * 
 * Storage layouts deduced automatically:
 * - If transA = NO_TRANS:         A stored as m×k (lda = m)
 * - If transA = TRANS/CONJ_TRANS: A stored as k×m (lda = k)
 * - If transB = NO_TRANS:         B stored as k×n (ldb = k)
 * - If transB = TRANS/CONJ_TRANS: B stored as n×k (ldb = n)
 * - C always stored as m×n (ldc = m)
 * 
 * @param transA Transpose operation on A
 * @param transB Transpose operation on B
 * @param m Number of rows in result C and op(A)
 * @param n Number of columns in result C and op(B)
 * @param k Contraction dimension (columns of op(A), rows of op(B))
 * @param alpha Scalar multiplier (complex<T>* or T*)
 * @param A Input matrix (column major)
 * @param B Input matrix (column major)
 * @param beta Scalar multiplier for C (complex<T>* or T*)
 * @param C Input/output matrix m×n (stored as m×n, column major)
 * @param precision FLOAT or DOUBLE
 * @param stream Optional stream for async execution
 */
void CudaLinearAlgebra::gemm(
    Transpose transA, Transpose transB,
    int m, int n, int k,
    const void* alpha,
    const void* A,
    const void* B,
    const void* beta,
    void* C,
    PrecisionType precision,
    IStream* stream
) {
    DeviceGuard guard(device_id_, handle_, stream);
    
    cublasOperation_t opA = toCublasOp(transA);
    cublasOperation_t opB = toCublasOp(transB);
    
    // Calculate leading dimensions based on storage
    // If NO_TRANS: A stored as m×k, lda = m
    // If TRANS/CONJ_TRANS: A stored as k×m, lda = k
    int lda = (transA == Transpose::NO_TRANS) ? m : k;
    int ldb = (transB == Transpose::NO_TRANS) ? k : n;
    int ldc = m;  // C is always m×n
    
    if (precision == PrecisionType::DOUBLE) {
        CUBLAS_CHECK(cublasZgemm(
            handle_, opA, opB,
            m, n, k,
            static_cast<const cuDoubleComplex*>(alpha),
            static_cast<const cuDoubleComplex*>(A), lda,
            static_cast<const cuDoubleComplex*>(B), ldb,
            static_cast<const cuDoubleComplex*>(beta),
            static_cast<cuDoubleComplex*>(C), ldc
        ));
    } else {
        CUBLAS_CHECK(cublasCgemm(
            handle_, opA, opB,
            m, n, k,
            static_cast<const cuComplex*>(alpha),
            static_cast<const cuComplex*>(A), lda,
            static_cast<const cuComplex*>(B), ldb,
            static_cast<const cuComplex*>(beta),
            static_cast<cuComplex*>(C), ldc
        ));
    }
}

void CudaLinearAlgebra::geam(
    Transpose transA, Transpose transB,
    int m, int n,
    const void* alpha,
    const void* A,
    const void* beta,
    const void* B,
    void* C,
    PrecisionType precision,
    IStream* stream
) {
    DeviceGuard guard(device_id_, handle_, stream);
    
    cublasOperation_t opA = toCublasOp(transA);
    cublasOperation_t opB = toCublasOp(transB);
    
    // Leading dimensions based on storage
    // If NO_TRANS: A stored as m×n, lda = m
    // If TRANS/CONJ_TRANS: A stored as n×m, lda = n
    int lda = (transA == Transpose::NO_TRANS) ? m : n;
    int ldb = (transB == Transpose::NO_TRANS) ? m : n;
    int ldc = m;  // C is always m×n
    
    if (precision == PrecisionType::DOUBLE) {
        CUBLAS_CHECK(cublasZgeam(
            handle_, opA, opB,
            m, n,
            static_cast<const cuDoubleComplex*>(alpha),
            static_cast<const cuDoubleComplex*>(A), lda,
            static_cast<const cuDoubleComplex*>(beta),
            static_cast<const cuDoubleComplex*>(B), ldb,
            static_cast<cuDoubleComplex*>(C), ldc
        ));
    } else {
        CUBLAS_CHECK(cublasCgeam(
            handle_, opA, opB,
            m, n,
            static_cast<const cuComplex*>(alpha),
            static_cast<const cuComplex*>(A), lda,
            static_cast<const cuComplex*>(beta),
            static_cast<const cuComplex*>(B), ldb,
            static_cast<cuComplex*>(C), ldc
        ));
    }
}
