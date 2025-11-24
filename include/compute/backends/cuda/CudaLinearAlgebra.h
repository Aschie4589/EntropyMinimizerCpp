#ifndef CUDA_LINEAR_ALGEBRA_H_
#define CUDA_LINEAR_ALGEBRA_H_

#include "compute/linalg/ILinearAlgebra.h"
#include <cublas_v2.h>
#include <memory>

/**
 * @brief CUDA implementation of linear algebra operations using cuBLAS
 * 
 * Wraps cuBLAS library with RAII handle management.
 * Automatically calculates leading dimensions based on transpose operations.
 */
class CudaLinearAlgebra : public ILinearAlgebra {
public:
    CudaLinearAlgebra();
    ~CudaLinearAlgebra() override;
    
    // Disable copy, enable move
    CudaLinearAlgebra(const CudaLinearAlgebra&) = delete;
    CudaLinearAlgebra& operator=(const CudaLinearAlgebra&) = delete;
    CudaLinearAlgebra(CudaLinearAlgebra&&) noexcept;
    CudaLinearAlgebra& operator=(CudaLinearAlgebra&&) noexcept;
    
    // Level 1 BLAS
    void axpy(
        int n,
        const void* alpha,
        const void* x,
        void* y,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    void dotc(
        int n,
        const void* x,
        const void* y,
        void* result,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    void norm2(
        int n,
        const void* x,
        void* result,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    void scal(
        int n,
        const void* alpha,
        void* x,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    // Level 2 BLAS
    void gemv(
        Transpose trans,
        int m, int n,
        const void* alpha,
        const void* A,
        const void* x,
        const void* beta,
        void* y,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    // Level 3 BLAS
    void gemm(
        Transpose transA, Transpose transB,
        int m, int n, int k,
        const void* alpha,
        const void* A,
        const void* B,
        const void* beta,
        void* C,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    void geam(
        Transpose transA, Transpose transB,
        int m, int n,
        const void* alpha,
        const void* A,
        const void* beta,
        const void* B,
        void* C,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    // Access to handle (for integration with existing code)
    cublasHandle_t handle() const { return handle_; }

private:
    cublasHandle_t handle_;
    
    // Helper to convert Transpose enum to cuBLAS operation
    static cublasOperation_t toCublasOp(Transpose trans);
};

#endif // CUDA_LINEAR_ALGEBRA_H_
