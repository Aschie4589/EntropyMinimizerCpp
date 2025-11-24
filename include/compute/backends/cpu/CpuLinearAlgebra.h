#ifndef CPU_LINEAR_ALGEBRA_H_
#define CPU_LINEAR_ALGEBRA_H_

#include "compute/linalg/ILinearAlgebra.h"
#include <complex>

/**
 * @brief CPU implementation of linear algebra operations using CBLAS
 * 
 * Uses CBLAS (OpenBLAS, MKL, or system BLAS) for all operations.
 * Automatically calculates leading dimensions based on transpose operations.
 */
class CpuLinearAlgebra : public ILinearAlgebra {
public:
    CpuLinearAlgebra() = default;
    ~CpuLinearAlgebra() override = default;
    
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

private:
    // Helper to convert Transpose enum to CBLAS operation
    static int toCblasOp(Transpose trans);
};

#endif // CPU_LINEAR_ALGEBRA_H_
