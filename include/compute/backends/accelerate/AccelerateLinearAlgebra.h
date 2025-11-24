#ifndef ACCELERATE_LINEAR_ALGEBRA_H_
#define ACCELERATE_LINEAR_ALGEBRA_H_

#include "compute/linalg/ILinearAlgebra.h"

#ifdef __APPLE__

/**
 * @brief macOS implementation using Apple's Accelerate framework
 * 
 * Uses optimized vecLib (BLAS) from Accelerate framework.
 * Provides optimal performance on Apple Silicon and Intel Macs.
 */
class AccelerateLinearAlgebra : public ILinearAlgebra {
public:
    AccelerateLinearAlgebra() = default;
    ~AccelerateLinearAlgebra() override = default;
    
    void axpy(int n, const void* alpha, const void* x, void* y,
              PrecisionType precision, IStream* stream = nullptr) override;
    void dotc(int n, const void* x, const void* y, void* result,
              PrecisionType precision, IStream* stream = nullptr) override;
    void norm2(int n, const void* x, void* result,
               PrecisionType precision, IStream* stream = nullptr) override;
    void scal(int n, const void* alpha, void* x,
              PrecisionType precision, IStream* stream = nullptr) override;
    void gemv(Transpose trans, int m, int n, const void* alpha, const void* A,
              const void* x, const void* beta, void* y,
              PrecisionType precision, IStream* stream = nullptr) override;
    void gemm(Transpose transA, Transpose transB, int m, int n, int k,
              const void* alpha, const void* A, const void* B,
              const void* beta, void* C,
              PrecisionType precision, IStream* stream = nullptr) override;
    void geam(Transpose transA, Transpose transB, int m, int n,
              const void* alpha, const void* A, const void* beta, const void* B,
              void* C, PrecisionType precision, IStream* stream = nullptr) override;

private:
    static int toAccelerateOp(Transpose trans);
};

#endif // __APPLE__
#endif // ACCELERATE_LINEAR_ALGEBRA_H_
