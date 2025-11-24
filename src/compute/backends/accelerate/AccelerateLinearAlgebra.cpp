#include "compute/backends/accelerate/AccelerateLinearAlgebra.h"

#ifdef __APPLE__

#include <Accelerate/Accelerate.h>
#include <complex>
#include <stdexcept>

int AccelerateLinearAlgebra::toAccelerateOp(Transpose trans) {
    switch (trans) {
        case Transpose::NO_TRANS:   return CblasNoTrans;
        case Transpose::TRANS:      return CblasTrans;
        case Transpose::CONJ_TRANS: return CblasConjTrans;
        default: throw std::invalid_argument("Invalid transpose operation");
    }
}

void AccelerateLinearAlgebra::axpy(int n, const void* alpha, const void* x, void* y,
                                   PrecisionType precision, IStream* stream) {
    if (precision == PrecisionType::DOUBLE) {
        cblas_zaxpy(n, static_cast<const double*>(alpha),
                    static_cast<const double*>(x), 1,
                    static_cast<double*>(y), 1);
    } else {
        cblas_caxpy(n, static_cast<const float*>(alpha),
                    static_cast<const float*>(x), 1,
                    static_cast<float*>(y), 1);
    }
}

void AccelerateLinearAlgebra::dotc(int n, const void* x, const void* y, void* result,
                                   PrecisionType precision, IStream* stream) {
    if (precision == PrecisionType::DOUBLE) {
        cblas_zdotc_sub(n, static_cast<const double*>(x), 1,
                        static_cast<const double*>(y), 1,
                        static_cast<double*>(result));
    } else {
        cblas_cdotc_sub(n, static_cast<const float*>(x), 1,
                        static_cast<const float*>(y), 1,
                        static_cast<float*>(result));
    }
}

void AccelerateLinearAlgebra::norm2(int n, const void* x, void* result,
                                    PrecisionType precision, IStream* stream) {
    if (precision == PrecisionType::DOUBLE) {
        *static_cast<double*>(result) = cblas_dznrm2(n, static_cast<const double*>(x), 1);
    } else {
        *static_cast<float*>(result) = cblas_scnrm2(n, static_cast<const float*>(x), 1);
    }
}

void AccelerateLinearAlgebra::scal(int n, const void* alpha, void* x,
                                   PrecisionType precision, IStream* stream) {
    if (precision == PrecisionType::DOUBLE) {
        cblas_zscal(n, static_cast<const double*>(alpha),
                    static_cast<double*>(x), 1);
    } else {
        cblas_cscal(n, static_cast<const float*>(alpha),
                    static_cast<float*>(x), 1);
    }
}

void AccelerateLinearAlgebra::gemv(Transpose trans, int m, int n, const void* alpha,
                                   const void* A, const void* x, const void* beta,
                                   void* y, PrecisionType precision, IStream* stream) {
    CBLAS_TRANSPOSE op = static_cast<CBLAS_TRANSPOSE>(toAccelerateOp(trans));
    int lda = (trans == Transpose::NO_TRANS) ? m : n;
    
    if (precision == PrecisionType::DOUBLE) {
        cblas_zgemv(CblasColMajor, op, m, n,
                    static_cast<const double*>(alpha),
                    static_cast<const double*>(A), lda,
                    static_cast<const double*>(x), 1,
                    static_cast<const double*>(beta),
                    static_cast<double*>(y), 1);
    } else {
        cblas_cgemv(CblasColMajor, op, m, n,
                    static_cast<const float*>(alpha),
                    static_cast<const float*>(A), lda,
                    static_cast<const float*>(x), 1,
                    static_cast<const float*>(beta),
                    static_cast<float*>(y), 1);
    }
}

void AccelerateLinearAlgebra::gemm(Transpose transA, Transpose transB, int m, int n, int k,
                                   const void* alpha, const void* A, const void* B,
                                   const void* beta, void* C,
                                   PrecisionType precision, IStream* stream) {
    CBLAS_TRANSPOSE opA = static_cast<CBLAS_TRANSPOSE>(toAccelerateOp(transA));
    CBLAS_TRANSPOSE opB = static_cast<CBLAS_TRANSPOSE>(toAccelerateOp(transB));
    int lda = (transA == Transpose::NO_TRANS) ? m : k;
    int ldb = (transB == Transpose::NO_TRANS) ? k : n;
    int ldc = m;
    
    if (precision == PrecisionType::DOUBLE) {
        cblas_zgemm(CblasColMajor, opA, opB, m, n, k,
                    static_cast<const double*>(alpha),
                    static_cast<const double*>(A), lda,
                    static_cast<const double*>(B), ldb,
                    static_cast<const double*>(beta),
                    static_cast<double*>(C), ldc);
    } else {
        cblas_cgemm(CblasColMajor, opA, opB, m, n, k,
                    static_cast<const float*>(alpha),
                    static_cast<const float*>(A), lda,
                    static_cast<const float*>(B), ldb,
                    static_cast<const float*>(beta),
                    static_cast<float*>(C), ldc);
    }
}

void AccelerateLinearAlgebra::geam(Transpose transA, Transpose transB, int m, int n,
                                   const void* alpha, const void* A,
                                   const void* beta, const void* B, void* C,
                                   PrecisionType precision, IStream* stream) {
    // Same implementation as CpuLinearAlgebra - CBLAS doesn't have geam
    int lda = (transA == Transpose::NO_TRANS) ? m : n;
    int ldb = (transB == Transpose::NO_TRANS) ? m : n;
    
    if (precision == PrecisionType::DOUBLE) {
        using Complex = std::complex<double>;
        auto* C_ptr = static_cast<Complex*>(C);
        auto* A_ptr = static_cast<const Complex*>(A);
        auto* B_ptr = static_cast<const Complex*>(B);
        auto alpha_val = *static_cast<const Complex*>(alpha);
        auto beta_val = *static_cast<const Complex*>(beta);
        
        if (transA == Transpose::NO_TRANS) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    C_ptr[j*m + i] = alpha_val * A_ptr[j*lda + i];
        } else if (transA == Transpose::TRANS) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    C_ptr[j*m + i] = alpha_val * A_ptr[i*lda + j];
        } else {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    C_ptr[j*m + i] = alpha_val * std::conj(A_ptr[i*lda + j]);
        }
        
        if (B_ptr) {
            if (transB == Transpose::NO_TRANS) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        C_ptr[j*m + i] += beta_val * B_ptr[j*ldb + i];
            } else if (transB == Transpose::TRANS) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        C_ptr[j*m + i] += beta_val * B_ptr[i*ldb + j];
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        C_ptr[j*m + i] += beta_val * std::conj(B_ptr[i*ldb + j]);
            }
        }
    } else {
        using Complex = std::complex<float>;
        auto* C_ptr = static_cast<Complex*>(C);
        auto* A_ptr = static_cast<const Complex*>(A);
        auto* B_ptr = static_cast<const Complex*>(B);
        auto alpha_val = *static_cast<const Complex*>(alpha);
        auto beta_val = *static_cast<const Complex*>(beta);
        
        if (transA == Transpose::NO_TRANS) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    C_ptr[j*m + i] = alpha_val * A_ptr[j*lda + i];
        } else if (transA == Transpose::TRANS) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    C_ptr[j*m + i] = alpha_val * A_ptr[i*lda + j];
        } else {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    C_ptr[j*m + i] = alpha_val * std::conj(A_ptr[i*lda + j]);
        }
        
        if (B_ptr) {
            if (transB == Transpose::NO_TRANS) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        C_ptr[j*m + i] += beta_val * B_ptr[j*ldb + i];
            } else if (transB == Transpose::TRANS) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        C_ptr[j*m + i] += beta_val * B_ptr[i*ldb + j];
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        C_ptr[j*m + i] += beta_val * std::conj(B_ptr[i*ldb + j]);
            }
        }
    }
}

#endif // __APPLE__
