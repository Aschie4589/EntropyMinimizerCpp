#include "compute/backends/accelerate/AccelerateSolver.h"

#ifdef __APPLE__

#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include <algorithm>
#include <vector>

// Note: Accelerate framework provides the same LAPACK interface as standard LAPACKE

void AccelerateSolver::geqrf(int m, int n, void* A, void* tau, void* workspace,
                             int workspace_size, PrecisionType precision, IStream* stream) {
    int lda = m;
    __CLPK_integer info;
    __CLPK_integer m_int = m;
    __CLPK_integer n_int = n;
    __CLPK_integer lda_int = lda;
    __CLPK_integer lwork = workspace_size;
    
    if (precision == PrecisionType::DOUBLE) {
        zgeqrf_(&m_int, &n_int,
                static_cast<__CLPK_doublecomplex*>(A), &lda_int,
                static_cast<__CLPK_doublecomplex*>(tau),
                static_cast<__CLPK_doublecomplex*>(workspace), &lwork,
                &info);
    } else {
        cgeqrf_(&m_int, &n_int,
                static_cast<__CLPK_complex*>(A), &lda_int,
                static_cast<__CLPK_complex*>(tau),
                static_cast<__CLPK_complex*>(workspace), &lwork,
                &info);
    }
    
    if (info != 0) {
        throw std::runtime_error("Accelerate geqrf failed with info = " + std::to_string(info));
    }
}

void AccelerateSolver::orgqr(int m, int n, int k, void* A, const void* tau,
                             void* workspace, int workspace_size,
                             PrecisionType precision, IStream* stream) {
    int lda = m;
    __CLPK_integer info;
    __CLPK_integer m_int = m;
    __CLPK_integer n_int = n;
    __CLPK_integer k_int = k;
    __CLPK_integer lda_int = lda;
    __CLPK_integer lwork = workspace_size;
    
    if (precision == PrecisionType::DOUBLE) {
        zungqr_(&m_int, &n_int, &k_int,
                static_cast<__CLPK_doublecomplex*>(A), &lda_int,
                static_cast<const __CLPK_doublecomplex*>(tau),
                static_cast<__CLPK_doublecomplex*>(workspace), &lwork,
                &info);
    } else {
        cungqr_(&m_int, &n_int, &k_int,
                static_cast<__CLPK_complex*>(A), &lda_int,
                static_cast<const __CLPK_complex*>(tau),
                static_cast<__CLPK_complex*>(workspace), &lwork,
                &info);
    }
    
    if (info != 0) {
        throw std::runtime_error("Accelerate orgqr failed with info = " + std::to_string(info));
    }
}

void AccelerateSolver::gesvd(char jobu, char jobvt, int m, int n, void* A, void* S,
                             void* U, void* VT, void* workspace, int workspace_size,
                             PrecisionType precision, IStream* stream) {
    int lda = m;
    int ldu = (jobu == 'A' || jobu == 'S') ? m : 1;
    int ldvt = 0;
    if (jobvt == 'A') ldvt = n;
    else if (jobvt == 'S') ldvt = std::min(m, n);
    else ldvt = 1;
    
    __CLPK_integer info;
    __CLPK_integer m_int = m;
    __CLPK_integer n_int = n;
    __CLPK_integer lda_int = lda;
    __CLPK_integer ldu_int = ldu;
    __CLPK_integer ldvt_int = ldvt;
    __CLPK_integer lwork = workspace_size;
    
    if (precision == PrecisionType::DOUBLE) {
        std::vector<__CLPK_doublereal> rwork(5 * std::min(m, n));
        zgesvd_(&jobu, &jobvt,
                &m_int, &n_int,
                static_cast<__CLPK_doublecomplex*>(A), &lda_int,
                static_cast<__CLPK_doublereal*>(S),
                static_cast<__CLPK_doublecomplex*>(U), &ldu_int,
                static_cast<__CLPK_doublecomplex*>(VT), &ldvt_int,
                static_cast<__CLPK_doublecomplex*>(workspace), &lwork,
                rwork.data(),
                &info);
    } else {
        std::vector<__CLPK_real> rwork(5 * std::min(m, n));
        cgesvd_(&jobu, &jobvt,
                &m_int, &n_int,
                static_cast<__CLPK_complex*>(A), &lda_int,
                static_cast<__CLPK_real*>(S),
                static_cast<__CLPK_complex*>(U), &ldu_int,
                static_cast<__CLPK_complex*>(VT), &ldvt_int,
                static_cast<__CLPK_complex*>(workspace), &lwork,
                rwork.data(),
                &info);
    }
    
    if (info != 0) {
        throw std::runtime_error("Accelerate gesvd failed with info = " + std::to_string(info));
    }
}

int AccelerateSolver::geqrf_workspace_size(int m, int n, PrecisionType precision) {
    __CLPK_integer m_int = m;
    __CLPK_integer n_int = n;
    __CLPK_integer lda_int = m;
    __CLPK_integer lwork = -1;
    __CLPK_integer info;
    
    if (precision == PrecisionType::DOUBLE) {
        __CLPK_doublecomplex work_query;
        std::vector<__CLPK_doublecomplex> A_dummy(m * n);
        std::vector<__CLPK_doublecomplex> tau(std::min(m, n));
        
        zgeqrf_(&m_int, &n_int,
                A_dummy.data(), &lda_int,
                tau.data(),
                &work_query, &lwork,
                &info);
        
        if (info == 0) {
            return static_cast<int>(work_query.r);
        }
    } else {
        __CLPK_complex work_query;
        std::vector<__CLPK_complex> A_dummy(m * n);
        std::vector<__CLPK_complex> tau(std::min(m, n));
        
        cgeqrf_(&m_int, &n_int,
                A_dummy.data(), &lda_int,
                tau.data(),
                &work_query, &lwork,
                &info);
        
        if (info == 0) {
            return static_cast<int>(work_query.r);
        }
    }
    
    return std::max(m, n);
}

int AccelerateSolver::orgqr_workspace_size(int m, int n, int k, PrecisionType precision) {
    __CLPK_integer m_int = m;
    __CLPK_integer n_int = n;
    __CLPK_integer k_int = k;
    __CLPK_integer lda_int = m;
    __CLPK_integer lwork = -1;
    __CLPK_integer info;
    
    if (precision == PrecisionType::DOUBLE) {
        __CLPK_doublecomplex work_query;
        std::vector<__CLPK_doublecomplex> A_dummy(m * n);
        std::vector<__CLPK_doublecomplex> tau(k);
        
        zungqr_(&m_int, &n_int, &k_int,
                A_dummy.data(), &lda_int,
                tau.data(),
                &work_query, &lwork,
                &info);
        
        if (info == 0) {
            return static_cast<int>(work_query.r);
        }
    } else {
        __CLPK_complex work_query;
        std::vector<__CLPK_complex> A_dummy(m * n);
        std::vector<__CLPK_complex> tau(k);
        
        cungqr_(&m_int, &n_int, &k_int,
                A_dummy.data(), &lda_int,
                tau.data(),
                &work_query, &lwork,
                &info);
        
        if (info == 0) {
            return static_cast<int>(work_query.r);
        }
    }
    
    return n;
}

int AccelerateSolver::gesvd_workspace_size(int m, int n, PrecisionType precision) {
    __CLPK_integer m_int = m;
    __CLPK_integer n_int = n;
    __CLPK_integer lda_int = m;
    __CLPK_integer ldu_int = m;
    __CLPK_integer ldvt_int = n;
    __CLPK_integer lwork = -1;
    __CLPK_integer info;
    char jobu = 'A';
    char jobvt = 'A';
    
    if (precision == PrecisionType::DOUBLE) {
        __CLPK_doublecomplex work_query;
        std::vector<__CLPK_doublecomplex> A_dummy(m * n);
        std::vector<__CLPK_doublereal> S(std::min(m, n));
        std::vector<__CLPK_doublecomplex> U(m * m);
        std::vector<__CLPK_doublecomplex> VT(n * n);
        std::vector<__CLPK_doublereal> rwork(5 * std::min(m, n));
        
        zgesvd_(&jobu, &jobvt,
                &m_int, &n_int,
                A_dummy.data(), &lda_int,
                S.data(),
                U.data(), &ldu_int,
                VT.data(), &ldvt_int,
                &work_query, &lwork,
                rwork.data(),
                &info);
        
        if (info == 0) {
            return static_cast<int>(work_query.r);
        }
    } else {
        __CLPK_complex work_query;
        std::vector<__CLPK_complex> A_dummy(m * n);
        std::vector<__CLPK_real> S(std::min(m, n));
        std::vector<__CLPK_complex> U(m * m);
        std::vector<__CLPK_complex> VT(n * n);
        std::vector<__CLPK_real> rwork(5 * std::min(m, n));
        
        cgesvd_(&jobu, &jobvt,
                &m_int, &n_int,
                A_dummy.data(), &lda_int,
                S.data(),
                U.data(), &ldu_int,
                VT.data(), &ldvt_int,
                &work_query, &lwork,
                rwork.data(),
                &info);
        
        if (info == 0) {
            return static_cast<int>(work_query.r);
        }
    }
    
    return 2 * std::min(m, n) + std::max(m, n);
}

// ============================================================================
// Polar SVD (falls back to standard gesvd on macOS)
// ============================================================================

void AccelerateSolver::gesvdp(
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
    // Accelerate doesn't have polar SVD, fall back to standard gesvd
    char jobu = (jobz == 1) ? (econ ? 'S' : 'A') : 'N';
    char jobvt = (jobz == 1) ? (econ ? 'S' : 'A') : 'N';
    
    gesvd(jobu, jobvt, m, n, A, S, U, V, workspace, workspace_size, precision, stream);
}

int AccelerateSolver::gesvdp_workspace_size(int m, int n, int econ, int jobz, PrecisionType precision) {
    // Fall back to standard gesvd workspace size
    // jobz parameter ignored since we fall back to gesvd
    return gesvd_workspace_size(m, n, precision);
}

// ============================================================================
// Randomized SVD (falls back to standard gesvd on macOS)
// ============================================================================

void AccelerateSolver::gesvdr(
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
    // Accelerate doesn't have randomized SVD, fall back to standard gesvd
    char jobu_full = (jobu == 'S') ? 'S' : 'N';
    char jobvt_full = (jobv == 'S') ? 'S' : 'N';
    
    gesvd(jobu_full, jobvt_full, m, n, A, S, U, V, workspace, workspace_size, precision, stream);
}

int AccelerateSolver::gesvdr_workspace_size(int m, int n, int k, int oversampling, int niters, PrecisionType precision) {
    return gesvd_workspace_size(m, n, precision);
}

#endif // __APPLE__
