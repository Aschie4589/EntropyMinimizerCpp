#ifndef CPU_SOLVER_H_
#define CPU_SOLVER_H_

#include "compute/linalg/ISolver.h"

/**
 * @brief CPU implementation of solver operations using LAPACKE
 * 
 * Uses LAPACKE (OpenBLAS, MKL, or system LAPACK) for QR and SVD.
 */
class CpuSolver : public ISolver {
public:
    CpuSolver() = default;
    ~CpuSolver() override = default;
    
    void geqrf(
        int m, int n,
        void* A,
        void* tau,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    void orgqr(
        int m, int n, int k,
        void* A,
        const void* tau,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    void gesvd(
        char jobu, char jobvt,
        int m, int n,
        void* A,
        void* S,
        void* U,
        void* VT,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    // Polar SVD (falls back to standard gesvd on CPU)
    void gesvdp(
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
        IStream* stream = nullptr
    ) override;
    
    // Randomized SVD (falls back to standard gesvd on CPU)
    void gesvdr(
        char jobu, char jobv,
        int m, int n, int k,
        void* A,
        void* S,
        void* U,
        void* V,
        void* workspace,
        int workspace_size,
        PrecisionType precision,
        int oversampling = -1,
        int niters = 2,
        IStream* stream = nullptr
    ) override;
    
    int geqrf_workspace_size(int m, int n, PrecisionType precision) override;
    int orgqr_workspace_size(int m, int n, int k, PrecisionType precision) override;
    int gesvd_workspace_size(int m, int n, PrecisionType precision) override;
    int gesvdp_workspace_size(int m, int n, int econ, PrecisionType precision) override;
    int gesvdr_workspace_size(int m, int n, int k, int oversampling, int niters, PrecisionType precision) override;
};

#endif // CPU_SOLVER_H_
