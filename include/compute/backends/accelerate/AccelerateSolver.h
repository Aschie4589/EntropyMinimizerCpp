#ifndef ACCELERATE_SOLVER_H_
#define ACCELERATE_SOLVER_H_

#include "compute/linalg/ISolver.h"

#ifdef __APPLE__

/**
 * @brief macOS implementation using Apple's Accelerate framework (LAPACK)
 */
class AccelerateSolver : public ISolver {
public:
    AccelerateSolver() = default;
    ~AccelerateSolver() override = default;
    
    void geqrf(int m, int n, void* A, void* tau, void* workspace, int workspace_size,
               PrecisionType precision, IStream* stream = nullptr) override;
    void orgqr(int m, int n, int k, void* A, const void* tau, void* workspace,
               int workspace_size, PrecisionType precision, IStream* stream = nullptr) override;
    void gesvd(char jobu, char jobvt, int m, int n, void* A, void* S, void* U, void* VT,
               void* workspace, int workspace_size, PrecisionType precision,
               IStream* stream = nullptr) override;
    void gesvdp(int jobz, int econ, int m, int n, void* A, void* S, void* U, void* V,
                void* workspace, int workspace_size, PrecisionType precision,
                IStream* stream = nullptr) override;
    void gesvdr(char jobu, char jobv, int m, int n, int k, void* A, void* S, void* U, void* V,
                void* workspace, int workspace_size, PrecisionType precision,
                int oversampling = -1, int niters = 2, IStream* stream = nullptr) override;
    int geqrf_workspace_size(int m, int n, PrecisionType precision) override;
    int orgqr_workspace_size(int m, int n, int k, PrecisionType precision) override;
    int gesvd_workspace_size(int m, int n, PrecisionType precision) override;
    int gesvdp_workspace_size(int m, int n, int econ, PrecisionType precision) override;
    int gesvdr_workspace_size(int m, int n, int k, int oversampling, int niters, PrecisionType precision) override;
};

#endif // __APPLE__
#endif // ACCELERATE_SOLVER_H_
