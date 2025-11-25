#ifndef CUDA_SOLVER_H_
#define CUDA_SOLVER_H_

#include "compute/linalg/ISolver.h"
#include <cusolverDn.h>
#include <memory>

/**
 * @brief CUDA implementation of solver operations using cuSOLVER
 * 
 * Wraps cuSOLVER library with RAII handle management.
 * Provides QR factorization and SVD operations.
 */
class CudaSolver : public ISolver {
public:
    explicit CudaSolver(int device_id);
    ~CudaSolver() override;
    
    // Disable copy, enable move
    CudaSolver(const CudaSolver&) = delete;
    CudaSolver& operator=(const CudaSolver&) = delete;
    CudaSolver(CudaSolver&&) noexcept;
    CudaSolver& operator=(CudaSolver&&) noexcept;
    
    // QR Factorization
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
    
    // SVD - Standard QR-based (uses new 64-bit API)
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
    
    // SVD - Polar decomposition based (more accurate)
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
    
    // SVD - Randomized for low-rank approximation (faster)
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
    
    // Workspace queries
    int geqrf_workspace_size(
        int m, int n,
        PrecisionType precision
    ) override;
    
    int orgqr_workspace_size(
        int m, int n, int k,
        PrecisionType precision
    ) override;
    
    int gesvd_workspace_size(
        int m, int n,
        PrecisionType precision
    ) override;
    
    int gesvdp_workspace_size(
        int m, int n,
        int econ,
        PrecisionType precision
    ) override;
    
    int gesvdr_workspace_size(
        int m, int n, int k,
        int oversampling,
        int niters,
        PrecisionType precision
    ) override;
    
    // Access to handle (for integration with existing code)
    cusolverDnHandle_t handle() const { return handle_; }

private:
    cusolverDnHandle_t handle_;
    cusolverDnParams_t params_;
    int device_id_;
    
    // RAII helper for device and stream management
    class DeviceGuard {
    public:
        DeviceGuard(int device_id, cusolverDnHandle_t handle, IStream* stream);
        ~DeviceGuard();
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;
    private:
        int previous_device_;
        cusolverDnHandle_t handle_;
        bool stream_was_set_;
    };
};

#endif // CUDA_SOLVER_H_
