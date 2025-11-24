#ifndef CUDA_SVD_SOLVER_H_
#define CUDA_SVD_SOLVER_H_

#include "compute/linalg/ISVDSolver.h"
#include "compute/memory/IScratchMemory.h"
#include "compute/memory/IDeviceMemory.h"
#include "compute/device/IComputeDevice.h"
#include <cusolverDn.h>
#include <memory>

/**
 * @brief CUDA implementation of stateful SVD solver
 * 
 * Manages SVD computations for fixed-size problems using cuSOLVER library.
 * Automatically manages workspace allocation via device scratch pools.
 * 
 * Algorithm selection:
 * - AUTO: cusolverDnXgesvd (QR-based, handles tall/wide matrices)
 * - QR: cusolverDnXgesvd (QR-based, works for m>=n or n>=m, falls back for wide matrices)
 * - POLAR: cusolverDnXgesvdp (polar decomposition, more accurate)
 * - RANDOMIZED: cusolverDnXgesvdr (randomized, fast for low-rank)
 * 
 * Thread safety: NOT thread-safe. Create separate instances for concurrent use.
 */
class CudaSVDSolver : public ISVDSolver {
public:
    /**
     * @brief Construct CUDA SVD solver for fixed-size problem
     * 
     * Creates its own cuSOLVER handle and params. Borrows scratch memory from device.
     * 
     * @param device Back-reference to parent device (for scratch pool access)
     * @param m Number of rows
     * @param n Number of columns
     * @param spec SVD specification (vectors, algorithm, parameters)
     * @param precision FLOAT or DOUBLE
     * 
     * @throws std::invalid_argument if m <= 0 or n <= 0
     * @throws std::runtime_error if handle/params creation or workspace query fails
     */
    CudaSVDSolver(
        IComputeDevice* device,
        int m, int n,
        const SVDSpec& spec,
        PrecisionType precision
    );
    
    ~CudaSVDSolver() override;
    
    // ISVDSolver interface implementation
    void compute(void* A, void* S, void* U, void* VT, IStream* stream = nullptr) override;
    void setSpec(const SVDSpec& spec) override;
    const SVDSpec& getSpec() const override { return spec_; }
    void getDimensions(int& m, int& n) const override { m = m_; n = n_; }
    void getWorkspaceSizes(size_t& device_bytes, size_t& host_bytes) const override;
    PrecisionType getPrecision() const override { return precision_; }
    DeviceBackend getBackend() const override { return DeviceBackend::CUDA; }
    
private:
    /**
     * @brief Query workspace requirements from cuSOLVER
     * 
     * Called during construction and when spec changes.
     * Updates workspace_device_bytes_ and workspace_host_bytes_.
     */
    void queryWorkspace();
    
    /**
     * @brief Compute SVD using cusolverDnXgesvd (QR-based)
     * 
     * Standard QR-based SVD. Works for tall (m>=n) or wide (n>=m) matrices.
     */
    void computeQR(void* A, void* S, void* U, void* VT);
    
    /**
     * @brief Compute SVD using cusolverDnXgesvdp (polar decomposition)
     * 
     * Polar decomposition based SVD. Generally more accurate than QR.
     * Note: Returns V, not V^H (unlike QR which returns V^H).
     */
    void computePolar(void* A, void* S, void* U, void* VT);
    
    /**
     * @brief Compute SVD using cusolverDnXgesvdr (randomized)
     * 
     * Randomized SVD for low-rank approximation.
     * Uses spec_.rank, spec_.oversampling, spec_.power_iterations.
     */
    void computeRandomized(void* A, void* S, void* U, void* VT);
    
    // Problem specification (immutable dimensions, mutable spec)
    int m_;
    int n_;
    SVDSpec spec_;
    PrecisionType precision_;
    
    // Workspace requirements (updated by queryWorkspace)
    size_t workspace_device_bytes_;
    size_t workspace_host_bytes_;
    
    IComputeDevice* device_;
    cusolverDnHandle_t cusolver_handle_;  // Owned by this instance
    cusolverDnParams_t cusolver_params_;   // Owned by this instance
    
    // Internal resources (owned)
    std::unique_ptr<IDeviceMemory> devinfo_;  // Device int for error status
    double h_err_sigma_;                       // Host variable for gesvdp
    
    // Stream management
    cudaStream_t original_stream_;  // For stream restoration
};

#endif // CUDA_SVD_SOLVER_H_
