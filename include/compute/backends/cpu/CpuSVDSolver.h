#ifndef CPU_SVD_SOLVER_H_
#define CPU_SVD_SOLVER_H_

#include "compute/linalg/ISVDSolver.h"
#include "compute/memory/IScratchMemory.h"
#include "compute/memory/IDeviceMemory.h"
#include "compute/device/IComputeDevice.h"
#include <memory>

/**
 * @brief CPU implementation of stateful SVD solver using LAPACK
 * 
 * Manages SVD computations for fixed-size problems using LAPACK zgesvd.
 * Automatically manages workspace allocation via scratch pools.
 * 
 * Algorithm selection:
 * - AUTO, QR, POLAR: Uses LAPACK zgesvd (all map to same implementation)
 * - RANDOMIZED: Currently falls back to full SVD (TODO: implement randomized)
 * 
 * Thread safety: NOT thread-safe. Create separate instances for concurrent use.
 */
class CpuSVDSolver : public ISVDSolver {
public:
    /**
     * @brief Construct CPU SVD solver for fixed-size problem
     * 
     * @param device Back-reference to parent device (for scratch pool access)
     * @param m Number of rows
     * @param n Number of columns
     * @param spec SVD specification (vectors, algorithm, parameters)
     * @param precision FLOAT or DOUBLE
     * 
     * @throws std::invalid_argument if m <= 0 or n <= 0
     * @throws std::runtime_error if workspace query fails
     */
    CpuSVDSolver(
        IComputeDevice* device,
        int m, int n,
        const SVDSpec& spec,
        PrecisionType precision
    );
    
    ~CpuSVDSolver() override = default;
    
    // ISVDSolver interface implementation
    void compute(void* A, void* S, void* U, void* VT, IStream* stream = nullptr) override;
    void setSpec(const SVDSpec& spec) override;
    const SVDSpec& getSpec() const override { return spec_; }
    void getDimensions(int& m, int& n) const override { m = m_; n = n_; }
    void getWorkspaceSizes(size_t& device_bytes, size_t& host_bytes) const override;
    PrecisionType getPrecision() const override { return precision_; }
    DeviceBackend getBackend() const override { return DeviceBackend::CPU; }
    
private:
    /**
     * @brief Query workspace requirements from LAPACK
     * 
     * Called during construction and when spec changes.
     * Updates workspace_bytes_.
     */
    void queryWorkspace();
    
    /**
     * @brief Compute SVD using LAPACK zgesvd
     */
    void computeLAPACK(void* A, void* S, void* U, void* VT);
    
    // Problem specification (immutable dimensions, mutable spec)
    int m_;
    int n_;
    SVDSpec spec_;
    PrecisionType precision_;
    
    // Workspace requirements (updated by queryWorkspace)
    size_t workspace_bytes_;
    
    IComputeDevice* device_;
};

#endif // CPU_SVD_SOLVER_H_
