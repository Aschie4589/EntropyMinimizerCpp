#ifndef ISVD_SOLVER_H_
#define ISVD_SOLVER_H_

#include "compute/core/ComputeTypes.h"
#include "compute/linalg/SVDTypes.h"
#include "compute/stream/IComputeStream.h"
#include <cstddef>

/**
 * @brief Stateful SVD solver for fixed-size problems
 * 
 * This interface represents a configured SVD solver for a specific matrix size
 * and computation mode. Unlike the stateless ISolver::gesvd() methods, ISVDSolver:
 * 
 * - Owns its workspace memory (device + host)
 * - Can be reused for multiple SVD computations of the same size
 * - Automatically manages scratch memory allocation
 * - Queries workspace requirements only once (at construction or spec update)
 * - Provides introspection of workspace sizes and configuration
 * 
 * Design rationale:
 * This follows established HPC patterns (MAGMA workspace, Eigen temporary pool,
 * cuBLAS workspace) where repeated allocation/deallocation is expensive.
 * For workflows that compute many SVDs of the same size, this provides
 * significant performance benefits.
 * 
 * Thread safety:
 * NOT thread-safe. Create separate instances for concurrent operations.
 * 
 * Lifecycle:
 * 1. Create via IComputeDevice::createSVDSolver() (queries workspace, allocates)
 * 2. Call compute() repeatedly (reuses workspace)
 * 3. Optionally call setSpec() to change vectors/algorithm (re-queries workspace)
 * 4. Destroy (automatic cleanup)
 * 
 * Example usage:
 * @code
 * auto device = getDevice();
 * SVDSpec spec{SVDVectors::THIN, SVDAlgorithm::RANDOMIZED};
 * auto svd_solver = device->createSVDSolver(100, 50, spec, PrecisionType::DOUBLE);
 * 
 * for (int i = 0; i < 1000; i++) {
 *     // Fill matrix A...
 *     svd_solver->compute(A, S, U, VT);
 *     // Use results...
 * }
 * @endcode
 */
class ISVDSolver {
public:
    virtual ~ISVDSolver() = default;
    
    /**
     * @brief Compute SVD: A = U * Σ * V^H
     * 
     * Matrix dimensions and computation mode are fixed at construction.
     * All matrices must be pre-allocated by caller with correct sizes.
     * 
     * Matrix A is modified in-place (may be overwritten depending on algorithm).
     * 
     * Expected sizes based on spec.vectors:
     * - SVDVectors::ALL:  U is m×m, VT is n×n
     * - SVDVectors::THIN: U is m×min(m,n), VT is min(m,n)×n
     * - SVDVectors::NONE: U and VT are nullptr
     * - S is always min(m,n)
     * 
     * All matrices stored in column-major order.
     * 
     * @param A Input matrix m×n, may be overwritten (column-major)
     * @param S Output singular values, size min(m,n), descending order
     * @param U Output left singular vectors (or nullptr if vectors=NONE)
     * @param VT Output right singular vectors^H (or nullptr if vectors=NONE)
     * @param stream Optional stream for async execution (nullptr = default stream)
     * 
     * @throws std::runtime_error if computation fails
     * @throws std::invalid_argument if matrix sizes don't match specification
     */
    virtual void compute(
        void* A,
        void* S,
        void* U,
        void* VT,
        IStream* stream = nullptr
    ) = 0;
    
    /**
     * @brief Update SVD specification (vectors, algorithm, etc.)
     * 
     * Changes the computation mode and/or algorithm preference.
     * This triggers workspace re-query if needed.
     * 
     * Matrix dimensions CANNOT be changed (fixed at construction).
     * 
     * Use this to switch between computing vectors vs singular values only,
     * or to try different algorithms.
     * 
     * @param spec New SVD specification
     * @throws std::invalid_argument if spec is invalid for this problem size
     */
    virtual void setSpec(const SVDSpec& spec) = 0;
    
    /**
     * @brief Get current specification
     * 
     * @return Current SVD spec (vectors, algorithm, parameters)
     */
    virtual const SVDSpec& getSpec() const = 0;
    
    /**
     * @brief Get problem dimensions
     * 
     * @param m Output: number of rows
     * @param n Output: number of columns
     */
    virtual void getDimensions(int& m, int& n) const = 0;
    
    /**
     * @brief Get current workspace sizes in bytes
     * 
     * Returns the workspace memory requirements for current configuration.
     * Useful for debugging, memory monitoring, and optimization.
     * 
     * Note: Actual memory usage may be higher due to:
     * - Scratch memory growth strategy (over-allocation for future requests)
     * - Internal buffers (devInfo, params, etc.)
     * 
     * @param device_bytes Output: device memory workspace size
     * @param host_bytes Output: host memory workspace size
     */
    virtual void getWorkspaceSizes(size_t& device_bytes, size_t& host_bytes) const = 0;
    
    /**
     * @brief Get precision type
     */
    virtual PrecisionType getPrecision() const = 0;
    
    /**
     * @brief Get backend type
     */
    virtual DeviceBackend getBackend() const = 0;
    
    // Non-copyable (SVD solver has unique workspace state)
    ISVDSolver(const ISVDSolver&) = delete;
    ISVDSolver& operator=(const ISVDSolver&) = delete;
    
protected:
    ISVDSolver() = default;
};

#endif // ISVD_SOLVER_H_
