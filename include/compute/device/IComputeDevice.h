#ifndef ICOMPUTEDEVICE_H_
#define ICOMPUTEDEVICE_H_

#include <memory>
#include <string>

#include "compute/core/ComputeTypes.h"

#include "compute/memory/IDeviceMemory.h"
#include "compute/memory/IScratchMemory.h"
#include "compute/stream/IComputeStream.h"
#include "compute/linalg/ILinearAlgebra.h"
#include "compute/linalg/ISolver.h"
#include "compute/linalg/ISVDSolver.h"
#include "compute/random/IRandomGenerator.h"

class IComputeDevice {
public:
    virtual ~IComputeDevice() = default;
    
    // Device identification
    virtual DeviceBackend getBackend() const = 0;
    virtual int getDeviceID() const = 0;
    virtual std::string getName() const = 0;
    
    // Memory management
    virtual std::unique_ptr<IDeviceMemory> allocate(size_t bytes) = 0;
    virtual void synchronize() = 0;
    
    // Stream management
    virtual std::unique_ptr<IStream> createStream() = 0;
    
    // Get specialized interfaces
    virtual ILinearAlgebra* getLinearAlgebra() = 0;
    virtual ISolver* getSolver() = 0;
    virtual IRandomGenerator* getRandomGenerator() = 0;
    
    /**
     * @brief Get device scratch memory pool
     * 
     * Returns the device's scratch memory pool for temporary workspace.
     * For CPU backends, this returns nullptr (no device memory).
     * For CUDA backends, returns a CudaScratchMemory instance.
     * 
     * The scratch pool is owned by the device and shared across operations.
     * 
     * @return Pointer to device scratch pool (nullptr for CPU)
     */
    virtual IScratchMemory* getDeviceScratch() = 0;
    
    /**
     * @brief Get host scratch memory pool
     * 
     * Returns the device's host scratch memory pool for temporary workspace.
     * All backends provide host scratch memory.
     * 
     * The scratch pool is owned by the device and shared across operations.
     * 
     * @return Pointer to host scratch pool (never nullptr)
     */
    virtual IScratchMemory* getHostScratch() = 0;
    
    /**
     * @brief Create a stateful SVD solver for fixed-size problems
     * 
     * Creates an SVD solver instance configured for a specific matrix size
     * and computation mode. The solver manages its own workspace memory and
     * can be reused for multiple SVD computations of the same dimensions.
     * 
     * Benefits of stateful solver:
     * - Workspace allocated once, reused across multiple compute() calls
     * - Automatic scratch memory management (grows as needed)
     * - Configuration introspection (workspace sizes, current spec)
     * - Better performance for repeated SVD of same size
     * 
     * @param m Number of rows in input matrix
     * @param n Number of columns in input matrix
     * @param spec SVD specification (vectors, algorithm, parameters)
     * @param precision FLOAT or DOUBLE
     * @return Configured SVD solver instance (unique ownership)
     * 
     * @throws std::runtime_error if solver creation fails
     * @throws std::invalid_argument if m <= 0 or n <= 0
     * 
     * Example:
     * @code
     * SVDSpec spec{SVDVectors::THIN, SVDAlgorithm::AUTO};
     * auto svd = device->createSVDSolver(100, 50, spec, PrecisionType::DOUBLE);
     * 
     * // Reuse for many computations
     * for (int i = 0; i < 1000; i++) {
     *     svd->compute(A, S, U, VT);
     * }
     * @endcode
     */
    virtual std::unique_ptr<ISVDSolver> createSVDSolver(
        int m, int n,
        const SVDSpec& spec,
        PrecisionType precision
    ) = 0;
    
    // Utility: data type conversion
    virtual void convertPrecision(
        const IDeviceMemory* src,
        IDeviceMemory* dst,
        PrecisionType src_type,
        PrecisionType dst_type,
        size_t count
    ) = 0;
};

#endif