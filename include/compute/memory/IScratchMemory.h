#ifndef ISCRATCH_MEMORY_H_
#define ISCRATCH_MEMORY_H_

#include <cstddef>
#include "compute/core/ComputeTypes.h"

/**
 * @brief Interface for reusable scratch memory
 * 
 * Scratch memory is workspace that:
 * - Persists across multiple operations (avoiding repeated alloc/free)
 * - Automatically expands when more space is needed
 * - Is NOT thread-safe (each thread should have its own instance)
 * - Does NOT guarantee zeroed memory
 * 
 * Usage pattern:
 * 1. Create once (typically owned by IComputeDevice)
 * 2. Request memory as needed (auto-expands)
 * 3. Reuse across multiple operations
 * 4. Optionally release to free memory back to system
 * 
 * This is inspired by established HPC patterns (MAGMA workspace, Eigen 
 * temporary pool, cuBLAS workspace) for high-performance computing where
 * repeated allocation/deallocation is expensive.
 */
class IScratchMemory {
public:
    virtual ~IScratchMemory() = default;
    
    /**
     * @brief Request scratch memory of at least `bytes` size
     * 
     * Returns pointer to scratch memory. If current allocation is insufficient,
     * reallocates to requested size (with growth strategy).
     * 
     * Memory is NOT zeroed. Contents are undefined.
     * Memory lifetime is managed by this object - do not free the pointer.
     * 
     * Thread safety: NOT thread-safe. Create separate instances for concurrent use.
     * 
     * @param bytes Minimum size in bytes (must be > 0)
     * @return void* Pointer to scratch memory (lifetime managed by this object)
     * @throws std::runtime_error if allocation fails
     * @throws std::invalid_argument if bytes == 0
     */
    virtual void* request(size_t bytes) = 0;
    
    /**
     * @brief Get current allocated capacity in bytes
     * 
     * @return Current allocation size. May be larger than last request() due to growth strategy.
     */
    virtual size_t capacity() const = 0;
    
    /**
     * @brief Explicitly free scratch memory
     * 
     * Releases memory back to the system. After calling release(), capacity() returns 0.
     * Memory will be reallocated on next request().
     * 
     * This is optional - destructor will automatically release.
     * Use this to explicitly control memory lifetime (e.g., free after peak usage).
     */
    virtual void release() = 0;
    
    /**
     * @brief Get backend type (CUDA, CPU, etc.)
     */
    virtual DeviceBackend getBackend() const = 0;
    
    // Non-copyable (scratch memory should have unique ownership)
    IScratchMemory(const IScratchMemory&) = delete;
    IScratchMemory& operator=(const IScratchMemory&) = delete;
    
    // Movable (allow transfer of ownership)
    IScratchMemory(IScratchMemory&&) = default;
    IScratchMemory& operator=(IScratchMemory&&) = default;
    
protected:
    IScratchMemory() = default;
};

#endif // ISCRATCH_MEMORY_H_
