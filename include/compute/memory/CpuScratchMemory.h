#ifndef CPU_SCRATCH_MEMORY_H_
#define CPU_SCRATCH_MEMORY_H_

#include "compute/memory/IScratchMemory.h"
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <new>

/**
 * @brief CPU host scratch memory implementation
 * 
 * Manages reusable workspace on CPU host with automatic growth.
 * Uses standard C++ allocation (aligned allocation for SIMD compatibility).
 * 
 * Memory allocation strategy:
 * - First request: Allocate exact size
 * - Growth: new_capacity = max(requested, current + current/2)
 * - Uses 64-byte alignment for cache line / SIMD optimization
 * 
 * Thread safety: NOT thread-safe. Each thread should have separate instance.
 */
class CpuScratchMemory : public IScratchMemory {
public:
    /**
     * @brief Construct empty scratch memory (no allocation)
     */
    CpuScratchMemory() : data_(nullptr), capacity_(0) {}
    
    /**
     * @brief Construct with pre-allocated capacity
     * 
     * @param initial_capacity Initial allocation size in bytes
     * @throws std::runtime_error if allocation fails
     */
    explicit CpuScratchMemory(size_t initial_capacity) 
        : data_(nullptr), capacity_(0) 
    {
        if (initial_capacity > 0) {
            request(initial_capacity);
        }
    }
    
    /**
     * @brief Destructor - automatically frees memory
     */
    ~CpuScratchMemory() override {
        release();
    }
    
    /**
     * @brief Request host scratch memory
     * 
     * If requested size exceeds current capacity, reallocates with growth strategy.
     * Memory is 64-byte aligned for cache/SIMD optimization.
     * 
     * @param bytes Requested size in bytes (must be > 0)
     * @return Pointer to host memory (64-byte aligned)
     * @throws std::invalid_argument if bytes == 0
     * @throws std::runtime_error if allocation fails
     */
    void* request(size_t bytes) override {
        if (bytes == 0) {
            throw std::invalid_argument("CpuScratchMemory::request: bytes must be > 0");
        }
        
        if (bytes > capacity_) {
            // Growth strategy: max(requested, 1.5x current)
            size_t new_capacity = std::max(bytes, capacity_ + capacity_ / 2);
            
            // Allocate aligned memory (64-byte alignment for cache lines / AVX-512)
            constexpr size_t alignment = 64;
            
            void* new_data = nullptr;
            
#if defined(_WIN32) || defined(_WIN64)
            // Windows: use _aligned_malloc
            new_data = _aligned_malloc(new_capacity, alignment);
            if (!new_data) {
                throw std::runtime_error(
                    std::string("CpuScratchMemory: _aligned_malloc failed for ") +
                    std::to_string(new_capacity) + " bytes"
                );
            }
#else
            // POSIX: use posix_memalign
            int result = posix_memalign(&new_data, alignment, new_capacity);
            if (result != 0) {
                throw std::runtime_error(
                    std::string("CpuScratchMemory: posix_memalign failed for ") +
                    std::to_string(new_capacity) + " bytes (error " +
                    std::to_string(result) + ")"
                );
            }
#endif
            
            // Free old allocation (if any)
            if (data_) {
#if defined(_WIN32) || defined(_WIN64)
                _aligned_free(data_);
#else
                free(data_);
#endif
            }
            
            data_ = new_data;
            capacity_ = new_capacity;
        }
        
        return data_;
    }
    
    /**
     * @brief Get current capacity
     */
    size_t capacity() const override { 
        return capacity_; 
    }
    
    /**
     * @brief Free host memory
     * 
     * After calling, capacity() returns 0 and data pointer is null.
     * Safe to call multiple times.
     */
    void release() override {
        if (data_) {
#if defined(_WIN32) || defined(_WIN64)
            _aligned_free(data_);
#else
            free(data_);
#endif
            data_ = nullptr;
            capacity_ = 0;
        }
    }
    
    /**
     * @brief Get backend type
     */
    DeviceBackend getBackend() const override { 
        return DeviceBackend::CPU; 
    }
    
    // Move constructor
    CpuScratchMemory(CpuScratchMemory&& other) noexcept
        : data_(other.data_), capacity_(other.capacity_) 
    {
        other.data_ = nullptr;
        other.capacity_ = 0;
    }
    
    // Move assignment
    CpuScratchMemory& operator=(CpuScratchMemory&& other) noexcept {
        if (this != &other) {
            // Release current resources
            release();
            
            // Transfer ownership
            data_ = other.data_;
            capacity_ = other.capacity_;
            
            // Clear other
            other.data_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }
    
private:
    void* data_;        // Host pointer (64-byte aligned)
    size_t capacity_;   // Current allocation size in bytes
};

#endif // CPU_SCRATCH_MEMORY_H_
