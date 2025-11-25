#ifndef CUDA_SCRATCH_MEMORY_H_
#define CUDA_SCRATCH_MEMORY_H_

#include "compute/memory/IScratchMemory.h"
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

/**
 * @brief CUDA device scratch memory implementation
 * 
 * Manages reusable workspace on CUDA device with automatic growth.
 * Uses a growth factor strategy (1.5x) to reduce reallocation frequency.
 * 
 * Memory allocation strategy:
 * - First request: Allocate exact size
 * - Growth: new_capacity = max(requested, current + current/2)
 * - This 1.5x growth balances memory efficiency vs reallocation overhead
 * 
 * Thread safety: NOT thread-safe. Each thread/stream should have separate instance.
 */
class CudaScratchMemory : public IScratchMemory {
public:
    // RAII helper to ensure operations happen on correct device
    class DeviceGuard {
    public:
        explicit DeviceGuard(int device_id) {
            cudaGetDevice(&previous_device_);
            if (previous_device_ != device_id) {
                cudaSetDevice(device_id);
            }
        }
        
        ~DeviceGuard() {
            cudaSetDevice(previous_device_);
        }
        
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;
        
    private:
        int previous_device_;
    };
    
    /**
     * @brief Construct empty scratch memory (no allocation)
     */
    explicit CudaScratchMemory(int device_id) : data_(nullptr), capacity_(0), device_id_(device_id) {}
    
    /**
     * @brief Construct with pre-allocated capacity
     * 
     * @param initial_capacity Initial allocation size in bytes
     * @throws std::runtime_error if allocation fails
     */
    CudaScratchMemory(size_t initial_capacity, int device_id) 
        : data_(nullptr), capacity_(0), device_id_(device_id) 
    {
        if (initial_capacity > 0) {
            request(initial_capacity);
        }
    }
    
    /**
     * @brief Destructor - automatically frees device memory
     */
    ~CudaScratchMemory() override {
        DeviceGuard guard(device_id_);
        release();
    }
    
    /**
     * @brief Request device scratch memory
     * 
     * If requested size exceeds current capacity, reallocates with growth strategy.
     * Otherwise, returns existing allocation.
     * 
     * @param bytes Requested size in bytes (must be > 0)
     * @return Pointer to device memory
     * @throws std::invalid_argument if bytes == 0
     * @throws std::runtime_error if cudaMalloc fails
     */
    void* request(size_t bytes) override {
        DeviceGuard guard(device_id_);
        
        if (bytes == 0) {
            throw std::invalid_argument("CudaScratchMemory::request: bytes must be > 0");
        }
        
        if (bytes > capacity_) {
            // Growth strategy: max(requested, 1.5x current)
            // This reduces reallocation frequency for incrementally growing workloads
            size_t new_capacity = std::max(bytes, capacity_ + capacity_ / 2);
            
            void* new_data = nullptr;
            cudaError_t err = cudaMalloc(&new_data, new_capacity);
            
            if (err != cudaSuccess) {
                // Provide detailed error message
                throw std::runtime_error(
                    std::string("CudaScratchMemory: cudaMalloc failed for ") +
                    std::to_string(new_capacity) + " bytes: " +
                    cudaGetErrorString(err)
                );
            }
            
            // Free old allocation (if any)
            if (data_) {
                cudaFree(data_);
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
     * @brief Free device memory
     * 
     * After calling, capacity() returns 0 and data pointer is null.
     * Safe to call multiple times.
     */
    void release() override {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
            capacity_ = 0;
        }
    }
    
    /**
     * @brief Get backend type
     */
    DeviceBackend getBackend() const override { 
        return DeviceBackend::CUDA; 
    }
    
    // Move constructor
    CudaScratchMemory(CudaScratchMemory&& other) noexcept
        : data_(other.data_), capacity_(other.capacity_), device_id_(other.device_id_) 
    {
        other.data_ = nullptr;
        other.capacity_ = 0;
    }
    
    // Move assignment
    CudaScratchMemory& operator=(CudaScratchMemory&& other) noexcept {
        if (this != &other) {
            // Release current resources
            DeviceGuard guard(device_id_);
            release();
            
            // Transfer ownership
            data_ = other.data_;
            capacity_ = other.capacity_;
            device_id_ = other.device_id_;
            
            // Clear other
            other.data_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }
    
private:
    void* data_;        // Device pointer
    size_t capacity_;   // Current allocation size in bytes
    int device_id_;     // CUDA device ID
};

#endif // CUDA_SCRATCH_MEMORY_H_
