#ifndef CUDA_RANDOM_H_
#define CUDA_RANDOM_H_

#include "compute/random/IRandomGenerator.h"
#include <curand.h>
#include <cuda_runtime.h>

class CudaRandom : public IRandomGenerator {
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
    
    explicit CudaRandom(int device_id);
    ~CudaRandom() override;
    
    void setSeed(unsigned long long seed) override;
    
    void generateUniform(
        void* output,
        size_t count,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    void generateNormal(
        void* output,
        size_t count,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    void generateComplexNormal(
        void* output,
        size_t count,
        PrecisionType precision,
        IStream* stream = nullptr
    ) override;
    
    // CUDA-specific: get the curand generator
    curandGenerator_t getGenerator() const { return generator_; }
    
private:
    curandGenerator_t generator_;
    unsigned long long current_seed_;
    int device_id_;
};

#endif // CUDA_RANDOM_H_
