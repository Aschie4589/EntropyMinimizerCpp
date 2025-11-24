#ifndef CUDA_RANDOM_H_
#define CUDA_RANDOM_H_

#include "compute/random/IRandomGenerator.h"
#include <curand.h>
#include <cuda_runtime.h>

class CudaRandom : public IRandomGenerator {
public:
    CudaRandom();
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
};

#endif // CUDA_RANDOM_H_
