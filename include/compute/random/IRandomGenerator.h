#ifndef IRANDOMGENERATOR_H_
#define IRANDOMGENERATOR_H_

#include <cstddef>

#include "compute/stream/IComputeStream.h"
#include "compute/core/ComputeTypes.h"

class IRandomGenerator {
public:
    virtual ~IRandomGenerator() = default;
    
    // Set seed
    virtual void setSeed(unsigned long long seed) = 0;
    
    // Generate uniform random numbers [0, 1)
    virtual void generateUniform(
        void* output,
        size_t count,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    // Generate normal distribution N(0, 1)
    virtual void generateNormal(
        void* output,
        size_t count,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
    
    // Generate complex normal (for Haar unitaries)
    virtual void generateComplexNormal(
        void* output,
        size_t count,
        PrecisionType precision,
        IStream* stream = nullptr
    ) = 0;
};

#endif // IRANDOMGENERATOR_H_