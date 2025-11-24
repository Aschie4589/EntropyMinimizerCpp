#ifndef CPU_RANDOM_H_
#define CPU_RANDOM_H_

#include "compute/random/IRandomGenerator.h"
#include <random>

class CpuRandom : public IRandomGenerator {
public:
    CpuRandom();
    ~CpuRandom() override = default;
    
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
    
private:
    std::mt19937_64 rng_;
    unsigned long long current_seed_;
};

#endif // CPU_RANDOM_H_
