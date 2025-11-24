#include "compute/backends/cpu/CpuRandom.h"
#include <complex>
#include <stdexcept>

CpuRandom::CpuRandom() : current_seed_(1234ULL) {
    rng_.seed(current_seed_);
}

void CpuRandom::setSeed(unsigned long long seed) {
    current_seed_ = seed;
    rng_.seed(seed);
}

void CpuRandom::generateUniform(
    void* output,
    size_t count,
    PrecisionType precision,
    IStream* stream
) {
    // CPU ignores stream parameter (operations are synchronous)
    
    if (precision == PrecisionType::DOUBLE) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double* out = static_cast<double*>(output);
        for (size_t i = 0; i < count; ++i) {
            out[i] = dist(rng_);
        }
    } else {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float* out = static_cast<float*>(output);
        for (size_t i = 0; i < count; ++i) {
            out[i] = dist(rng_);
        }
    }
}

void CpuRandom::generateNormal(
    void* output,
    size_t count,
    PrecisionType precision,
    IStream* stream
) {
    // CPU ignores stream parameter
    
    if (precision == PrecisionType::DOUBLE) {
        std::normal_distribution<double> dist(0.0, 1.0);
        double* out = static_cast<double*>(output);
        for (size_t i = 0; i < count; ++i) {
            out[i] = dist(rng_);
        }
    } else {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float* out = static_cast<float*>(output);
        for (size_t i = 0; i < count; ++i) {
            out[i] = dist(rng_);
        }
    }
}

void CpuRandom::generateComplexNormal(
    void* output,
    size_t count,
    PrecisionType precision,
    IStream* stream
) {
    // CPU ignores stream parameter
    
    if (precision == PrecisionType::DOUBLE) {
        std::normal_distribution<double> dist(0.0, 1.0);
        std::complex<double>* out = static_cast<std::complex<double>*>(output);
        for (size_t i = 0; i < count; ++i) {
            double real = dist(rng_);
            double imag = dist(rng_);
            out[i] = std::complex<double>(real, imag);
        }
    } else {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::complex<float>* out = static_cast<std::complex<float>*>(output);
        for (size_t i = 0; i < count; ++i) {
            float real = dist(rng_);
            float imag = dist(rng_);
            out[i] = std::complex<float>(real, imag);
        }
    }
}
