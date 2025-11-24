#include "compute/backends/cuda/CudaRandom.h"
#include "compute/backends/cuda/CudaStream.h"
#include "utilities/cuda/error_handling.h"
#include <stdexcept>
#include <cuComplex.h>

CudaRandom::CudaRandom() : current_seed_(1234ULL) {
    // Create generator (default: PSEUDO_DEFAULT which is XORWOW)
    curandStatus_t status = curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuRAND generator: " + std::to_string(status));
    }
    
    // Set default seed
    setSeed(current_seed_);
}

CudaRandom::~CudaRandom() {
    if (generator_) {
        curandDestroyGenerator(generator_);
    }
}

void CudaRandom::setSeed(unsigned long long seed) {
    current_seed_ = seed;
    curandStatus_t status = curandSetPseudoRandomGeneratorSeed(generator_, seed);
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to set cuRAND seed: " + std::to_string(status));
    }
}

void CudaRandom::generateUniform(
    void* output,
    size_t count,
    PrecisionType precision,
    IStream* stream
) {
    // Set stream if provided
    if (stream != nullptr) {
        auto* cuda_stream = dynamic_cast<CudaStream*>(stream);
        if (cuda_stream) {
            curandSetStream(generator_, cuda_stream->getCudaStream());
        }
    }
    
    curandStatus_t status;
    
    if (precision == PrecisionType::DOUBLE) {
        status = curandGenerateUniformDouble(generator_, static_cast<double*>(output), count);
    } else {
        status = curandGenerateUniform(generator_, static_cast<float*>(output), count);
    }
    
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to generate uniform random numbers: " + std::to_string(status));
    }
}

void CudaRandom::generateNormal(
    void* output,
    size_t count,
    PrecisionType precision,
    IStream* stream
) {
    // Set stream if provided
    if (stream != nullptr) {
        auto* cuda_stream = dynamic_cast<CudaStream*>(stream);
        if (cuda_stream) {
            curandSetStream(generator_, cuda_stream->getCudaStream());
        }
    }
    
    curandStatus_t status;
    
    // cuRAND requires even number of samples for normal distribution
    size_t adjusted_count = (count % 2 == 0) ? count : count + 1;
    
    if (precision == PrecisionType::DOUBLE) {
        status = curandGenerateNormalDouble(
            generator_, 
            static_cast<double*>(output), 
            adjusted_count,
            0.0,  // mean
            1.0   // stddev
        );
    } else {
        status = curandGenerateNormal(
            generator_, 
            static_cast<float*>(output), 
            adjusted_count,
            0.0f,  // mean
            1.0f   // stddev
        );
    }
    
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to generate normal random numbers: " + std::to_string(status));
    }
}

void CudaRandom::generateComplexNormal(
    void* output,
    size_t count,
    PrecisionType precision,
    IStream* stream
) {
    // Set stream if provided
    if (stream != nullptr) {
        auto* cuda_stream = dynamic_cast<CudaStream*>(stream);
        if (cuda_stream) {
            curandSetStream(generator_, cuda_stream->getCudaStream());
        }
    }
    
    // Complex numbers have real and imaginary parts, so we need 2*count random numbers
    // cuRAND requires even count, so 2*count is always even
    size_t real_count = 2 * count;
    
    curandStatus_t status;
    
    if (precision == PrecisionType::DOUBLE) {
        // Generate directly into the complex array (interleaved real/imag)
        // cuDoubleComplex is compatible with double[2] layout
        status = curandGenerateNormalDouble(
            generator_,
            reinterpret_cast<double*>(output),
            real_count,
            0.0,  // mean
            1.0   // stddev (for each component)
        );
    } else {
        // cuComplex is compatible with float[2] layout
        status = curandGenerateNormal(
            generator_,
            reinterpret_cast<float*>(output),
            real_count,
            0.0f,  // mean
            1.0f   // stddev (for each component)
        );
    }
    
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to generate complex normal random numbers: " + std::to_string(status));
    }
}
