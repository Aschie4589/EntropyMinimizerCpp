#ifndef COMPUTE_TYPES_H_
#define COMPUTE_TYPES_H_

// Common types used across compute interfaces

enum class DeviceBackend {
    CUDA,
    CPU
    // POTENTIAL FUTURE BACKENDS: ROCM, METAL, etc.
};

enum class PrecisionType {
    FLOAT,
    DOUBLE
};

#endif // COMPUTE_TYPES_H_
