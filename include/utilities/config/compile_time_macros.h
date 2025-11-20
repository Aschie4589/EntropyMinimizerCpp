#ifndef COMPILE_TIME_MACROS_H
#define COMPILE_TIME_MACROS_H

/* MACROS */
// Custom macro to handle cudaMalloc errors
#define CUDA_MALLOC_CHECK(ptr, size, name) \
    do { \
        ptr = nullptr; \
        cudaError_t err = cudaMalloc((void**)&ptr, size); \
        if (err != cudaSuccess) { \
            std::cerr << "Error allocating memory for " << name << ": " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("Failed to allocate memory for " + std::string(name)); \
        } \
    } while(0)

#define CUDA_MEMCPY_CHECK(dest, src, size, kind, name) \
    do { \
        cudaError_t err = cudaMemcpy(dest, src, size, kind); \
        if (err != cudaSuccess) { \
            std::cerr << "Error copying memory for " << name << ": " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("Failed to copy memory for " + std::string(name)); \
        } \
    } while(0)



#endif