#ifndef DEFS_H
#define DEFS_H

/* 
DEFS - These are definitions that are used throughout the program, and shouldn't be changed
*/

/*
Cuda Minimizer
*/
#define CUDA_MINIMIZER_CREATED 0    // The minimizer has been created, but the vector not initialized. The kraus ops are correct. Memory is allocated but not initialized.
#define CUDA_MINIMIZER_STAGE_0 1    // At stage 0, the input vector is initialized and correct. Also kraus is initialized and correct. d_vecs_1 and d_vecs_2 are initialized but empty. d_sv_1 and d_sv_2 are initialized but empty. The entropy is initialized to -1.
#define CUDA_MINIMIZER_STAGE_1 2    // At stage 1, same as stage zero. On top of that, d_vecs_1 and d_sv_1 are filled with the results of the first SVD. It is possible to compute entropy.
#define CUDA_MINIMIZER_STAGE_2 3    // At stage 2, also d_vecs_2 and d_sv_2 are filled with the results of the second SVD. d_vec is updated to the new vector, so the entropy can't be computed for this new vector yet. Computing the entropy will give the old one.




/*
Return codes
*/
#define ENTROPY_MINIMIZER_CONTINUE 0
#define ENTROPY_MINIMIZER_CONVERGED 1
#define ENTROPY_MINIMIZER_MAX_ITERS 2
#define ENTROPY_MINIMIZER_NUMERICAL_INST 3
#define ENTROPY_MINIMIZER_TERMINATED 4
#define ENTROPY_MINIMIZER_MOE_PREDICTION 5


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