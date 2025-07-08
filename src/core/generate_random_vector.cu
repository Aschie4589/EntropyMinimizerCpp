#include "common_includes.h"
#include "core/generate_random_vector.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <cusolverDn.h>

std::vector<std::complex<double> >* generateUniformRandomVector(int N){
    // Step 0: Initialize output
    std::vector<std::complex<double> >* out = new std::vector<std::complex<double> >(N);
    // Step 1: Set up random number generator
    std::random_device rd;                  // Random device to seed the generator
    std::mt19937 gen(rd());                 // Mersenne Twister generator, seeded by rd()
    std::normal_distribution<double> dist(0.0f, 1.0f); // Normal distribution with mean and stddev

    // Step 2: Generate random real and imaginary parts
    for (int i=0; i < N; i++){
        double real_part = dist(gen);  // Generate the real part (normal distribution)
        double imag_part = dist(gen);  // Generate the imaginary part (normal distribution)
        out->at(i) = std::complex<double>(real_part, imag_part);
    }

    // Step 3: Renormalize the vector
    double norm = 0;
    for (int i=0; i < N; i++){
        norm += std::pow(std::abs(out->at(i)),2);
    }
    norm = std::sqrt(norm);
    for (int i=0; i < N; i++){
        out->at(i) /= norm;
    }
    // Return the pointer    
    return out;
}

// Kernel to init curand states
__global__ void init_curand_states_v(curandState *states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// Kernel to generate complex normal random numbers
template<typename T>
__global__ void generate_complex_normals_v(curandState *states, typename CudaTraits<T>::Complex* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    curandState localState = states[idx];
    double real = CudaTraits<T>::rand_nor(&localState);
    double imag = CudaTraits<T>::rand_nor(&localState);
    out[idx] = CudaTraits<T>::make_complex(real, imag);
    states[idx] = localState;
}

template<typename T>
__device__ double complex_abs_squared(typename CudaTraits<T>::Complex z) {
    return z.x * z.x + z.y * z.y;
}

__global__ void calculate_norms_float(cuComplex* v, float* norms, int N, int num_vectors) {
    /*
    Kernel to compute the norm of a set of complex vectors, stored sequentially in a 1D array.
    Each block (with however many threads it contains) is responsible for calculating one of the norms of the vectors.
    Parameters:
        v: Pointer to device memory containing the complex vectors, stored sequentially.
        norms: Pointer to device memory where the computed norms will be stored.
        N: Size of each vector.
        num_vectors: Number of vectors.
    */

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (bid >= num_vectors) return; // One block per vector
    // Use memory coalescing. Every thread reads one entry, and these entries are adjacent. This is memory efficient since GPU has to 
    // access memory in fewer "transactions".
    extern __shared__ float smem[];    
    float p_norm = 0.0; // Local array to store partial norm squared calculated by the thread

    for (int i = bid * N + tid; i < (bid + 1) * N; i += num_threads) {
        cuComplex val = v[i];
        p_norm += complex_abs_squared<float>(val); // Accumulate the squared absolute value
    }
    smem[tid] = p_norm; // Store the square root of the sum of squares

    // Synchronize threads within the block to ensure all threads have written their partial norms
    __syncthreads();

    // Now do reduction.
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    // Save the norm
    if (tid == 0) {
        norms[bid] = sqrt(smem[0]); // Store the final norm in the output array
    }
}

__global__ void calculate_norms_double(cuDoubleComplex* v, double* norms, int N, int num_vectors) {
    /*
    Kernel to compute the norm of a set of complex vectors, stored sequentially in a 1D array.
    Each block (with however many threads it contains) is responsible for calculating one of the norms of the vectors.
    Parameters:
        v: Pointer to device memory containing the complex vectors, stored sequentially.
        norms: Pointer to device memory where the computed norms will be stored.
        N: Size of each vector.
        num_vectors: Number of vectors.
    */

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (bid >= num_vectors) return; // One block per vector
    // Use memory coalescing. Every thread reads one entry, and these entries are adjacent. This is memory efficient since GPU has to 
    // access memory in fewer "transactions".
    extern __shared__ double smem2[];    
    double p_norm = 0.0; // Local array to store partial norm squared calculated by the thread

    for (int i = bid * N + tid; i < (bid + 1) * N; i += num_threads) {
        cuDoubleComplex val = v[i];
        p_norm += complex_abs_squared<double>(val); // Accumulate the squared absolute value
    }
    smem2[tid] = p_norm; // Store the square root of the sum of squares

    // Synchronize threads within the block to ensure all threads have written their partial norms
    __syncthreads();

    // Now do reduction.
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem2[tid] += smem2[tid + s];
        }
        __syncthreads();
    }
    // Save the norm
    if (tid == 0) {
        norms[bid] = sqrt(smem2[0]); // Store the final norm in the output array
    }
}

template<typename T>
__global__ void normalize_vectors(typename CudaTraits<T>::Complex* v, typename CudaTraits<T>::Real* norms, int N, int num_vectors) {
    /*
    Kernel to normalize complex vectors stored sequentially in a 1D array.
    Each block (with however many threads it contains) is responsible for normalizing one of the vectors.
    Parameters:
        v: Pointer to device memory containing the complex vectors, stored sequentially.
        norms: Pointer to device memory containing the norms of the vectors.
        N: Size of each vector.
        num_vectors: Number of vectors.
    */

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (bid >= num_vectors) return; // One block per vector

    for (int i = bid * N + tid; i < (bid + 1) * N; i += num_threads) {
        typename CudaTraits<T>::Complex val = v[i];
        // Normalize the vector by dividing each component by its norm
        val.x /= norms[bid];
        val.y /= norms[bid];
        // Write the normalized value back to the vector
        v[i] = val; 
    }
}

// This version accepts a pre-allocated output vector on the GPU and does not copy the haar unitary back to the host.
// This is useful for performance reasons, as it avoids the overhead of copying the matrix back to the host.
template <typename T>
cudaError_t generateUniformRandomVectorsCuda<T>(typename CudaTraits<T>::Complex *v, int N, int num_vectors) {
    /*
    Generates Haar random unitary matrices on the GPU.
    Parameters:
        v: Pointer to device memory where the generated vectors will be stored. 
              The memory should be allocated with size N * num_vectors * sizeof(typename CudaTraits<T>::Complex).
              Vectors are stored sequentially.
        N: Size of the vectors.
        num_vectors: Number of vectors to generate.
    Returns:
        cudaError_t: CUDA error code. Returns cudaSuccess on success, or an error code if an error occurs.
    TODO: 
        - Use approach more similar to Haar unitary generation
        - Reformat (no duplicate functions)
    */

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Random Number Generation section.
    // Allocate device memory for curand states
    curandState* d_states = nullptr;
    cudaMalloc(&d_states, num_vectors * N * sizeof(curandState)); // Each entry of each matrix will have its own state.

    // Get a random seed. This happens on CPU. 
    unsigned long long random_seed = std::random_device{}();

    // Initialize RNG states. Make sure not to exceed the maximum block size.
    cudaDeviceProp prop;
    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);

    int blockSize = min(prop.maxThreadsPerBlock, 256); // Use the maximum threads per block or 256, whichever is smaller.
    int gridSize = (num_vectors * N + blockSize - 1) / blockSize; // This can be as high as 2^31-1, the tasks are scheduled and dispatched to the hardware as resources become available.

    // Execute kernel to initialize curand states
    init_curand_states_v<<<gridSize, blockSize>>>(d_states, random_seed, num_vectors * N);
    cudaDeviceSynchronize();

    // Generate random complex vector
    generate_complex_normals_v<T><<<gridSize, blockSize>>>(d_states, v, num_vectors * N);
    cudaDeviceSynchronize();

    // Free the curand states memory
    cudaFree(d_states);

    // Calculate norms of the vectors
    typename CudaTraits<T>::Real* d_norms = nullptr;
    cudaMalloc(&d_norms, num_vectors * sizeof(typename CudaTraits<T>::Real));
   
    // Select the threads per block. This should be the closest multiple of 32 higher than the size of the vector, or 256.
    int norm_block_size = min(256, (N + 31) / 32 * 32); // Ensure the block size is a multiple of 32 for optimal performance.
    int norm_grid_size = num_vectors; // This can be as high as 2^31-1, the tasks are
    // scheduled and dispatched to the hardware as resources become available.
    size_t shared_mem_size = norm_block_size * sizeof(typename CudaTraits<T>::Real);
    if constexpr (std::is_same<T, float>::value) {
        calculate_norms_float<<<norm_grid_size, norm_block_size, shared_mem_size>>>(v, d_norms, N, num_vectors);
    } else if constexpr (std::is_same<T, double>::value) {
        calculate_norms_double<<<norm_grid_size, norm_block_size, shared_mem_size>>>(v, d_norms, N, num_vectors);
    }
    
    cudaDeviceSynchronize();

    // Normalize the vectors
    normalize_vectors<T><<<norm_grid_size, norm_block_size>>>(v, d_norms, N, num_vectors);
    cudaDeviceSynchronize();

    // Free the norms memory
    cudaFree(d_norms);
    // Destroy the cusolver handle
    cusolverDnDestroy(handle);
    // Return success
    return cudaSuccess;

}

template cudaError_t generateUniformRandomVectorsCuda<double>(cuDoubleComplex*, int, int);
template cudaError_t generateUniformRandomVectorsCuda<float>(cuComplex*, int, int);