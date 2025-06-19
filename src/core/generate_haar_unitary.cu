#include "common_includes.h"
#include "core/matrix_operations.h"
#include "core/generate_haar_unitary.h"


#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <cusolverDn.h>


#include <vector>
#include <iostream>
#include <stdexcept>
#include <random> // Used to prime the cuRAND generator with a seed
#include <algorithm>


// Kernel to init curand states
__global__ void init_curand_states(curandState *states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// Kernel to generate complex normal random numbers
__global__ void generate_complex_normals(curandState *states, cuDoubleComplex* out, int num_elements, int stride) {
    int blockSize = blockDim.x;
    int idx = blockIdx.x * blockSize + threadIdx.x;

    curandState localState = states[idx];
    
    for (int i = idx; i < num_elements; i += stride) {
        // Generate a random complex number
        double real = curand_normal_double(&localState);
        double imag = curand_normal_double(&localState);
        out[i] = make_cuDoubleComplex(real, imag);
    }


}

// Dummy kernel to use for occupancy query
__global__ void dummyKernel() {
    // Empty kernel for occupancy calculation
}


// This is the "expensive" version, which copies the haar unitary back to the host after creation.
std::vector<std::complex<double>> generateHaarRandomUnitary(int N) {
    int matrixSize = N * N;

    // Use device 1
    cudaSetDevice(1);

    // Allocate device memory
    cuDoubleComplex* d_A = nullptr;
    curandState* d_states = nullptr;
    cudaMalloc(&d_A, matrixSize * sizeof(cuDoubleComplex));
    cudaMalloc(&d_states, matrixSize * sizeof(curandState));

    // Get a random seed. This happens on CPU.
    unsigned long long random_seed = std::random_device{}();

    // Initialize RNG states. Make sure not to exceed the maximum block size.
    cudaDeviceProp prop;
    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);

    int blockSize = min(prop.maxThreadsPerBlock, 256); // Use the maximum threads per block or 256, whichever is smaller.
    int gridSize = (matrixSize + blockSize - 1) / blockSize; // This can be as high as 2^31-1, the tasks are scheduled and dispatched to the hardware as resources become available.

    // Execute kernel to initialize curand states
    init_curand_states<<<gridSize, blockSize>>>(d_states, random_seed, matrixSize);
    cudaDeviceSynchronize();

    // Generate random complex matrix
    generate_complex_normals<<<gridSize, blockSize>>>(d_states, d_A, matrixSize, 1);
    cudaDeviceSynchronize();

    // cuSOLVER handles (This function initializes the cuSolverDN library and creates a handle on the cuSolverDN context)
    cusolverDnHandle_t cusolverH = nullptr;
    cusolverDnCreate(&cusolverH);

    // QR decomposition
    int work_size = 0;
    int *devInfo = nullptr;
    cuDoubleComplex* d_tau = nullptr;

    cudaMalloc(&devInfo, sizeof(int));
    cudaMalloc(&d_tau, N * sizeof(cuDoubleComplex));

    // Query workspace size for geqrf (QR factorization)
    cusolverDnZgeqrf_bufferSize(cusolverH, N, N, d_A, N, &work_size);
    // Allocate workspace for QR factorization
    cuDoubleComplex* d_work = nullptr;
    cudaMalloc(&d_work, work_size * sizeof(cuDoubleComplex));
    // Compute QR factorization
    cusolverDnZgeqrf(cusolverH, N, N, d_A, N, d_tau, d_work, work_size, devInfo);
    cudaDeviceSynchronize();
    // Check for successful factorization, copy info back to host
    int info_gpu = 0;
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_gpu != 0) {
        throw std::runtime_error("QR factorization failed on GPU");
    }

    // Generate Q from the factorization
    // Query workspace size for ungqr (generate unitary Q)
    cusolverDnZungqr_bufferSize(cusolverH, N, N, N, d_A, N, d_tau, &work_size);
    // Allocate workspace for Q generation
    cudaFree(d_work);
    cudaMalloc(&d_work, work_size * sizeof(cuDoubleComplex));
    // Generate Q matrix
    cusolverDnZungqr(cusolverH, N, N, N, d_A, N, d_tau, d_work, work_size, devInfo);
    cudaDeviceSynchronize();
    // Check for successful Q generation, copy info back to host
    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_gpu != 0) {
        throw std::runtime_error("Q generation failed on GPU");
    }

    // Copy result back to host
    std::vector<std::complex<double>> out(matrixSize);
    cudaMemcpy(out.data(), d_A, matrixSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_tau);
    cudaFree(d_work);
    cudaFree(devInfo);
    cudaFree(d_states);
    cusolverDnDestroy(cusolverH);

    return out;
}

// This version accepts a pre-allocated output vector on the GPU and does not copy the haar unitary back to the host.
// This is useful for performance reasons, as it avoids the overhead of copying the matrix back to the host.
cudaError_t generateHaarRandomUnitaries(cuDoubleComplex *d_A, int N, int num_matrices, int num_streams) {
    /*
    Generates Haar random unitary matrices on the GPU.
    Parameters:
        d_A: Pointer to device memory where the generated matrices will be stored. 
              The memory should be allocated with size N * N * num_matrices * sizeof(cuDoubleComplex).
              Matrices are stored sequentially.
        N: Size of the matrices (N x N).
        num_matrices: Number of matrices to generate.
        num_streams: Number of streams to use for parallel generation. 
                      This should be set based on the available GPU memory and the number of matrices to generate.
                      If set to 1, a single stream will be used.
                      Streams are run in parallel and each generate some of the matrices.
    Returns:
        cudaError_t: CUDA error code. Returns cudaSuccess on success, or an error code if an error occurs.
    Note:
        The function splits the unitaries generation into streams. 
        Each stream is responsible of generating a certain number of unitaries, and generation happens in parallel.
    
    */
    int matrixSize = N * N;

    cudaStream_t streams[num_streams];
    cusolverDnHandle_t handles[num_streams];
    // Create streams and handles
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cusolverDnCreate(&handles[i]);
        cusolverDnSetStream(handles[i], streams[i]);
    }

    /*
        RNG section - doesnt use streams
    */

    // Understand the number of threads per block that are sensible. Defaults to max=256 (hardcoded)
    cudaDeviceProp prop;
    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);

    int blockSize = min(prop.maxThreadsPerBlock, 256); // Use the maximum threads per block or 256, whichever is smaller.
    
    int maxBlocksPerSM;
    cudaError_t err2 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, dummyKernel, blockSize, 0);
    if (err2 != cudaSuccess) {
        printf("cudaOccupancyMaxActiveBlocksPerMultiprocessor failed: %s\n", cudaGetErrorString(err));
        return err2;
    }
    int concurrentBlocksMax = prop.multiProcessorCount * maxBlocksPerSM;

    // Allocate device memory for curand states. Each curandState is 48 bytes. Limit to 3GB the memory used.
    curandState* d_states = nullptr;
    int num_curand_states = blockSize*concurrentBlocksMax;
    cudaMalloc(&d_states, num_curand_states * sizeof(curandState)); // Each entry of each matrix will have its own state.
 
    // Log allocated resources
    std::cout << "Allocated d_states size: " << num_curand_states * sizeof(curandState) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

    // Get a random seed. This happens on CPU. 
    unsigned long long random_seed = std::random_device{}();


    // Execute kernel to initialize curand states
    init_curand_states<<<concurrentBlocksMax, blockSize>>>(d_states, random_seed, num_curand_states);
    cudaDeviceSynchronize();

    // Generate random complex matrix
    generate_complex_normals<<<concurrentBlocksMax, blockSize>>>(d_states, d_A, matrixSize*num_matrices, concurrentBlocksMax);
    cudaDeviceSynchronize();

    // At this point, all entries are initialized to random complex numbers. We need to generate the unitaries by performing QR decomposition.


    /*
        QR section
    */

    // Step 1: get the work size for each computation. 
    // Allocate memory for workspace size (one per matrix)
    int work_size[num_matrices] = {0}; // Host
    // Query worksize
    for (int i = 0; i < num_matrices; i++){
        // Get the offset from d_A where the current matrix starts
        cuDoubleComplex* d_A_offset = d_A + i * matrixSize;
        // Query workspace size for geqrf (QR factorization)
        cusolverDnZgeqrf_bufferSize(handles[i % num_streams], N, N, d_A_offset, N, &work_size[i]);
    }
    // Sync all streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }


    // Step 2: allocate workspace. Allocate max memory needed.
    cuDoubleComplex* d_work[num_streams] = {nullptr};
    for (int i = 0; i < num_streams; i++) {
        // max memory needed for stream i
        int work_size_stream = 0;
        for (int j = 0; j < num_matrices; j++) {
            if (j % num_streams == i) {
                work_size_stream = std::max(work_size_stream, work_size[j]);
            }
        }
        cudaMalloc(&d_work[i], work_size_stream * sizeof(cuDoubleComplex));
        // Debug
        std::cout << "Allocated work size for stream " << i << ": " << work_size_stream * sizeof(cuDoubleComplex) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    }

    // Step 3: compute QR factorization for each matrix in parallel.
    // Devinfo (Contains information about the success of the computation)
    int *devInfo[num_matrices] = {nullptr};
    // Tau (contains the scaling factors used to reconstruct Q which has weird normalization)
    cuDoubleComplex* d_tau[num_matrices] = {nullptr};
    // Allocate them
    for (int i = 0; i < num_matrices; i++) {
        cudaMalloc(&devInfo[i], sizeof(int));
        cudaMalloc(&d_tau[i], N * sizeof(cuDoubleComplex));
    }
    // Log allocated resources, cumulative
    int total_devInfo_size = num_matrices * sizeof(int);
    int total_tau_size = num_matrices * N * sizeof(cuDoubleComplex);
    std::cout << "Total allocated devInfo size: " << total_devInfo_size / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "Total allocated d_tau size: " << total_tau_size / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

    // Now perform QR
    // Launch all jobs (1 per stream)
    for (int i = 0; i < num_matrices; ++i) {
        int stream_id = i % num_streams;
        cuDoubleComplex* d_A_offset = d_A + i * matrixSize;
        // Perform QR factorization
        cusolverDnZgeqrf(handles[stream_id], N, N, d_A_offset, N, d_tau[i], d_work[stream_id], work_size[i], devInfo[i]);
    }
    // Wait for all streams to finish
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    // Reconstruct Q from the factorization
    // Query workspace size for ungqr (generate unitary Q)
    for (int i = 0; i < num_matrices; i++){
        // Get the offset from d_A where the current matrix starts
        cuDoubleComplex* d_A_offset = d_A + i * matrixSize;
        // Query workspace size
        cusolverDnZungqr_bufferSize(handles[i % num_streams], N, N, N, d_A_offset, N, d_tau[i], &work_size[i]);
    }
    // Sync all streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Allocate workspace for Q generation
    for (int i = 0; i < num_streams; i++) {
        // max memory needed for stream i
        int work_size_stream = 0;
        for (int j = 0; j < num_matrices; j++) {
            if (j % num_streams == i) {
                work_size_stream = std::max(work_size_stream, work_size[j]);
            }
        }
        cudaFree(d_work[i]);
        cudaMalloc(&d_work[i], work_size_stream * sizeof(cuDoubleComplex));
        // Debug
        std::cout << "Now reconstructing Q. Allocated work size for stream " << i << ": " << work_size_stream * sizeof(cuDoubleComplex) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    }
    // Generate Q matrix
    for (int i = 0; i < num_matrices; i++){
        // Get the offset from d_A where the current matrix starts
        cuDoubleComplex* d_A_offset = d_A + i * matrixSize;
        // Generate Q matrix
        cusolverDnZungqr(handles[i % num_streams], N, N, N, d_A_offset, N, d_tau[i], d_work[i % num_streams], work_size[i], devInfo[i]);
    }
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
        cusolverDnDestroy(handles[i]);
        cudaFree(d_work[i]);
    }

    for (int i = 0; i < num_matrices; i++) {
        cudaFree(devInfo[i]);
        cudaFree(d_tau[i]);
    }
    return cudaSuccess; // Return success code
}