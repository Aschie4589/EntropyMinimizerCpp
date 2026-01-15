#include "channel/utilities/generate_haar_unitary.h"

#include <vector>
#include <complex>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <cusolverDn.h>

// RAII wrappers
#include "utilities/cuda/cuda_device_memory.h"
#include "utilities/cuda/cuda_stream.h"
#include "utilities/cuda/cuda_solver_handle.h"
#include "utilities/cuda/error_handling.h"

// Messaging
#include "utilities/messaging/message_handler.h"


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

namespace channel {

cudaError_t generateHaarRandomUnitaries(
    cuDoubleComplex *d_A, 
    int N, 
    int num_matrices, 
    int num_streams,
    utils::MessageHandler* msg_handler
) {
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
    try {
        const int matrixSize = N * N;

        // RAII: Create streams and solver handles (automatically destroyed)
        std::vector<CudaStream> streams(num_streams);
        std::vector<CudaSolverHandle> handles(num_streams);
        
        // Associate each handle with its stream
        for (int i = 0; i < num_streams; i++) {
            handles[i].setStream(streams[i].get());
        }

        /*
            RNG section - doesnt use streams
        */

        // Get device properties for optimal occupancy
        cudaDeviceProp prop;
        int device_id = -1;
        CUDA_CHECK(cudaGetDevice(&device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

        int blockSize = std::min(prop.maxThreadsPerBlock, 256);
        
        int maxBlocksPerSM;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM, dummyKernel, blockSize, 0));
        int concurrentBlocksMax = prop.multiProcessorCount * maxBlocksPerSM;

        // RAII: Allocate device memory for curand states
        int num_curand_states = blockSize * concurrentBlocksMax;
        CudaDeviceMemory<curandState> d_states(num_curand_states);
     
        // Optional debug logging
        if (msg_handler) {
            double size_gib = (num_curand_states * sizeof(curandState)) / (1024.0 * 1024.0 * 1024.0);
            msg_handler->debug("Allocated d_states size: " + std::to_string(size_gib) + " GB");
        }

        // Get a random seed (on CPU)
        unsigned long long random_seed = std::random_device{}();

        // Initialize curand states
        init_curand_states<<<concurrentBlocksMax, blockSize>>>(
            d_states.get(), random_seed, num_curand_states);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Generate random complex matrix
        generate_complex_normals<<<concurrentBlocksMax, blockSize>>>(
            d_states.get(), d_A, matrixSize * num_matrices, concurrentBlocksMax);
        CUDA_CHECK(cudaDeviceSynchronize());

        // At this point, all entries are initialized to random complex numbers. We need to generate the unitaries by performing QR decomposition.

        /*
            QR section
        */

        // Step 1: Query workspace size for each matrix
        std::vector<int> work_size(num_matrices, 0);
        
        for (int i = 0; i < num_matrices; i++) {
            cuDoubleComplex* d_A_offset = d_A + i * matrixSize;
            CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(
                handles[i % num_streams].get(), N, N, d_A_offset, N, &work_size[i]));
        }
        
        // Synchronize all streams
        for (auto& stream : streams) {
            stream.synchronize();
        }

        // Step 2: Allocate workspace for each stream (max size needed per stream)
        std::vector<CudaDeviceMemory<cuDoubleComplex>> d_work;
        for (int i = 0; i < num_streams; i++) {
            int work_size_stream = 0;
            for (int j = 0; j < num_matrices; j++) {
                if (j % num_streams == i) {
                    work_size_stream = std::max(work_size_stream, work_size[j]);
                }
            }
            d_work.emplace_back(work_size_stream);
            if (msg_handler) {
                double size_gib = (work_size_stream * sizeof(cuDoubleComplex)) / (1024.0 * 1024.0 * 1024.0);
                msg_handler->debug("Allocated work size for stream " + std::to_string(i) + 
                                   ": " + std::to_string(size_gib) + " GB");
            }
        }

        // Step 3: Allocate devInfo and tau for each matrix
        std::vector<CudaDeviceMemory<int>> devInfo;
        std::vector<CudaDeviceMemory<cuDoubleComplex>> d_tau;
        for (int i = 0; i < num_matrices; i++) {
            devInfo.emplace_back(1);
            d_tau.emplace_back(N);
        }
        
        // Optional debug logging for allocated resources
        if (msg_handler) {
            int total_devInfo_size = num_matrices * sizeof(int);
            int total_tau_size = num_matrices * N * sizeof(cuDoubleComplex);
            double devInfo_gib = total_devInfo_size / (1024.0 * 1024.0 * 1024.0);
            double tau_gib = total_tau_size / (1024.0 * 1024.0 * 1024.0);
            msg_handler->debug("Total allocated devInfo size: " + std::to_string(devInfo_gib) + " GB");
            msg_handler->debug("Total allocated d_tau size: " + std::to_string(tau_gib) + " GB");
        }

        // Step 4: Perform QR factorization
        for (int i = 0; i < num_matrices; i++) {
            int stream_id = i % num_streams;
            cuDoubleComplex* d_A_offset = d_A + i * matrixSize;
            CUSOLVER_CHECK(cusolverDnZgeqrf(
                handles[stream_id].get(), N, N, d_A_offset, N,
                d_tau[i].get(), d_work[stream_id].get(), work_size[i],
                devInfo[i].get()));
        }
        
        // Wait for all streams to finish
        for (auto& stream : streams) {
            stream.synchronize();
        }

        // Step 5: Query workspace size for Q generation
        for (int i = 0; i < num_matrices; i++) {
            cuDoubleComplex* d_A_offset = d_A + i * matrixSize;
            CUSOLVER_CHECK(cusolverDnZungqr_bufferSize(
                handles[i % num_streams].get(), N, N, N, d_A_offset, N,
                d_tau[i].get(), &work_size[i]));
        }
        
        // Synchronize all streams
        for (auto& stream : streams) {
            stream.synchronize();
        }

        // Step 6: Reallocate workspace for Q generation
        d_work.clear();
        for (int i = 0; i < num_streams; i++) {
            int work_size_stream = 0;
            for (int j = 0; j < num_matrices; j++) {
                if (j % num_streams == i) {
                    work_size_stream = std::max(work_size_stream, work_size[j]);
                }
            }
            d_work.emplace_back(work_size_stream);
            if (msg_handler) {
                double size_gib = (work_size_stream * sizeof(cuDoubleComplex)) / (1024.0 * 1024.0 * 1024.0);
                msg_handler->debug("Reconstructing Q. Allocated work size for stream " + 
                                   std::to_string(i) + ": " + std::to_string(size_gib) + " GB");
            }
        }

        // Step 7: Generate Q matrix
        for (int i = 0; i < num_matrices; i++) {
            int stream_id = i % num_streams;
            cuDoubleComplex* d_A_offset = d_A + i * matrixSize;
            CUSOLVER_CHECK(cusolverDnZungqr(
                handles[stream_id].get(), N, N, N, d_A_offset, N,
                d_tau[i].get(), d_work[stream_id].get(), work_size[i],
                devInfo[i].get()));
        }
        
        // Final synchronization
        for (auto& stream : streams) {
            stream.synchronize();
        }

        // All RAII objects automatically cleaned up here
        return cudaSuccess;
        
    } catch (const std::runtime_error& e) {
        if (msg_handler) {
            msg_handler->error("CUDA error in generateHaarRandomUnitaries: " + std::string(e.what()));
        } else {
            std::cerr << "CUDA error in generateHaarRandomUnitaries: " << e.what() << std::endl;
        }
        return cudaErrorUnknown;
    } catch (const std::exception& e) {
        if (msg_handler) {
            msg_handler->error("Unexpected error in generateHaarRandomUnitaries: " + std::string(e.what()));
        } else {
            std::cerr << "Unexpected error in generateHaarRandomUnitaries: " << e.what() << std::endl;
        }
        return cudaErrorUnknown;
    }
}

} // namespace channel