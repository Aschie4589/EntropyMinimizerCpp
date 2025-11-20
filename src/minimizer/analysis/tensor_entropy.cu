#include "minimizer/analysis/tensor_entropy.h"
#include "utilities/config/compile_time_macros.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cmath>

template<typename T>
TensorEntropyEstimator<T>::TensorEntropyEstimator() {
    // Constructor implementation (if needed)
}

template<typename T>
TensorEntropyEstimator<T>::~TensorEntropyEstimator() {
    // Destructor implementation (if needed)
}

// Helper functions for complex arithmetic on device
template<typename T>
__device__ typename CudaTraits<T>::Complex complex_add(typename CudaTraits<T>::Complex a, typename CudaTraits<T>::Complex b);

template<>
__device__ cuDoubleComplex complex_add<double>(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCadd(a, b);
}

template<>
__device__ cuComplex complex_add<float>(cuComplex a, cuComplex b) {
    return cuCaddf(a, b);
}

// Kernel to compute traces for all pairs of A and B matrices
// A_batch: array of pointers to A matrices (N x N)
// B_batch: array of pointers to B matrices (N x N)
// traces: output array of size batchA_size x batchB_size
template<typename T>
__global__ void compute_traces_kernel(
    typename CudaTraits<T>::Complex** A_batch,
    typename CudaTraits<T>::Complex** B_batch,
    typename CudaTraits<T>::Complex* traces,
    int N,
    int batchA_size,
    int batchB_size
){
    int idxA = blockIdx.x;
    int idxB = threadIdx.x;

    if (idxA >= batchA_size || idxB >= batchB_size) return;

    typename CudaTraits<T>::Complex* A = A_batch[idxA];
    typename CudaTraits<T>::Complex* B = B_batch[idxB];

    typename CudaTraits<T>::Complex sum = CudaTraits<T>::make_complex(0.0, 0.0);

    // Compute Tr(B*A) = sum over diagonal of (B*A)
    // For matrices in column-major: (B*A)_{ii} = sum_k B_{ik} * A_{ki}
    for(int i = 0; i < N; i++){
        for(int k = 0; k < N; k++){
            // B[i,k] in column-major: B[i + k*N]
            // A[k,i] in column-major: A[k + i*N]
            typename CudaTraits<T>::Complex b_ik = B[i + k*N];
            typename CudaTraits<T>::Complex a_ki = A[k + i*N];
            typename CudaTraits<T>::Complex prod = CudaTraits<T>::complex_mult(b_ik, a_ki);
            sum = complex_add<T>(sum, prod);
        }
    }

    traces[idxA * batchB_size + idxB] = sum;
}

// Kernel to populate M matrix with normalized traces
// traces: input traces of size batchA*d x batchB*d
// M: output matrix d^2 x d^2 (row-major indexing: M[i*d+j, k*d+l])
// i0, k0: batch offsets
// batchA, batchB: batch sizes
// d, N: dimensions
template<typename T>
__global__ void populate_M_kernel(
    typename CudaTraits<T>::Complex* traces,
    typename CudaTraits<T>::Complex* M,
    int i0, int k0,
    int batchA, int batchB,
    int d, int N
){
    int idx_i = blockIdx.x * blockDim.x + threadIdx.x; // index within batchA
    int idx_k = blockIdx.y * blockDim.y + threadIdx.y; // index within batchB
    
    if(idx_i >= batchA*d || idx_k >= batchB*d) return;
    
    // Decode (i, j) from idx_i: i = i0 + idx_i/d, j = idx_i%d
    int i = i0 + idx_i / d;
    int j = idx_i % d;
    
    // Decode (k, l) from idx_k: k = k0 + idx_k/d, l = idx_k%d
    int k = k0 + idx_k / d;
    int l = idx_k % d;
    
    // M index: row = i*d + j, col = k*d + l
    int row = i * d + j;
    int col = k * d + l;
    
    // Get trace and normalize by 1/(N)
    typename CudaTraits<T>::Complex trace_val = traces[idx_i * (batchB*d) + idx_k];
    typename CudaTraits<T>::Real norm_factor = 1.0 / static_cast<typename CudaTraits<T>::Real>(N);
    
    typename CudaTraits<T>::Complex normalized;
    if constexpr (std::is_same_v<T, double>) {
        normalized = make_cuDoubleComplex(trace_val.x * norm_factor, trace_val.y * norm_factor);
    } else {
        normalized = make_cuFloatComplex(trace_val.x * norm_factor, trace_val.y * norm_factor);
    }
    
    // Store in M (column-major: M[row + col * d*d])
    M[row + col * d * d] = normalized;
}

// Helper to create array of device pointers for batched operations
// Uses async memcpy with a stream for overlapping with computation
template<typename T>
void create_device_ptr_array_async(typename CudaTraits<T>::Complex* d_data, 
                                     typename CudaTraits<T>::Complex** d_ptrs, 
                                     int batch_size, 
                                     int N,
                                     cudaStream_t stream){
    typename CudaTraits<T>::Complex** h_ptrs = new typename CudaTraits<T>::Complex*[batch_size];
    for(int i=0; i<batch_size; i++){
        h_ptrs[i] = d_data + i*N*N;
    }
    cudaMemcpyAsync(d_ptrs, h_ptrs, batch_size*sizeof(typename CudaTraits<T>::Complex*), cudaMemcpyHostToDevice, stream);
    delete[] h_ptrs;
}


template<typename T>
int TensorEntropyEstimator<T>::computeEntropy(typename CudaTraits<T>::Complex* d_kraus_input, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, typename CudaTraits<T>::Real eps) {
    // Implementation of entropy computation using CUDA kernels
    
    // Alias for clarity
    const int d = kraus_number;
    const int N = kraus_out_dimension;
    
    // Determine batching parameters based on available memory
    // We compute A_ij matrices in batches: for i ∈ [i0, i0+num_i_per_batch), j ∈ [0, d)
    // This gives num_i_per_batch * d matrices per batch
    // Each matrix is N×N complex numbers
    size_t bytes_per_matrix = N * N * sizeof(typename CudaTraits<T>::Complex);
    size_t target_batch_bytes = 2ULL * 1024 * 1024 * 1024; // 2 GB target per batch
    int max_matrices_per_batch = target_batch_bytes / bytes_per_matrix;
    int num_i_per_batch = std::max(1, std::min(8, max_matrices_per_batch / d)); // How many i values per batch
    
    const int num_streams = 1; // Use single stream for simplicity
    
    std::cout << "Tensor entropy computation parameters:" << std::endl;
    std::cout << "  N = " << N << ", d = " << d << std::endl;
    std::cout << "  Number of 'i' values per batch: " << num_i_per_batch << std::endl;
    std::cout << "  Matrices per batch: " << num_i_per_batch * d << " (A matrices) + " << num_i_per_batch * d << " (B matrices)" << std::endl;
    std::cout << "  Memory per batch: " << (2.0 * num_i_per_batch * d * bytes_per_matrix) / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << "  Total batches: " << (d + num_i_per_batch - 1) / num_i_per_batch << std::endl;

    // Allocate device memory for Kraus operators and copy over
    typename CudaTraits<T>::Complex* d_kraus;
    CUDA_MALLOC_CHECK(d_kraus, sizeof(typename CudaTraits<T>::Complex)*d*kraus_in_dimension*N, "d_kraus");
    CUDA_MEMCPY_CHECK(d_kraus, d_kraus_input, sizeof(typename CudaTraits<T>::Complex)*d*kraus_in_dimension*N, cudaMemcpyHostToDevice, "d_kraus");

    // Create cuBLAS handles and CUDA streams for parallel execution
    cublasHandle_t handles[num_streams];
    cudaStream_t streams[num_streams];
    for(int s=0; s<num_streams; s++){
        cublasCreate(&handles[s]);
        cudaStreamCreate(&streams[s]);
        cublasSetStream(handles[s], streams[s]);
    }

    // Allocate temporary memory for A_batch and B_batch (per stream to allow overlap)
    // These hold num_i_per_batch * d matrices of size N×N each
    typename CudaTraits<T>::Complex* d_Abatch[num_streams];
    typename CudaTraits<T>::Complex* d_Bbatch[num_streams];
    typename CudaTraits<T>::Complex** d_Aptrs[num_streams];
    typename CudaTraits<T>::Complex** d_Bptrs[num_streams];
    
    for(int s=0; s<num_streams; s++){
        CUDA_MALLOC_CHECK(d_Abatch[s], sizeof(typename CudaTraits<T>::Complex)*num_i_per_batch*d*N*N, "d_Abatch");
        CUDA_MALLOC_CHECK(d_Bbatch[s], sizeof(typename CudaTraits<T>::Complex)*num_i_per_batch*d*N*N, "d_Bbatch");
        CUDA_MALLOC_CHECK(d_Aptrs[s], num_i_per_batch*d*sizeof(typename CudaTraits<T>::Complex*), "d_Aptrs");
        CUDA_MALLOC_CHECK(d_Bptrs[s], num_i_per_batch*d*sizeof(typename CudaTraits<T>::Complex*), "d_Bptrs");
    }

    // Allocate output M (d^2 x d^2 matrix)
    typename CudaTraits<T>::Complex* d_M;
    CUDA_MALLOC_CHECK(d_M, sizeof(typename CudaTraits<T>::Complex)*d*d*d*d, "d_M");
    
    // Loop over batches of A_ij (outer loop for i)
    int total_batches = (d + num_i_per_batch - 1) / num_i_per_batch;
    int current_batch = 0;
    for(int i0=0; i0<d; i0+=num_i_per_batch){
        current_batch++;
        int num_i_this_batch = std::min(num_i_per_batch, d-i0);
        int stream_id = (i0/num_i_per_batch) % num_streams;
        
        std::cout << "\n=== Processing outer batch " << current_batch << "/" << total_batches 
                  << " (i=" << i0 << " to " << (i0+num_i_this_batch-1) << ") ===" << std::endl;
        
        int idx = 0;
        for(int i=i0; i<i0+num_i_this_batch; i++){
            for(int j=0; j<d; j++){
                // Compute A_ij = K_i * K_j^* -> store in d_Abatch[stream_id][idx]
                const typename CudaTraits<T>::Complex* Ki = d_kraus + i*kraus_in_dimension*N;
                const typename CudaTraits<T>::Complex* Kj = d_kraus + j*kraus_in_dimension*N;
                typename CudaTraits<T>::Complex alpha = CudaTraits<T>::make_complex_h(1.0, 0.0);
                typename CudaTraits<T>::Complex beta  = CudaTraits<T>::make_complex_h(0.0, 0.0);

                // Compute K_i * K_j^H (conjugate transpose)
                // Output is N x N matrix (since K is N x kraus_in_dimension)
                CudaTraits<T>::gemm3m(handles[stream_id], 
                            CUBLAS_OP_N,           // transA: no transpose for Ki
                            CUBLAS_OP_C,           // transB: conjugate transpose for Kj
                            N,                     // m: rows of output (rows of Ki)
                            N,                     // n: cols of output (rows of Kj after transpose)
                            kraus_in_dimension,    // k: inner dimension
                            &alpha, 
                            Ki, N,                 // A: Ki is N x kraus_in_dimension, lda = N
                            Kj, N,                 // B: Kj is N x kraus_in_dimension, ldb = N
                            &beta, 
                            d_Abatch[stream_id] + idx * N * N, N  // C: output is N x N, ldc = N
                        );
                idx++;
            }
        }
        
        std::cout << "  Computed " << (num_i_this_batch * d) << " A_ij matrices" << std::endl;
        
        // Create pointer array for A batch (async with stream)
        create_device_ptr_array_async<T>(d_Abatch[stream_id], d_Aptrs[stream_id], num_i_this_batch*d, N, streams[stream_id]);

        // Loop over batches of B_kl (inner loop for k)
        int inner_batches = (d + num_i_per_batch - 1) / num_i_per_batch;
        int current_inner_batch = 0;
        for(int k0=0; k0<d; k0+=num_i_per_batch){
            current_inner_batch++;
            int num_k_this_batch = std::min(num_i_per_batch, d-k0);
            
            std::cout << "  Inner batch " << current_inner_batch << "/" << inner_batches 
                      << " (k=" << k0 << " to " << (k0+num_k_this_batch-1) << ")" << std::endl;
            idx = 0;
            
            for(int k=k0; k<k0+num_k_this_batch; k++){
                for(int l=0; l<d; l++){
                    // Compute B_kl = K_l * K_k^* -> store in d_Bbatch[stream_id][idx]
                    const typename CudaTraits<T>::Complex* Kl = d_kraus + l*kraus_in_dimension*N;
                    const typename CudaTraits<T>::Complex* Kk = d_kraus + k*kraus_in_dimension*N;
                    typename CudaTraits<T>::Complex alpha = CudaTraits<T>::make_complex_h(1.0, 0.0);
                    typename CudaTraits<T>::Complex beta  = CudaTraits<T>::make_complex_h(0.0, 0.0);

                    // Compute K_l * K_k^H (conjugate transpose)
                    CudaTraits<T>::gemm3m(handles[stream_id], 
                                CUBLAS_OP_N,           // transA: no transpose for Kl
                                CUBLAS_OP_C,           // transB: conjugate transpose for Kk
                                N,                     // m: rows of output
                                N,                     // n: cols of output
                                kraus_in_dimension,    // k: inner dimension
                                &alpha,
                                Kl, N,                 // A: Kl is N x kraus_in_dimension, lda = N
                                Kk, N,                 // B: Kk is N x kraus_in_dimension, ldb = N
                                &beta,
                                d_Bbatch[stream_id] + idx*N*N, N  // C: output is N x N, ldc = N
                            );
                    idx++;
                }
            }
            
            std::cout << "    Computed " << (num_k_this_batch * d) << " B_kl matrices" << std::endl;
            
            // Create pointer array for B batch (async with stream)
            create_device_ptr_array_async<T>(d_Bbatch[stream_id], d_Bptrs[stream_id], num_k_this_batch*d, N, streams[stream_id]);

            // Allocate traces for this batch
            typename CudaTraits<T>::Complex* d_traces;
            CUDA_MALLOC_CHECK(d_traces, sizeof(typename CudaTraits<T>::Complex)*(num_i_this_batch*d)*(num_k_this_batch*d), "d_traces");

            std::cout << "    Computing " << (num_i_this_batch*d) << " x " << (num_k_this_batch*d) 
                      << " = " << ((num_i_this_batch*d)*(num_k_this_batch*d)) << " traces..." << std::endl;

            // Launch kernel: grid=num_i_this_batch*d, block=num_k_this_batch*d
            // Each block computes one trace for a pair (A_ij, B_kl)
            compute_traces_kernel<T><<<num_i_this_batch*d, num_k_this_batch*d, 0, streams[stream_id]>>>(
                d_Aptrs[stream_id], d_Bptrs[stream_id], d_traces, N, num_i_this_batch*d, num_k_this_batch*d);

            std::cout << "    Populating M matrix..." << std::endl;

            // Populate M matrix with normalized traces
            // Use 2D grid to handle all (num_i_this_batch*d) x (num_k_this_batch*d) entries
            dim3 block_dim(16, 16);
            dim3 grid_dim((num_i_this_batch*d + 15)/16, (num_k_this_batch*d + 15)/16);
            populate_M_kernel<T><<<grid_dim, block_dim, 0, streams[stream_id]>>>(
                d_traces, d_M, i0, k0, num_i_this_batch, num_k_this_batch, d, N);
            
            // Free traces (consider reusing this allocation)
            cudaFree(d_traces);
        }
    }

    // Synchronize all streams before cleanup
    for(int s=0; s<num_streams; s++){
        cudaStreamSynchronize(streams[s]);
    }
    
    std::cout << "\nAll batches completed. Total M entries computed: " << (d*d*d*d) << std::endl;

    // =========================================================================
    // STEP 2: Apply epsilon shifting to M to get M_eps
    // M_eps = (1 - eps) * M + (eps / (d*d)) * I
    // =========================================================================
    
    // Create a kernel to apply the shift, or do it on CPU
    // Since d is typically small, copy M to host and work on CPU
    typename CudaTraits<T>::Complex* h_M = new typename CudaTraits<T>::Complex[d*d*d*d];
    CUDA_MEMCPY_CHECK(h_M, d_M, sizeof(typename CudaTraits<T>::Complex)*d*d*d*d, cudaMemcpyDeviceToHost, "h_M");
    
    // Apply shifting: M_eps = (1-eps)*M + (eps/(d^2))*I
    typename CudaTraits<T>::Real one_minus_eps = 1.0 - eps;
    typename CudaTraits<T>::Real eps_over_d2 = eps / static_cast<typename CudaTraits<T>::Real>(d*d);
    
    for(int i=0; i<d*d*d*d; i++){
        if constexpr (std::is_same_v<T, double>) {
            h_M[i].x *= one_minus_eps;
            h_M[i].y *= one_minus_eps;
        } else {
            h_M[i].x *= one_minus_eps;
            h_M[i].y *= one_minus_eps;
        }
    }
    
    // Add eps/(d^2) to diagonal
    for(int i=0; i<d*d; i++){
        if constexpr (std::is_same_v<T, double>) {
            h_M[i + i*d*d].x += eps_over_d2;
        } else {
            h_M[i + i*d*d].x += eps_over_d2;
        }
    }
    
    // =========================================================================
    // STEP 3: Diagonalize M_eps to get eigenvalues
    // =========================================================================
    
    // Use cuSOLVER to diagonalize the matrix on GPU
    // Copy shifted M back to device
    CUDA_MEMCPY_CHECK(d_M, h_M, sizeof(typename CudaTraits<T>::Complex)*d*d*d*d, cudaMemcpyHostToDevice, "d_M_shifted");
    
    // Create cuSOLVER handle
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    
    // Allocate eigenvalues array
    typename CudaTraits<T>::Real* d_eigenvalues;
    CUDA_MALLOC_CHECK(d_eigenvalues, sizeof(typename CudaTraits<T>::Real)*d*d, "d_eigenvalues");
    
    // Query workspace size
    int lwork;
    cusolverStatus_t cusolver_status = CudaTraits<T>::gesvd_buffer(
        cusolverH, d*d, d*d, &lwork
    );
    
    if(cusolver_status != CUSOLVER_STATUS_SUCCESS){
        std::cerr << "Error querying cuSOLVER buffer size" << std::endl;
        delete[] h_M;
        cusolverDnDestroy(cusolverH);
        cudaFree(d_eigenvalues);
        cudaFree(d_kraus);
        for(int s=0; s<num_streams; s++){
            cudaFree(d_Abatch[s]);
            cudaFree(d_Bbatch[s]);
            cudaFree(d_Aptrs[s]);
            cudaFree(d_Bptrs[s]);
            cublasDestroy(handles[s]);
            cudaStreamDestroy(streams[s]);
        }
        cudaFree(d_M);
        return -1;
    }
    
    // Allocate workspace
    typename CudaTraits<T>::Complex* d_work;
    CUDA_MALLOC_CHECK(d_work, sizeof(typename CudaTraits<T>::Complex)*lwork, "d_work");
    
    typename CudaTraits<T>::Real* d_rwork;
    CUDA_MALLOC_CHECK(d_rwork, sizeof(typename CudaTraits<T>::Real)*(d*d-1), "d_rwork");
    
    int* d_info;
    CUDA_MALLOC_CHECK(d_info, sizeof(int), "d_info");
    
    // Perform SVD (which gives eigenvalues for Hermitian matrix)
    // For Hermitian matrices, we use zheevd or can use gesvd
    // Since we have gesvd in traits, use that (singular values = sqrt of eigenvalues for positive semidefinite)
    typename CudaTraits<T>::Complex* d_U;
    typename CudaTraits<T>::Complex* d_VT;
    CUDA_MALLOC_CHECK(d_U, sizeof(typename CudaTraits<T>::Complex)*d*d*d*d, "d_U");
    CUDA_MALLOC_CHECK(d_VT, sizeof(typename CudaTraits<T>::Complex)*d*d*d*d, "d_VT");
    
    cusolver_status = CudaTraits<T>::gesvd(
        cusolverH, 'N', 'N',
        d*d, d*d,
        d_M, d*d,
        d_eigenvalues,
        d_U, d*d,
        d_VT, d*d,
        d_work, lwork,
        d_rwork,
        d_info
    );
    
    if(cusolver_status != CUSOLVER_STATUS_SUCCESS){
        std::cerr << "Error in cuSOLVER gesvd" << std::endl;
        delete[] h_M;
        cudaFree(d_work);
        cudaFree(d_rwork);
        cudaFree(d_info);
        cudaFree(d_U);
        cudaFree(d_VT);
        cusolverDnDestroy(cusolverH);
        cudaFree(d_eigenvalues);
        cudaFree(d_kraus);
        for(int s=0; s<num_streams; s++){
            cudaFree(d_Abatch[s]);
            cudaFree(d_Bbatch[s]);
            cudaFree(d_Aptrs[s]);
            cudaFree(d_Bptrs[s]);
            cublasDestroy(handles[s]);
            cudaStreamDestroy(streams[s]);
        }
        cudaFree(d_M);
        return -1;
    }
    
    // =========================================================================
    // STEP 4: Compute entropy = -Tr[M_eps * log(M_eps)] = -sum(lambda_i * log(lambda_i))
    // =========================================================================
    
    // Copy eigenvalues to host
    typename CudaTraits<T>::Real* h_eigenvalues = new typename CudaTraits<T>::Real[d*d];
    CUDA_MEMCPY_CHECK(h_eigenvalues, d_eigenvalues, sizeof(typename CudaTraits<T>::Real)*d*d, cudaMemcpyDeviceToHost, "h_eigenvalues");
    
    // Randomly print the sum of eigenvalues for logging
    typename CudaTraits<T>::Real eigen_sum = static_cast<typename CudaTraits<T>::Real>(0);
    for(int i=0; i<d*d; i++){
        eigen_sum += h_eigenvalues[i];
    }
    std::cout << "Sum of eigenvalues (should be 1): " << eigen_sum << std::endl;

    // Compute epsilon_entropy = -sum(lambda_i * log(lambda_i))
    this->epsilon_entropy = static_cast<typename CudaTraits<T>::Real>(0);
    for(int i=0; i<d*d; i++){
        typename CudaTraits<T>::Real lambda = h_eigenvalues[i];
        if(lambda > 1e-15){ // Avoid log(0)
            this->epsilon_entropy -= lambda * std::log(lambda);
        }
    }
    
    // =========================================================================
    // STEP 5: Estimate error and unshifted entropy
    // 
    // We have the inequality:
    //   S(ρ) ≤ S(ρ_eps) ≤ S(ρ) + ε·log(d²) + H(ε)
    // 
    // where H(ε) is the binary entropy: H(ε) = -ε·log(ε) - (1-ε)·log(1-ε)
    //
    // Estimated entropy:
    //   estimated_entropy = [S(ρ_eps) - ε·log(d²)]/(1-ε) - H(ε)/(2(1-ε))
    //
    // Maximum error:
    //   estimated_error = H(ε)/(2(1-ε))
    // =========================================================================
    
    typename CudaTraits<T>::Real bin_entropy;
    if(eps > 1e-15 && eps < 1.0 - 1e-15){
        bin_entropy = -eps * std::log(eps) - (1.0 - eps) * std::log(1.0 - eps);
    } else {
        bin_entropy = static_cast<typename CudaTraits<T>::Real>(0);
    }
    
    // Compute epsilon correction term: ε·log(d²)
    typename CudaTraits<T>::Real eps_log_dim = eps * std::log(static_cast<typename CudaTraits<T>::Real>(d*d));
    
    typename CudaTraits<T>::Real entropy_correction;
    
    if(eps < 1.0 - 1e-15){
        typename CudaTraits<T>::Real one_minus_eps_val = 1.0 - eps;
        // [S(ρ_eps) - ε·log(d²)]/(1-ε)
        typename CudaTraits<T>::Real corrected_entropy = (this->epsilon_entropy - eps_log_dim) / one_minus_eps_val;
        
        // H(ε)/(2(1-ε))
        entropy_correction = bin_entropy / (2.0 * one_minus_eps_val);
        
        // Estimated entropy
        this->estimated_entropy = corrected_entropy - entropy_correction;
    } else {
        // Fallback for eps ≈ 1 (avoid division by zero)
        this->estimated_entropy = this->epsilon_entropy;
        entropy_correction = static_cast<typename CudaTraits<T>::Real>(0);
    }
    
    // Maximum error estimate
    this->estimated_error = entropy_correction;
    
    this->epsilon = eps;
    
    // Clean up
    delete[] h_M;
    delete[] h_eigenvalues;
    cudaFree(d_work);
    cudaFree(d_rwork);
    cudaFree(d_info);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_eigenvalues);
    cusolverDnDestroy(cusolverH);
    
    cudaFree(d_kraus);
    for(int s=0; s<num_streams; s++){
        cudaFree(d_Abatch[s]);
        cudaFree(d_Bbatch[s]);
        cudaFree(d_Aptrs[s]);
        cudaFree(d_Bptrs[s]);
        cublasDestroy(handles[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_M);

    return 0; // Return 0 on success
}

template<typename T>
typename CudaTraits<T>::Real TensorEntropyEstimator<T>::getEpsilonEntropy() {
    return this->epsilon_entropy;
}

template<typename T>
typename CudaTraits<T>::Real TensorEntropyEstimator<T>::getEstimatedEntropy() {
    return this->estimated_entropy;
}

template<typename T>
typename CudaTraits<T>::Real TensorEntropyEstimator<T>::getEstimatedError() {
    return this->estimated_error;
}

// Explicit template instantiation for float and double
template class TensorEntropyEstimator<float>;
template class TensorEntropyEstimator<double>;