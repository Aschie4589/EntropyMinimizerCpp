#include "common_includes.h"
#include "core/strategies/high_memory_strategy.h"
#include "core/cuda_minimizer.h"
#include "core/cuda_kernels.h"
#include "core/cuda_traits.h"
#include "config/defs.h"      
#include <nvtx3/nvtx3.hpp>

template<typename T>
cudaError_t HighMemoryStrategy<T>::initialize(CudaMinimizer<T>* minimizer) {
    // Allocate the memory necessary for scratch calculations using this strategy. Do it once and only free it when strategy is destroyed.
    // Vectors after first SVD, there are d of them and they have size M
    d_vecs_1 = nullptr;
    CUDA_MALLOC_CHECK(d_vecs_1, minimizer->M * minimizer->d * sizeof(typename CudaTraits<T>::Complex), "d_vecs_1");
    // Vectors after second SVD, there are d * d of them and they have size N
    d_vecs_2 = nullptr;
    CUDA_MALLOC_CHECK(d_vecs_2, minimizer->N * minimizer->d * minimizer->d * sizeof(typename CudaTraits<T>::Complex), "d_vecs_2");
    // Singular values of d_vecs_1
    d_sv_1 = nullptr;
    CUDA_MALLOC_CHECK(d_sv_1, minimizer->d * sizeof(typename CudaTraits<T>::Real), "d_sv_1");
    // Singular values of d_vecs_2
    d_sv_2 = nullptr;
    CUDA_MALLOC_CHECK(d_sv_2, minimizer->d * minimizer->d * sizeof(typename CudaTraits<T>::Real), "d_sv_2");
    // Query the work size for the SVD operations, and allocate d_scratch accordingly
    work_size_1 = 0;
    CudaTraits<T>::gesvd_buffer(minimizer->cusolver_handle, minimizer->M, minimizer->d, &work_size_1);
    work_size_2 = 0;
    CudaTraits<T>::gesvd_buffer(minimizer->cusolver_handle, minimizer->N, minimizer->d * minimizer->d, &work_size_2);
    // Actual scratch space from work sizes
    d_scratch = nullptr;
    CUDA_MALLOC_CHECK(d_scratch, std::max(work_size_1, work_size_2) * sizeof(typename CudaTraits<T>::Complex), "d_scratch");
    // Randomized SVD workspace
    Srand = nullptr; 
    Urand = nullptr; 
    Vrand = nullptr;
    CUDA_MALLOC_CHECK(Srand, sizeof(typename CudaTraits<T>::Real), "Srand for SVD");
    CUDA_MALLOC_CHECK(Urand, minimizer->N * sizeof(typename CudaTraits<T>::Complex), "Urand for SVD");
    CUDA_MALLOC_CHECK(Vrand, minimizer->d * minimizer->d * sizeof(typename CudaTraits<T>::Complex), "Vrand for SVD");
    // This is the high memory strategy so allocate one extra d_kraus_trans of size N*N*d
    d_kraus_1 = nullptr;
    d_kraus_2 = nullptr;
    CUDA_MALLOC_CHECK(d_kraus_1, minimizer->N * minimizer->N * minimizer->d * sizeof(typename CudaTraits<T>::Complex), "d_kraus_1");
    CUDA_MALLOC_CHECK(d_kraus_2, minimizer->N * minimizer->N * minimizer->d * sizeof(typename CudaTraits<T>::Complex), "d_kraus_2");
    
    typename CudaTraits<T>::Complex one = CudaTraits<T>::make_complex_h(1.0, 0.0);
    typename CudaTraits<T>::Complex zero = CudaTraits<T>::make_complex_h(0.0, 0.0);
    
    // Build d_kraus_1, which should be the vertical stack of the K_i.
    {

        nvtx3::scoped_range r1{"Build kraus_1. Adjoint inner Kraus"};
        // One operation per stream
        for (int i=0; i < minimizer->num_streams; i++){
            // Each stream transposes one of the K_i
            for (int j = i; j < minimizer->d; j += minimizer->num_streams) {
                // you should transpose the kraus matrix at position i of d_kraus into d_kraus_1
                /*
                leading dimension = stride between consecutive cols (in col major format) = number of rows
                Kraus[i] is MxN, so Op[Kraus[i]] is NxM. so m=N and n=M.
                */
                CudaTraits<T>::geam(minimizer->blas_handles[i], // Cublas handle for the stream
                        CUBLAS_OP_C,                                                // Conjugate transpose matrix A
                        CUBLAS_OP_N,                                                // No transpose matrix B
                        minimizer->N,                                               // Rows of C. Since A = Kraus[i] is MxN, A^H is NxM, and C has N columns
                        minimizer->M,                                               // Columns of C. M
                        &one,                                                       // alpha in C = alpha * op(A) + beta * op(B)
                        minimizer->d_kraus + j * minimizer->N * minimizer->M,       // A, which is the jth kraus operator
                        minimizer->M,                                               // leading dimension of a = number of rows
                        &zero,                                                      // beta
                        nullptr,                                                    // B, does not have to be valid 
                        minimizer->N,                                               // leading dimension of B
                        d_kraus_2 + j * minimizer->M * minimizer->N,            // C, the output matrix
                        minimizer->N);                                              // leading dimension of C, in this case N since C is NxM
            } // d_kraus_2 has nested dimensions (d, N, M) in col major order. I.e. it can be interpreted as a BIG NxdM matrix in col major order.
        }
        // Synchronize streams to ensure all operations are complete
        cudaDeviceSynchronize();
    }  
    
    // Now hermitian conjugate it back it as a MxdN matrix. I.e. : minimizer->d_kraus_2 = d_kraus_1 ^H as a whole matrix.
    {
        nvtx3::scoped_range r1{"Build kraus_1. Adjoint outer Kraus"};
        // Only one stream necessary, since we consider the kraus as a column major matrix of size N x (M*d).
        // Since Kraus is Nx(M*d), Op[Kraus] is (M*d)xN. So m = M*d and n = N.
        CudaTraits<T>::geam(minimizer->blas_handles[0],  // Cublas handle
                    CUBLAS_OP_C,                                    // Conjugate transpose matrix A
                    CUBLAS_OP_N,                                    // No transpose matrix B
                    minimizer->M * minimizer->d,                    // Rows of C. We want C to be a (M*d) x N matrix, so M*d.
                    minimizer->N,                                   // Columns of C. N.
                    &one,                                           // alpha
                    d_kraus_2,                                      // A
                    minimizer->N,                                   // leading dimension of A = number of rows of A, which is Nx(M*d)
                    &zero,                                          // beta
                    nullptr,                                        // B, does not have to be valid
                    minimizer->M * minimizer->d,                    // leading dimension of B
                    d_kraus_1,                             // C
                    minimizer->M * minimizer->d);                   // leading dimension of C
        cudaDeviceSynchronize();
    } // d_kraus_1 should be treated as a (N*d)xN matrix, which can be multiplied to the given vector to produce the full matrix X.



    // Now build d_kraus_2. Since we look for K_i^*, we can use as source the original d_kraus from the minimizer, and only adjoin that as a big Mx(N*d) matrix.
    {
        nvtx3::scoped_range r1{"Build kraus_2. Adjoint original Kraus"};
        // Only one stream necessary, since we consider the kraus as a column major matrix of size M x (N*d).
        // Since Kraus is Mx(N*d), Op[Kraus] is (N*d)xM. So m = N*d and n = M.
        CudaTraits<T>::geam(minimizer->blas_handles[0],  // Cublas handle
                    CUBLAS_OP_C,                                    // Conjugate transpose matrix A
                    CUBLAS_OP_N,                                    // No transpose matrix B
                    minimizer->N * minimizer->d,                    // Rows of C. We want C to be a (N*d) x M matrix, so N*d.
                    minimizer->M,                                   // Columns of C. M.
                    &one,                                           // alpha
                    minimizer->d_kraus,                             // A
                    minimizer->M,                                   // leading dimension of A = number of rows of A, which is Mx(N*d)
                    &zero,                                          // beta
                    nullptr,                                        // B, does not have to be valid
                    minimizer->N * minimizer->d,                    // leading dimension of B
                    d_kraus_2,                                      // C
                    minimizer->N * minimizer->d);                   // leading dimension of C
        cudaDeviceSynchronize();
    } // d_kraus_2 should be treated as a (N*d)xM matrix, which can be multiplied to the given vector to produce the full matrix X.
    return cudaSuccess;
}

template<typename T>
size_t HighMemoryStrategy<T>::getMemoryRequired(CudaMinimizer<T>* minimizer) const {
    // Return the amount of memory required by this strategy in bytes
    size_t total_memory = 0;
    total_memory += minimizer->M * minimizer->d * sizeof(typename CudaTraits<T>::Complex); // d_vecs_1
    total_memory += minimizer->N * minimizer->d * minimizer->d * sizeof(typename CudaTraits<T>::Complex); // d_vecs_2
    total_memory += minimizer->d * sizeof(typename CudaTraits<T>::Real); // d_sv_1
    total_memory += minimizer->d * minimizer->d * sizeof(typename CudaTraits<T>::Real); // d_sv_2
    total_memory += std::max(work_size_1, work_size_2) * sizeof(typename CudaTraits<T>::Complex); // d_scratch
    total_memory += sizeof(typename CudaTraits<T>::Real); // Srand
    total_memory += minimizer->N * sizeof(typename CudaTraits<T>::Complex); // Urand
    total_memory += minimizer->d * minimizer->d * sizeof(typename CudaTraits<T>::Complex); // Vrand
    total_memory += 2 * minimizer->N * minimizer->N * minimizer->d * sizeof(typename CudaTraits<T>::Complex); // d_kraus_1 and 2
    return total_memory;
}

template<typename T>
cudaError_t HighMemoryStrategy<T>::stepAlgorithm(CudaMinimizer<T>* minimizer) {
    /*
    Only responsible for stepping through and updating the d_vec
    */
    cudaError_t err;
    err = step_1(minimizer); // Step 1: compute {K_i |d_vec>}_i and perform SVD to obtain {lambda_i} and {psi_i}. Multiply the psi_i by (log((1-eps)lambda_i^2 + eps) - log(eps)). At the end of this stage, the lambda_i are the square roots of Phi(rho).
    if (err != cudaSuccess) {
        std::cerr << "Error in step 1: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if step 1 failed
    }
    err = step_2(minimizer); // Step 2: compute {K_j |psi_i>}_ij and perform SVD to obtain {mu_ij} and {phi_ij}. The phi_ij corresponding to the largest mu_ij is the new vector state. Save it in d_vec.
    if (err != cudaSuccess) {
        std::cerr << "Error in step 2: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if step 2 failed
    }
    return cudaSuccess;
}

template<typename T>
cudaError_t HighMemoryStrategy<T>::step(CudaMinimizer<T>* minimizer) {
    /*
    Steps the algorithm and updates the entropy value in the minimizer.
    */
    cudaError_t err;
    err = step_1(minimizer); // Step 1: compute {K_i |d_vec>}_i and perform SVD to obtain {lambda_i} and {psi_i}. Multiply the psi_i by (log((1-eps)lambda_i^2 + eps) - log(eps)). At the end of this stage, the lambda_i are the square roots of Phi(rho).
    if (err != cudaSuccess) {
        std::cerr << "Error in step 1: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if step 1 failed
    }

    err = calculateEpsilonEntropy(minimizer, true);
    if (err != cudaSuccess) {
        std::cerr << "Error calculating epsilon entropy: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if the entropy calculation failed
    }

    err = step_2(minimizer); // Step 2: compute {K_j |psi_i>}_ij and perform SVD to obtain {mu_ij} and {phi_ij}. The phi_ij corresponding to the largest mu_ij is the new vector state. Save it in d_vec.
    if (err != cudaSuccess) {
        std::cerr << "Error in step 2: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if step 2 failed
    }
    return cudaSuccess;

}


template<typename T>
cudaError_t HighMemoryStrategy<T>::step_1(CudaMinimizer<T>* minimizer) {
    /*
    Perform the first step in the algorithm. That is, do the following:
    - Compute {K_i |d_vec>}_i, where {K_i}_i are the kraus operators. Save the result in d_vecs_1.
    - Treat d_vecs_1 as a matrix in column major order and perform SVD on it to obtain {lambda_i}_i and {psi_i}_i.
    - Multiply the psi_i by (log((1-eps)lambda_i^2 + eps) - log(eps)). Save the result in d_vecs_2.

    Returns:
        - cudaSuccess if the step was performed successfully.

    Note:
    - At the end of step 1, the d_sv_1 will contain the lambda_i which are used for entropy calculation
    - d_vecs_1 will contain the correct vectors (the rescaled right singular vectors of {K_i |d_vec>}_i).
    */

    // Step 1: Compute {K_i |d_vec>}_i and save the result in d_vecs_1
    typename CudaTraits<T>::Complex one = CudaTraits<T>::make_complex_h(1.0, 0.0);
    typename CudaTraits<T>::Complex zero = CudaTraits<T>::make_complex_h(0.0, 0.0);
    {
    nvtx3::scoped_range r1{"Step 1: K_i |d_vec> multiplication"};
    // Directly multiply minimizer->d_kraus (which is now the adjoint of the kraus operators) by d_vec to get d_vecs_1
    // minimizer->d_kraus is (M*d)xN, d_vec is Nx1. The output is X in col major order.
    CudaTraits<T>::gemv(minimizer->blas_handles[0], // cublas handle
        CUBLAS_OP_N,                                // no transpose for A
        minimizer->M * minimizer->d,                // rows of A. A is (M*d)xN since we multiply it to a Nx1 vector
        minimizer->N,                               // columns of A
        &one,                                       // alpha
        d_kraus_1,                         // A
        minimizer->M * minimizer->d,                // leading dimension of A = number of rows of A
        minimizer->d_vec,                           // x
        1,                                          // incx (stride between elements of the vector)
        &zero,                                      // beta
        d_vecs_1,                                   // output
        1);                                         // stride between elements of d_vecs_1
    
    // Synchronize streams to ensure all operations are complete
    cudaDeviceSynchronize();
    }
    {
    nvtx3::scoped_range r2{"Step 1: SVD of K_i |v>_i"};
    // Step 2: find the SVD of X = {K_i |d_vec>}_i. X is Mxd
    // So X = U S V^H means that V^H has d columns, and U has M rows.
    //Don't need to have more streams as its only one decomposition.
    // Scratch space is already allocated

    int* devinfo = nullptr;
    CUDA_MALLOC_CHECK(devinfo, sizeof(int), "devinfo for SVD");

    cusolverStatus_t st = CudaTraits<T>::gesvd(minimizer->cusolver_handle, 
                     'O', // Overwrite input matrix with left singular vectors (U in USV^H)
                     'N', // Don't compute right singular vectors (V^H in USV^H)
                     minimizer->M, minimizer->d, 
                     d_vecs_1, minimizer->M, 
                     d_sv_1, 
                     nullptr, minimizer->M, // U matrix not needed. Still, ldu >= min(M, d).
                     nullptr, 1, // VT matrix, not needed. ldvt = 1 since we don't compute it.
                     d_scratch, work_size_1, // Scratch space
                     nullptr, devinfo); // 'rwork can be a null pointer if the user does not want info about superdiagonal'

    cudaDeviceSynchronize();
    
    cudaFree(devinfo); // Free device info  

    if (st != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error in SVD computation: " << st << std::endl;
        return cudaErrorUnknown; // Return error
    }
    }

    // At this point we can rescale the vectors in Y=d_sv_1 by the right amounts (so we have log(Phi_eps(rho))-log(eps)*1 = YY^H)
    int total_elems = minimizer->d * minimizer->M;
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;
    {
    nvtx3::scoped_range r3{"Step 1: Rescale vectors in d_vecs_1"};
    rescale_vecs_1<T><<<blocks, threads>>>(d_vecs_1, d_sv_1, minimizer->epsilon, minimizer->d, minimizer->M);
    cudaDeviceSynchronize();
    }
    return cudaGetLastError(); // Returns success if the kernel launched successfully
}

template<typename T>
cudaError_t HighMemoryStrategy<T>::step_2(CudaMinimizer<T>* minimizer) {
    /*
    Perform the second step in the algorithm. That is, do the following:
    - Compute {K_j^H |psi_i>}_ij, where {K_j}_j are the kraus operators and {psi_i}_i are the vectors obtained in step 1. Save the result in d_vecs_2.
    - Treat d_vecs_2 as a matrix in column major order and perform SVD on it to obtain {mu_ij}_ij and {phi_ij}_ij.
    - The phi_ij corresponding to the largest mu_ij is the new vector state. Save it in d_vec.
    
    Returns:
        - cudaSuccess if the step was performed successfully.
    Note:
    - At the end of step 2, d_vec will contain the new vector state.
    - The state of d_vecs_1 and d_sv_1 will not change. 
    - d_vecs_2 and d_sv_2 will be updated. 
    */
 
    // Step 1: Compute {K_i |d_vecs_j>}_ij and save the result in d_vecs_2
    typename CudaTraits<T>::Complex one = CudaTraits<T>::make_complex_h(1.0, 0.0);
    typename CudaTraits<T>::Complex zero = CudaTraits<T>::make_complex_h(0.0, 0.0);

    {
    nvtx3::scoped_range r{"Step 2: K_i^H |psi_j> multiplication"};
    // Again we can just multiply the d_kraus_trans directly to d_vecs_1!
    // minimizer->d_kraus_trans is (N*d)xM, d_vecs_1 is Mxd. The output is X2 of size (N*d)xd in col major order.
    /*
        template<> static constexpr inline cublasStatus_t (*CudaTraits<...>::gemm)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
    */
    CudaTraits<T>::gemm(minimizer->blas_handles[0], 
        CUBLAS_OP_N,                            // do not transpose A
        CUBLAS_OP_N,                            // do not transpose B
        minimizer->N*minimizer->d,              // number of rows of output matrix C.
        minimizer->d,                           // number of columns of C
        minimizer->M,                           // number of columns of A = number of rows of B = contraction dimension
        &one,                                   // alpha
        d_kraus_2,                             // A, which is (N*d)xM
        minimizer->N*minimizer->d,              // leading dimension of A
        d_vecs_1,                               // B, which is Mxd
        minimizer->M,                           // leading dimension of B
        &zero,                                  // beta
        d_vecs_2,                               // C
        minimizer->N*minimizer->d);             // leading dimension of C
    // Synchronize streams to ensure all operations are complete
    cudaDeviceSynchronize();
    }
    // Step 2: extract the largest left singular vector of d_vecs_2. This is the new state.
    // For now, compute full SVD of the N * (dxd) matrix d_vecs_2. Only store the left singular vectors. In reality only care about the first!

    {
    nvtx3::scoped_range r{"Step 2: SVD of K_j^H |psi_i>_ij"};

    // Use first the randomized method. If that fails, proceed with full svd (but use the one based on polar decomposition!)

    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    size_t workspace_device_bytes = 0;
    size_t workspace_host_bytes = 0;

    // Query buffer size
    cusolverStatus_t svd_status = cusolverDnXgesvdr_bufferSize (
        minimizer->cusolver_handle, // cusolver handle
        params, // cusolver parameters
        'S', // Compute the first k left singular vectors 
        'N', // Don't bother with the right singular vectors
        static_cast<int64_t>(minimizer->N), // Number of rows
        static_cast<int64_t>(minimizer->d * minimizer->d), // Number of columns (dxd)
        static_cast<int64_t>(1),  // k=1
        static_cast<int64_t>(6),  // oversampling (= larger subspace). Suggested: 2k, so let's say 3 = 3k.
        static_cast<int64_t>(8),  // number of iterations, n_iter. Suggested:2
        CudaTraits<T>::CUDA_C, // data type of A
        static_cast<void*>(d_vecs_2), // pointer to the input matrix A
        static_cast<int64_t>(minimizer->N), // leading dimension of A
        CudaTraits<T>::CUDA_R, // data type of Srand
        static_cast<void*>(Srand), // pointer to the output vector Srand
        CudaTraits<T>::CUDA_C, // data type of Urand
        static_cast<void*>(Urand), // pointer to the output matrix Urand
        static_cast<int64_t>(minimizer->N),
        CudaTraits<T>::CUDA_C, // Type of V
        static_cast<void*>(Vrand), 
        static_cast<int64_t>(minimizer->d * minimizer->d), // Leading dimension of V (1 since nullptr)
        CudaTraits<T>::CUDA_C, // Type of work
        &workspace_device_bytes, // Pointer to the workspace on device
        &workspace_host_bytes // Pointer to workspace on host 
    );
    if (svd_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying buffer size for SVD computation: " << svd_status << std::endl;
        return cudaErrorUnknown; // Return an error if the query failed
    }
    // Check for errors in the buffer size query
    if (workspace_device_bytes == 0 || workspace_host_bytes == 0) {
        std::cerr << "Error querying buffer size for SVD computation: workspace_device_bytes = " 
                  << workspace_device_bytes << ", workspace_host_bytes = " 
                  << workspace_host_bytes << std::endl;
        return cudaErrorUnknown; // Return an error if the buffer size is zero
    }

    // Allocate workspace on device
    if (workspace_device_bytes > std::max(work_size_1, work_size_2)*sizeof(typename CudaTraits<T>::Complex)){
        cudaFree(d_scratch); // Free the old scratch space
        d_scratch = nullptr; // Set to null to avoid dangling pointer
        CUDA_MALLOC_CHECK(d_scratch, workspace_device_bytes, "d_scratch for randomized SVD");
    }

    // Allocate host workspace
    void* h_scratch = nullptr;
    if (workspace_host_bytes > 0) {
        h_scratch = malloc(workspace_host_bytes);
        if (h_scratch == nullptr) {
            std::cerr << "Error allocating memory for host scratch space." << std::endl;
            return cudaErrorMemoryAllocation; // Return memory allocation error
        }
    }

    // get dev info 2
    // allocate
    // memory on device for devinfo
    int* devinfo_2 = nullptr;
    cudaMalloc((void**)&devinfo_2, sizeof(int)); // Device info for SVD
    if (devinfo_2 == nullptr) {
        std::cerr << "Error allocating memory for device info." << std::endl;
        if (h_scratch != nullptr) {
            free(h_scratch); // Free host scratch space if allocated
        }
        return cudaErrorMemoryAllocation; // Return memory allocation error
    }

    // Perform the SVD using the randomized method
    cusolverDnXgesvdr(
        minimizer->cusolver_handle, // cusolver handle
        params, // cusolver parameters
        'S', // Compute the first k left singular vectors 
        'N', // Don't bother with the right singular vectors
        static_cast<int64_t>(minimizer->N), // Number of rows
        static_cast<int64_t>(minimizer->d * minimizer->d), // Number of columns (dxd)
        static_cast<int64_t>(1),  // k=1
        static_cast<int64_t>(6), // oversampling (= larger subspace). Suggested: 2k, so let's say 3 = 3k.
        static_cast<int64_t>(15),  // number of iterations, n_iter. Suggested:2
        CudaTraits<T>::CUDA_C, // data type of A
        static_cast<void*>(d_vecs_2), // pointer to the input matrix A
        static_cast<int64_t>(minimizer->N), // leading dimension of A
        CudaTraits<T>::CUDA_R, // data type of Srand
        static_cast<void*>(Srand), // pointer to the output vector Srand
        CudaTraits<T>::CUDA_C, // data type of Urand
        static_cast<void*>(Urand), // pointer to the output matrix Urand
        static_cast<int64_t>(minimizer->N),
        CudaTraits<T>::CUDA_C, // Type of V
        static_cast<void*>(Vrand),   // V (nullptr)
        static_cast<int64_t>(minimizer->d * minimizer->d), // Leading dimension of V (1 since nullptr)
        CudaTraits<T>::CUDA_C, // Type of work
        static_cast<void*>(d_scratch),
        workspace_device_bytes,
        h_scratch,
        workspace_host_bytes,
        devinfo_2 // Pointer to device info
    );


    cudaDeviceSynchronize();
    cudaFree(devinfo_2); // Free device info

    if (svd_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error in SVD computation: " << svd_status << std::endl;
        if (h_scratch != nullptr) {
            free(h_scratch); // Free host scratch space if allocated
        }
        return cudaErrorUnknown; // Return error
    }

    // Free h_scratch after computation
    if (h_scratch != nullptr) {
        free(h_scratch);
    }

    }
    // Step 3: Update the d_vec state. (TODO: actually, I could avoid this copy...)
    {
    nvtx3::scoped_range r{"Step 2: Copy largest left singular vector to d_vec"};

    // Now copy the vector over
    CUDA_MEMCPY_CHECK(minimizer->d_vec, Urand, minimizer->N * sizeof(typename CudaTraits<T>::Complex), cudaMemcpyDeviceToDevice, "d_vec from Urand");

    }
    return cudaSuccess;
}

template<typename T>
cudaError_t HighMemoryStrategy<T>::cleanup(CudaMinimizer<T>* minimizer) {
    // Free all allocated memory
    if (d_vecs_1) {
        cudaFree(d_vecs_1);
        d_vecs_1 = nullptr;
    }
    if (d_vecs_2) {
        cudaFree(d_vecs_2);
        d_vecs_2 = nullptr;
    }
    if (d_sv_1) {
        cudaFree(d_sv_1);
        d_sv_1 = nullptr;
    }
    if (d_sv_2) {
        cudaFree(d_sv_2);
        d_sv_2 = nullptr;
    }
    if (d_scratch) {
        cudaFree(d_scratch);
        d_scratch = nullptr;
    }
    if (Srand) {
        cudaFree(Srand);
        Srand = nullptr;
    }
    if (Urand) {
        cudaFree(Urand);
        Urand = nullptr;
    }
    if (Vrand) {
        cudaFree(Vrand);
        Vrand = nullptr;
    }
    return cudaSuccess;
}


template<typename T>
cudaError_t HighMemoryStrategy<T>::calculateEpsilonEntropy(CudaMinimizer<T>* minimizer, bool force) {
    /*
        Calculates the von Neumann entropy of the output matrix after applying the epsilon channel.
        When in stage 1, the square roots of eigenvalues lamvda_i of Phi(rho) are saved in d_sv_1.
        The formula then is:

            S(Phi_e(rho)) = [- sum_{i<=d} ((1-eps) lambda_i^2 + eps) log((1-eps) lambda_i^2 + eps)] - (M - d) * eps * log(eps)
        
        Arguments:
            - minimizer: pointer to the CudaMinimizer instance.
            - force: if true, forces recomputation of step 1 to ensure d_sv_1 is up to date.

        Returns:
            - cudaSuccess if the entropy was calculated successfully.
        Note:
            - If the minimizer is not in stage 1 but has been initialized, step_1 is run to bring it to stage_1.
            - The entropy is stored in the member variable `entropy`.

    */
    if (force) {
        // Force recomputation of step 1 to ensure d_sv_1 is up to date
        cudaError_t err = step_1(minimizer);
        if (err != cudaSuccess) {
            std::cerr << "Error in step 1 while forcing entropy calculation: " << cudaGetErrorString(err) << std::endl;
            return err; // Return the error if step 1 failed
        }
    }

    // Now perform the computation of entropy. Since d is assumed small, it is best to just copy values to host and work on CPU.
    typename CudaTraits<T>::Real* h_sv_1 = new typename CudaTraits<T>::Real[minimizer->d];
    cudaError_t err_cpy = cudaMemcpy(h_sv_1, d_sv_1, minimizer->d * sizeof(typename CudaTraits<T>::Real), cudaMemcpyDeviceToHost);
    if (err_cpy != cudaSuccess) {
        std::cerr << "Error copying eigenvalues from device to host: " << cudaGetErrorString(err_cpy) << std::endl;
        delete[] h_sv_1; // Free the host memory before returning
        return err_cpy; // Return the error if the copy failed
    }
    minimizer->entropy = static_cast<typename CudaTraits<T>::Real>(0.0);
    for (int i = 0; i < minimizer->d; ++i) {
        typename CudaTraits<T>::Real lambda_i = h_sv_1[i];
        typename CudaTraits<T>::Real val = (1.0 - minimizer->epsilon) * lambda_i * lambda_i + minimizer->epsilon/minimizer->M; // epsilon is assumed small but positive
        minimizer->entropy -= val * log(val);
    }
    minimizer->entropy -= static_cast<typename CudaTraits<T>::Real>(minimizer->M - minimizer->d)/minimizer->M * minimizer->epsilon * (log(minimizer->epsilon) - log(minimizer->M)); // Add the contribution from the zero eigenvalues

    delete[] h_sv_1; // Free the host memory
    return cudaSuccess; // Return success if the entropy was calculated successfully    
}




// Explicit instantiations
template class HighMemoryStrategy<float>;
template class HighMemoryStrategy<double>;