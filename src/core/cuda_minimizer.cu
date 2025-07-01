// Minimizer.cpp
#include "common_includes.h"
#include "core/cuda_minimizer.h"
#include "config/config.h"
#include "core/matrix_operations.h"
#include "core/generate_random_vector.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>

CudaMinimizer::CudaMinimizer(cuDoubleComplex* d_kraus_p, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, double eps) {
    /*
    CudaMinimizer constructor.
    
    Arguments:
    - d_kraus (device): pointer to an array of double precision complex numbers (cuComplexDouble), which contains the Kraus operators.
        - Promise: the channel is trace preserving (up to a multiplicative factor). Else won't work.
    - kraus_number (host): pointer to number of Kraus operators.
    - kraus_in_dimension (host): pointer to the input dimension of the Kraus operators.
    - kraus_out_dimension (host): pointer to the output dimension of the Kraus operators
    - eps (host): pointer to the precision of the minimization algorithm.

    Notes:
    - The d_kraus array should be an array of d contiguous NxN matrices, each stored in column-major order.
      So entry (i,j) of matrix k corresponds to index k*N*N+j*N+i.
    - CudaMinimizer is NOT responsible for management of these resources and just receives pointers to them. No data is copied.    
    - The prefix d_ is used for device data. If nothing is specified, the data is stored on the host.
    */

    // Step 1: save the pointers that were passed.
    d_kraus = d_kraus_p;
    d = kraus_number;
    N = kraus_in_dimension;
    M = kraus_out_dimension;
    epsilon = eps; 
    // SVD only supports the case cols >= rows.
    /* TODO: Implement a better algorithm for the second SVD, where only the top eigenvector is actually needed!*/
    if (d * d > N || d > M){
        std::cerr << "Error: d * d > N or d > M. This is not supported by the SVD algorithm." << std::endl;
        throw std::runtime_error("Invalid dimensions for SVD");
    }

    /* OLD
    // Step 2: initialize the other members.
    // - Vector state: this is the rank one vector that we want to improve
    d_vecstate = nullptr;
    cudaMalloc((void**)&d_vecstate, N * sizeof(cuDoubleComplex));
    // - Input matrix: this will be initialized to the rank one projector |v><v|, where v is the vector state.
    d_inmatrix = nullptr;
    cudaMalloc((void**)&d_inmatrix, N * N * sizeof(cuDoubleComplex));   
    // - Output matrix: this will be the output of the channel, which is a M x M matrix.
    d_outmatrix = nullptr;
    cudaMalloc((void**)&d_outmatrix, M * M * sizeof(cuDoubleComplex));
    // - Scratch space for zgemm channel application: this will be a temporary matrix used in the channel application.
    // I need to store ZGEMM_BATCH_SIZE_CHANNEL M*N matrices for first intermediate step, and Z_GEMM_BATCH_SIZE_CHANNEL M*M or NxN matrices for the second intermediate step.
    d_tmpmat = nullptr;
    int max_dim = std::max(N,M);
    cudaMalloc((void**)&d_tmpmat, 2*ZGEMM_BATCH_SIZE_CHANNEL * (max_dim * max_dim) * sizeof(cuDoubleComplex)); // There is always space for at least 2 matrices, used for algorithm step
    // - Scratch space for storing eigenvalues
    d_eigs = nullptr;
    cudaMalloc((void**)&d_eigs, std::max(N, M) * sizeof(double));
    // - Device side entropy calculation space
    d_entropy_scratch = nullptr;
    cudaMalloc((void**)&d_entropy_scratch, sizeof(double) * M); // Scratch space for entropy calculation
    */

    // Step 2: create handles for cuBLAS, cuSolver and appropriate streams
    num_streams = MINIMIZER_CUDA_STREAMS; // Number of streams to use for parallel execution of minimizer operations
    streams = new cudaStream_t[num_streams]; // Array of streams for parallel execution
    blas_handles = new cublasHandle_t[num_streams];

    cudaError_t err;

    // Create streams and blas_handles
    for (int i = 0; i < num_streams; i++) {
        // Streams
        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "Error creating CUDA stream: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Failed to create CUDA stream");
        }
        // cuBLAS handles
        cublasStatus_t cublas_status;
        cublas_status = cublasCreate(&blas_handles[i]);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS initialization failed for stream " << i << "!" << std::endl;
            throw std::runtime_error("CUBLAS initialization failed for stream");
        }
        cublas_status = cublasSetStream(blas_handles[i], streams[i]);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS failed to set stream for stream " << i << "!" << std::endl;
            throw std::runtime_error("CUBLAS failed to set stream");
        }
    }
 
    // create and bind cusolver_handle to streams[0]
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization failed!" << std::endl;
        throw std::runtime_error("CUSOLVER initialization failed");
    }
    cusolver_status = cusolverDnSetStream(cusolver_handle, streams[0]);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER failed to set stream for stream 0!" << std::endl;
        throw std::runtime_error("CUSOLVER failed to set stream");
    }

    // Step 3: allocate memory on device

    // State vector
    d_vec = nullptr;
    err = cudaMalloc((void**)&d_vec, N * sizeof(cuDoubleComplex));   
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_vec: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory for d_vec");
    }
    // Vectors after first SVD, there are d of them and they have size M
    d_vecs_1 = nullptr;
    err = cudaMalloc((void**)&d_vecs_1, M * d * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_vecs_1: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory for d_vecs_1");
    }
    // Vectors after second SVD, there are d * d of them and they have size N
    d_vecs_2 = nullptr;
    err = cudaMalloc((void**)&d_vecs_2, N * d * d * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_vecs_2: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory for d_vecs_2");
    }
    // Singular values of d_vecs_1
    d_sv_1 = nullptr;
    err = cudaMalloc((void**)&d_sv_1, d * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_sv_1: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory for d_sv_1");
    }
    // Singular values of d_vecs_2
    d_sv_2 = nullptr;
    err = cudaMalloc((void**)&d_sv_2, d * d * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_sv_2: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory for d_sv_2");
    }
    // Query the work size for the SVD operations, and allocate d_scratch accordinglt
    work_size_1 = 0;
    cusolverDnZgesvd_bufferSize(cusolver_handle, M, d, &work_size_1);
    work_size_2 = 0;
    cusolverDnZgesvd_bufferSize(cusolver_handle, N, d*d, &work_size_2);
    // Scratch space. 
    d_scratch = nullptr;
    err = cudaMalloc((void**)&d_scratch, std::max(work_size_1, work_size_2) *sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_scratch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory for d_scratch");
    }



    // OTHER USEFUL CONSTANTS and FLAGS
    entropy = -1;
    /*OLD    
    // We need to know the binary entropy of epsilon, which is used in obtaining the error in the approximation of the entropy
    bin_entropy = -epsilon * std::log(epsilon) - (1-epsilon) * std::log(1-epsilon);
    // The corresponding error is quantified by bin_ent(eps)/2*(1-eps)
    entropy_error = bin_entropy / (2*(1-epsilon));
    */

    /*OLD
    // Initiate cublas handle
    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed!" << std::endl;
        throw std::runtime_error("CUBLAS initialization failed");
    }
    // Initiate cusolver handle
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization failed!" << std::endl;
        throw std::runtime_error("CUSOLVER initialization failed");
    }
    */



    // Update internal state
    minimizer_state = CUDA_MINIMIZER_STAGE_0;
}

cudaError_t CudaMinimizer::initializeVectorFromDevice(cuDoubleComplex* v) {
    /*
    Initializes the vector state to a given vector v.

    Arguments:
        - v (device): pointer to a vector of double precision complex numbers (cuComplexDouble), which is the vector to initialize the state to.
    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The dimension is not checked! If the dimension does not match, undefined behavior.
        - The data in v is copied to d_vec.
        - No memory management is performed - in particular, memory for v is not freed.
    */
    cudaError_t err = cudaMemcpy(d_vec, v, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying vector from device to device: " << cudaGetErrorString(err) << std::endl;
    }
    minimizer_state = CUDA_MINIMIZER_STAGE_0; // Reset the state to stage 0
    return err;
}

cudaError_t CudaMinimizer::initializeVectorFromHost(cuDoubleComplex* v){
    /*
    Initializes the vector state to a given vector v, which is stored on the host.
    Arguments:
        - v (host): pointer to a vector of double precision complex numbers (cuComplexDouble), which is the vector to initialize the state to.
    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The dimension is not checked! If the dimension does not match, undefined behavior.
        - The data in v is simply copied to the d_vec.
        - No memory management is performed - in particular, memory for v is not freed.
    */
    cudaError_t err = cudaMemcpy(d_vec, v, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying vector from host to device: " << cudaGetErrorString(err) << std::endl;
    }
    minimizer_state = CUDA_MINIMIZER_STAGE_0; // Reset the state to stage 0

    return err; // Returns success if the copy was successful

}

cudaError_t CudaMinimizer::initializeRandomVector(){
    /*
    Use the generate_random_vector function to initialize the vector state to a random vector.

    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The vector state is initialized to a random vector of dimension N.
        - The data in the d_vec is overwritten.
    */

    // Generate a random vector on the device
    cudaError_t err = generateUniformRandomVectorsCuda(d_vec, N, 1);
    minimizer_state = CUDA_MINIMIZER_STAGE_0; // Reset the state to stage 0

    return err;
}



/*
Idea on how to improve performance significantly.

The current setup is the following:
- I compute |v><v|
- I apply the channel by first computing K_i |v><v|, then also K_i |v><v| K_i^*. This is the super expensive part, occupying about 45% of the time
- I perturb by epsilon.
- Then I diagonalize Phi(rho), compute log. Also expensive (maaybe 5-8% of runtime), but less so.
- Then I apply the dual channel. Same idea as before, 45% of the time.
- Finally I diagonalize Phi^*(log(Phi(|v><v|))) and choose the highest eigenvector.

Areas of improvement:
1) Instead of computing K_i |v><v|, I can compute K_i |v> and then compute the outer product (K_i |v>)(K_i |v>)^*. This is a lot faster, since I only need to compute N^2 complex numbers instead of N^3.
2) Instead of diagonalizing Phi(rho), I can maybe just do an SVD of the individual results of K_i |v> and then compute the outer product of the singular vectors. This is a lot faster, since I only need to compute N^2 complex numbers instead of N^3.
3) 

*/

/*

We define 2 main steps of the algorithm:

1) Given a vector, compute {K_i |v>}_i and perform SVD to obtain {lambda_i} and {psi_i}. Multiply the psi_i by (log((1-eps)lambda_i^2 + eps) - log(eps)). At the end of this stage, the lambda_i are the square roots of Phi(rho).
2) Next, compute {K_j |psi_i>}_ij and perform SVD to obtain {mu_ij} and {phi_ij}. The phi_ij corresponding to the largest mu_ij is the new vector state.

At the end of step 1, we are able to compute the entropy of Phi_e(rho) as follows:

S(Phi_e(rho)) = [- sum_{i<=d} ((1-eps) lambda_i^2 + eps) log((1-eps) lambda_i^2 + eps)] - (M - d) * eps * log(eps)
*/

__global__ void rescale_vecs_1(cuDoubleComplex* vecs, double* lambdas, double epsilon, int num_vecs, int vec_size){
    /*
    Rescale the vectors in d_vecs_1 by the right amounts (so we have log(Phi_eps(rho))-log(eps)*1 = YY^H).
    This is done by multiplying each vector by (log((1-eps)lambda_i^2 + eps) - log(eps)).
    */

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Memory coalescing: each thread is responsible for just one operation. Then consecutive threads will load from consecutive parts of memory
    if (idx >= num_vecs * vec_size) {
        return;
    }
    // Get the entry
    cuDoubleComplex entry = vecs[idx];
    // Get the lambda_i
    double lambda_i = lambdas[idx / vec_size];
    // Compute the rescaling factor
    cuDoubleComplex rescale_factor = make_cuDoubleComplex(sqrt(log((1 - epsilon) * lambda_i * lambda_i + epsilon) - log(epsilon)),0.0);
    // Rescale
    entry = cuCmul(entry, rescale_factor);
    // Write back the result
    vecs[idx] = entry;    
}

cudaError_t CudaMinimizer::step_1(){
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
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // One operation per stream
    for (int i=0; i < num_streams; i++){
        // Each stream multiplies K_i |v> for a range of i. No need for them to be consecutive: 
        // Since K_i is column_major, we cannot simply do one mat vet multiplication with all the K_i at once :(
        for (int j = i; j < d; j += num_streams) {
            // Operation: T[i] = Kraus[i] * d_vec
            // Kraus[i] is MxN, d_vec is Nx1, and T[i] is Mx1.
            cublasZgemv(blas_handles[i], CUBLAS_OP_N, M, N, &one, d_kraus + j * N * M, M, d_vec, 1, &zero, d_vecs_1 + j * M, 1);
        }
    }
    // Synchronize streams to ensure all operations are complete
    cudaDeviceSynchronize();

    // Step 2: find the SVD of X = {K_i |d_vec>}_i. X is Mxd
    // So X = U S V^H means that V^H has d columns, and U has M rows.
    //Don't need to have more streams as its only one decomposition.
    // Scratch space is already allocated
    cusolverDnZgesvd(cusolver_handle, 
                     'O', // Overwrite input matrix with left singular vectors (U in USV^H)
                     'N', // Don't compute right singular vectors (V^H in USV^H)
                     M, d, 
                     d_vecs_1, M, 
                     d_sv_1, 
                     nullptr, M, // U matrix not needed. Still, ldu >= min(M, d).
                     nullptr, 1, // VT matrix, not needed. ldvt = 1 since we don't compute it.
                     d_scratch, work_size_1, // Scratch space
                     nullptr, 0); // 'rwork can be a null pointer if the user does not want info about superdiagonal'

    cudaDeviceSynchronize();
    // At this point we can rescale the vectors in Y=d_sv_1 by the right amounts (so we have log(Phi_eps(rho))-log(eps)*1 = YY^H)
    int total_elems = d * M;
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;
    rescale_vecs_1<<<blocks, threads>>>(d_vecs_1, d_sv_1, epsilon, d, M);
    cudaDeviceSynchronize();
    minimizer_state = CUDA_MINIMIZER_STAGE_1; // Reset the state to stage 0

    return cudaGetLastError(); // Returns success if the kernel launched successfully
}

cudaError_t CudaMinimizer::step_2(){
    /*
    Perform the second step in the algorithm. That is, do the following:
    - Compute {K_j |psi_i>}_ij, where {K_j}_j are the kraus operators and {psi_i}_i are the vectors obtained in step 1. Save the result in d_vecs_2.
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
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // One operation per stream
    for (int i=0; i < num_streams; i++){
        // Each stream performs matrix multiplication K_i (|v_j>_j) for a range of i. No need for them to be consecutive: 
        // Since K_i is column_major, we cannot simply do one mat vet multiplication with all the K_i at once :(
        for (int j = i; j < d; j += num_streams) {
            // Operation: T[i] = Kraus[i]^H * d_vecs_1
            // Kraus[i] is MxN (so op(K_i) is NxM), d_vecs_1 is Mxd, and T[i] is Nxd.
            // Save in d_vecs_2 with right offset
            cublasZgemm(blas_handles[i], 
                CUBLAS_OP_C, 
                CUBLAS_OP_N, 
                N, 
                d, 
                M, 
                &one, 
                d_kraus + j * N * M, 
                M, 
                d_vecs_1 + j * M, 
                M, 
                &zero, 
                d_vecs_2 + j * N * d, 
                N);
        }
    }
    // Synchronize streams to ensure all operations are complete
    cudaDeviceSynchronize();

    // Step 2: extract the largest left singular vector of d_vecs_2. This is the new state.
    // For now, compute full SVD of the N * (dxd) matrix d_vecs_2. Only store the left singular vectors. In reality only care about the first!
    cusolverDnZgesvd(cusolver_handle, 
                     'O', // Overwrite input matrix with left singular vectors (U in USV^H)
                     'N', // Don't compute right singular vectors (V^H in USV^H)
                     N, d*d, 
                     d_vecs_2, N, 
                     d_sv_2, 
                     nullptr, N, // U matrix not needed. Still, ldu >= min(M, d).
                     nullptr, 1, // VT matrix, not needed. ldvt = 1 since we don't compute it.
                     d_scratch, work_size_2, // Scratch space
                     nullptr, 0); // 'rwork can be a null pointer if the user does not want info about superdiagonal'

    cudaDeviceSynchronize();

    // Step 3: Update the d_vec state. (TODO: actually, I could avoid this copy...)
    cudaMemcpy(d_vec, d_vecs_2, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    // No further need to sync since memCpy waits for transfer to be done
    minimizer_state = CUDA_MINIMIZER_STAGE_2; // Reset the state to stage 2 (same as stage 0, but algorithm has started running!)
    return cudaSuccess;
}





/*

OLD STUFF

*/


// cudaError_t CudaMinimizer::updateProjector() {
//     /*
//     Updates the d_inmatrix to be the projector |v><v|, where v is the vector state.
    
//     Returns:d
//         - cudaSuccess if the projector was updated successfully.
//     Note:
//         - The d_inmatrix is updated in place.
//         - The d_inmatrix is a NxN matrix, where N is the dimension of the vector state.
//     */

//     // I need a cublas Handle    

//     cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0); // Scalar alpha for the outer product
//     int incx = 1, incy = 1; // Stride for x vector

//     // Initialize the d_inmatrix to zero
//     cudaMemset(d_inmatrix, 0, N * N * sizeof(cuDoubleComplex));
//     // Outer product: A = alpha * x * y^H + A 
//     cublasStatus_t status = cublasZgerc(handle, N, N, &alpha, d_vecstate, incx, d_vecstate, incy, d_inmatrix, N);
//     if (status != CUBLAS_STATUS_SUCCESS) {
//         std::cerr << "CUBLAS Zher failed!" << std::endl;
//         return cudaErrorUnknown; // Return an error code
//     }
//     // Set the flag
//     input_matrix_scrambled = false; // The input matrix is now the projector |v><v|, so it is not scrambled
//     return cudaSuccess; // Return success
// }

// // Custom kernel to sum a sequence of matrices. TODO: this can probably be made more efficient by using shared memory or other techniques, but this is a good starting point.
// __global__ void reduce_matrix_sum(const cuDoubleComplex* matrices, cuDoubleComplex* output, int rows, int cols, int num_matrices) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_elements = rows * cols;
//     if (idx >= total_elements) return;

//     cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
//     for (int i = 0; i < num_matrices; ++i) {
//         cuDoubleComplex val = matrices[i * total_elements + idx];
//         sum = cuCadd(sum, val);
//     }
//     output[idx] = cuCadd(output[idx],sum);
// }

// cudaError_t CudaMinimizer::applyChannel(){
//     /*
    
//     Applies the channel represented by the Kraus operators to input matrix. Saves the result in d_outmatrix.
    
//     Returns:
//         - cudaSuccess if the channel was applied successfully.
//     Note:   
//         - The input matrix is not checked or updated.
//         - The output matrix is updated in place. Its value is overwritten by Phi(d_inmatrix).
//         - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
//             ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
        
//     */

//     // Set d_outmatrix to zero
//     cudaMemset(d_outmatrix, 0, M * M * sizeof(cuDoubleComplex));

//     cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
//     cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

//     for (int i=0; i < d; i+= ZGEMM_BATCH_SIZE_CHANNEL) {
//         // Determine the number of Kraus operators to process in this batch
//         int batch_size = std::min(ZGEMM_BATCH_SIZE_CHANNEL, d - i);
//         // batched matrix multiplication 1. Use strided one because matrices are contiguous
//         // Operation: T[i] = Kraus[i] * d_inmatrix
//         // Kraus[i] is MxN, d_inmatrix is NxN, and T[i] is MxN.
//         cublasZgemmStridedBatched(handle,
//             CUBLAS_OP_N, // No transpose for A
//             CUBLAS_OP_N, // No transpose for B
//             M,           // First output dimension
//             N,           // Second output dimension
//             N,           // Common dimension (contracted)
//             &one,        // Scalar alpha
//             d_kraus + i * M * N, // Pointer to the first element of batch of Kraus operators
//             M,           // Leading dimension of Kraus operators (no op)
//             M * N,       // Stride for the next batch of Kraus operators
//             d_inmatrix,  // Pointer to the input matrix
//             N,           // Leading dimension of input matrix
//             0,           // Stride for the next input matrix (0 because we are using the same input matrix for all batches)
//             &zero,       // Scalar beta
//             d_tmpmat,    // Pointer to the first temporary output matrix for this batch. It is always reused!
//             M,           // Leading dimension of temporary output matrix
//             M * N,       // Stride for the next temporary output matrix
//             batch_size   // Number of matrices in this batch
//         );

//         // batched matrix multiplication 2.
//         // Operation: T[i] = T[i] * conj(Kraus[i]^T) (Notice that the T[i] on RHS is MxN, on the LHS it is MxM!!! But the memory has been allocated so there shouldn't be any problems)
//         // T[i] is MxN, conj(Kraus[i]^T) is NxM, and d_outmatrix is MxM.
//         cublasZgemmStridedBatched(handle,
//             CUBLAS_OP_N, // No transpose for A
//             CUBLAS_OP_C, // Conjugate transpose for B
//             M,           // First output dimension
//             M,           // Second output dimension
//             N,           // Contracted dimension
//             &one,        // Scalar alpha
//             d_tmpmat, // Pointer to the first temporary output matrix for this batch
//             M,           // Leading dimension of temporary output matrix
//             M * N,       // Stride for the next temporary output matrix
//             d_kraus + i * M * N, // Pointer to the first element of batch of Kraus operators
//             M,           // Leading dimension of Kraus operators (no op)
//             M*N,           // Stride for the next batch of Kraus operators
//             &zero,        // Scalar beta
//             d_tmpmat + ZGEMM_BATCH_SIZE_CHANNEL * M * N, // Pointer to the first temporary output matrix for this batch
//             M,           // Leading dimension of output matrix
//             M * M,           // Stride for the next output matrix (0 because we are using the same output matrix for all batches)
//             batch_size   // Number of matrices in this batch
//         );

//         // Now reduce the results to a single matrix
//         int total_elements = M * M; // Size of a single matrix, will spawn one thread per entry
//         int threads = 256;
//         int blocks = (total_elements + threads - 1) / threads;

//         reduce_matrix_sum<<<blocks, threads>>>(d_tmpmat + ZGEMM_BATCH_SIZE_CHANNEL * M * N, d_outmatrix, M, M, ZGEMM_BATCH_SIZE_CHANNEL);
//         cudaDeviceSynchronize();


//     }
//     return cudaGetLastError(); // Return the last error from CUDA operations
// }

// cudaError_t CudaMinimizer::applyDualChannel(){
//     /*
    
//     Applies the channel dual to the one represented by the Kraus operators to d_outmatrix. Saves the result in d_inmatrix.
    
//     Returns:
//         - cudaSuccess if the channel was applied successfully.
//     Note:   
//         - The d_outmatrix is not checked or updated.
//         - The d_inmatrix is updated in place. Its value is overwritten by Phi^*(d_outmatrix).
//         - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
//             ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
        
//     */

//     // Set d_outmatrix to zero
//     cudaMemset(d_inmatrix, 0, N * N * sizeof(cuDoubleComplex));

//     cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
//     cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

//     for (int i=0; i < d; i+= ZGEMM_BATCH_SIZE_CHANNEL) {
//         // Determine the number of Kraus operators to process in this batch
//         int batch_size = std::min(ZGEMM_BATCH_SIZE_CHANNEL, d - i);
//         // batched matrix multiplication 1. Use strided one because matrices are contiguous
//         // Operation: T[i] = Kraus[i]^H * d_outmatrix
//         // Kraus[i] is MxN, d_outmatrix is MxM, and T[i] is NxM.
//         cublasZgemmStridedBatched(handle,
//             CUBLAS_OP_C, // Conjugate transpose for A
//             CUBLAS_OP_N, // No transpose for B
//             N,           // First output dimension
//             M,           // Second output dimension
//             M,           // Common dimension
//             &one,        // Scalar alpha
//             d_kraus + i * M * N, // Pointer to the first element of batch of Kraus operators
//             M,           // Leading dimension of Kraus operators
//             M * N,       // Stride for the next batch of Kraus operators
//             d_outmatrix, // Pointer to the output matrix
//             M,           // Leading dimension of output matrix
//             0,           // Stride for the next output matrix (0 because we are using the same matrix for all batches)
//             &zero,       // Scalar beta
//             d_tmpmat, // Pointer to the first temporary output matrix for this batch. It is always reused!
//             N,           // Leading dimension of temporary output matrix
//             N * M,       // Stride for the next temporary output matrix
//             batch_size   // Number of matrices in this batch
//         );

//         // batched matrix multiplication 2.
//         // Operation: T'[i] = T[i] * Kraus[i] 
//         // T[i] is NxM, Kraus[i] is MxN, and T'[i] is NxN.
//         cublasZgemmStridedBatched(handle,
//             CUBLAS_OP_N, // No transpose for A
//             CUBLAS_OP_N, // No transpose for B
//             N,           // Number of rows of op(T[i]) (i.e. first output dimension)
//             N,           // Number of columns of op(Kraus[i]) (i.e. second output dimension)
//             M,           // Common dimension
//             &one,        // Scalar alpha
//             d_tmpmat, // Pointer to the first temporary output matrix for this batch
//             N,           // Leading dimension of temporary output matrix
//             N * M,       // Stride for the next temporary output matrix
//             d_kraus + i * N * M, // Pointer to the first element of batch of Kraus operators
//             M,           // Leading dimension of Kraus operators (before op)
//             M*N,         // Stride for the next batch of Kraus operators
//             &zero,       // Scalar beta
//             d_tmpmat + ZGEMM_BATCH_SIZE_CHANNEL * M * N, // Pointer to the first temporary output matrix for this batch
//             N,           // Leading dimension of output matrix
//             N * N,       // Stride for the next output matrix (0 because we are using the same output matrix for all batches)
//             batch_size   // Number of matrices in this batch
//         );

//         // Now reduce the results to a single matrix
//         int total_elements = M * M; // Size of a single matrix, will spawn one thread per entry
//         int threads = 256;
//         int blocks = (total_elements + threads - 1) / threads;

//         reduce_matrix_sum<<<blocks, threads>>>(d_tmpmat + ZGEMM_BATCH_SIZE_CHANNEL * M * N, d_inmatrix, N, N, ZGEMM_BATCH_SIZE_CHANNEL);
//         cudaDeviceSynchronize();


//     }
//     return cudaGetLastError(); // Return the last error from CUDA operations
// }

// __global__ void perturbByEpsilon(cuDoubleComplex* matrix, int dimension, double epsilon) {
//     /*
//     Kernel to perturb a square matrix in the follwing way:

//             A -> (1-epsilon)* A + epsilon/d * I
    
//     Arguments:
//         - matrix (device): pointer to the matrix to perturb.
//         - dimension (host): dimension of the matrix.
//         - epsilon (host): perturbation value.
//     */
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < dimension * dimension) {
//         int row = idx / dimension;
//         int col = idx % dimension;
//         // Apply the perturbation
//         matrix[idx] = cuCmul(matrix[idx], make_cuDoubleComplex(1.0 - epsilon, 0.0));
//         if (row == col) {
//             matrix[idx] = cuCadd(matrix[idx], make_cuDoubleComplex(epsilon / dimension, 0.0));
//         }
//     }
// }

// cudaError_t CudaMinimizer::applyEpsilonChannel(){
//     /*
//     Applies the channel represented by the Kraus operators to input matrix, perturbing it by epsilon.
//     Saves the result in d_outmatrix.
//     Returns:
//         - cudaSuccess if the channel was applied successfully.
//     Note:
//         - The input matrix is not checked or updated.
//         - The output matrix is updated in place. Its value is overwritten by Phi(d_inmatrix).
//         - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
//             ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
//     */

//     // Step 1: Apply the channel to the input matrix
//     applyChannel();
//     // Print error
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cerr << "Error applying channel: " << cudaGetErrorString(err) << std::endl;
//         return err; // Return the error if the channel application failed
//     }
//     // Step 2: Perturb the output matrix by epsilon
//     int threads = 256;
//     int blocks = (M * M + threads - 1) / threads;
//     // Launch the kernel to perturb the output matrix
//     perturbByEpsilon<<<blocks, threads>>>(d_outmatrix, M, epsilon);
//     return cudaGetLastError();
// }

// cudaError_t CudaMinimizer::applyEpsilonDualChannel(){
//     /*
//     Applies the channel dual to the one represented by the Kraus operators to d_outmatrix, perturbing it by epsilon.
//     Saves the result in d_inmatrix.
//     Returns:
//         - cudaSuccess if the channel was applied successfully.
//     Note:
//         - The d_outmatrix is not checked or updated.
//         - The d_inmatrix is updated in place. Its value is overwritten by Phi^*(d_outmatrix).
//         - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
//             ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
//     */
//     // Step 1: Apply the dual channel to the output matrix
//     applyDualChannel();
//     // Step 2: Perturb the input matrix by epsilon
//     int threads = 256;
//     int blocks = (N * N + threads - 1) / threads;
//     perturbByEpsilon<<<blocks, threads>>>(d_inmatrix, N, epsilon);
//     return cudaGetLastError(); // Return the last error from CUDA operations
// }


/*

            Algorithm

*/


// __global__ void computeLog(double* eigs, int dimension) {
//     /*
//     Kernel to compute the logarithm of the eigenvalues in place
    
//     Arguments:
//         - eigs (device): pointer to the eigenvalues.
//         - dimension (host): number of eigenvalues.
//     */
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < dimension) {
//         double eig = eigs[idx];
//         if (eig > 0) {
//             eigs[idx] = std::log(eig);
//         } else {
//             eigs[idx] = -INFINITY; // Handle non-positive eigenvalues
//         }
//     }
// }

// __global__ void scale_columns(cuDoubleComplex* A, double* scalars, cuDoubleComplex* out, int rows, int cols) {
//     /*
//     Kernel to scale each column of a matrix by a corresponding scalar value.
//     Arguments:
//         - A (device): pointer to the matrix to scale.
//         - scalars (device): pointer to an array of scalars, one for each column.
//         - out (device): pointer to the output matrix where the scaled values will be stored.
//         - rows (host): number of rows in the matrix.
//         - cols (host): number of columns in the matrix.
//     */
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < rows && col < cols) {
//         cuDoubleComplex scale = make_cuDoubleComplex(scalars[col], 0.0); // Get the scalar for the current column
//         cuDoubleComplex val = A[col * rows + row];  // column-major
//         out[col * rows + row] = cuCmul(val, scale);
//     }
// }

cudaError_t CudaMinimizer::stepAlgorithm(){
    /*
        Run through one minimization pass of the algorithm. Updatre d_vec with a new one which provides better entropy.

        Returns:
            - cudaSuccess if the step was successful.
    */
    cudaError_t err;
    err = step_1(); // Step 1: compute {K_i |d_vec>}_i and perform SVD to obtain {lambda_i} and {psi_i}. Multiply the psi_i by (log((1-eps)lambda_i^2 + eps) - log(eps)). At the end of this stage, the lambda_i are the square roots of Phi(rho).
    if (err != cudaSuccess) {
        std::cerr << "Error in step 1: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if step 1 failed
    }
    err = step_2(); // Step 2: compute {K_j |psi_i>}_ij and perform SVD to obtain {mu_ij} and {phi_ij}. The phi_ij corresponding to the largest mu_ij is the new vector state. Save it in d_vec.
    if (err != cudaSuccess) {
        std::cerr << "Error in step 2: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if step 2 failed
    }
    return cudaSuccess;
}

// cudaError_t CudaMinimizer::stepAlgorithm(){

//     /*
//     Steps the minimization algorithm by performing the following operations:
//         1. Update the projector based on the vector.
//         2. Compute Phi_e(rho) by applying the epsilon channel to the input matrix.
//         3. Compute log(Phi_e(rho)) by diagonalizing the output matrix and reconstructing it.
//         4. Compute Phi_e^*(log(Phi_e(rho))) by applying the dual channel to the output matrix.
//         5. Find the eigenvector with the highest eigenvalue in the input matrix and update the vector state.

//     Returns:
//         - cudaSuccess if the step was successful.
//     */
//     // Step 1: update the projector
//     if (input_matrix_scrambled) {
//         cudaError_t err_update = updateProjector();
//         if (err_update != cudaSuccess) {
//             std::cerr << "Error updating projector: " << cudaGetErrorString(err_update) << std::endl;
//             return err_update; // Return the error if the projector update failed
//         }
//     }

//     // Step 2: compute Phi_e(rho)
//     cudaError_t err_eps_ch = applyEpsilonChannel();
//     if (err_eps_ch != cudaSuccess) {
//         std::cerr << "Error applying epsilon channel: " << cudaGetErrorString(err_eps_ch) << std::endl;
//         return err_eps_ch;
//     }

//     // Step 3: compute log(Phi_e(rho))
//     // 3.1 diagonalize output matrix

//     // 3.1.1 - Query buffer size
//     int lwork = 0; // Workspace size
//     cusolverDnZheevd_bufferSize(
//         cusolver_handle,
//         CUSOLVER_EIG_MODE_VECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
//         CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
//         M,                          // Size of the matrix (M x M)
//         d_outmatrix,              // Pointer to the matrix to be diagonalized (d_outmatrix). It gets overwritten (on success) with the eigenvectors.
//         M,                          // Leading dimension of the matrix (M)
//         d_eigs,                       // Vector of eigenvalues of the matrix.
//         &lwork
//     );                    // Pointer to the workspace size (output parameter). Stored on host.

//     // 3.1.2 - Check for available workspace. Allocate remaining resources.
//     // For now, reuse the scratch space "temp_matrix" that was allocated for the channel application.
//     if (2 * ZGEMM_BATCH_SIZE_CHANNEL * (std::max(N,M)*std::max(N,M)) < lwork) {
//         // Try to allocate more memory for the temporary matrix
//         cudaFree(d_tmpmat);
//         cudaError_t err_alloc = cudaMalloc((void**)&d_tmpmat, lwork * sizeof(cuDoubleComplex));
//         if (err_alloc != cudaSuccess) {
//             std::cerr << "Error allocating memory for temporary matrix: " << cudaGetErrorString(err_alloc) << std::endl;
//             return err_alloc; // Return the error if memory allocation fails
//         }
//     }
//     int* devInfo = nullptr;
//     cudaMalloc(&devInfo, sizeof(int)); // Allocate device memory for the info variable


//     // 3.1.3 - Call the solver
//     cusolverDnZheevd(
//         cusolver_handle,            // Handle to the cuSOLVER context
//         CUSOLVER_EIG_MODE_VECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
//         CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
//         M,                          // Size of the matrix (M x M)
//         d_outmatrix,              // Pointer to the matrix to be diagonalized (d_outmatrix). It gets overwritten (on success) with the eigenvectors.
//         M,                          // Leading dimension of the matrix (M) 
//         d_eigs,                       // Vector of eigenvalues of the matrix.
//         d_tmpmat,                 // Pointer to the workspace (temporary matrix). It is used to store intermediate results.
//         lwork,                      // Size of the workspace (temporary matrix)
//         devInfo);                   // Pointer to the info variable (output parameter). It indicates success or failure of the operation.

//     // 3.1.4 - Check for errors
//     int h_info = 0;
//     cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//     if (h_info != 0) {
//         std::cerr << "Error in cusolverDnZheevd: " << h_info << std::endl;
//         return cudaErrorUnknown; // Return an error code if the operation failed
//     }

//     // 3.2 - compute log(d_eigs)
//     // We can use a kernel to compute the logarithm of the eigenvalues in place    
//     int threads = 256;
//     int blocks = (M + threads - 1) / threads; // Number of blocks
//     computeLog<<<blocks, threads>>>(d_eigs, M); // Launch the kernel to compute the logarithm of the eigenvalues
//     cudaDeviceSynchronize(); // Wait for the kernel to finish
//     cudaError_t err_log = cudaGetLastError(); // Get the last error from CUDA operations
//     if (err_log != cudaSuccess) {
//         std::cerr << "Error computing logarithm of eigenvalues: " << cudaGetErrorString(err_log) << std::endl;
//         return err_log; // Return the error if the kernel failed
//     }

//     // 3.3 - reconstruct the matrix
//     cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
//     cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

//     // 3.3.1: First perform diag * d_eigs, store result in the first entries of d_tmpmat
//     dim3 threadsPerBlock(16, 16);
//     dim3 numBlocks((M + 15) / 16, (M + 15) / 16);
//     scale_columns<<<numBlocks, threadsPerBlock>>>(d_outmatrix, d_eigs, d_tmpmat, M, M);
//     cudaDeviceSynchronize(); // Wait for the kernel to finish
//     cudaError_t err_scale = cudaGetLastError(); // Get the last error from CUDA operations
//     if (err_scale != cudaSuccess) {
//         std::cerr << "Error scaling columns: " << cudaGetErrorString(err_scale) << std::endl;
//         return err_scale; // Return the error if the kernel failed
//     }
//     // 3.3.2: Then do d_tmpmat*conj(d_outmatrix^T) -> d_tmpmat'
//     cublasStatus_t status = cublasZgemm(
//         handle,
//         CUBLAS_OP_N, // Conjugate transpose for A
//         CUBLAS_OP_C, // No transpose for B
//         M,           // Number of rows of op(A)
//         M,           // Number of columns of op(B)
//         M,           // Common dimension
//         &one,        // Scalar alpha
//         d_tmpmat, // Pointer to the first matrix (d_outmatrix)
//         M,           // Leading dimension of d_outmatrix
//         d_outmatrix,  // Pointer to the second matrix (d_tmpmat)
//         M,           // Leading dimension of d_tmpmat
//         &zero,       // Scalar beta
//         d_tmpmat + M * M, // Pointer to the output matrix (d_tmpmat')
//         M            // Leading dimension of output matrix (d_tmpmat')
//     );
//     if (status != CUBLAS_STATUS_SUCCESS) {
//         std::cerr << "CUBLAS Zgemm failed, when recombining log(Phi(rho))!" << std::endl;
//         return cudaErrorUnknown; // Return an error code if the operation failed
//     }
//     // 3.3.3: Copy the result back to d_outmatrix
//     cudaMemcpy(d_outmatrix, d_tmpmat + M * M, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
//     // Now d_outmatrix contains log(Phi_e(rho)) in the form of a Hermitian matrix.

//     // Step 4: compute Phi_e^*(log(Phi_e(rho)))
//     cudaError_t err_dual_ch = applyEpsilonDualChannel();
//     if (err_dual_ch != cudaSuccess) {
//         std::cerr << "Error applying dual epsilon channel: " << cudaGetErrorString(err_dual_ch) << std::endl;
//         return err_dual_ch; 
//     }

//     // Step 5: find eigenvector with highest eigenvalue
//     // 5.1 - Diagonalize the d_inmatrix to find the eigenvalues and eigenvectors
//     // 5.1.1 - Query buffer size
//     lwork = 0; // Reset workspace size
//     cusolverDnZheevd_bufferSize(
//         cusolver_handle,
//         CUSOLVER_EIG_MODE_VECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
//         CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
//         N,                          // Size of the matrix (N x N)
//         d_inmatrix,               // Pointer to the matrix to be diagonalized (d_inmatrix). It gets overwritten (on success) with the eigenvectors.
//         N,                          // Leading dimension of the matrix (N)
//         d_eigs,                       // Vector of eigenvalues of the matrix.
//         &lwork                       // Pointer to the workspace size (output parameter). Stored on host.
//     );
//     // 5.1.2 - Check for available workspace. Allocate remaining resources.
//     // For now, reuse the scratch space "temp_matrix" that was allocated for the channel application.
//     if (2 * ZGEMM_BATCH_SIZE_CHANNEL * std::max(N,M) * std::max(N,M) < lwork) {
//         // Try to allocate more memory for the temporary matrix
//         cudaFree(d_tmpmat);
//         cudaError_t err_alloc = cudaMalloc((void**)&d_tmpmat, lwork * sizeof(cuDoubleComplex));
//         if (err_alloc != cudaSuccess) {
//             std::cerr << "Error allocating memory for temporary matrix: " << cudaGetErrorString(err_alloc) << std::endl;
//             return err_alloc; // Return the error if memory allocation fails
//         }
//     }
//     // 5.1.3 - Call the solver
//     cusolverDnZheevd(
//         cusolver_handle,            // Handle to the cuSOLVER context
//         CUSOLVER_EIG_MODE_VECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
//         CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
//         N,                          // Size of the matrix (N x N)
//         d_inmatrix,               // Pointer to the matrix to be diagonalized (d_inmatrix). It gets overwritten (on success) with the eigenvectors.
//         N,                          // Leading dimension of the matrix (N)
//         d_eigs,                       // Vector of eigenvalues of the matrix.
//         d_tmpmat,                 // Pointer to the workspace (temporary matrix). It is used to store intermediate results.
//         lwork,                      // Size of the workspace (temporary matrix)
//         devInfo);                   // Pointer to the info variable (output parameter). It indicates success or failure of the operation.

//     // Set flag
//     input_matrix_scrambled = true; // The input matrix is now scrambled, as it has been diagonalized and contains the eigenvectors. Projector should be computed again.

//     // 5.1.4 - Check for errors
//     h_info = 0;
//     cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//     if (h_info != 0) {
//         std::cerr << "Error in cusolverDnZheevd: " << h_info << std::endl;
//         return cudaErrorUnknown; // Return an error code if the operation failed
//     }
//     // 5.2 - Copy the eigenvector corresponding to the highest eigenvalue to the vector state
//     // The eigenvectors are stored in the columns of d_inmatrix, so we need to copy
//     // the last column (corresponding to the highest eigenvalue) to the d_vecstate
//     cudaMemcpy(d_vecstate, d_inmatrix + (N - 1) * N, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
//     // Check for success
//     if (cudaGetLastError() != cudaSuccess) {
//         std::cerr << "Error copying eigenvector to d_vecstate: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
//         return cudaGetLastError(); // Return the error if the copy failed
//     }

//     // Step 6: Clean up
//     cudaFree(devInfo); // Free the device memory for the info variable

//     // Return success
//     return cudaSuccess;
    

    
// }

cudaError_t CudaMinimizer::step(){
    /*
        Will compute one step of the minimization algorithm.

        1. Runs "stepAlgorithm" to update d_vecstate. This also updates the projector if necessary.
        2. Recomputes the projector
        3. Computes entropy

        Returns:
            - cudaSuccess if the step was successful.
        Note: 
            - d_vecstate must be initialized to a valid vector, no checks are performed

    */
    // Step 1
    if (minimizer_state != CUDA_MINIMIZER_STAGE_1){
        cudaError_t err_step_1 = step_1();
        if (err_step_1 != cudaSuccess){
            std::cerr << "Error in step_1: " << cudaGetErrorString(err_step_1) << std::endl;
            return err_step_1; // Return the error if the step failed 
        }
    }
    // Step 2
    if (minimizer_state == CUDA_MINIMIZER_STAGE_2){
        cudaError_t err_step_2 = step_2();
        if (err_step_2 != cudaSuccess) {
            std::cerr << "Error in step_2: " << cudaGetErrorString(err_step_2) << std::endl;
            return err_step_2; // Return the error if the projector update failed
        }
    } else {
        std::cerr << "Error in running the algorithm: minimizer is not in correct stage!" << std::endl;
        return cudaErrorUnknown;
    }


    // Step 3: Compute the entropy
    cudaError_t err_entropy = calculateEpsilonEntropy();
    if (err_entropy != cudaSuccess) {
        std::cerr << "Error calculating epsilon entropy: " << cudaGetErrorString(err_entropy) << std::endl;
        return err_entropy; // Return the error if the entropy calculation failed
    }

    return cudaSuccess; // Return success if all steps were successful


}


/*

        UPDATERS

*/




cudaError_t CudaMinimizer::calculateEpsilonEntropy(){
/*
    Calculates the von Neumann entropy of the output matrix after applying the epsilon channel.
    When in stage 1, the square roots of eigenvalues lamvda_i of Phi(rho) are saved in d_sv_1.

    The formula then is:

        S(Phi_e(rho)) = [- sum_{i<=d} ((1-eps) lambda_i^2 + eps) log((1-eps) lambda_i^2 + eps)] - (M - d) * eps * log(eps)
    
    Returns:
        - cudaSuccess if the entropy was calculated successfully.
    Note:
        - If the minimizer is not in stage 1 but has been initialized, step_1 is run to bring it to stage_1.
        - The entropy is stored in the member variable `entropy`.

*/
    if (minimizer_state != CUDA_MINIMIZER_STAGE_1) {
        if (minimizer_state == CUDA_MINIMIZER_STAGE_0 || minimizer_state == CUDA_MINIMIZER_STAGE_2) {
        cudaError_t err_step_1 = step_1();
        if (err_step_1 != cudaSuccess) {
            std::cerr << "Error in step_1: " << cudaGetErrorString(err_step_1) << std::endl;
            return err_step_1; // Return the error if the step failed 
        }
        } else {
            std::cerr << "Error: Minimizer is in an uninitialized or unknown state, cannot calculate entropy." << std::endl;
            return cudaErrorUnknown; // Return an error if the minimizer is in an unknown state
        }
    }

    // Now perform the computation of entropy. Since d is assumed small, it is best to just copy values to host and work on CPU.
    double* h_sv_1 = new double[d];
    cudaError_t err_cpy = cudaMemcpy(h_sv_1, d_sv_1, d * sizeof(double), cudaMemcpyDeviceToHost);
    if (err_cpy != cudaSuccess) {
        std::cerr << "Error copying eigenvalues from device to host: " << cudaGetErrorString(err_cpy) << std::endl;
        delete[] h_sv_1; // Free the host memory before returning
        return err_cpy; // Return the error if the copy failed
    }

    entropy = 0.0;
    for (int i = 0; i < d; ++i) {
        double lambda_i = h_sv_1[i];
        double val = (1.0 - epsilon) * lambda_i * lambda_i + epsilon; // epsilon is assumed small but positive
        entropy -= val * log(val);
    }
    entropy -= (M - d) * epsilon * log(epsilon); // Add the contribution from the zero eigenvalues
    delete[] h_sv_1; // Free the host memory
    return cudaSuccess; // Return success if the entropy was calculated successfully    
}


// __global__ void compute_entropy_multi(double* eigs, double* scratch, int size) {
//     extern __shared__ double sdata[];
//     int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x + tid;

//     double val = 0.0;
//     if (i < size) {
//         double p = eigs[i];
//         if (p > 1e-15)
//             val = -p * log(p);
//     }

//     sdata[tid] = val;
//     __syncthreads();

//     // Block-level reduction
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s)
//             sdata[tid] += sdata[tid + s];
//         __syncthreads();
//     }

//     // First thread of block writes block result to global scratch
//     if (tid == 0)
//         scratch[blockIdx.x] = sdata[0];
// }

// cudaError_t CudaMinimizer::calculateEpsilonEntropy(){
//     /*
//     Calculates the von Neumann entropy of the output matrix after applying the epsilon channel.
//     The entropy is computed as the von Neumann entropy of the output matrix.

//     Returns:
//         - cudaSuccess if the entropy was calculated successfully.
//     Note:
//         - The input matrix is NOT updated. Check therefore that it contains the correct state!
//         - The entropy is stored in the member variable `entropy`.
//     */

//     // Step 1: Apply the epsilon channel
//     cudaError_t err_eps_ch = applyEpsilonChannel();
//     if (err_eps_ch != cudaSuccess) {
//         std::cerr << "Error applying epsilon channel: " << cudaGetErrorString(err_eps_ch) << std::endl;
//         return err_eps_ch;
//     }

//     // Step 2: Diagonalize the output matrix to get eigenvalues
//     int lwork = 0; // Workspace size
//     cusolverDnZheevd_bufferSize(
//         cusolver_handle,
//         CUSOLVER_EIG_MODE_NOVECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
//         CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
//         M,                          // Size of the matrix (M x M)
//         d_outmatrix,              // Pointer to the matrix to be diagonalized (d_outmatrix). It gets overwritten (on success) with the eigenvectors.
//         M,                          // Leading dimension of the matrix (M)
//         d_eigs,                       // Vector of eigenvalues of the matrix.
//         &lwork                      // Pointer to the workspace size (output parameter). Stored on host.
//     );

//     // Check for available workspace. Allocate remaining resources.
//     if (2 * ZGEMM_BATCH_SIZE_CHANNEL * M * M < lwork) {
//         // Try to allocate more memory for the temporary matrix
//         cudaFree(d_tmpmat);
//         cudaError_t err_alloc = cudaMalloc((void**)&d_tmpmat, lwork * sizeof(cuDoubleComplex));
//         if (err_alloc != cudaSuccess) {
//             std::cerr << "Error allocating memory for temporary matrix: " << cudaGetErrorString(err_alloc) << std::endl;
//             return err_alloc; // Return the error if memory allocation
//         }
//     }
//     int* devInfo = nullptr;
//     cudaMalloc(&devInfo, sizeof(int)); // Allocate device memory for the info variable
//     // Call the solver to diagonalize the output matrix
//     cusolverDnZheevd(
//         cusolver_handle,            // Handle to the cuSOLVER context
//         CUSOLVER_EIG_MODE_NOVECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
//         CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
//         M,                          // Size of the matrix (M x M)
//         d_outmatrix,              // Pointer to the matrix to be diagonalized (d_outmatrix). It gets overwritten (on success) with the eigenvectors.
//         M,                          // Leading dimension of the matrix (M)
//         d_eigs,                       // Vector of eigenvalues of the matrix.
//         d_tmpmat,                 // Pointer to the workspace (temporary matrix). It is used to store intermediate results.
//         lwork,                      // Size of the workspace (temporary matrix)
//         devInfo);                   // Pointer to the info variable (output parameter). It indicates success or failure of the operation.
//     // Check for errors
//     int h_info = 0;
//     cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//     if (h_info != 0) {
//         std::cerr << "Error in cusolverDnZheevd: " << h_info << std::endl;
//         cudaFree(devInfo); // Free the device memory for the info variable
//         return cudaErrorUnknown; // Return an error code if the operation failed   
//     }
//     // Step 3: Compute the von Neumann entropy
//     // 3.1 - Compute the entropy using a kernel
//     int threadsPerBlock = 256;
//     int blocks = (M + threadsPerBlock - 1) / threadsPerBlock;

//     // d_entropy_scratch must be at least `blocks` doubles long. It is since N > 1+N/threadsPerBlock.
//     compute_entropy_multi<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_eigs, d_entropy_scratch, M);
//     cudaDeviceSynchronize();

//     // Now do final sum on host (since blocks is small)
//     std::vector<double> host_scratch(blocks);
//     cudaMemcpy(host_scratch.data(), d_entropy_scratch, blocks * sizeof(double), cudaMemcpyDeviceToHost);
//     entropy = std::accumulate(host_scratch.begin(), host_scratch.end(), 0.0);

//     // Step 4: Clean up
//     cudaFree(devInfo); // Free the device memory for the info variable
//     // Return success
//     return cudaSuccess;
// }

/*

                GETTERS

*/
cuDoubleComplex* CudaMinimizer::getVector() {
    /*
    Returns a pointer to the vector state on the device.

    Returns:
        - Pointer to the vector state on the device.
    */
    return d_vec;
}

// cuDoubleComplex* CudaMinimizer::getInputState() {
//     /*
//     Returns a pointer to the input matrix.

//     Returns:
//         - Pointer to the input matrix on the device.
//     Note:
//         - There is no guarantee that the input matrix is the rank one projector.
//         - To ensure this, run updateProjector() before calling getState().
//     */
//     return d_inmatrix;
// }

// cuDoubleComplex* CudaMinimizer::getOutputState() {
//     /*
//     Returns a pointer to the output matrix.

//     Returns:
//         - Pointer to the output matrix on the device.
//     Note:
//         - The output matrix is the result of applying the channel to the input matrix.
//         - To ensure this, run applyChannel() before calling getOutputState().
//     */
//     return d_outmatrix;
// } 

double CudaMinimizer::getEntropy() {
    /*
    Returns the current von Neumann entropy.

    Returns:
        - Pointer to the entropy value on the device.
    Note:
        - The entropy is computed as the von Neumann entropy of the output matrix.
        - This method does not update the entropy.
    */
    return entropy;
}

CudaMinimizer::~CudaMinimizer() {
    /*
    CudaMinimizer destructor.
    
    Notes:
    - The destructor frees the memory allocated for the internally allocated GPU resources.
    - The d_kraus are not freed, as they are not managed by this class.
    */

    /*OLD
    if (d_vecstate != nullptr) {
        cudaFree(d_vecstate);
    }
    if (d_inmatrix != nullptr) {
        cudaFree(d_inmatrix);
    }
    if (d_outmatrix != nullptr) {
        cudaFree(d_outmatrix);
    }
    if (d_tmpmat != nullptr) {
        cudaFree(d_tmpmat);
    }
    if (d_eigs != nullptr) {
        cudaFree(d_eigs);
    }
    */
    // Free the device memory for the matrices and vectors
    if (d_vec != nullptr) cudaFree(d_vec);
    if (d_vecs_1 != nullptr) cudaFree(d_vecs_1);
    if (d_vecs_2 != nullptr) cudaFree(d_vecs_2);
    if (d_scratch != nullptr) cudaFree(d_scratch);
    if (d_sv_1 != nullptr) cudaFree(d_sv_1);
    if (d_sv_2 != nullptr) cudaFree(d_sv_2);
    // Free the streams and handles
    for (int i = 0; i < num_streams; i++) {
        cublasDestroy(blas_handles[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete[] blas_handles;
    delete[] streams;
}

