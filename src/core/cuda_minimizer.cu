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

    // OTHER USEFUL CONSTANTS and FLAGS
    entropy = -2;
    // We need to know the binary entropy of epsilon, which is used in obtaining the error in the approximation of the entropy
    bin_entropy = -epsilon * std::log(epsilon) - (1-epsilon) * std::log(1-epsilon);
    // The corresponding error is quantified by bin_ent(eps)/2*(1-eps)
    entropy_error = bin_entropy / (2*(1-epsilon));

    // Flags
    input_matrix_scrambled = true; // Flag to indicate if the input matrix has been scrambled

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
        - The data in v is copied to the d_vecstate.
        - No memory management is performed - in particular, memory for v is not freed.
    */
    cudaError_t err = cudaMemcpy(d_vecstate, v, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
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
        - The data in v is simply copied to the d_vecstate.
        - No memory management is performed - in particular, memory for v is not freed.
    */
    cudaError_t err = cudaMemcpy(d_vecstate, v, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying vector from host to device: " << cudaGetErrorString(err) << std::endl;
    }
    return err; // Returns success if the copy was successful

}

cudaError_t CudaMinimizer::initializeRandomVector(){
    /*
    Use the generate_random_vector function to initialize the vector state to a random vector.

    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The vector state is initialized to a random vector of dimension N.
        - The data in the d_vecstate is overwritten.
    */

    // Generate a random vector on the device
    cudaError_t err = generateUniformRandomVectorsCuda(d_vecstate, N, 1);
    return err;
}



cudaError_t CudaMinimizer::updateProjector() {
    /*
    Updates the d_inmatrix to be the projector |v><v|, where v is the vector state.
    
    Returns:d
        - cudaSuccess if the projector was updated successfully.
    Note:
        - The d_inmatrix is updated in place.
        - The d_inmatrix is a NxN matrix, where N is the dimension of the vector state.
    */

    // I need a cublas Handle    

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0); // Scalar alpha for the outer product
    int incx = 1, incy = 1; // Stride for x vector

    // Initialize the d_inmatrix to zero
    cudaMemset(d_inmatrix, 0, N * N * sizeof(cuDoubleComplex));
    // Outer product: A = alpha * x * y^H + A 
    cublasStatus_t status = cublasZgerc(handle, N, N, &alpha, d_vecstate, incx, d_vecstate, incy, d_inmatrix, N);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS Zher failed!" << std::endl;
        return cudaErrorUnknown; // Return an error code
    }
    // Set the flag
    input_matrix_scrambled = false; // The input matrix is now the projector |v><v|, so it is not scrambled
    return cudaSuccess; // Return success
}

// Custom kernel to sum a sequence of matrices. TODO: this can probably be made more efficient by using shared memory or other techniques, but this is a good starting point.
__global__ void reduce_matrix_sum(const cuDoubleComplex* matrices, cuDoubleComplex* output,
                                  int rows, int cols, int num_matrices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    if (idx >= total_elements) return;

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < num_matrices; ++i) {
        cuDoubleComplex val = matrices[i * total_elements + idx];
        sum = cuCadd(sum, val);
    }
    output[idx] = cuCadd(output[idx],sum);
}

cudaError_t CudaMinimizer::applyChannel(){
    /*
    
    Applies the channel represented by the Kraus operators to input matrix. Saves the result in d_outmatrix.
    
    Returns:
        - cudaSuccess if the channel was applied successfully.
    Note:   
        - The input matrix is not checked or updated.
        - The output matrix is updated in place. Its value is overwritten by Phi(d_inmatrix).
        - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
            ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
        
    */

    // Set d_outmatrix to zero
    cudaMemset(d_outmatrix, 0, M * M * sizeof(cuDoubleComplex));

    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    for (int i=0; i < d; i+= ZGEMM_BATCH_SIZE_CHANNEL) {
        // Determine the number of Kraus operators to process in this batch
        int batch_size = std::min(ZGEMM_BATCH_SIZE_CHANNEL, d - i);
        // batched matrix multiplication 1. Use strided one because matrices are contiguous
        // Operation: T[i] = Kraus[i] * d_inmatrix
        // Kraus[i] is MxN, d_inmatrix is NxN, and T[i] is MxN.
        cublasZgemmStridedBatched(handle,
            CUBLAS_OP_N, // No transpose for A
            CUBLAS_OP_N, // No transpose for B
            M,           // First output dimension
            N,           // Second output dimension
            N,           // Common dimension (contracted)
            &one,        // Scalar alpha
            d_kraus + i * M * N, // Pointer to the first element of batch of Kraus operators
            M,           // Leading dimension of Kraus operators (no op)
            M * N,       // Stride for the next batch of Kraus operators
            d_inmatrix, // Pointer to the input matrix
            N,           // Leading dimension of input matrix
            0,           // Stride for the next input matrix (0 because we are using the same input matrix for all batches)
            &zero,       // Scalar beta
            d_tmpmat, // Pointer to the first temporary output matrix for this batch. It is always reused!
            M,           // Leading dimension of temporary output matrix
            M * N,       // Stride for the next temporary output matrix
            batch_size   // Number of matrices in this batch
        );

        // batched matrix multiplication 2.
        // Operation: T[i] = T[i] * conj(Kraus[i]^T) (Notice that the T[i] on RHS is MxN, on the LHS it is MxM!!! But the memory has been allocated so there shouldn't be any problems)
        // T[i] is MxN, conj(Kraus[i]^T) is NxM, and d_outmatrix is MxM.
        cublasZgemmStridedBatched(handle,
            CUBLAS_OP_N, // No transpose for A
            CUBLAS_OP_C, // Conjugate transpose for B
            M,           // First output dimension
            M,           // Second output dimension
            N,           // Contracted dimension
            &one,        // Scalar alpha
            d_tmpmat, // Pointer to the first temporary output matrix for this batch
            M,           // Leading dimension of temporary output matrix
            M * N,       // Stride for the next temporary output matrix
            d_kraus + i * M * N, // Pointer to the first element of batch of Kraus operators
            M,           // Leading dimension of Kraus operators (no op)
            M*N,           // Stride for the next batch of Kraus operators
            &zero,        // Scalar beta
            d_tmpmat + ZGEMM_BATCH_SIZE_CHANNEL * M * N, // Pointer to the first temporary output matrix for this batch
            M,           // Leading dimension of output matrix
            M * M,           // Stride for the next output matrix (0 because we are using the same output matrix for all batches)
            batch_size   // Number of matrices in this batch
        );

        // Now reduce the results to a single matrix
        int total_elements = M * M; // Size of a single matrix, will spawn one thread per entry
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        reduce_matrix_sum<<<blocks, threads>>>(d_tmpmat + ZGEMM_BATCH_SIZE_CHANNEL * M * N, d_outmatrix, M, M, ZGEMM_BATCH_SIZE_CHANNEL);
        cudaDeviceSynchronize();


    }
    return cudaGetLastError(); // Return the last error from CUDA operations
}

cudaError_t CudaMinimizer::applyDualChannel(){
    /*
    
    Applies the channel dual to the one represented by the Kraus operators to d_outmatrix. Saves the result in d_inmatrix.
    
    Returns:
        - cudaSuccess if the channel was applied successfully.
    Note:   
        - The d_outmatrix is not checked or updated.
        - The d_inmatrix is updated in place. Its value is overwritten by Phi^*(d_outmatrix).
        - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
            ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
        
    */

    // Set d_outmatrix to zero
    cudaMemset(d_inmatrix, 0, N * N * sizeof(cuDoubleComplex));

    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    for (int i=0; i < d; i+= ZGEMM_BATCH_SIZE_CHANNEL) {
        // Determine the number of Kraus operators to process in this batch
        int batch_size = std::min(ZGEMM_BATCH_SIZE_CHANNEL, d - i);
        // batched matrix multiplication 1. Use strided one because matrices are contiguous
        // Operation: T[i] = Kraus[i]^H * d_outmatrix
        // Kraus[i] is MxN, d_outmatrix is MxM, and T[i] is NxM.
        cublasZgemmStridedBatched(handle,
            CUBLAS_OP_C, // Conjugate transpose for A
            CUBLAS_OP_N, // No transpose for B
            N,           // First output dimension
            M,           // Second output dimension
            M,           // Common dimension
            &one,        // Scalar alpha
            d_kraus + i * M * N, // Pointer to the first element of batch of Kraus operators
            M,           // Leading dimension of Kraus operators
            M * N,       // Stride for the next batch of Kraus operators
            d_outmatrix, // Pointer to the output matrix
            M,           // Leading dimension of output matrix
            0,           // Stride for the next output matrix (0 because we are using the same matrix for all batches)
            &zero,       // Scalar beta
            d_tmpmat, // Pointer to the first temporary output matrix for this batch. It is always reused!
            N,           // Leading dimension of temporary output matrix
            N * M,       // Stride for the next temporary output matrix
            batch_size   // Number of matrices in this batch
        );

        // batched matrix multiplication 2.
        // Operation: T'[i] = T[i] * Kraus[i] 
        // T[i] is NxM, Kraus[i] is MxN, and T'[i] is NxN.
        cublasZgemmStridedBatched(handle,
            CUBLAS_OP_N, // No transpose for A
            CUBLAS_OP_N, // No transpose for B
            N,           // Number of rows of op(T[i]) (i.e. first output dimension)
            N,           // Number of columns of op(Kraus[i]) (i.e. second output dimension)
            M,           // Common dimension
            &one,        // Scalar alpha
            d_tmpmat, // Pointer to the first temporary output matrix for this batch
            N,           // Leading dimension of temporary output matrix
            N * M,       // Stride for the next temporary output matrix
            d_kraus + i * N * M, // Pointer to the first element of batch of Kraus operators
            M,           // Leading dimension of Kraus operators (before op)
            M*N,         // Stride for the next batch of Kraus operators
            &zero,       // Scalar beta
            d_tmpmat + ZGEMM_BATCH_SIZE_CHANNEL * M * N, // Pointer to the first temporary output matrix for this batch
            N,           // Leading dimension of output matrix
            N * N,       // Stride for the next output matrix (0 because we are using the same output matrix for all batches)
            batch_size   // Number of matrices in this batch
        );

        // Now reduce the results to a single matrix
        int total_elements = M * M; // Size of a single matrix, will spawn one thread per entry
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        reduce_matrix_sum<<<blocks, threads>>>(d_tmpmat + ZGEMM_BATCH_SIZE_CHANNEL * M * N, d_inmatrix, N, N, ZGEMM_BATCH_SIZE_CHANNEL);
        cudaDeviceSynchronize();


    }
    return cudaGetLastError(); // Return the last error from CUDA operations
}

__global__ void perturbByEpsilon(cuDoubleComplex* matrix, int dimension, double epsilon) {
    /*
    Kernel to perturb a square matrix in the follwing way:

            A -> (1-epsilon)* A + epsilon/d * I
    
    Arguments:
        - matrix (device): pointer to the matrix to perturb.
        - dimension (host): dimension of the matrix.
        - epsilon (host): perturbation value.
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dimension * dimension) {
        int row = idx / dimension;
        int col = idx % dimension;
        // Apply the perturbation
        matrix[idx] = cuCmul(matrix[idx], make_cuDoubleComplex(1.0 - epsilon, 0.0));
        if (row == col) {
            matrix[idx] = cuCadd(matrix[idx], make_cuDoubleComplex(epsilon / dimension, 0.0));
        }
    }
}




cudaError_t CudaMinimizer::applyEpsilonChannel(){
    /*
    Applies the channel represented by the Kraus operators to input matrix, perturbing it by epsilon.
    Saves the result in d_outmatrix.
    Returns:
        - cudaSuccess if the channel was applied successfully.
    Note:
        - The input matrix is not checked or updated.
        - The output matrix is updated in place. Its value is overwritten by Phi(d_inmatrix).
        - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
            ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
    */

    // Step 1: Apply the channel to the input matrix
    applyChannel();
    // Print error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error applying channel: " << cudaGetErrorString(err) << std::endl;
        return err; // Return the error if the channel application failed
    }
    // Step 2: Perturb the output matrix by epsilon
    int threads = 256;
    int blocks = (M * M + threads - 1) / threads;
    // Launch the kernel to perturb the output matrix
    perturbByEpsilon<<<blocks, threads>>>(d_outmatrix, M, epsilon);
    return cudaGetLastError();
}

cudaError_t CudaMinimizer::applyEpsilonDualChannel(){
    /*
    Applies the channel dual to the one represented by the Kraus operators to d_outmatrix, perturbing it by epsilon.
    Saves the result in d_inmatrix.
    Returns:
        - cudaSuccess if the channel was applied successfully.
    Note:
        - The d_outmatrix is not checked or updated.
        - The d_inmatrix is updated in place. Its value is overwritten by Phi^*(d_outmatrix).
        - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
            ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
    */
    // Step 1: Apply the dual channel to the output matrix
    applyDualChannel();
    // Step 2: Perturb the input matrix by epsilon
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    perturbByEpsilon<<<blocks, threads>>>(d_inmatrix, N, epsilon);
    return cudaGetLastError(); // Return the last error from CUDA operations
}


/*

            Algorithm

*/


__global__ void computeLog(double* eigs, int dimension) {
    /*
    Kernel to compute the logarithm of the eigenvalues in place
    
    Arguments:
        - eigs (device): pointer to the eigenvalues.
        - dimension (host): number of eigenvalues.
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dimension) {
        double eig = eigs[idx];
        if (eig > 0) {
            eigs[idx] = std::log(eig);
        } else {
            eigs[idx] = -INFINITY; // Handle non-positive eigenvalues
        }
    }
}

__global__ void scale_columns(cuDoubleComplex* A, double* scalars, cuDoubleComplex* out, int rows, int cols) {
    /*
    Kernel to scale each column of a matrix by a corresponding scalar value.
    Arguments:
        - A (device): pointer to the matrix to scale.
        - scalars (device): pointer to an array of scalars, one for each column.
        - out (device): pointer to the output matrix where the scaled values will be stored.
        - rows (host): number of rows in the matrix.
        - cols (host): number of columns in the matrix.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        cuDoubleComplex scale = make_cuDoubleComplex(scalars[col], 0.0); // Get the scalar for the current column
        cuDoubleComplex val = A[col * rows + row];  // column-major
        out[col * rows + row] = cuCmul(val, scale);
    }
}

cudaError_t CudaMinimizer::stepAlgorithm(){

    /*
    Steps the minimization algorithm by performing the following operations:
        1. Update the projector based on the vector.
        2. Compute Phi_e(rho) by applying the epsilon channel to the input matrix.
        3. Compute log(Phi_e(rho)) by diagonalizing the output matrix and reconstructing it.
        4. Compute Phi_e^*(log(Phi_e(rho))) by applying the dual channel to the output matrix.
        5. Find the eigenvector with the highest eigenvalue in the input matrix and update the vector state.

    Returns:
        - cudaSuccess if the step was successful.
    */
    // Step 1: update the projector
    if (input_matrix_scrambled) {
        cudaError_t err_update = updateProjector();
        if (err_update != cudaSuccess) {
            std::cerr << "Error updating projector: " << cudaGetErrorString(err_update) << std::endl;
            return err_update; // Return the error if the projector update failed
        }
    }

    // Step 2: compute Phi_e(rho)
    cudaError_t err_eps_ch = applyEpsilonChannel();
    if (err_eps_ch != cudaSuccess) {
        std::cerr << "Error applying epsilon channel: " << cudaGetErrorString(err_eps_ch) << std::endl;
        return err_eps_ch;
    }

    // Step 3: compute log(Phi_e(rho))
    // 3.1 diagonalize output matrix

    // 3.1.1 - Query buffer size
    int lwork = 0; // Workspace size
    cusolverDnZheevd_bufferSize(
        cusolver_handle,
        CUSOLVER_EIG_MODE_VECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
        CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
        M,                          // Size of the matrix (M x M)
        d_outmatrix,              // Pointer to the matrix to be diagonalized (d_outmatrix). It gets overwritten (on success) with the eigenvectors.
        M,                          // Leading dimension of the matrix (M)
        d_eigs,                       // Vector of eigenvalues of the matrix.
        &lwork
    );                    // Pointer to the workspace size (output parameter). Stored on host.

    // 3.1.2 - Check for available workspace. Allocate remaining resources.
    // For now, reuse the scratch space "temp_matrix" that was allocated for the channel application.
    if (2 * ZGEMM_BATCH_SIZE_CHANNEL * (std::max(N,M)*std::max(N,M)) < lwork) {
        // Try to allocate more memory for the temporary matrix
        cudaFree(d_tmpmat);
        cudaError_t err_alloc = cudaMalloc((void**)&d_tmpmat, lwork * sizeof(cuDoubleComplex));
        if (err_alloc != cudaSuccess) {
            std::cerr << "Error allocating memory for temporary matrix: " << cudaGetErrorString(err_alloc) << std::endl;
            return err_alloc; // Return the error if memory allocation fails
        }
    }
    int* devInfo = nullptr;
    cudaMalloc(&devInfo, sizeof(int)); // Allocate device memory for the info variable


    // 3.1.3 - Call the solver
    cusolverDnZheevd(
        cusolver_handle,            // Handle to the cuSOLVER context
        CUSOLVER_EIG_MODE_VECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
        CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
        M,                          // Size of the matrix (M x M)
        d_outmatrix,              // Pointer to the matrix to be diagonalized (d_outmatrix). It gets overwritten (on success) with the eigenvectors.
        M,                          // Leading dimension of the matrix (M) 
        d_eigs,                       // Vector of eigenvalues of the matrix.
        d_tmpmat,                 // Pointer to the workspace (temporary matrix). It is used to store intermediate results.
        lwork,                      // Size of the workspace (temporary matrix)
        devInfo);                   // Pointer to the info variable (output parameter). It indicates success or failure of the operation.

    // 3.1.4 - Check for errors
    int h_info = 0;
    cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "Error in cusolverDnZheevd: " << h_info << std::endl;
        return cudaErrorUnknown; // Return an error code if the operation failed
    }

    // 3.2 - compute log(d_eigs)
    // We can use a kernel to compute the logarithm of the eigenvalues in place    
    int threads = 256;
    int blocks = (M + threads - 1) / threads; // Number of blocks
    computeLog<<<blocks, threads>>>(d_eigs, M); // Launch the kernel to compute the logarithm of the eigenvalues
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    cudaError_t err_log = cudaGetLastError(); // Get the last error from CUDA operations
    if (err_log != cudaSuccess) {
        std::cerr << "Error computing logarithm of eigenvalues: " << cudaGetErrorString(err_log) << std::endl;
        return err_log; // Return the error if the kernel failed
    }

    // 3.3 - reconstruct the matrix
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // 3.3.1: First perform diag * d_eigs, store result in the first entries of d_tmpmat
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + 15) / 16, (M + 15) / 16);
    scale_columns<<<numBlocks, threadsPerBlock>>>(d_outmatrix, d_eigs, d_tmpmat, M, M);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    cudaError_t err_scale = cudaGetLastError(); // Get the last error from CUDA operations
    if (err_scale != cudaSuccess) {
        std::cerr << "Error scaling columns: " << cudaGetErrorString(err_scale) << std::endl;
        return err_scale; // Return the error if the kernel failed
    }
    // 3.3.2: Then do d_tmpmat*conj(d_outmatrix^T) -> d_tmpmat'
    cublasStatus_t status = cublasZgemm(
        handle,
        CUBLAS_OP_N, // Conjugate transpose for A
        CUBLAS_OP_C, // No transpose for B
        M,           // Number of rows of op(A)
        M,           // Number of columns of op(B)
        M,           // Common dimension
        &one,        // Scalar alpha
        d_tmpmat, // Pointer to the first matrix (d_outmatrix)
        M,           // Leading dimension of d_outmatrix
        d_outmatrix,  // Pointer to the second matrix (d_tmpmat)
        M,           // Leading dimension of d_tmpmat
        &zero,       // Scalar beta
        d_tmpmat + M * M, // Pointer to the output matrix (d_tmpmat')
        M            // Leading dimension of output matrix (d_tmpmat')
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS Zgemm failed, when recombining log(Phi(rho))!" << std::endl;
        return cudaErrorUnknown; // Return an error code if the operation failed
    }
    // 3.3.3: Copy the result back to d_outmatrix
    cudaMemcpy(d_outmatrix, d_tmpmat + M * M, M * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    // Now d_outmatrix contains log(Phi_e(rho)) in the form of a Hermitian matrix.

    // Step 4: compute Phi_e^*(log(Phi_e(rho)))
    cudaError_t err_dual_ch = applyEpsilonDualChannel();
    if (err_dual_ch != cudaSuccess) {
        std::cerr << "Error applying dual epsilon channel: " << cudaGetErrorString(err_dual_ch) << std::endl;
        return err_dual_ch; 
    }

    // Step 5: find eigenvector with highest eigenvalue
    // 5.1 - Diagonalize the d_inmatrix to find the eigenvalues and eigenvectors
    // 5.1.1 - Query buffer size
    lwork = 0; // Reset workspace size
    cusolverDnZheevd_bufferSize(
        cusolver_handle,
        CUSOLVER_EIG_MODE_VECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
        CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
        N,                          // Size of the matrix (N x N)
        d_inmatrix,               // Pointer to the matrix to be diagonalized (d_inmatrix). It gets overwritten (on success) with the eigenvectors.
        N,                          // Leading dimension of the matrix (N)
        d_eigs,                       // Vector of eigenvalues of the matrix.
        &lwork                       // Pointer to the workspace size (output parameter). Stored on host.
    );
    // 5.1.2 - Check for available workspace. Allocate remaining resources.
    // For now, reuse the scratch space "temp_matrix" that was allocated for the channel application.
    if (2 * ZGEMM_BATCH_SIZE_CHANNEL * std::max(N,M) * std::max(N,M) < lwork) {
        // Try to allocate more memory for the temporary matrix
        cudaFree(d_tmpmat);
        cudaError_t err_alloc = cudaMalloc((void**)&d_tmpmat, lwork * sizeof(cuDoubleComplex));
        if (err_alloc != cudaSuccess) {
            std::cerr << "Error allocating memory for temporary matrix: " << cudaGetErrorString(err_alloc) << std::endl;
            return err_alloc; // Return the error if memory allocation fails
        }
    }
    // 5.1.3 - Call the solver
    cusolverDnZheevd(
        cusolver_handle,            // Handle to the cuSOLVER context
        CUSOLVER_EIG_MODE_VECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
        CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
        N,                          // Size of the matrix (N x N)
        d_inmatrix,               // Pointer to the matrix to be diagonalized (d_inmatrix). It gets overwritten (on success) with the eigenvectors.
        N,                          // Leading dimension of the matrix (N)
        d_eigs,                       // Vector of eigenvalues of the matrix.
        d_tmpmat,                 // Pointer to the workspace (temporary matrix). It is used to store intermediate results.
        lwork,                      // Size of the workspace (temporary matrix)
        devInfo);                   // Pointer to the info variable (output parameter). It indicates success or failure of the operation.

    // Set flag
    input_matrix_scrambled = true; // The input matrix is now scrambled, as it has been diagonalized and contains the eigenvectors. Projector should be computed again.

    // 5.1.4 - Check for errors
    h_info = 0;
    cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "Error in cusolverDnZheevd: " << h_info << std::endl;
        return cudaErrorUnknown; // Return an error code if the operation failed
    }
    // 5.2 - Copy the eigenvector corresponding to the highest eigenvalue to the vector state
    // The eigenvectors are stored in the columns of d_inmatrix, so we need to copy
    // the last column (corresponding to the highest eigenvalue) to the d_vecstate
    cudaMemcpy(d_vecstate, d_inmatrix + (N - 1) * N, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    // Check for success
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Error copying eigenvector to d_vecstate: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return cudaGetLastError(); // Return the error if the copy failed
    }

    // Step 6: Clean up
    cudaFree(devInfo); // Free the device memory for the info variable

    // Return success
    return cudaSuccess;
    

    
}

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
    // Step 1: Run the algorithm step
    cudaError_t err_step = stepAlgorithm();
    if (err_step != cudaSuccess) {
        std::cerr << "Error in stepAlgorithm: " << cudaGetErrorString(err_step) << std::endl;
        return err_step; // Return the error if the step failed 
    }

    // Step 2: Recompute the projector
    cudaError_t err_projector = updateProjector();
    if (err_projector != cudaSuccess) {
        std::cerr << "Error updating projector: " << cudaGetErrorString(err_projector) << std::endl;
        return err_projector; // Return the error if the projector update failed
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

__global__ void compute_entropy_multi(double* eigs, double* scratch, int size) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    double val = 0.0;
    if (i < size) {
        double p = eigs[i];
        if (p > 1e-15)
            val = -p * log(p);
    }

    sdata[tid] = val;
    __syncthreads();

    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // First thread of block writes block result to global scratch
    if (tid == 0)
        scratch[blockIdx.x] = sdata[0];
}

cudaError_t CudaMinimizer::calculateEpsilonEntropy(){
    /*
    Calculates the von Neumann entropy of the output matrix after applying the epsilon channel.
    The entropy is computed as the von Neumann entropy of the output matrix.

    Returns:
        - cudaSuccess if the entropy was calculated successfully.
    Note:
        - The input matrix is NOT updated. Check therefore that it contains the correct state!
        - The entropy is stored in the member variable `entropy`.
    */

    // Step 1: Apply the epsilon channel
    cudaError_t err_eps_ch = applyEpsilonChannel();
    if (err_eps_ch != cudaSuccess) {
        std::cerr << "Error applying epsilon channel: " << cudaGetErrorString(err_eps_ch) << std::endl;
        return err_eps_ch;
    }

    // Step 2: Diagonalize the output matrix to get eigenvalues
    int lwork = 0; // Workspace size
    cusolverDnZheevd_bufferSize(
        cusolver_handle,
        CUSOLVER_EIG_MODE_NOVECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
        CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
        M,                          // Size of the matrix (M x M)
        d_outmatrix,              // Pointer to the matrix to be diagonalized (d_outmatrix). It gets overwritten (on success) with the eigenvectors.
        M,                          // Leading dimension of the matrix (M)
        d_eigs,                       // Vector of eigenvalues of the matrix.
        &lwork                      // Pointer to the workspace size (output parameter). Stored on host.
    );

    // Check for available workspace. Allocate remaining resources.
    if (2 * ZGEMM_BATCH_SIZE_CHANNEL * M * M < lwork) {
        // Try to allocate more memory for the temporary matrix
        cudaFree(d_tmpmat);
        cudaError_t err_alloc = cudaMalloc((void**)&d_tmpmat, lwork * sizeof(cuDoubleComplex));
        if (err_alloc != cudaSuccess) {
            std::cerr << "Error allocating memory for temporary matrix: " << cudaGetErrorString(err_alloc) << std::endl;
            return err_alloc; // Return the error if memory allocation
        }
    }
    int* devInfo = nullptr;
    cudaMalloc(&devInfo, sizeof(int)); // Allocate device memory for the info variable
    // Call the solver to diagonalize the output matrix
    cusolverDnZheevd(
        cusolver_handle,            // Handle to the cuSOLVER context
        CUSOLVER_EIG_MODE_NOVECTOR,   // CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR (no eigenvectors)
        CUBLAS_FILL_MODE_UPPER,     // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER (upper or lower triangular part of the matrix is stored)
        M,                          // Size of the matrix (M x M)
        d_outmatrix,              // Pointer to the matrix to be diagonalized (d_outmatrix). It gets overwritten (on success) with the eigenvectors.
        M,                          // Leading dimension of the matrix (M)
        d_eigs,                       // Vector of eigenvalues of the matrix.
        d_tmpmat,                 // Pointer to the workspace (temporary matrix). It is used to store intermediate results.
        lwork,                      // Size of the workspace (temporary matrix)
        devInfo);                   // Pointer to the info variable (output parameter). It indicates success or failure of the operation.
    // Check for errors
    int h_info = 0;
    cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "Error in cusolverDnZheevd: " << h_info << std::endl;
        cudaFree(devInfo); // Free the device memory for the info variable
        return cudaErrorUnknown; // Return an error code if the operation failed   
    }
    // Step 3: Compute the von Neumann entropy
    // 3.1 - Compute the entropy using a kernel
    int threadsPerBlock = 256;
    int blocks = (M + threadsPerBlock - 1) / threadsPerBlock;

    // d_entropy_scratch must be at least `blocks` doubles long. It is since N > 1+N/threadsPerBlock.
    compute_entropy_multi<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_eigs, d_entropy_scratch, M);
    cudaDeviceSynchronize();

    // Now do final sum on host (since blocks is small)
    std::vector<double> host_scratch(blocks);
    cudaMemcpy(host_scratch.data(), d_entropy_scratch, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    entropy = std::accumulate(host_scratch.begin(), host_scratch.end(), 0.0);

    // Step 4: Clean up
    cudaFree(devInfo); // Free the device memory for the info variable
    // Return success
    return cudaSuccess;
}

/*

                GETTERS

*/
cuDoubleComplex* CudaMinimizer::getVector() {
    /*
    Returns a pointer to the vector state on the device.

    Returns:
        - Pointer to the vector state on the device.
    */
    return d_vecstate;
}

cuDoubleComplex* CudaMinimizer::getInputState() {
    /*
    Returns a pointer to the input matrix.

    Returns:
        - Pointer to the input matrix on the device.
    Note:
        - There is no guarantee that the input matrix is the rank one projector.
        - To ensure this, run updateProjector() before calling getState().
    */
    return d_inmatrix;
}

cuDoubleComplex* CudaMinimizer::getOutputState() {
    /*
    Returns a pointer to the output matrix.

    Returns:
        - Pointer to the output matrix on the device.
    Note:
        - The output matrix is the result of applying the channel to the input matrix.
        - To ensure this, run applyChannel() before calling getOutputState().
    */
    return d_outmatrix;
} 

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

/*





int Minimizer::step(){
    stepAlgorithm();
    calculateEntropy();
    return 0;
}

/// GETTERS

int Minimizer::getN(){
    return N;
}

int Minimizer::getD(){
    return d;
}



*/
CudaMinimizer::~CudaMinimizer() {
    /*
    CudaMinimizer destructor.
    
    Notes:
    - The destructor frees the memory allocated for the internally allocated GPU resources.
    - The d_kraus are not freed, as they are not managed by this class.
    */

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

}

