// Minimizer.cpp
#include "common_includes.h"
#include "core/cuda_minimizer.h"
#include "config/config.h"
#include "core/matrix_operations.h"
#include "core/generate_random_vector.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>

CudaMinimizer::CudaMinimizer(cuDoubleComplex* kraus_ops, int kraus_number, int kraus_in_dimension, int kraus_out_dimension, double eps) {
    /*
    CudaMinimizer constructor.
    
    Arguments:
    - kraus_ops (device): pointer to an array of double precision complex numbers (cuComplexDouble), which contains the Kraus operators.
    - kraus_number (host): pointer to number of Kraus operators.
    - kraus_in_dimension (host): pointer to the input dimension of the Kraus operators.
    - kraus_out_dimension (host): pointer to the output dimension of the Kraus operators
    - eps (host): pointer to the precision of the minimization algorithm.

    Notes:
    - The kraus_ops array should be an array of d contiguous NxN matrices, each stored in column-major order.
      So entry (i,j) of matrix k corresponds to index k*N*N+j*N+i.
    - CudaMinimizer is NOT responsible for management of these resources and just receives pointers to them. No data is copied.    
    */

    // Step 1: save the pointers that were passed.
    kraus_operators = kraus_ops;
    d = kraus_number;
    N = kraus_in_dimension;
    M = kraus_out_dimension;
    epsilon = eps; 

    // Step 2: initialize the other members.
    // - Vector state: this is the rank one vector that we want to improve
    vector_state = nullptr;
    cudaMalloc((void**)&vector_state, N * sizeof(cuDoubleComplex));
    // - Input matrix: this will be initialized to the rank one projector |v><v|, where v is the vector state.
    input_matrix = nullptr;
    cudaMalloc((void**)&input_matrix, N * N * sizeof(cuDoubleComplex));   
    // - Output matrix: this will be the output of the channel, which is a M x M matrix.
    output_matrix = nullptr;
    cudaMalloc((void**)&output_matrix, M * M * sizeof(cuDoubleComplex));
    // - Scratch space for zgemm channel application: this will be a temporary matrix used in the channel application.
    // I need to store ZGEMM_BATCH_SIZE_CHANNEL M*N matrices for first intermediate step, and Z_GEMM_BATCH_SIZE_CHANNEL M*M or NxN matrices for the second intermediate step.
    tmp_matrix = nullptr;
    int max_dim = std::max(N,M);
    cudaMalloc((void**)&tmp_matrix, ZGEMM_BATCH_SIZE_CHANNEL * ( M * N + max_dim * max_dim) * sizeof(cuDoubleComplex));

    // OTHER USEFUL CONSTANTS
    entropy = -1;
    // We need to know the binary entropy of epsilon, which is used in obtaining the error in the approximation of the entropy
    bin_entropy = -epsilon * std::log(epsilon) - (1-epsilon) * std::log(1-epsilon);
    // The corresponding error is quantified by bin_ent(eps)/2*(1-eps)
    entropy_error = bin_entropy / (2*(1-epsilon));
    // Initiate cublas handle
    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed!" << std::endl;
        throw std::runtime_error("CUBLAS initialization failed");
    }
}

cudaError_t CudaMinimizer::initializeVector(cuDoubleComplex* v) {
    /*
    Initializes the vector state to a given vector v.

    Arguments:
        - v (device): pointer to a vector of double precision complex numbers (cuComplexDouble), which is the vector to initialize the state to.
    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The dimension is not checked! If the dimension does not match, undefined behavior.
        - The data in v is copied to the vector_state.
        - No memory management is performed - in particular, memory for v is not freed.
    */
    cudaError_t err = cudaMemcpy(vector_state, v, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    return err;
}

cudaError_t CudaMinimizer::initializeRandomVector(){
    /*
    Use the generate_random_vector function to initialize the vector state to a random vector.

    Returns:
        - cudaSuccess if the vector was initialized successfully.
    Note:
        - The vector state is initialized to a random vector of dimension N.
        - The data in the vector_state is overwritten.
    */

    // Generate a random vector on the device
    cudaError_t err = generateUniformRandomVectorsCuda(vector_state, N, 1);
    return err;
}



cudaError_t CudaMinimizer::updateProjector() {
    /*
    Updates the input_matrix to be the projector |v><v|, where v is the vector state.
    
    Returns:d
        - cudaSuccess if the projector was updated successfully.
    Note:
        - The input_matrix is updated in place.
        - The input_matrix is a NxN matrix, where N is the dimension of the vector state.
    */

    // I need a cublas Handle    

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0); // Scalar alpha for the outer product
    int incx = 1, incy = 1; // Stride for x vector

    // Initialize the input_matrix to zero
    cudaMemset(input_matrix, 0, N * N * sizeof(cuDoubleComplex));
    // Outer product: A = alpha * x * y^H + A 
    cublasStatus_t status = cublasZgerc(handle, N, N, &alpha, vector_state, incx, vector_state, incy, input_matrix, N);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS Zher failed!" << std::endl;
        return cudaErrorUnknown; // Return an error code
    }
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
    
    Applies the channel represented by the Kraus operators to input matrix. Saves the result in output_matrix.
    
    Returns:
        - cudaSuccess if the channel was applied successfully.
    Note:   
        - The input matrix is not checked or updated.
        - The output matrix is updated in place. Its value is overwritten by Phi(input_matrix).
        - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
            ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
        
    */

    // Set output_matrix to zero
    cudaMemset(output_matrix, 0, M * M * sizeof(cuDoubleComplex));

    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    for (int i=0; i < d; i+= ZGEMM_BATCH_SIZE_CHANNEL) {
        // Determine the number of Kraus operators to process in this batch
        int batch_size = std::min(ZGEMM_BATCH_SIZE_CHANNEL, d - i);
        // batched matrix multiplication 1. Use strided one because matrices are contiguous
        // Operation: T[i] = Kraus[i] * input_matrix
        // Kraus[i] is MxN, input_matrix is NxN, and T[i] is MxN.
        cublasZgemmStridedBatched(handle,
            CUBLAS_OP_N, // No transpose for A
            CUBLAS_OP_N, // No transpose for B
            M,           // First output dimension
            N,           // Second output dimension
            N,           // Common dimension (contracted)
            &one,        // Scalar alpha
            kraus_operators + i * M * N, // Pointer to the first element of batch of Kraus operators
            M,           // Leading dimension of Kraus operators (no op)
            M * N,       // Stride for the next batch of Kraus operators
            input_matrix, // Pointer to the input matrix
            N,           // Leading dimension of input matrix
            0,           // Stride for the next input matrix (0 because we are using the same input matrix for all batches)
            &zero,       // Scalar beta
            tmp_matrix, // Pointer to the first temporary output matrix for this batch. It is always reused!
            M,           // Leading dimension of temporary output matrix
            M * N,       // Stride for the next temporary output matrix
            batch_size   // Number of matrices in this batch
        );

        // batched matrix multiplication 2.
        // Operation: T[i] = T[i] * conj(Kraus[i]^T) (Notice that the T[i] on RHS is MxN, on the LHS it is MxM!!! But the memory has been allocated so there shouldn't be any problems)
        // T[i] is MxN, conj(Kraus[i]^T) is NxM, and output_matrix is MxM.
        cublasZgemmStridedBatched(handle,
            CUBLAS_OP_N, // No transpose for A
            CUBLAS_OP_C, // Conjugate transpose for B
            M,           // First output dimension
            M,           // Second output dimension
            N,           // Contracted dimension
            &one,        // Scalar alpha
            tmp_matrix, // Pointer to the first temporary output matrix for this batch
            M,           // Leading dimension of temporary output matrix
            M * N,       // Stride for the next temporary output matrix
            kraus_operators + i * M * N, // Pointer to the first element of batch of Kraus operators
            M,           // Leading dimension of Kraus operators (no op)
            M*N,           // Stride for the next batch of Kraus operators
            &zero,        // Scalar beta
            tmp_matrix + ZGEMM_BATCH_SIZE_CHANNEL * M * N, // Pointer to the first temporary output matrix for this batch
            M,           // Leading dimension of output matrix
            M * M,           // Stride for the next output matrix (0 because we are using the same output matrix for all batches)
            batch_size   // Number of matrices in this batch
        );

        // Now reduce the results to a single matrix
        int total_elements = M * M; // Size of a single matrix, will spawn one thread per entry
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        reduce_matrix_sum<<<blocks, threads>>>(tmp_matrix + ZGEMM_BATCH_SIZE_CHANNEL * M * N, output_matrix, M, M, ZGEMM_BATCH_SIZE_CHANNEL);
        cudaDeviceSynchronize();


    }
    return cudaGetLastError(); // Return the last error from CUDA operations
}

cudaError_t CudaMinimizer::applyDualChannel(){
    /*
    
    Applies the channel dual to the one represented by the Kraus operators to output_matrix. Saves the result in input_matrix.
    
    Returns:
        - cudaSuccess if the channel was applied successfully.
    Note:   
        - The output_matrix is not checked or updated.
        - The input_matrix is updated in place. Its value is overwritten by Phi^*(output_matrix).
        - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
            ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
        
    */

    // Set output_matrix to zero
    cudaMemset(input_matrix, 0, N * N * sizeof(cuDoubleComplex));

    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    for (int i=0; i < d; i+= ZGEMM_BATCH_SIZE_CHANNEL) {
        // Determine the number of Kraus operators to process in this batch
        int batch_size = std::min(ZGEMM_BATCH_SIZE_CHANNEL, d - i);
        // batched matrix multiplication 1. Use strided one because matrices are contiguous
        // Operation: T[i] = Kraus[i]^H * Output_matrix
        // Kraus[i] is MxN, output_matrix is MxM, and T[i] is NxM.
        cublasZgemmStridedBatched(handle,
            CUBLAS_OP_C, // Conjugate transpose for A
            CUBLAS_OP_N, // No transpose for B
            N,           // First output dimension
            M,           // Second output dimension
            M,           // Common dimension
            &one,        // Scalar alpha
            kraus_operators + i * M * N, // Pointer to the first element of batch of Kraus operators
            M,           // Leading dimension of Kraus operators
            M * N,       // Stride for the next batch of Kraus operators
            output_matrix, // Pointer to the output matrix
            M,           // Leading dimension of output matrix
            0,           // Stride for the next output matrix (0 because we are using the same matrix for all batches)
            &zero,       // Scalar beta
            tmp_matrix, // Pointer to the first temporary output matrix for this batch. It is always reused!
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
            tmp_matrix, // Pointer to the first temporary output matrix for this batch
            N,           // Leading dimension of temporary output matrix
            N * M,       // Stride for the next temporary output matrix
            kraus_operators + i * N * M, // Pointer to the first element of batch of Kraus operators
            M,           // Leading dimension of Kraus operators (before op)
            M*N,         // Stride for the next batch of Kraus operators
            &zero,       // Scalar beta
            tmp_matrix + ZGEMM_BATCH_SIZE_CHANNEL * M * N, // Pointer to the first temporary output matrix for this batch
            N,           // Leading dimension of output matrix
            N * N,       // Stride for the next output matrix (0 because we are using the same output matrix for all batches)
            batch_size   // Number of matrices in this batch
        );

        // Now reduce the results to a single matrix
        int total_elements = M * M; // Size of a single matrix, will spawn one thread per entry
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        reduce_matrix_sum<<<blocks, threads>>>(tmp_matrix + ZGEMM_BATCH_SIZE_CHANNEL * M * N, input_matrix, N, N, ZGEMM_BATCH_SIZE_CHANNEL);
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
    Saves the result in output_matrix.
    Returns:
        - cudaSuccess if the channel was applied successfully.
    Note:
        - The input matrix is not checked or updated.
        - The output matrix is updated in place. Its value is overwritten by Phi(input_matrix).
        - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
            ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
    */
    // Step 1: Apply the channel to the input matrix
    applyChannel();
    // Step 2: Perturb the output matrix by epsilon
    int threads = 256;
    int blocks = (M * M + threads - 1) / threads;
    perturbByEpsilon<<<blocks, threads>>>(output_matrix, M, epsilon);
    return cudaGetLastError(); // Return the last error from CUDA operations    
}

cudaError_t CudaMinimizer::applyEpsilonDualChannel(){
    /*
    Applies the channel dual to the one represented by the Kraus operators to output_matrix, perturbing it by epsilon.
    Saves the result in input_matrix.
    Returns:
        - cudaSuccess if the channel was applied successfully.
    Note:
        - The output_matrix is not checked or updated.
        - The input_matrix is updated in place. Its value is overwritten by Phi^*(output_matrix).
        - The matrix multiplication process is batched. There is a tradeoff in memory vs. speed.
            ZGEMM_BATCH_SIZE_CHANNEL is the number of batches to use for the matrix multiplication
    */
    // Step 1: Apply the dual channel to the output matrix
    applyDualChannel();
    // Step 2: Perturb the input matrix by epsilon
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    perturbByEpsilon<<<blocks, threads>>>(input_matrix, N, epsilon);
    return cudaGetLastError(); // Return the last error from CUDA operations
}


/*

            Algorithm

*/

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
    updateProjector();

    // Step 2: compute Phi_e(rho)
    cudaError_t err_eps_ch = applyEpsilonChannel();
    if (err_eps_ch != cudaSuccess) {
        std::cerr << "Error applying epsilon channel: " << cudaGetErrorString(err_eps_ch) << std::endl;
        return err_eps_ch;
    }

    // Step 3: compute log(Phi_e(rho))

    
    
}

/*
int Minimizer::stepAlgorithm(){
    // Step 1: update the projector based on the vector
    updateProjector();

    // Step 2: compute Phi_e(rho)
    applyEpsilonChannel(kraus_operators,input_matrix, output_matrix, d, N, M, epsilon);

    // Step 3: compute log(Phi_e(rho))
    // Step 3.1: diagonalize output matrix
    std::vector<double> eigvals(M);
    zheev_wrapper('V', 'U', M, output_matrix,M,&eigvals);
    // Step 3.2: compute log(eigs)
    std::vector<std::complex<double> >* eig_mat = new std::vector<std::complex<double> >(N*N,std::complex<double>(0.0f,0.0f));
    for (int i=0; i<M; i++){
        eig_mat->at(N*i+i) = std::log(eigvals.at(i));
    }

    // Step 3.3: reconstruct the matrix
    //output matrix contains the eigenvectors as columns. Perform matrix multiplication.
    // First perform diag * eigs, store result in another temp mat
    std::vector<std::complex<double> >* tmp_mat = new std::vector<std::complex<double> >(N*N);
    std::complex<double> one(1.0f,0.0f);
    std::complex<double> zero(0.0f,0.0f);
    cblas_zgemm(CblasColMajor,CblasNoTrans,CblasConjTrans,M,M,M,&one,reinterpret_cast<lapack_complex_t*>(eig_mat->data()),M,reinterpret_cast<lapack_complex_t*>(output_matrix->data()),M,&zero,reinterpret_cast<lapack_complex_t*>(tmp_mat->data()),M);
    // Then do conj(eigs^T)*tmp_mat
    cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,M,M,&one,reinterpret_cast<lapack_complex_t*>(output_matrix->data()),M,reinterpret_cast<lapack_complex_t*>(tmp_mat->data()),M,&zero,reinterpret_cast<lapack_complex_t*>(eig_mat->data()),M);
    for (int i=0; i< M*M; i++){
        output_matrix->at(i) = eig_mat->at(i);
    }
    delete eig_mat;
    delete tmp_mat;

    // As far as I can tell, the matrix logarithm is calculated correctly.

    // Step 4: compute Phi_e^*(log(Phi_e(rho)))
    applyDualChannel(kraus_operators,output_matrix,input_matrix,d,M,N);

    // Step 5: find eigenvector with highest eigenvalues
    eigvals.resize(N);
    zheev_wrapper('V', 'U', N, input_matrix,N,&eigvals);

    

    // Update the vector state to the last column, which corresponds to the highest eigenvalue.
    for (int i=0; i< N; i++){
        vector_state->at(i) = input_matrix->at(N*(N-1)+i);
    }



    return 0;
}
*/

/*

                GETTERS

*/
cuDoubleComplex* CudaMinimizer::getVector() {
    /*
    Returns a pointer to the vector state on the device.

    Returns:
        - Pointer to the vector state on the device.
    */
    return vector_state;
}

cuDoubleComplex* CudaMinimizer::getState() {
    /*
    Returns a pointer to the input matrix.

    Returns:
        - Pointer to the input matrix on the device.
    Note:
        - There is no guarantee that the input matrix is the rank one projector.
        - To ensure this, run updateProjector() before calling getState().
    */
    return input_matrix;
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
    return output_matrix;
} 

/*




int Minimizer::stepAlgorithm(){
    // Step 1: update the projector based on the vector
    updateProjector();

    // Step 2: compute Phi_e(rho)
    applyEpsilonChannel(kraus_operators,input_matrix, output_matrix, d, N, M, epsilon);

    // Step 3: compute log(Phi_e(rho))
    // Step 3.1: diagonalize output matrix
    std::vector<double> eigvals(M);
    zheev_wrapper('V', 'U', M, output_matrix,M,&eigvals);
    // Step 3.2: compute log(eigs)
    std::vector<std::complex<double> >* eig_mat = new std::vector<std::complex<double> >(N*N,std::complex<double>(0.0f,0.0f));
    for (int i=0; i<M; i++){
        eig_mat->at(N*i+i) = std::log(eigvals.at(i));
    }

    // Step 3.3: reconstruct the matrix
    //output matrix contains the eigenvectors as columns. Perform matrix multiplication.
    // First perform diag * eigs, store result in another temp mat
    std::vector<std::complex<double> >* tmp_mat = new std::vector<std::complex<double> >(N*N);
    std::complex<double> one(1.0f,0.0f);
    std::complex<double> zero(0.0f,0.0f);
    cblas_zgemm(CblasColMajor,CblasNoTrans,CblasConjTrans,M,M,M,&one,reinterpret_cast<lapack_complex_t*>(eig_mat->data()),M,reinterpret_cast<lapack_complex_t*>(output_matrix->data()),M,&zero,reinterpret_cast<lapack_complex_t*>(tmp_mat->data()),M);
    // Then do conj(eigs^T)*tmp_mat
    cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,M,M,&one,reinterpret_cast<lapack_complex_t*>(output_matrix->data()),M,reinterpret_cast<lapack_complex_t*>(tmp_mat->data()),M,&zero,reinterpret_cast<lapack_complex_t*>(eig_mat->data()),M);
    for (int i=0; i< M*M; i++){
        output_matrix->at(i) = eig_mat->at(i);
    }
    delete eig_mat;
    delete tmp_mat;

    // As far as I can tell, the matrix logarithm is calculated correctly.

    // Step 4: compute Phi_e^*(log(Phi_e(rho)))
    applyDualChannel(kraus_operators,output_matrix,input_matrix,d,M,N);

    // Step 5: find eigenvector with highest eigenvalues
    eigvals.resize(N);
    zheev_wrapper('V', 'U', N, input_matrix,N,&eigvals);

    

    // Update the vector state to the last column, which corresponds to the highest eigenvalue.
    for (int i=0; i< N; i++){
        vector_state->at(i) = input_matrix->at(N*(N-1)+i);
    }



    return 0;
}

int Minimizer::calculateEntropy(){
    // Get the vN entropy of Phi_e(state).
    // This works, pending verification on the application of the EpsilonChannel.

    updateProjector();
    applyEpsilonChannel(kraus_operators, input_matrix, output_matrix, d, N, M, epsilon);
    std::vector<double> eigvals = std::vector<double>(N);
    std::vector<std::complex<double> > tmp(M*M);
    for (int i=0; i < M*M; i++){
        tmp.at(i) = output_matrix->at(i);
    }
    zheev_wrapper('N', 'U', M,&tmp,M,&eigvals);
    entropy = 0.0f;
    for (int i = 0; i< M; i++){
        // WARNING: We are assuming that the diagonal here is real (which it is since it contains the eigs of a hermitian matrix)
        entropy -= eigvals.at(i)*std::log(eigvals.at(i));
    }
    //std::cout<< "Current entropy: " << std::fixed << std::setprecision(PRINT_PRECISION) <<entropy << std::endl;

    return 0;

}

int Minimizer::step(){
    stepAlgorithm();
    calculateEntropy();
    return 0;
}

/// GETTERS
std::vector<std::complex<double> >* Minimizer::getState(){
    updateProjector();
    return input_matrix;
}

std::vector<std::complex<double> > Minimizer::getVector(){
    // diagonalize the state and extract the largest eigenvector. No need to apply the channel.
    std::vector<double> eigvals(N);
    updateProjector();
    zheev_wrapper('V', 'U', N, input_matrix,N,&eigvals);
    std::vector<std::complex<double> > out(N);
    for (int i=0; i<N; i++){
        out.at(i) = input_matrix->at(N*(N-1)+i);
    }    
    return out;
}

double* Minimizer::getEntropy(){
    return &entropy;
}

int Minimizer::getN(){
    return N;
}

int Minimizer::getD(){
    return d;
}


/// DEBUGGING AND PRINTING AND ETC

int Minimizer::printVectorState(){
    // Print a list. Some variables are hard coded!
    if (vector_state == nullptr){
        std::cout << "No assigned value for vector..." << std::endl;
        return 1;
    } else {
        int size = vector_state->size();
        std::cout << "[";
        for (int i = 0; i < size; i++) {
                double r = (*vector_state)[i].real();
                double j = (*vector_state)[i].imag();
                std::cout <<std::fixed << std::setprecision(PRINT_PRECISION) << r;
                if (j>-1e-10){
                    std::cout << "+";
                }
                std::cout <<std::fixed << std::setprecision(PRINT_PRECISION)<<j << "j, ";
        }
        std::cout << "]"<< std::endl;
        return 0;
    }

}

int Minimizer::printMatrix(std::vector<std::complex<double> >* matrix_pointer, int n, int m) {
    // "Pretty" prints a nxm matrix in column-major order
    // it has to loop over every row, then print every entry of the row by looping over columns
    // there are n rows and m columns
    // To begin with, let's just print the entries and separate them by tabs or something
    // All variables are hardcoded here!
    for (int i=0; i<n; i++){
        for (int j=0;j<m; j++){
                double re = (*matrix_pointer)[j*m+i].real();
                double im = (*matrix_pointer)[j*m+i].imag();
                if (re>-1e-15){
                    std::cout << " ";
                }
                std::cout <<std::fixed << std::setprecision(PRINT_PRECISION)<< re;
                if (im>-1e-15){
                    std::cout << "+";
                }
                std::cout <<std::fixed << std::setprecision(PRINT_PRECISION)<<im << "j ";
        }
        std::cout << std::endl;
    }
    return 0;
}

int Minimizer::printState(){
    printMatrix(getState(), N, N);
    return 0;
}
*/

CudaMinimizer::~CudaMinimizer() {
    /*
    CudaMinimizer destructor.
    
    Notes:
    - The destructor frees the memory allocated for the vector_state, input_matrix, and output_matrix.
    - The kraus_operators are not freed, as they are not managed by this class.
    */

    if (vector_state != nullptr) {
        cudaFree(vector_state);
    }
    if (input_matrix != nullptr) {
        cudaFree(input_matrix);
    }
    if (output_matrix != nullptr) {
        cudaFree(output_matrix);
    }

}

