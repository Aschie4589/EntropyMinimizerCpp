// Minimizer.cpp
#include "common_includes.h"
#include "core/cuda_minimizer.h"
#include "config/config.h"
#include "core/matrix_operations.h"
#include "core/generate_random_vector.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>

// Profiling
#include <nvtx3/nvtx3.hpp>

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
    // Update internal state
    minimizer_state = CUDA_MINIMIZER_CREATED;

//    std::cout << "Current stage is: " << minimizer_state << std::endl;
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
//    std::cout << "I just created a vector. The new stage of the minimizer is " << minimizer_state << std::endl;

    return err;
}



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
    cuDoubleComplex rescale_factor = make_cuDoubleComplex(sqrt(log((1 - epsilon) * lambda_i * lambda_i + epsilon/ vec_size) - log(epsilon / vec_size)),0.0);
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
    {
    nvtx3::scoped_range r1{"Step 1: K_i |d_vec> multiplication"};
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
    }
    {
    nvtx3::scoped_range r2{"Step 1: SVD of K_i |v>_i"};
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
    }

    // At this point we can rescale the vectors in Y=d_sv_1 by the right amounts (so we have log(Phi_eps(rho))-log(eps)*1 = YY^H)
    int total_elems = d * M;
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;
    {
    nvtx3::scoped_range r3{"Step 1: Rescale vectors in d_vecs_1"};
    rescale_vecs_1<<<blocks, threads>>>(d_vecs_1, d_sv_1, epsilon, d, M);
    cudaDeviceSynchronize();
    }
    minimizer_state = CUDA_MINIMIZER_STAGE_1; // Reset the state to stage 0

    return cudaGetLastError(); // Returns success if the kernel launched successfully
}

cudaError_t CudaMinimizer::step_2(){
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
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    // One operation per stream
    {
    nvtx3::scoped_range r{"Step 2: K_i^H |psi_j> multiplication"};
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
                d_vecs_1, 
                M, 
                &zero, 
                d_vecs_2 + j * N * d, 
                N);
        }
    }
    // Synchronize streams to ensure all operations are complete
    cudaDeviceSynchronize();
    }
    // Step 2: extract the largest left singular vector of d_vecs_2. This is the new state.
    // For now, compute full SVD of the N * (dxd) matrix d_vecs_2. Only store the left singular vectors. In reality only care about the first!
    {
    nvtx3::scoped_range r{"Step 2: SVD of K_j^H |psi_i>_ij"};
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
    }
    // Step 3: Update the d_vec state. (TODO: actually, I could avoid this copy...)
    {
    nvtx3::scoped_range r{"Step 2: Copy largest left singular vector to d_vec"};
    cudaMemcpy(d_vec, d_vecs_2, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    // No further need to sync since memCpy waits for transfer to be done
    minimizer_state = CUDA_MINIMIZER_STAGE_2; // Reset the state to stage 2 (same as stage 0, but algorithm has started running!)
    }
    return cudaSuccess;
}




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
//    std::cout << "At the beginning of the step the stage is: " << minimizer_state << std::endl;
    if (minimizer_state != CUDA_MINIMIZER_STAGE_1){
//        std::cout << "Running step_1 to bring minimizer to stage 1..." << std::endl;
        cudaError_t err_step_1 = step_1();
//        std::cout << "Done. Current stage: " << minimizer_state << std::endl;
        if (err_step_1 != cudaSuccess){
            std::cerr << "Error in step_1: " << cudaGetErrorString(err_step_1) << std::endl;
            return err_step_1; // Return the error if the step failed 
        }
    }
    // Step 2
    if (minimizer_state == CUDA_MINIMIZER_STAGE_1){
//        std::cout << "Running step_2 to bring minimizer to stage 2..." << std::endl;
        cudaError_t err_step_2 = step_2();
//        std::cout << "Done. Current stage: " << minimizer_state << std::endl;
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
//            std::cout << "I was asked to compute entropy but first I need to know the output state of the channel..." <<std::endl;
        cudaError_t err_step_1 = step_1();
//        std::cout << "Done! Now I can calculate the entropy. The stage is:" << minimizer_state << std::endl;

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
        double val = (1.0 - epsilon) * lambda_i * lambda_i + epsilon/M; // epsilon is assumed small but positive
        entropy -= val * log(val);
    }

    entropy -= (M - d)/M * epsilon * (log(epsilon) - log(M)); // Add the contribution from the zero eigenvalues

    delete[] h_sv_1; // Free the host memory
    return cudaSuccess; // Return success if the entropy was calculated successfully    
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
    return d_vec;
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

CudaMinimizer::~CudaMinimizer() {
    /*
    CudaMinimizer destructor.
    
    Notes:
    - The destructor frees the memory allocated for the internally allocated GPU resources.
    - The d_kraus are not freed, as they are not managed by this class.
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

