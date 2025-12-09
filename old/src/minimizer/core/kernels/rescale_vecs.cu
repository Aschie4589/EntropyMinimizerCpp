#include <cmath>

#include "minimizer/core/cuda_traits.h"
#include "minimizer/core/kernels/rescale_vecs.h"

// You need template declaration for CUDA kernels
template<typename T>
__global__ void rescale_vecs_1(typename CudaTraits<T>::Complex* vecs, typename CudaTraits<T>::Real* lambdas, typename CudaTraits<T>::Real epsilon, int num_vecs, int vec_size){
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
    typename CudaTraits<T>::Complex entry = vecs[idx];
    // Get the lambda_i
    typename CudaTraits<T>::Real lambda_i = lambdas[idx / vec_size];
    // Compute the rescaling factor
    typename CudaTraits<T>::Complex rescale_factor = CudaTraits<T>::make_complex(sqrt(log((1 - epsilon) * lambda_i * lambda_i + epsilon/ vec_size) - log(epsilon / vec_size)),0.0);
    // Rescale
    entry = CudaTraits<T>::complex_mult(entry, rescale_factor);
    // Write back the result
    vecs[idx] = entry;    
}

// Explicit instantiations
template __global__ void rescale_vecs_1<float>(cuComplex* vecs, float* lambdas, float epsilon, int num_vecs, int vec_size);
template __global__ void rescale_vecs_1<double>(cuDoubleComplex* vecs, double* lambdas, double epsilon, int num_vecs, int vec_size);