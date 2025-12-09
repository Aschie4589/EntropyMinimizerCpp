#ifndef CUDATRAITS_H
#define CUDATRAITS_H

#include "cuComplex.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand_kernel.h>


// Make versions for different precisions. For example, maek CudaTraits<T>::Complex be either cuDoubleComplex or cuComplex depending on the argument
template<typename T> struct CudaTraits;

template<>
struct CudaTraits<double> {
    using Complex = cuDoubleComplex;
    using Real = double;
    // cuBLAS
    static constexpr auto gemv = cublasZgemv;
    static constexpr auto gemm = cublasZgemm;
    static constexpr auto gemm3m = cublasZgemm3m;
    static constexpr auto geam = cublasZgeam;
    // cuSOLVER
    static constexpr auto gesvd = cusolverDnZgesvd;
    static constexpr auto gesvd_buffer = cusolverDnZgesvd_bufferSize;
    static constexpr auto CUDA_C = CUDA_C_64F; // Define CUDA_C_64F for double precision
    static constexpr auto CUDA_R = CUDA_R_64F; // Define CUDA_R_64F for double precision
    
    // Complex number stuff
    static __device__ cuDoubleComplex make_complex(double real, double imag) {
        return make_cuDoubleComplex(real, imag);
    }
    static constexpr auto make_complex_h = make_cuDoubleComplex;
    static __device__ cuDoubleComplex complex_mult(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCmul(a, b);
    }
    static constexpr auto complex_mult_h = cuCmul;
    // Random number generation. This requires defining a device side function! No code slowdown.
    static __device__ double rand_nor(curandState* state) {
        return curand_normal_double(state);
    }
    static constexpr auto real_part_h = cuCreal;
    static constexpr auto imag_part_h = cuCimag;
        // ... etc
};

template<>
struct CudaTraits<float> {
    using Complex = cuComplex;
    using Real = float;
    // cuBLAS
    static constexpr auto gemv = cublasCgemv;
    static constexpr auto gemm = cublasCgemm;
    static constexpr auto gemm3m = cublasCgemm3m;
    static constexpr auto geam = cublasCgeam;
    // cuSOLVER
    static constexpr auto gesvd = cusolverDnCgesvd;
    static constexpr auto gesvd_buffer = cusolverDnCgesvd_bufferSize;
    static constexpr auto CUDA_C = CUDA_C_32F; // Define CUDA_C_32F for single precision
    static constexpr auto CUDA_R = CUDA_R_32F; // Define CUDA_R_32F for single precision
    // Complex number stuff
    static __device__ cuComplex make_complex(float real, float imag) {
        return make_cuComplex(real, imag);
    }
    static constexpr auto make_complex_h = make_cuComplex;
    static __device__ cuComplex complex_mult(cuComplex a, cuComplex b) {
        return cuCmulf(a, b);
    }
    static constexpr auto complex_mult_h = cuCmulf;
    // Random number generation. This requires defining a device side function! No code slowdown.
    static __device__ float rand_nor(curandState* state) {
        return curand_normal(state);
    }
    static constexpr auto real_part_h = cuCrealf;
    static constexpr auto imag_part_h = cuCimagf;
};


#endif // CUDATRAITS_H