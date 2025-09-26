#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "common_includes.h"
#include "core/cuda_traits.h"

template<typename T>
__global__ void rescale_vecs_1(typename CudaTraits<T>::Complex* vecs, typename CudaTraits<T>::Real* lambdas, typename CudaTraits<T>::Real epsilon, int num_vecs, int vec_size);


#endif