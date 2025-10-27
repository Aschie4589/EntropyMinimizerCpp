#pragma once
#include "minimizer/core/cuda_traits.h"
#include "minimizer/core/cuda_minimizer_strategy/base_minimizer_strategy.h"

template<typename T>
class LowMemoryStrategy : public MinimizationStrategy<T> {
public:
    cudaError_t initialize(CudaMinimizer<T>* minimizer) override;
    cudaError_t stepAlgorithm(CudaMinimizer<T>* minimizer) override;
    cudaError_t step(CudaMinimizer<T>* minimizer) override;
    cudaError_t step_1(CudaMinimizer<T>* minimizer);
    cudaError_t step_2(CudaMinimizer<T>* minimizer);
    cudaError_t calculateEpsilonEntropy(CudaMinimizer<T>* minimizer, bool force = false); // Force: force recomputation of eigenvectors?
    cudaError_t cleanup(CudaMinimizer<T>* minimizer) override;
    const char* getName() const override { return "LowMemory"; }
    size_t getMemoryRequired(CudaMinimizer<T>* minimizer) const override;

private:
    typename CudaTraits<T>::Complex* d_vecs_1;        // Pointer to memory for storing {K_i |d_vec>}_i
    typename CudaTraits<T>::Complex* d_vecs_2;        // Pointer to memory for storing {K_i |phi_j>}_ij where |phi_j> comes from SVD of d_vecs_1
    typename CudaTraits<T>::Complex* d_scratch;       // Pointer to scratch space for doing computations
    typename CudaTraits<T>::Real* d_sv_1;                   // Pointer to memory for storing singular values of {K_i |d_vec>}_i
    typename CudaTraits<T>::Real* d_sv_2;                   // Pointer to memory for storing singular values of {K_i |phi_j>}_ij

    typename CudaTraits<T>::Complex* Srand;          
    typename CudaTraits<T>::Complex* Urand; 
    typename CudaTraits<T>::Complex* Vrand;

    // Other relevant internal quantities
    int work_size_1;                  // For SVD
    int work_size_2;                  // For SVD

};