#ifndef CUDA_MINIMIZATION_STRATEGY_H_
#define CUDA_MINIMIZATION_STRATEGY_H_

#include "minimizer/algorithm/minimization_strategy.h"
#include "compute/device/IComputeDevice.h"
#include "compute/memory/IDeviceMemory.h"
#include "compute/linalg/ISVDSolver.h"
#include "compute/stream/IComputeStream.h"
#include <vector>
#include <memory>

/**
 * @brief CUDA implementation of minimization strategy
 * 
 * Implements the quantum channel minimization algorithm on NVIDIA GPUs using
 * cuBLAS and cuSOLVER. 
 * 
 * Algorithm per iteration:
 * 1. Compute {K_i|ψ⟩}_i via parallel matrix-vector multiplications
 * 2. SVD: {K_i|ψ⟩} = U Σ V^H to get singular values {λ_i}
 * 3. Apply logarithmic scaling: |φ_i⟩ = sqrt(log((1-ε)λ_i² + ε) - log(ε)) |φ_i⟩
 * 4. Compute {K_j^H|φ_i⟩}_{ij} for all pairs
 * 5. SVD to extract top eigenvector as new |ψ⟩
 * 6. Compute entropy from singular values
 * 
 * Memory strategy: Always uses high-memory approach (precomputes nothing,
 * recomputes Kraus applications each iteration - same as old LowMemoryStrategy).
 * 
 * Precision: Supports both FLOAT and DOUBLE via templated implementations.
 * 
 * This is a stateless strategy - it owns only workspace buffers. Kraus operators
 * and current vector are managed by AlgorithmManager.
 */
class CudaMinimizationStrategy : public IMinimizationStrategy {
public:
    /**
     * @brief Construct strategy for given device
     * 
     * @param device CUDA device to run computations on
     * @param num_streams Number of CUDA streams for parallel Kraus operations (default: 4)
     */
    explicit CudaMinimizationStrategy(IComputeDevice& device, int num_streams = 4);
    
    ~CudaMinimizationStrategy() override = default;
    
    // IMinimizationStrategy interface
    void initialize(
        int kraus_count,
        int input_dim,
        int output_dim,
        double epsilon,
        PrecisionType precision
    ) override;
    
    double stepOnce(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    ) override;
    
    PrecisionType getPrecision() const override;
    
    size_t getWorkspaceSize() const override;
    
private:
    // Device reference (non-owning)
    IComputeDevice& device_;
    
    // Configuration
    PrecisionType precision_;
    int num_streams_;
    int kraus_count_;   // d
    int input_dim_;     // N
    int output_dim_;    // M
    double epsilon_;
    
    // State
    bool initialized_;
    
    // Device memory - workspace only (RAII via unique_ptr)
    // Note: d_kraus and d_current_vector are now managed by AlgorithmManager
    std::unique_ptr<IDeviceMemory> d_vecs_1_;          // d×M intermediate after K_i|ψ⟩
    std::unique_ptr<IDeviceMemory> d_sing_1_;          // d×M singular vectors from first SVD
    std::unique_ptr<IDeviceMemory> d_vecs_2_;          // d²×N intermediate after K_j^H|φ_i⟩
    std::unique_ptr<IDeviceMemory> d_sing_2_;          // N×1 singular vector from second SVD
    std::unique_ptr<IDeviceMemory> d_sv_1_;            // d singular values from first SVD
    std::unique_ptr<IDeviceMemory> d_sv_2_;            // d² singular values from second SVD
    
    // Compute resources
    std::vector<std::unique_ptr<IStream>> streams_;
    std::unique_ptr<ISVDSolver> svd_solver_1_;  // For M×d first SVD
    std::unique_ptr<ISVDSolver> svd_solver_2_;  // For N×(d²) second SVD
    
    // Templated implementation methods (dispatch based on precision_)
    template<typename T>
    void initializeImpl(
        int kraus_count,
        int input_dim,
        int output_dim,
        double epsilon
    );
    
    template<typename T>
    double stepOnceImpl(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    );
    
    template<typename T>
    void step1Impl(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    );  // Compute {K_i|ψ⟩}, SVD, logarithmic scaling
    
    template<typename T>
    void step2Impl(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    );  // Compute {K_j^H|φ_i⟩}, SVD, extract eigenvector and update d_current_vector
    
    template<typename T>
    double computeEntropyImpl();  // Calculate S(Φ_ε(|ψ⟩⟨ψ|)) from singular values
    
    // Helper to convert double vector to target precision
    template<typename T>
    std::vector<std::complex<T>> convertVector(
        const std::vector<std::complex<double>>& vec
    ) const;
};

#endif // CUDA_MINIMIZATION_STRATEGY_H_
