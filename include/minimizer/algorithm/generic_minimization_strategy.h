#ifndef GENERIC_MINIMIZATION_STRATEGY_H_
#define GENERIC_MINIMIZATION_STRATEGY_H_

#include "minimizer/algorithm/minimization_strategy.h"
#include "compute/device/IComputeDevice.h"
#include "compute/memory/IDeviceMemory.h"
#include "compute/stream/IComputeStream.h"
#include "compute/linalg/ISVDSolver.h"
#include <memory>
#include <vector>

/**
 * @brief Generic device-agnostic implementation of the minimization algorithm
 * 
 * Works with any backend (CUDA, CPU, etc.) through the IComputeDevice abstraction.
 * Uses the device's linear algebra and SVD capabilities to perform the algorithm.
 * 
 * This implementation:
 * - Supports both float and double precision
 * - Uses device streams for parallelism when available
 * - Manages workspace memory through RAII DeviceMemory wrappers
 * - Optimized for devices with good parallel matrix-vector performance
 * 
 * This is a stateless strategy - it owns only workspace buffers. Kraus operators
 * and current vector are managed by AlgorithmManager.
 * 
 * Workspace layout:
 * - d_vecs_1_: Step 1 vectors {K_i|ψ⟩} (d × M elements)
 * - d_sing_1_: Step 1 singular vectors (d × M elements)
 * - d_vecs_2_: Step 2 vectors {K_j^H|φ_i⟩} (d² × N elements)
 * - d_sing_2_: Step 2 singular vectors (N elements)
 * - d_sv_1_: Step 1 singular values (d elements)
 * - d_sv_2_: Step 2 singular values (d² elements)
 */
class GenericMinimizationStrategy : public IMinimizationStrategy {
public:
    /**
     * @brief Construct with compute device
     * 
     * @param device Compute device (CUDA, CPU, etc.)
     * @param num_streams Number of parallel streams to use (default 4)
     * 
     * @throws std::invalid_argument if num_streams < 1
     */
    explicit GenericMinimizationStrategy(
        IComputeDevice& device,
        int num_streams = 4
    );
    
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
    // Device and configuration
    IComputeDevice& device_;
    PrecisionType precision_;
    int num_streams_;
    
    // Problem dimensions
    int kraus_count_;  // d
    int input_dim_;    // N
    int output_dim_;   // M
    double epsilon_;
    
    // State
    bool initialized_;
    
    // Device memory - workspace only
    // Note: d_kraus and d_current_vector are now managed by AlgorithmManager
    std::unique_ptr<IDeviceMemory> d_vecs_1_;
    std::unique_ptr<IDeviceMemory> d_sing_1_;
    std::unique_ptr<IDeviceMemory> d_vecs_2_;
    std::unique_ptr<IDeviceMemory> d_sing_2_;
    std::unique_ptr<IDeviceMemory> d_sv_1_;
    std::unique_ptr<IDeviceMemory> d_sv_2_;
    
    // Compute resources
    std::vector<std::unique_ptr<IStream>> streams_;
    std::unique_ptr<ISVDSolver> svd_solver_1_;
    std::unique_ptr<ISVDSolver> svd_solver_2_;
    
    // Templated implementation methods (defined in .cpp)
    template<typename T> void initializeImpl(
        int kraus_count,
        int input_dim,
        int output_dim,
        double epsilon
    );
    
    template<typename T> double stepOnceImpl(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    );
    
    template<typename T> void step1Impl(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    );
    
    template<typename T> void step2Impl(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    );
    
    template<typename T> double computeEntropyImpl();
    
    template<typename T> std::vector<std::complex<T>> convertVector(
        const std::vector<std::complex<double>>& vec
    ) const;
};

#endif // GENERIC_MINIMIZATION_STRATEGY_H_
