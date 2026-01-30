#ifndef MINIMIZATION_STRATEGY_H_
#define MINIMIZATION_STRATEGY_H_

#include <complex>
#include <vector>
#include <memory>
#include <cstring>
#include "compute/core/ComputeTypes.h"
#include "minimizer/orchestration/types.h"

// Forward declaration
class IDeviceMemory;


/**
 * @brief Abstract interface for quantum channel minimization algorithms
 * 
 * Represents the Strategy pattern for different device-specific implementations
 * of the minimization algorithm:
 * 
 * Algorithm: Find minimum of S(Φ_ε(|ψ⟩⟨ψ|)) where
 * - Φ is a quantum channel with Kraus operators {K_i}
 * - Φ_ε is the epsilon-perturbed channel
 * - S is the von Neumann entropy
 * 
 * Each iteration performs:
 * 1. Apply channel: compute {K_i|ψ⟩}_i
 * 2. SVD and logarithmic scaling
 * 3. Second SVD to extract new |ψ⟩
 * 
 * This is a stateless strategy - Kraus operators and current vector are managed
 * by AlgorithmManager and passed into each step. The strategy owns only its
 * workspace buffers via DeviceMemory<T> RAII wrappers.
 */
class IMinimizationStrategy {
public:
    virtual ~IMinimizationStrategy() = default;
    
    /**
     * @brief Initialize strategy workspace without Kraus operators or state
     * 
     * Allocates workspace memory for SVD operations and intermediate results.
     * Does NOT allocate or transfer Kraus operators or current vector - those
     * are managed by AlgorithmManager and passed into stepOnce().
     * 
     * @param kraus_count Number of Kraus operators (d)
     * @param input_dim Input dimension (N)
     * @param output_dim Output dimension (M)
     * @param epsilon Perturbation parameter (0 < ε < 1)
     * @param precision FLOAT or DOUBLE precision
     * 
     * @throws std::runtime_error if device doesn't support precision
     * @throws std::runtime_error if dimensions invalid (d > M constraint)
     * @throws std::runtime_error if device memory allocation fails
     */
    virtual void initialize(
        int kraus_count,
        int input_dim,
        int output_dim,
        double epsilon,
        PrecisionType precision
    ) = 0;
    
    /**
     * @brief Execute one iteration of the minimization algorithm
     * 
     * Performs the full algorithm step:
     * 1. Compute {K_i|ψ⟩}_i via matrix-vector multiplications
     * 2. SVD to get {λ_i, |φ_i⟩}
     * 3. Apply logarithmic transformation: log((1-ε)λ_i² + ε) - log(ε)
     * 4. Second SVD to extract new |ψ⟩ (top eigenvector)
     * 5. Compute entropy from singular values
     * 
     * Updates d_current_vector in-place with new state.
     * Does NOT cache entropy - returns it directly.
     * 
     * @param d_kraus Device memory containing Kraus operators (d×M×N, managed by caller)
     * @param d_current_vector Device memory containing current state (N, updated in-place)
     * @return Computed entropy S(Φ_ε(|ψ⟩⟨ψ|))
     * 
     * @throws std::runtime_error if called before initialize()
     * @throws std::runtime_error if device computation fails
     */
    virtual double stepOnce(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    ) = 0;
    
    /**
     * @brief Get the precision type used for computation
     * 
     * @return FLOAT or DOUBLE
     */
    virtual PrecisionType getPrecision() const = 0;
    
    /**
     * @brief Get workspace memory required by this strategy in bytes
     * 
     * Returns device memory needed for algorithm-specific workspace:
     * - Intermediate vectors for SVD operations
     * - Singular value storage
     * - SVD scratch space
     * 
     * Does NOT include Kraus operators or current vector - those are
     * managed by AlgorithmManager.
     * 
     * @return Total workspace bytes required on device
     */
    virtual size_t getWorkspaceSize() const = 0;
};

#endif // MINIMIZATION_STRATEGY_H_
