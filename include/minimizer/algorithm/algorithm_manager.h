#ifndef ALGORITHM_MANAGER_H_
#define ALGORITHM_MANAGER_H_

#include "minimizer/algorithm/minimization_strategy.h"
#include "minimizer/state/entropy_manager.h"
#include "compute/device/IComputeDevice.h"
#include "compute/memory/IDeviceMemory.h"
#include <memory>
#include <vector>
#include <complex>
#include <stdexcept>

namespace entropy {

/**
 * @brief Manages algorithm execution and problem state (Kraus operators, current vector)
 * 
 * Responsibilities:
 * - Owns device memory for Kraus operators and current vector (RAII)
 * - Coordinates MinimizationStrategy (stateless computation)
 * - Manages entropy computation via EntropyManager
 * - Provides getter/setter interface for state access
 * 
 * Ownership:
 * - OWNS: d_kraus_, d_current_vector_ (device memory)
 * - OWNS: strategy_ (computation engine)
 * - REFERENCES: device_, entropy_manager_ (non-owning)
 * 
 * Thread Safety: Not thread-safe. Each thread should have its own instance.
 */
class AlgorithmManager {
public:
    /**
     * @brief Construct AlgorithmManager
     * 
     * @param device Device for computation (non-owning reference)
     * @param entropy_manager Manages entropy transformations (non-owning reference)
     * @param kraus Host-side Kraus operators
     * @param epsilon Depolarization parameter for algorithm
     * @param precision Computation precision (FLOAT or DOUBLE)
     * 
     * @throws std::invalid_argument if epsilon not in (0,1)
     * @throws std::invalid_argument if dimensions are invalid
     */
    AlgorithmManager(
        IComputeDevice& device,
        EntropyManager& entropy_manager,
        const HostKrausOperators& kraus,
        double epsilon,
        PrecisionType precision
    );
    
    ~AlgorithmManager() = default;
    
    // Non-copyable, movable
    AlgorithmManager(const AlgorithmManager&) = delete;
    AlgorithmManager& operator=(const AlgorithmManager&) = delete;
    AlgorithmManager(AlgorithmManager&&) = default;
    AlgorithmManager& operator=(AlgorithmManager&&) = default;
    
    /**
     * @brief Initialize with specific starting vector
     * 
     * @param initial_vector Host vector (must be size input_dim)
     * 
     * @throws std::invalid_argument if vector size doesn't match input_dim
     * @throws std::runtime_error if strategy not set
     * 
     * Actions:
     * - Allocates and uploads d_kraus_ to device
     * - Allocates and uploads d_current_vector_ to device
     * - Initializes strategy with dimensions and epsilon
     */
    void initialize(const std::vector<std::complex<double>>& initial_vector);
    
    /**
     * @brief Initialize with random normalized vector
     * 
     * @throws std::runtime_error if strategy not set
     */
    void initializeRandom();
    
    /**
     * @brief Execute one minimization step
     * 
     * @throws std::runtime_error if not initialized
     * 
     * Actions:
     * - Calls strategy->stepOnce(d_kraus_, d_current_vector_)
     * - Updates entropy_manager with returned epsilon entropy
     * - d_current_vector_ is updated in-place by strategy
     */
    void stepOnce();
    
    // ========== State Access (Getters/Setters) ==========
    
    /**
     * @brief Get current vector from device (EXPENSIVE - device to host copy)
     * 
     * @return Current vector as host std::vector
     * @throws std::runtime_error if not initialized
     */
    std::vector<std::complex<double>> getCurrentVector() const;
    
    /**
     * @brief Set current vector on device (EXPENSIVE - host to device copy)
     * 
     * @param vec New vector (must be size input_dim)
     * @throws std::invalid_argument if vector size doesn't match
     * @throws std::runtime_error if not initialized
     */
    void setCurrentVector(const std::vector<std::complex<double>>& vec);
    
    // ========== Entropy Accessors (delegate to EntropyManager) ==========
    
    /**
     * @brief Get epsilon-perturbed entropy S(ρ_ε)
     * @return Last computed epsilon entropy
     */
    double getEpsilonEntropy() const;
    
    /**
     * @brief Get estimated true entropy S(ρ)
     * @return Corrected entropy estimate
     */
    double getEstimatedEntropy() const;
    
    /**
     * @brief Get entropy bounds [lower, upper]
     * @return Pair of (estimated - error, estimated + error)
     */
    std::pair<double, double> getEntropyBounds() const;
    
    // ========== Strategy Management ==========
    
    /**
     * @brief Set the minimization strategy
     * 
     * @param strategy Strategy instance (takes ownership)
     * 
     * @throws std::invalid_argument if strategy is null
     * 
     * Note: Must be called before initialize()
     */
    void setStrategy(std::unique_ptr<IMinimizationStrategy> strategy);
    
    // ========== Query Methods ==========
    
    /**
     * @brief Check if manager is initialized and ready for stepOnce()
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Get computation precision
     */
    PrecisionType getPrecision() const { return precision_; }
    
    /**
     * @brief Get problem dimensions
     */
    int getKrausCount() const { return kraus_count_; }
    int getInputDim() const { return input_dim_; }
    int getOutputDim() const { return output_dim_; }
    
    /**
     * @brief Get the device this manager is using
     */
    IComputeDevice& getDevice() const { return device_; }
    
private:
    // ========== Device Memory (RAII) ==========
    std::unique_ptr<IDeviceMemory> d_kraus_;           ///< Kraus operators on device
    std::unique_ptr<IDeviceMemory> d_current_vector_;  ///< Current vector on device
    
    // ========== Strategy (owns workspace) ==========
    std::unique_ptr<IMinimizationStrategy> strategy_;
    
    // ========== Non-owning References ==========
    IComputeDevice& device_;          ///< Computation device
    EntropyManager& entropy_manager_; ///< Entropy computation manager
    
    // ========== Problem Configuration ==========
    int kraus_count_;         ///< Number of Kraus operators
    int input_dim_;           ///< Input dimension N
    int output_dim_;          ///< Output dimension M
    double epsilon_;          ///< Depolarization parameter
    PrecisionType precision_; ///< Computation precision
    
    // ========== Host-side Kraus Data ==========
    HostKrausOperators host_kraus_;  ///< Cached for re-initialization
    
    // ========== State ==========
    bool initialized_;  ///< Whether initialize() has been called
    
    // ========== Device Validation ==========
    
    /**
     * @brief Validate that device memory belongs to this manager's device
     * 
     * @param memory Memory to validate
     * @throws std::invalid_argument if memory doesn't belong to our device
     */
    void validateDeviceMemory(const IDeviceMemory* memory) const;
};

} // namespace entropy

#endif // ALGORITHM_MANAGER_H_
