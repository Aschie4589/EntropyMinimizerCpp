#ifndef STRATEGY_FACTORY_H_
#define STRATEGY_FACTORY_H_

#include "minimizer/algorithm/minimization_strategy.h"
#include "compute/device/IComputeDevice.h"
#include "compute/core/ComputeTypes.h"
#include <memory>
#include <cstddef>

namespace entropy {

/**
 * @brief Strategy type selection mode
 */
enum class StrategyType {
    AUTO,     ///< Auto-select based on device backend + available memory
    GENERIC,  ///< Force GenericMinimizationStrategy (device-agnostic)
    CUDA      ///< Force CudaMinimizationStrategy (CUDA backend only)
};

/**
 * @brief Factory for creating memory-aware minimization strategies
 * 
 * Provides intelligent strategy selection based on:
 * - Device backend (CUDA vs CPU)
 * - Available GPU memory
 * - Estimated workspace requirements
 * 
 * The factory estimates memory requirements BEFORE allocating to prevent
 * out-of-memory errors on large quantum systems.
 * 
 * AUTO Selection Logic:
 * 1. Check device.getBackend() (CUDA vs CPU)
 * 2. If CUDA:
 *    - Estimate workspace via estimateWorkspaceSize()
 *    - Query free GPU memory via NVML (no context init) or cudaMemGetInfo()
 *    - If free >= 1.3 × workspace: CudaMinimizationStrategy (30% buffer)
 *    - Else: GenericMinimizationStrategy (fallback)
 * 3. If CPU: GenericMinimizationStrategy
 * 
 * Safety Margins:
 * - AUTO: 30% buffer (1.3×) for safe operation with fragmentation
 * - Explicit CUDA: 20% buffer (1.2×) for validation
 * - GENERIC: No memory check (fallback, works everywhere)
 * 
 * Thread Safety: Stateless factory - all methods are static and thread-safe
 * 
 * Example Usage:
 * @code
 * // AUTO selection (recommended)
 * auto strategy = StrategyFactory::create(
 *     device, kraus_count, input_dim, output_dim, precision
 * );
 * 
 * // Explicit CUDA (throws if insufficient memory)
 * auto cuda_strategy = StrategyFactory::create(
 *     StrategyType::CUDA, device, kraus_count, input_dim, output_dim, precision
 * );
 * @endcode
 */
class StrategyFactory {
public:
    /**
     * @brief Create strategy with AUTO selection
     * 
     * Automatically selects optimal strategy based on device backend and
     * available memory. Never throws - always falls back to Generic on failure.
     * 
     * @param device Compute device (CUDA or CPU)
     * @param kraus_count Number of Kraus operators (d)
     * @param input_dim Input dimension (N)
     * @param output_dim Output dimension (M)
     * @param precision FLOAT or DOUBLE
     * @param num_streams Number of parallel streams (default: 4)
     * @return Unique pointer to optimal strategy (never null)
     * 
     * @throws std::invalid_argument if dimensions are invalid (<= 0)
     */
    static std::unique_ptr<IMinimizationStrategy> create(
        IComputeDevice& device,
        int kraus_count,
        int input_dim,
        int output_dim,
        PrecisionType precision,
        int num_streams = 4
    );
    
    /**
     * @brief Create strategy with explicit type
     * 
     * Forces specific strategy type. Throws on failure (no fallback).
     * 
     * @param type Strategy type to create (AUTO/GENERIC/CUDA)
     * @param device Compute device
     * @param kraus_count Number of Kraus operators (d)
     * @param input_dim Input dimension (N)
     * @param output_dim Output dimension (M)
     * @param precision FLOAT or DOUBLE
     * @param num_streams Number of parallel streams (default: 4)
     * @return Unique pointer to requested strategy (never null)
     * 
     * @throws std::invalid_argument if:
     *   - Dimensions are invalid (<= 0)
     *   - CUDA strategy requested on non-CUDA device
     * @throws std::runtime_error if:
     *   - CUDA strategy requested but insufficient GPU memory (20% buffer)
     */
    static std::unique_ptr<IMinimizationStrategy> create(
        StrategyType type,
        IComputeDevice& device,
        int kraus_count,
        int input_dim,
        int output_dim,
        PrecisionType precision,
        int num_streams = 4
    );
    
    /**
     * @brief Estimate workspace memory required by strategy
     * 
     * Creates a temporary strategy on the given device, initializes it,
     * and queries the actual workspace size. This is more accurate than
     * manual calculation and ensures consistency with strategy implementation.
     * 
     * Note: Does NOT include Kraus operators or current vector (managed by
     * AlgorithmManager separately).
     * 
     * @param device Compute device to use for estimation
     * @param kraus_count Number of Kraus operators (d)
     * @param input_dim Input dimension (N)
     * @param output_dim Output dimension (M)
     * @param precision FLOAT or DOUBLE
     * @return Estimated workspace size in bytes
     * 
     * @throws std::invalid_argument if dimensions are invalid (<= 0)
     */
    static size_t estimateWorkspaceSize(
        IComputeDevice& device,
        int kraus_count,
        int input_dim,
        int output_dim,
        PrecisionType precision
    );
    
    /**
     * @brief Query available memory on device
     * 
     * For CUDA: Queries free GPU memory via NVML (doesn't initialize CUDA context)
     *           Falls back to cudaMemGetInfo() if NVML fails
     * For CPU: Returns SIZE_MAX (no practical memory limit)
     * 
     * @param device Compute device to query
     * @return Available memory in bytes
     * 
     * @throws std::runtime_error if CUDA memory query fails
     */
    static size_t getAvailableMemory(IComputeDevice& device);
    
private:
    /**
     * @brief Internal AUTO selection implementation
     * 
     * Implements the AUTO selection logic with fallback to Generic.
     * Never throws - always returns valid strategy.
     */
    static std::unique_ptr<IMinimizationStrategy> createAuto(
        IComputeDevice& device,
        int kraus_count,
        int input_dim,
        int output_dim,
        PrecisionType precision,
        int num_streams
    );
    
    /**
     * @brief Validate dimensions before creating strategy
     * 
     * @throws std::invalid_argument if dimensions are invalid
     */
    static void validateDimensions(
        int kraus_count,
        int input_dim,
        int output_dim
    );
    
    /**
     * @brief Check if CUDA strategy fits in available memory
     * 
     * @param device CUDA device
     * @param kraus_count Number of Kraus operators
     * @param input_dim Input dimension
     * @param output_dim Output dimension
     * @param precision FLOAT or DOUBLE
     * @param safety_factor Multiplicative buffer (1.2 or 1.3)
     * @return true if workspace fits with safety margin
     */
    static bool cudaStrategyFitsMemory(
        IComputeDevice& device,
        int kraus_count,
        int input_dim,
        int output_dim,
        PrecisionType precision,
        double safety_factor
    );
};

} // namespace entropy

#endif // STRATEGY_FACTORY_H_
