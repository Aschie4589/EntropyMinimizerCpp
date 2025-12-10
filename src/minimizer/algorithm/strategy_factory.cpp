#include "minimizer/algorithm/strategy_factory.h"
#include "minimizer/algorithm/generic_minimization_strategy.h"
#include "minimizer/algorithm/cuda_minimization_strategy.h"
#include <stdexcept>
#include <sstream>
#include <limits>
#include <complex>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Include for non-CUDA builds
#include <cuda_runtime_api.h>
#endif

namespace entropy {

// ============================================================================
// Public Interface - AUTO Selection
// ============================================================================

std::unique_ptr<IMinimizationStrategy> StrategyFactory::create(
    IComputeDevice& device,
    int kraus_count,
    int input_dim,
    int output_dim,
    PrecisionType precision,
    int num_streams
) {
    return create(
        StrategyType::AUTO,
        device,
        kraus_count,
        input_dim,
        output_dim,
        precision,
        num_streams
    );
}

// ============================================================================
// Public Interface - Explicit Type Selection
// ============================================================================

std::unique_ptr<IMinimizationStrategy> StrategyFactory::create(
    StrategyType type,
    IComputeDevice& device,
    int kraus_count,
    int input_dim,
    int output_dim,
    PrecisionType precision,
    int num_streams
) {
    // Validate dimensions first
    validateDimensions(kraus_count, input_dim, output_dim);
    
    switch (type) {
        case StrategyType::AUTO:
            return createAuto(
                device, kraus_count, input_dim, output_dim, precision, num_streams
            );
            
        case StrategyType::GENERIC:
            // Generic always works - no checks needed
            return std::make_unique<GenericMinimizationStrategy>(device, num_streams);
            
        case StrategyType::CUDA: {
            // Validate backend
            if (device.getBackend() != DeviceBackend::CUDA) {
                throw std::invalid_argument(
                    "CUDA strategy requires CUDA device backend (device is " +
                    std::to_string(static_cast<int>(device.getBackend())) + ")"
                );
            }
            
            // Check memory with 20% buffer (explicit CUDA requires tighter check)
            if (!cudaStrategyFitsMemory(
                device, kraus_count, input_dim, output_dim, precision, 1.2
            )) {
                // Query actual requirement from CUDA strategy for error message
                try {
                    auto test_strategy = std::make_unique<CudaMinimizationStrategy>(device, 1);
                    test_strategy->initialize(kraus_count, input_dim, output_dim, 0.01, precision);
                    size_t required = test_strategy->getWorkspaceSize();
                    size_t available = getAvailableMemory(device);
                    
                    std::ostringstream oss;
                    oss << "Insufficient GPU memory for CUDA strategy. "
                        << "Required: " << (required / (1024.0 * 1024.0)) << " MB, "
                        << "Available: " << (available / (1024.0 * 1024.0)) << " MB "
                        << "(with 20% safety buffer)";
                    throw std::runtime_error(oss.str());
                } catch (const std::runtime_error&) {
                    throw;  // Re-throw our formatted error
                } catch (...) {
                    throw std::runtime_error("Insufficient GPU memory for CUDA strategy");
                }
            }
            
            return std::make_unique<CudaMinimizationStrategy>(device, num_streams);
        }
        
        default:
            throw std::invalid_argument("Unknown StrategyType");
    }
}

// ============================================================================
// Memory Estimation
// ============================================================================

size_t StrategyFactory::estimateWorkspaceSize(
    IComputeDevice& device,
    int kraus_count,
    int input_dim,
    int output_dim,
    PrecisionType precision
) {
    validateDimensions(kraus_count, input_dim, output_dim);
    
    // Create a dummy strategy on the given device and initialize it
    // to get actual workspace size. This is better than duplicating
    // the memory calculation logic.
    auto dummy_strategy = std::make_unique<GenericMinimizationStrategy>(device, 1);
    
    // Initialize the strategy (allocates workspace buffers).
    // Use epsilon=0.01 as a representative value (doesn't affect workspace size).
    dummy_strategy->initialize(
        kraus_count,
        input_dim,
        output_dim,
        0.01,
        precision
    );
    
    // Query the actual workspace size from the strategy
    size_t workspace_size = dummy_strategy->getWorkspaceSize();
    
    // Strategy will be automatically destroyed here (RAII)
    return workspace_size;
}

size_t StrategyFactory::getAvailableMemory(IComputeDevice& device) {
    if (device.getBackend() == DeviceBackend::CUDA) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        
        cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to query CUDA memory: ") + 
                cudaGetErrorString(err)
            );
        }
        
        return free_bytes;
    } else {
        // CPU: No practical memory limit
        return std::numeric_limits<size_t>::max();
    }
}

// ============================================================================
// Private Helpers
// ============================================================================

std::unique_ptr<IMinimizationStrategy> StrategyFactory::createAuto(
    IComputeDevice& device,
    int kraus_count,
    int input_dim,
    int output_dim,
    PrecisionType precision,
    int num_streams
) {
    // AUTO logic: Try CUDA if backend supports it and memory fits
    // Otherwise fallback to Generic (never throw)
    
    if (device.getBackend() == DeviceBackend::CUDA) {
        try {
            // Check if CUDA strategy fits with 30% buffer
            if (cudaStrategyFitsMemory(
                device, kraus_count, input_dim, output_dim, precision, 1.3
            )) {
                // Memory check passed - create CUDA strategy
                return std::make_unique<CudaMinimizationStrategy>(device, num_streams);
            }
        } catch (...) {
            // Any error in CUDA path -> fallback to Generic
            // (e.g., cudaMemGetInfo failure, allocation failure, etc.)
        }
    }
    
    // Fallback: Generic strategy (works on all backends)
    return std::make_unique<GenericMinimizationStrategy>(device, num_streams);
}

void StrategyFactory::validateDimensions(
    int kraus_count,
    int input_dim,
    int output_dim
) {
    if (kraus_count <= 0) {
        throw std::invalid_argument(
            "Invalid kraus_count: " + std::to_string(kraus_count) + 
            " (must be > 0)"
        );
    }
    
    if (input_dim <= 0) {
        throw std::invalid_argument(
            "Invalid input_dim: " + std::to_string(input_dim) + 
            " (must be > 0)"
        );
    }
    
    if (output_dim <= 0) {
        throw std::invalid_argument(
            "Invalid output_dim: " + std::to_string(output_dim) + 
            " (must be > 0)"
        );
    }
}

bool StrategyFactory::cudaStrategyFitsMemory(
    IComputeDevice& device,
    int kraus_count,
    int input_dim,
    int output_dim,
    PrecisionType precision,
    double safety_factor
) {
    try {
        // Create dummy CUDA strategy (no allocation yet)
        auto test_strategy = std::make_unique<CudaMinimizationStrategy>(device, 1);
        
        // Initialize to allocate workspace and compute size
        // (Only allocates workspace buffers, not Kraus operators - those are owned by AlgorithmManager)
        test_strategy->initialize(
            kraus_count,
            input_dim,
            output_dim,
            0.01,  // epsilon doesn't affect workspace size
            precision
        );
        
        // Query actual workspace requirement from CUDA strategy
        size_t required = test_strategy->getWorkspaceSize();
        size_t available = getAvailableMemory(device);
        
        // Check with safety factor (1.2 or 1.3)
        // required * safety_factor <= available
        // Avoid overflow: required <= available / safety_factor
        size_t threshold = static_cast<size_t>(available / safety_factor);
        
        // Strategy destroyed here, workspace freed (RAII)
        return required <= threshold;
        
    } catch (...) {
        // Any error (allocation failure, invalid dimensions, etc.) -> doesn't fit
        return false;
    }
}

} // namespace entropy
