#ifndef ORCHESTRATION_TYPES_H_
#define ORCHESTRATION_TYPES_H_

#include <vector>
#include <complex>
#include <string>

namespace entropy {

/**
 * @brief Simple wrapper for host-side complex vectors
 * 
 * Used throughout the orchestration layer for device-agnostic
 * representation of quantum state vectors. This is the canonical
 * definition shared by ResultCollector, CheckpointManager, and
 * RunOrchestrator.
 * 
 * Thread-safety: Copyable value type, thread-safe if not mutated.
 */
struct HostVector {
    std::vector<std::complex<double>> data;  ///< Complex vector data
    int dimension;                           ///< Vector dimension (cached for convenience)
    
    /**
     * @brief Default constructor - empty vector
     */
    HostVector() : dimension(0) {}
    
    /**
     * @brief Construct from vector of complex numbers
     * @param vec Vector data to copy
     */
    explicit HostVector(const std::vector<std::complex<double>>& vec) 
        : data(vec), dimension(static_cast<int>(vec.size())) {}
};

/**
 * @brief Error types for minimization run failures
 * 
 * Categorizes different failure modes to enable proper error handling
 * and recovery strategies in the orchestration layer.
 */
enum class RunErrorType {
    NONE = 0,                ///< No error - successful execution
    INVALID_INPUT,           ///< Input validation failed (wrong dimensions, NaN, etc.)
    DEVICE_ERROR,            ///< GPU/compute device error (OOM, kernel failure, etc.)
    CHECKPOINT_FAILURE,      ///< Checkpoint save/load failed (non-fatal for run)
    TIMEOUT,                 ///< Run exceeded time limit (if implemented)
    ALGORITHM_FAILURE,       ///< Algorithm failed to converge or produced invalid results
    UNKNOWN                  ///< Unhandled exception or unexpected error
};

/**
 * @brief Convert RunErrorType to human-readable string
 * @param type Error type enum value
 * @return String representation
 */
inline std::string to_string(RunErrorType type) {
    switch (type) {
        case RunErrorType::NONE: return "NONE";
        case RunErrorType::INVALID_INPUT: return "INVALID_INPUT";
        case RunErrorType::DEVICE_ERROR: return "DEVICE_ERROR";
        case RunErrorType::CHECKPOINT_FAILURE: return "CHECKPOINT_FAILURE";
        case RunErrorType::TIMEOUT: return "TIMEOUT";
        case RunErrorType::ALGORITHM_FAILURE: return "ALGORITHM_FAILURE";
        case RunErrorType::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Task specification for a single minimization run
 * 
 * Shared between WorkerThreadPool (consumer) and MinimizerOrchestrator (producer).
 * Encapsulates all information needed to execute one minimization attempt.
 * 
 * Design Notes:
 * - config_id enables future support for heterogeneous run configurations
 * - priority reserved for future work-stealing prioritization
 * - initial_vector copied to avoid lifetime issues (small overhead vs. complexity)
 */
struct RunTask {
    int run_id;                      ///< Unique run identifier (0-based, sequential)
    int config_id;                   ///< Index into shared config array (future: multi-config support)
    HostVector initial_vector;       ///< Starting quantum state vector (copied)
    int priority = 0;                ///< Task priority for future scheduling (higher = more urgent)
    
    /**
     * @brief Default constructor - creates invalid task
     */
    RunTask() : run_id(-1), config_id(0), priority(0) {}
    
    /**
     * @brief Construct task with all parameters
     * @param id Unique run identifier
     * @param cfg_id Configuration index (default: 0 for single-config mode)
     * @param vec Initial quantum state vector
     * @param prio Priority level (default: 0)
     */
    RunTask(int id, int cfg_id, const HostVector& vec, int prio = 0)
        : run_id(id), config_id(cfg_id), initial_vector(vec), priority(prio) {}
};

/**
 * @brief Result of a single minimization run
 * 
 * Stores outcome of one minimization attempt, including
 * the final quantum state, entropy value, and performance metrics.
 * Error messages and types indicate failures.
 * 
 * Thread-safety: Copyable value type, thread-safe if not mutated.
 */
struct RunResult {
    int run_id;                      ///< Unique identifier for this run
    double final_entropy;            ///< Final entropy value achieved
    HostVector final_vector;         ///< Final quantum state vector
    int iterations_taken;            ///< Number of iterations performed
    double runtime_seconds;          ///< Wall-clock time in seconds
    RunErrorType error_type;         ///< Categorized error type (NONE if successful)
    std::string error_message;       ///< Empty if successful, error description otherwise
    
    /**
     * @brief Default constructor - successful result
     */
    RunResult() : run_id(-1), final_entropy(0.0), iterations_taken(0), 
                  runtime_seconds(0.0), error_type(RunErrorType::NONE) {}
    
    /**
     * @brief Check if this run succeeded
     * @return true if no error occurred
     */
    bool isSuccess() const {
        return error_type == RunErrorType::NONE;
    }
};

} // namespace entropy

#endif // ORCHESTRATION_TYPES_H_
