#ifndef ORCHESTRATION_TYPES_H_
#define ORCHESTRATION_TYPES_H_

#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdexcept>
#include "compute/core/ComputeTypes.h"

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
    RunResult() : run_id(-1), final_entropy(std::numeric_limits<double>::infinity()), iterations_taken(0), 
                  runtime_seconds(0.0), error_type(RunErrorType::NONE) {}
    
    /**
     * @brief Check if this run succeeded
     * @return true if no error occurred
     */
    bool isSuccess() const {
        return error_type == RunErrorType::NONE;
    }
};


/**
 * @brief Host-side representation of Kraus operators with type-erased storage
 * 
 * Contains d Kraus operators K_i, each of dimension M×N (output × input).
 * Stored as a contiguous byte array in column-major order to support both
 * float and double precision without conversion overhead.
 * 
 * Layout: [K_0][K_1]...[K_{d-1}] where each K_i is M×N complex numbers
 * - Entry (row, col) of K_i is at byte offset: 
 *   (i*M*N + col*M + row) * sizeof(complex<T>)
 * 
 * Invariant: data.size() == kraus_count * output_dim * input_dim * complex_size
 */
struct HostKrausOperators {
    std::vector<char> data;  ///< Type-erased byte storage for all Kraus operators
    PrecisionType precision; ///< FLOAT or DOUBLE precision
    int kraus_count;         ///< Number of Kraus operators (d)
    int input_dim;           ///< Input dimension (N)
    int output_dim;          ///< Output dimension (M)
    
    /**
     * @brief Default constructor - creates empty operator set
     */
    HostKrausOperators() 
        : precision(PrecisionType::DOUBLE),
          kraus_count(0), 
          input_dim(0), 
          output_dim(0) {}
    
    /**
     * @brief Create from float precision data
     * 
     * @param kraus_data Source data in float precision
     * @param d Number of Kraus operators
     * @param N Input dimension
     * @param M Output dimension
     * @return HostKrausOperators with FLOAT precision
     */
    static HostKrausOperators fromFloat(
        const std::vector<std::complex<float>>& kraus_data,
        int d, int N, int M
    ) {
        HostKrausOperators result;
        result.precision = PrecisionType::FLOAT;
        result.kraus_count = d;
        result.input_dim = N;
        result.output_dim = M;
        
        // Validate input size
        if (kraus_data.size() != static_cast<size_t>(d * N * M)) {
            throw std::invalid_argument(
                "HostKrausOperators::fromFloat: data size mismatch. Expected " + 
                std::to_string(d * N * M) + ", got " + std::to_string(kraus_data.size())
            );
        }
        
        // Copy data as bytes
        size_t byte_size = kraus_data.size() * sizeof(std::complex<float>);
        result.data.resize(byte_size);
        std::memcpy(result.data.data(), kraus_data.data(), byte_size);
        
        return result;
    }
    
    /**
     * @brief Create from double precision data
     * 
     * @param kraus_data Source data in double precision
     * @param d Number of Kraus operators
     * @param N Input dimension
     * @param M Output dimension
     * @return HostKrausOperators with DOUBLE precision
     */
    static HostKrausOperators fromDouble(
        const std::vector<std::complex<double>>& kraus_data,
        int d, int N, int M
    ) {
        HostKrausOperators result;
        result.precision = PrecisionType::DOUBLE;
        result.kraus_count = d;
        result.input_dim = N;
        result.output_dim = M;
        
        // Validate input size
        if (kraus_data.size() != static_cast<size_t>(d * N * M)) {
            throw std::invalid_argument(
                "HostKrausOperators::fromDouble: data size mismatch. Expected " + 
                std::to_string(d * N * M) + ", got " + std::to_string(kraus_data.size())
            );
        }
        
        // Copy data as bytes
        size_t byte_size = kraus_data.size() * sizeof(std::complex<double>);
        result.data.resize(byte_size);
        std::memcpy(result.data.data(), kraus_data.data(), byte_size);
        
        return result;
    }
    
    /**
     * @brief Get type-erased pointer to all Kraus data
     * 
     * Strategy implementations can cast this to the appropriate type
     * based on the precision field.
     * 
     * @return Pointer to raw byte data
     */
    const void* getData() const {
        return data.data();
    }
    
    /**
     * @brief Get typed pointer to i-th Kraus operator (float precision)
     * 
     * @param i Index of Kraus operator (0 <= i < kraus_count)
     * @return Pointer to start of K_i data
     * @throws std::logic_error if precision is not FLOAT
     */
    const std::complex<float>* getKrausFloat(int i) const {
        if (precision != PrecisionType::FLOAT) {
            throw std::logic_error("getKrausFloat called but precision is not FLOAT");
        }
        const auto* typed_data = reinterpret_cast<const std::complex<float>*>(data.data());
        return typed_data + i * output_dim * input_dim;
    }
    
    /**
     * @brief Get typed pointer to i-th Kraus operator (double precision)
     * 
     * @param i Index of Kraus operator (0 <= i < kraus_count)
     * @return Pointer to start of K_i data
     * @throws std::logic_error if precision is not DOUBLE
     */
    const std::complex<double>* getKrausDouble(int i) const {
        if (precision != PrecisionType::DOUBLE) {
            throw std::logic_error("getKrausDouble called but precision is not DOUBLE");
        }
        const auto* typed_data = reinterpret_cast<const std::complex<double>*>(data.data());
        return typed_data + i * output_dim * input_dim;
    }
    
    /**
     * @brief Get total size in bytes of all Kraus operators
     */
    size_t sizeBytes() const {
        return data.size();
    }
    
    /**
     * @brief Get size of a single complex number in current precision
     */
    size_t complexSize() const {
        return (precision == PrecisionType::FLOAT) ? 
               sizeof(std::complex<float>) : 
               sizeof(std::complex<double>);
    }
};





} // namespace entropy




#endif // ORCHESTRATION_TYPES_H_
