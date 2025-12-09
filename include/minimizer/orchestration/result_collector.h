#ifndef RESULT_COLLECTOR_H_
#define RESULT_COLLECTOR_H_

#include <vector>
#include <string>
#include <complex>
#include <mutex>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace entropy {

/**
 * @brief Simple wrapper for host-side complex vectors
 * Used for storing final quantum state vectors (device-agnostic)
 */
struct HostVector {
    std::vector<std::complex<double>> data;
    int dimension;
    
    HostVector() : dimension(0) {}
    explicit HostVector(const std::vector<std::complex<double>>& vec) 
        : data(vec), dimension(static_cast<int>(vec.size())) {}
};

/**
 * @brief Result of a single minimization run
 * 
 * Stores outcome of one minimization attempt, including
 * the final quantum state, entropy value, and performance metrics.
 * Error messages indicate failures.
 */
struct RunResult {
    int run_id;                      ///< Unique identifier for this run
    double final_entropy;            ///< Final entropy value achieved
    HostVector final_vector;         ///< Final quantum state vector
    int iterations_taken;            ///< Number of iterations performed
    double runtime_seconds;          ///< Wall-clock time in seconds
    std::string error_message;       ///< Empty if successful, error description otherwise
    
    /**
     * @brief Check if this run succeeded
     * @return true if no error occurred
     */
    bool isSuccess() const {
        return error_message.empty();
    }
    
    /**
     * @brief Default constructor - creates invalid result
     */
    RunResult() 
        : run_id(-1), 
          final_entropy(std::numeric_limits<double>::infinity()),
          iterations_taken(0),
          runtime_seconds(0.0) {}
    
    /**
     * @brief Construct successful result
     */
    RunResult(int id, double entropy, const HostVector& vec, int iters, double runtime)
        : run_id(id),
          final_entropy(entropy),
          final_vector(vec),
          iterations_taken(iters),
          runtime_seconds(runtime) {}
    
    /**
     * @brief Construct error result
     */
    RunResult(int id, const std::string& error)
        : run_id(id),
          final_entropy(std::numeric_limits<double>::infinity()),
          iterations_taken(0),
          runtime_seconds(0.0),
          error_message(error) {}
};

/**
 * @brief Thread-safe collector for minimization run results
 * 
 * Aggregates results from multiple concurrent minimization runs,
 * tracks errors, and provides statistics. Used by orchestration
 * layer to collect outcomes from worker threads.
 * 
 * Features:
 * - Thread-safe result addition
 * - Find minimum entropy result
 * - Track success/error counts
 * - Compute statistics (mean, std dev, success rate)
 * 
 * Usage:
 * @code
 * ResultCollector collector;
 * 
 * // From worker threads
 * collector.addResult(run_id, result);
 * // or
 * collector.addError(run_id, "Device failure");
 * 
 * // After completion
 * auto best = collector.getMinimum();
 * auto stats = collector.getStatistics();
 * @endcode
 */
class ResultCollector {
public:
    /**
     * @brief Statistics computed from all successful runs
     */
    struct Statistics {
        double mean_entropy;         ///< Mean of final entropies
        double std_entropy;          ///< Standard deviation of entropies
        double min_entropy;          ///< Minimum entropy achieved
        double max_entropy;          ///< Maximum entropy achieved
        double mean_iterations;      ///< Mean iterations taken
        double mean_runtime;         ///< Mean runtime in seconds
        size_t num_successful;       ///< Number of successful runs
        size_t num_failed;           ///< Number of failed runs
        double success_rate;         ///< Fraction of successful runs
    };
    
    /**
     * @brief Construct empty collector
     */
    ResultCollector() = default;
    
    // Non-copyable (contains mutex)
    ResultCollector(const ResultCollector&) = delete;
    ResultCollector& operator=(const ResultCollector&) = delete;
    
    // Movable
    ResultCollector(ResultCollector&&) = default;
    ResultCollector& operator=(ResultCollector&&) = default;
    
    /**
     * @brief Add a successful result (thread-safe)
     * 
     * @param run_id Unique run identifier
     * @param result Result to add (must have empty error_message)
     */
    void addResult(int run_id, const RunResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        results_.push_back(result);
    }
    
    /**
     * @brief Add an error result (thread-safe)
     * 
     * Creates a RunResult with the given error message and
     * adds it to the error list.
     * 
     * @param run_id Unique run identifier
     * @param error Error description
     */
    void addError(int run_id, const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        errors_.emplace_back(run_id, error);
    }
    
    /**
     * @brief Get result with minimum entropy (thread-safe)
     * 
     * Returns the successful result with the lowest final entropy.
     * Only considers results with empty error_message.
     * 
     * @return Result with minimum entropy
     * @throws std::runtime_error if no successful results exist
     */
    RunResult getMinimum() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (results_.empty()) {
            throw std::runtime_error("No successful results available");
        }
        
        auto min_it = results_.begin();
        double min_entropy = min_it->final_entropy;
        
        for (auto it = results_.begin(); it != results_.end(); ++it) {
            if (it->isSuccess() && it->final_entropy < min_entropy) {
                min_entropy = it->final_entropy;
                min_it = it;
            }
        }
        
        return *min_it;
    }
    
    /**
     * @brief Get all results (thread-safe)
     * 
     * Returns copy of all results (both successful and errors).
     * 
     * @return Vector of all results
     */
    std::vector<RunResult> getAllResults() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<RunResult> all;
        all.reserve(results_.size() + errors_.size());
        all.insert(all.end(), results_.begin(), results_.end());
        all.insert(all.end(), errors_.begin(), errors_.end());
        return all;
    }
    
    /**
     * @brief Get number of completed successful runs (thread-safe)
     * 
     * @return Number of results with empty error_message
     */
    size_t numCompleted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return results_.size();
    }
    
    /**
     * @brief Get number of errors (thread-safe)
     * 
     * @return Number of runs that failed
     */
    size_t numErrors() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return errors_.size();
    }
    
    /**
     * @brief Get total number of runs (successful + errors)
     * 
     * @return Total runs recorded
     */
    size_t numTotal() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return results_.size() + errors_.size();
    }
    
    /**
     * @brief Check if any results have been collected
     * 
     * @return true if no results or errors recorded
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return results_.empty() && errors_.empty();
    }
    
    /**
     * @brief Clear all results and errors (thread-safe)
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        results_.clear();
        errors_.clear();
    }
    
    /**
     * @brief Compute statistics from successful results (thread-safe)
     * 
     * @return Statistics object with computed metrics
     * @throws std::runtime_error if no successful results exist
     */
    Statistics getStatistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (results_.empty()) {
            throw std::runtime_error("No successful results for statistics");
        }
        
        Statistics stats;
        stats.num_successful = results_.size();
        stats.num_failed = errors_.size();
        stats.success_rate = static_cast<double>(stats.num_successful) / 
                            (stats.num_successful + stats.num_failed);
        
        // Compute entropy statistics
        double sum_entropy = 0.0;
        double sum_iterations = 0.0;
        double sum_runtime = 0.0;
        stats.min_entropy = std::numeric_limits<double>::infinity();
        stats.max_entropy = -std::numeric_limits<double>::infinity();
        
        for (const auto& result : results_) {
            if (result.isSuccess()) {
                sum_entropy += result.final_entropy;
                sum_iterations += result.iterations_taken;
                sum_runtime += result.runtime_seconds;
                stats.min_entropy = std::min(stats.min_entropy, result.final_entropy);
                stats.max_entropy = std::max(stats.max_entropy, result.final_entropy);
            }
        }
        
        stats.mean_entropy = sum_entropy / stats.num_successful;
        stats.mean_iterations = sum_iterations / stats.num_successful;
        stats.mean_runtime = sum_runtime / stats.num_successful;
        
        // Compute standard deviation
        double sum_squared_diff = 0.0;
        for (const auto& result : results_) {
            if (result.isSuccess()) {
                double diff = result.final_entropy - stats.mean_entropy;
                sum_squared_diff += diff * diff;
            }
        }
        stats.std_entropy = std::sqrt(sum_squared_diff / stats.num_successful);
        
        return stats;
    }

private:
    mutable std::mutex mutex_;          ///< Protects results_ and errors_
    std::vector<RunResult> results_;    ///< Successful results
    std::vector<RunResult> errors_;     ///< Failed runs
};

} // namespace entropy

#endif // RESULT_COLLECTOR_H_
