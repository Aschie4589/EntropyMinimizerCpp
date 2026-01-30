#ifndef PROGRESS_TRACKER_H_
#define PROGRESS_TRACKER_H_

#include <mutex>
#include <unordered_map>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

namespace entropy {

/**
 * @brief Thread-safe progress tracker for monitoring minimization runs
 * 
 * Provides a centralized, thread-safe way for worker threads to report
 * progress updates during minimization runs. Updates include:
 * - Current iteration number
 * - Current entropy value
 * - Timestamp of last update
 * 
 * Design Rationale:
 * - Lives OUTSIDE worker thread scope (owned by orchestrator/main)
 * - Thread-safe via mutex (workers update concurrently)
 * - Non-blocking updates (fast lock, no I/O under mutex)
 * - Can be queried from main thread for monitoring/logging
 * - Supports multiple concurrent runs (keyed by run_id)
 * 
 * Thread Safety:
 * - All public methods are thread-safe
 * - Uses std::mutex for synchronization
 * - Lock is held only during data structure updates (minimal contention)
 * - Safe to call from any thread (workers, main thread, etc.)
 * 
 * Performance Considerations:
 * - Minimal overhead per update (~microseconds)
 * - No I/O or heavy computation under lock
 * - Suitable for frequent updates (every iteration)
 * - Memory usage: O(number of concurrent runs)
 * 
 * Typical Usage:
 * @code
 * // In main thread
 * ProgressTracker tracker;
 * 
 * // Pass to worker pool (will be forwarded to RunOrchestrator)
 * WorkerThreadPool pool(...);
 * pool.setProgressTracker(&tracker);
 * 
 * // Workers automatically update progress during execution
 * pool.start();
 * 
 * // Query progress from main thread
 * while (work_in_progress) {
 *     tracker.printSummary();
 *     std::this_thread::sleep_for(std::chrono::seconds(5));
 * }
 * @endcode
 */
class ProgressTracker {
public:
    /**
     * @brief Progress information for a single run
     */
    struct RunProgress {
        int run_id;
        int current_iteration;
        double current_entropy;
        std::chrono::system_clock::time_point last_update_time;
        
        RunProgress()
            : run_id(-1)
            , current_iteration(0)
            , current_entropy(std::numeric_limits<double>::infinity())
            , last_update_time(std::chrono::system_clock::now())
        {}
        
        RunProgress(int id, int iter, double entropy)
            : run_id(id)
            , current_iteration(iter)
            , current_entropy(entropy)
            , last_update_time(std::chrono::system_clock::now())
        {}
    };
    
    /**
     * @brief Construct empty progress tracker
     */
    ProgressTracker() = default;
    
    /**
     * @brief Destructor
     */
    ~ProgressTracker() = default;
    
    // Non-copyable (contains mutex)
    ProgressTracker(const ProgressTracker&) = delete;
    ProgressTracker& operator=(const ProgressTracker&) = delete;
    
    // Non-movable (intended to live at fixed location for reference passing)
    ProgressTracker(ProgressTracker&&) = delete;
    ProgressTracker& operator=(ProgressTracker&&) = delete;
    
    /**
     * @brief Update progress for a specific run
     * 
     * Thread-safe. Can be called from any worker thread.
     * Updates are fast (no I/O under lock).
     * 
     * @param run_id Unique run identifier
     * @param iteration Current iteration number
     * @param entropy Current entropy value
     */
    void updateProgress(int run_id, int iteration, double entropy) {
        std::lock_guard<std::mutex> lock(mutex_);
        progress_map_[run_id] = RunProgress(run_id, iteration, entropy);
    }
    
    /**
     * @brief Get progress for a specific run
     * 
     * Thread-safe. Returns copy of progress data.
     * 
     * @param run_id Run identifier
     * @return RunProgress if found, default-constructed RunProgress otherwise
     */
    RunProgress getProgress(int run_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = progress_map_.find(run_id);
        if (it != progress_map_.end()) {
            return it->second;
        }
        return RunProgress();  // Not found, return default
    }
    
    /**
     * @brief Get all current progress data
     * 
     * Thread-safe. Returns copy of all progress entries.
     * Useful for displaying summary of all active runs.
     * 
     * @return Vector of RunProgress for all tracked runs
     */
    std::vector<RunProgress> getAllProgress() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<RunProgress> result;
        result.reserve(progress_map_.size());
        for (const auto& pair : progress_map_) {
            result.push_back(pair.second);
        }
        return result;
    }
    
    /**
     * @brief Remove progress entry for a completed run
     * 
     * Thread-safe. Call when run completes to free memory.
     * Optional - progress map will grow with active runs only.
     * 
     * @param run_id Run identifier to remove
     */
    void removeRun(int run_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        progress_map_.erase(run_id);
    }
    
    /**
     * @brief Clear all progress data
     * 
     * Thread-safe. Removes all tracked runs.
     * Useful for resetting between different phases.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        progress_map_.clear();
    }
    
    /**
     * @brief Get number of currently tracked runs
     * 
     * Thread-safe.
     * 
     * @return Number of runs being tracked
     */
    size_t getActiveRunCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return progress_map_.size();
    }
    
    /**
     * @brief Print summary of all active runs to stdout
     * 
     * Thread-safe. Useful for monitoring progress from main thread.
     * Format: one line per run with run_id, iteration, entropy, time since update.
     */
    void printSummary() const {
        auto all_progress = getAllProgress();
        
        if (all_progress.empty()) {
            std::cout << "[ProgressTracker] No active runs" << std::endl;
            return;
        }
        
        std::cout << "\n[ProgressTracker] Active runs: " << all_progress.size() << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << std::left 
                  << std::setw(10) << "Run ID"
                  << std::setw(12) << "Iteration"
                  << std::setw(18) << "Entropy"
                  << std::setw(20) << "Last Update"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        auto now = std::chrono::system_clock::now();
        for (const auto& progress : all_progress) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - progress.last_update_time
            );
            
            std::cout << std::left
                      << std::setw(10) << progress.run_id
                      << std::setw(12) << progress.current_iteration
                      << std::setw(18) << std::fixed << std::setprecision(8) << progress.current_entropy
                      << std::setw(20) << (std::to_string(elapsed.count()) + "s ago")
                      << std::endl;
        }
        std::cout << std::string(80, '=') << std::endl;
    }
    
    /**
     * @brief Print detailed progress for a specific run
     * 
     * Thread-safe.
     * 
     * @param run_id Run identifier
     */
    void printRunProgress(int run_id) const {
        auto progress = getProgress(run_id);
        
        if (progress.run_id == -1) {
            std::cout << "[ProgressTracker] Run " << run_id << " not found" << std::endl;
            return;
        }
        
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - progress.last_update_time
        );
        
        std::cout << "[ProgressTracker] Run " << run_id << ":" << std::endl;
        std::cout << "  Iteration: " << progress.current_iteration << std::endl;
        std::cout << "  Entropy: " << std::fixed << std::setprecision(10) 
                  << progress.current_entropy << std::endl;
        std::cout << "  Last update: " << elapsed.count() << "s ago" << std::endl;
    }
    
private:
    mutable std::mutex mutex_;  ///< Protects progress_map_
    std::unordered_map<int, RunProgress> progress_map_;  ///< run_id -> progress
};

}  // namespace entropy

#endif  // PROGRESS_TRACKER_H_
