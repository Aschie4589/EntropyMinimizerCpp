#ifndef RUN_ORCHESTRATOR_H_
#define RUN_ORCHESTRATOR_H_

#include "minimizer/config/minimizer_config.h"
#include "minimizer/algorithm/algorithm_manager.h"
#include "minimizer/stopping/conditions_checker.h"
#include "minimizer/state/entropy_predictor.h"
#include "minimizer/orchestration/types.h"
#include "compute/device/IComputeDevice.h"
#include <functional>
#include <memory>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace entropy {

// Forward declarations
class CheckpointManager;

/**
 * @brief Executes a single minimization run from start to finish
 * 
 * Responsibilities:
 * - Execute one minimization run on a single device
 * - Manage run lifecycle: initialization → iteration loop → finalization
 * - Own and coordinate: AlgorithmManager, ConditionsChecker, EntropyPredictor, CheckpointManager
 * - Report progress via optional callback
 * - Handle exceptions gracefully and return error results
 * 
 * Ownership Model:
 * - OWNS: AlgorithmManager, ConditionsChecker, EntropyPredictor, CheckpointManager
 * - REFERENCES: IComputeDevice (non-owning, provided by DevicePool)
 * - USES: MinimizerConfig (const reference, lives in orchestrator)
 * 
 * Thread Safety:
 * - Not thread-safe internally
 * - Each worker thread should have its own RunOrchestrator instance
 * - Safe to call execute() from different threads on different instances
 * 
 * Typical Usage:
 * @code
 * // Worker thread
 * RunOrchestrator orchestrator(config, device, [](int id, int iter, double S) {
 *     std::cout << "Run " << id << " iter " << iter << " S=" << S << std::endl;
 * });
 * 
 * HostVector initial = generateRandomVector(dim);
 * RunResult result = orchestrator.execute(run_id, initial);
 * 
 * if (result.isSuccess()) {
 *     std::cout << "Success! Final entropy: " << result.final_entropy << std::endl;
 * } else {
 *     std::cerr << "Error: " << result.error_message << std::endl;
 * }
 * @endcode
 */
class RunOrchestrator {
public:
    /**
     * @brief Progress callback signature
     * 
     * Called periodically during minimization run from the worker thread.
     * 
     * Thread Safety:
     * - Invoked from worker thread, NOT main thread
     * - Implementation MUST be thread-safe (use atomics, mutexes, or lock-free structures)
     * - Should be non-blocking and fast (avoid I/O, heavy computation)
     * - Typical use: atomic counter updates, lock-free logging, condition variable signals
     * 
     * Exception Safety:
     * - Callbacks are protected by try-catch; exceptions will be logged but won't crash the run
     * - However, prefer noexcept implementations to avoid overhead
     * 
     * @param run_id Unique run identifier
     * @param iteration Current iteration number (0-based)
     * @param entropy Current entropy value
     */
    using ProgressCallback = std::function<void(int run_id, int iteration, double entropy)>;
    
    /**
     * @brief Construct RunOrchestrator
     * 
     * Initializes all owned components based on config.
     * 
     * Ownership Model:
     * - config: Non-owning reference (must outlive this object)
     * - kraus_ops: Copied by AlgorithmManager (not stored here to avoid duplication)
     * - device: Non-owning reference (owned by DevicePool, must outlive this object)
     * - progress_cb: Copied (std::function ownership semantics)
     * 
     * @param config Minimizer configuration (must remain valid during lifetime)
     * @param kraus_ops Kraus operators for the quantum channel (copied by AlgorithmManager)
     * @param input_dim Input dimension for quantum state vectors
     * @param device Compute device for execution (non-owning reference)
     * @param progress_cb Optional progress callback (default: nullptr)
     * 
     * @throws std::invalid_argument if config is invalid
     * @throws std::runtime_error if component initialization fails
     */
    RunOrchestrator(
        const MinimizerConfig& config,
        const HostKrausOperators& kraus_ops,
        int input_dim,
        IComputeDevice& device,
        ProgressCallback progress_cb = nullptr
    );
    
    /**
     * @brief Destructor
     * 
     * Must be defined in .cpp file due to forward-declared CheckpointManager
     * in unique_ptr (incomplete type issue with default destructor)
     */
    ~RunOrchestrator();
    
    // Non-copyable (owns unique resources)
    RunOrchestrator(const RunOrchestrator&) = delete;
    RunOrchestrator& operator=(const RunOrchestrator&) = delete;
    
    // Movable
    RunOrchestrator(RunOrchestrator&&) = default;
    RunOrchestrator& operator=(RunOrchestrator&&) = default;
    
    /**
     * @brief Execute minimization run
     * 
     * Main entry point for run execution. Handles full lifecycle:
     * 1. Initialize algorithm with starting vector
     * 2. Reset stopping conditions and predictor
     * 3. Main iteration loop (minimize, check conditions, checkpoint)
     * 4. Finalize and return result
     * 
     * Error Handling:
     * - Catches all exceptions during execution
     * - Returns RunResult with error_message populated on failure
     * - Partial results (e.g., iterations_taken) are preserved on error
     * 
     * @param run_id Unique identifier for this run
     * @param initial_vector Starting quantum state vector
     * @return RunResult with outcome (success or error)
     */
    RunResult execute(int run_id, const HostVector& initial_vector);
    
private:
    /**
     * @brief Main minimization loop
     * 
     * Iterates until stopping condition met:
     * - Perform one iteration (algorithm step)
     * - Update entropy tracking
     * - Check stopping conditions
     * - Handle checkpoints
     * - Invoke progress callback
     * 
     * @param run_id Run identifier for callbacks/checkpoints
     * @return Final entropy value
     */
    double mainLoop(int run_id);
    
    /**
     * @brief Check stopping conditions
     * 
     * @param iteration Current iteration number
     * @param current_entropy Current entropy value
     * @param previous_entropy Previous iteration's entropy
     * @return StopReason (CONTINUE or specific stop reason)
     */
    StopReason checkStoppingConditions(
        size_t iteration,
        double current_entropy,
        double previous_entropy
    );
    
    /**
     * @brief Handle checkpoint saving if needed
     * 
     * Checks if checkpoint should be saved based on:
     * - Checkpoint interval (if configured)
     * - Iteration number
     * 
     * @param run_id Run identifier
     * @param iteration Current iteration number
     * @param current_entropy Current entropy value
     * @param min_entropy Minimum entropy seen so far
     */
    void handleCheckpoint(
        int run_id,
        int iteration,
        double current_entropy,
        double min_entropy
    );
    
    /**
     * @brief Build RunResult from execution outcome
     * 
     * @param run_id Run identifier
     * @param final_entropy Final entropy achieved
     * @param iterations Number of iterations performed
     * @param runtime_seconds Elapsed wall-clock time
     * @return Complete RunResult
     */
    RunResult buildResult(
        int run_id,
        double final_entropy,
        int iterations,
        double runtime_seconds
    ) const;
    
    // Configuration and device
    const MinimizerConfig& config_;   ///< Minimizer configuration (non-owning)
    IComputeDevice& device_;          ///< Compute device (non-owning)
    int input_dim_;                   ///< Input dimension for quantum states
    
    // Owned components
    std::unique_ptr<EntropyManager> entropy_manager_;     ///< Entropy calculation
    std::unique_ptr<AlgorithmManager> algorithm_mgr_;     ///< Manages minimization algorithm
    std::unique_ptr<ConditionsChecker> conditions_;       ///< Stopping condition checker
    std::unique_ptr<EntropyPredictor> predictor_;         ///< Entropy prediction
    std::unique_ptr<CheckpointManager> checkpoint_mgr_;   ///< Checkpoint management
    
    // Callback
    ProgressCallback progress_cb_;    ///< Optional progress reporting
    
    // Tracking
    int current_iteration_;           ///< Current iteration in main loop
    double min_entropy_seen_;         ///< Minimum entropy encountered so far
};

} // namespace entropy

#endif // RUN_ORCHESTRATOR_H_
