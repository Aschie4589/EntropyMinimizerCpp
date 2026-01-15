#ifndef MINIMIZER_ORCHESTRATOR_H_
#define MINIMIZER_ORCHESTRATOR_H_

#include "minimizer/orchestration/types.h"
#include "minimizer/orchestration/device_pool.h"
#include "minimizer/orchestration/worker_thread_pool.h"
#include "minimizer/orchestration/concurrent_queue.h"
#include "minimizer/orchestration/result_collector.h"
#include "minimizer/config/minimizer_config.h"
#include "minimizer/config/resource_config.h"
#include "minimizer/algorithm/minimization_strategy.h"
#include <random>
#include <memory>

namespace entropy {

/**
 * @brief Top-level orchestrator for multi-run minimization workflows
 * 
 * Coordinates all orchestration components to execute the Minimum-of-Entropies (MOE)
 * algorithm: runs N independent minimization attempts with different random initial
 * vectors and returns the result with minimum entropy.
 * 
 * Architecture:
 * - Creates and manages DevicePool (GPU/CPU device allocation)
 * - Creates and manages WorkerThreadPool (parallel task execution)
 * - Uses ConcurrentQueue for task distribution
 * - Uses ResultCollector for result aggregation
 * - Delegates single-run execution to RunOrchestrator (via WorkerThreadPool)
 * 
 * Design Principles:
 * - Stateless: Components created/destroyed per findMOE() call
 * - Thread-safe: All orchestration components are thread-safe
 * - Exception-safe: Proper cleanup on error
 * - Resource-aware: Respects configuration resource limits
 * 
 * Typical Workflow:
 * 1. User creates MinimizerConfig with Kraus operators, resources, stopping criteria
 * 2. Calls findMOE(config)
 * 3. Orchestrator creates all components
 * 4. Generates N random initial vectors
 * 5. Distributes tasks to WorkerThreadPool
 * 6. Workers execute minimization runs in parallel
 * 7. Results collected and minimum entropy identified
 * 8. Returns best result
 * 
 * Usage:
 * @code
 * MinimizerOrchestrator orchestrator;
 * 
 * MinimizerConfig config;
 * config.multi_run.num_attempts = 100;
 * config.resources.max_gpus = 2;
 * config.resources.max_cpus = 4;
 * // ... configure Kraus operators, stopping criteria, etc.
 * 
 * RunResult best = orchestrator.findMOE(config);
 * std::cout << "Minimum entropy: " << best.final_entropy << std::endl;
 * @endcode
 */
class MinimizerOrchestrator {
public:
    /**
     * @brief Default constructor (stateless orchestrator)
     * 
     * Creates orchestrator in default state. All components are created
     * and destroyed within findMOE() calls.
     */
    MinimizerOrchestrator() = default;
    
    /**
     * @brief Destructor
     */
    ~MinimizerOrchestrator() = default;
    
    // Non-copyable (to avoid accidental copies of potentially expensive state)
    MinimizerOrchestrator(const MinimizerOrchestrator&) = delete;
    MinimizerOrchestrator& operator=(const MinimizerOrchestrator&) = delete;
    
    // Movable
    MinimizerOrchestrator(MinimizerOrchestrator&&) = default;
    MinimizerOrchestrator& operator=(MinimizerOrchestrator&&) = default;
    
    // ========================================================================
    // Orchestration Statistics (Phase 4)
    // ========================================================================
    
    /**
     * @brief Statistics for monitoring orchestration
     */
    struct OrchestrationStats {
        int total_workers = 0;                              ///< Total worker threads
        int active_workers = 0;                             ///< Workers executing tasks
        int completed_runs = 0;                             ///< Successfully completed runs
        int failed_runs = 0;                                ///< Failed runs
        int pending_tasks = 0;                              ///< Tasks in queue
        std::vector<DevicePool::DeviceStatus> device_status; ///< Device information
    };
    
    /**
     * @brief Get orchestration statistics
     * 
     * Returns current state of orchestration (only valid during findMOE execution).
     * 
     * @return Statistics structure
     */
    OrchestrationStats getStats() const;
    
    // ========================================================================
    // Main API
    // ========================================================================
    
    /**
     * @brief Execute multi-run minimization workflow with ResourceConfig (new)
     * 
     * New API that uses ResourceConfig for dynamic device management.
     * Supports runtime adjustment of GPU allocation based on gpu_config.yaml.
     * 
     * @param config Complete minimizer configuration
     * @param kraus_ops Kraus operators defining the quantum channel
     * @param input_dim Input dimension for quantum state vectors
     * @param resource_config Resource configuration for dynamic device management
     * @return RunResult with minimum entropy across all attempts
     * 
     * @throws std::invalid_argument if config invalid
     * @throws std::runtime_error if no successful runs or orchestration fails
     */
    RunResult findMOE(
        const MinimizerConfig& config,
        const HostKrausOperators& kraus_ops,
        int input_dim,
        const ResourceConfig& resource_config
    );
    
private:
    /**
     * @brief Generate random initial quantum state vector
     * 
     * Creates a random normalized quantum state vector for use as
     * initial guess in minimization run.
     * 
     * Generation Strategy:
     * - Use uniform random distribution [0,1] for both real and imaginary parts
     * - Normalize to unit norm (quantum state requirement)
     * - Thread-safe random number generation
     * 
     * @param dim Dimension of quantum state vector
     * @return Normalized random quantum state vector
     */
    HostVector generateInitialVector(int dim);
    
    /**
     * @brief Fill work queue with run tasks
     * 
     * Generates config.multi_run.num_attempts tasks with unique
     * initial vectors and pushes them to the work queue.
     * 
     * @param queue Task queue (thread-safe)
     * @param num_attempts Number of tasks to generate
     * @param input_dim Dimension for initial vectors
     */
    void fillWorkQueue(
        ConcurrentQueue<RunTask>& queue,
        int num_attempts,
        int input_dim
    );
    
    /**
     * @brief Configuration change callback (Phase 4)
     * 
     * Called by DeviceRegistry when gpu_config.yaml changes.
     * Adjusts DevicePool and WorkerThreadPool to match new configuration.
     * 
     * @param enabled_gpus Vector of currently enabled GPU device IDs
     */
    void onConfigurationChanged(const std::vector<int>& enabled_gpus);
    
    /**
     * @brief Adjust resources to match new configuration (Phase 4)
     * 
     * Synchronizes DevicePool and WorkerThreadPool with new GPU list.
     * 
     * @param enabled_gpus Vector of enabled GPU device IDs
     */
    void adjustResources(const std::vector<int>& enabled_gpus);
    
    // Random number generator (per-orchestrator instance)
    std::mt19937 rng_{std::random_device{}()};
    
    // Component pointers (for callbacks during findMOE execution)
    DevicePool* device_pool_ = nullptr;
    WorkerThreadPool* worker_pool_ = nullptr;
    ResultCollector* result_collector_ = nullptr;
    ConcurrentQueue<RunTask>* work_queue_ = nullptr;
};

} // namespace entropy

#endif // MINIMIZER_ORCHESTRATOR_H_
