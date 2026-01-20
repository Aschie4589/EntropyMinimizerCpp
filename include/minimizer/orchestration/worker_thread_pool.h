#ifndef WORKER_THREAD_POOL_H_
#define WORKER_THREAD_POOL_H_

#include "minimizer/orchestration/types.h"
#include "minimizer/orchestration/concurrent_queue.h"
#include "minimizer/orchestration/result_collector.h"
#include "minimizer/orchestration/device_pool.h"
#include "minimizer/config/minimizer_config.h"
#include "minimizer/algorithm/minimization_strategy.h"
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

namespace entropy {

/**
 * @brief Thread pool for parallel minimization run execution
 * 
 * Manages worker threads that:
 * 1. Pull RunTask items from shared ConcurrentQueue
 * 2. Assign device via DevicePool (round-robin)
 * 3. Execute task via RunOrchestrator
 * 4. Collect results in ResultCollector
 * 5. Support dynamic resizing (add/remove workers)
 * 6. Handle graceful shutdown
 * 
 * Architecture:
 * - Each worker thread owns one RunOrchestrator instance
 * - DevicePool provides device assignment (non-owning reference)
 * - ConcurrentQueue distributes tasks (thread-safe FIFO)
 * - ResultCollector aggregates outcomes (thread-safe)
 * - Shared MinimizerConfig across all workers
 * 
 * Thread Safety:
 * - All public methods are thread-safe
 * - Workers execute independently (no shared state except via synchronized containers)
 * - RunOrchestrator instances are NOT shared between threads
 * - Shutdown is cooperative: workers check shutdown_requested_ atomically
 * 
 * Lifecycle:
 * 1. Construction: Initialize with references, no threads yet
 * 2. start(): Spawn worker threads
 * 3. Workers pull tasks until queue empty + shutdown signaled
 * 4. stop(): Signal shutdown (cooperative)
 * 5. waitAll(): Block until all workers exit
 * 6. Destruction: Ensure all threads joined
 * 
 * Error Handling:
 * - Worker exceptions caught in workerLoop()
 * - Errors added to ResultCollector with run_id
 * - One worker error does NOT crash other workers
 * - Continue processing remaining tasks after error
 * 
 * Typical Usage:
 * @code
 * DevicePool device_pool(limits);
 * ConcurrentQueue<RunTask> work_queue;
 * ResultCollector results;
 * std::vector<MinimizerConfig> configs = {config};
 * HostKrausOperators kraus_ops = ...;
 * int input_dim = 5;
 * 
 * WorkerThreadPool pool(device_pool, work_queue, results, configs, kraus_ops, input_dim);
 * 
 * // Enqueue tasks
 * for (int i = 0; i < 100; ++i) {
 *     work_queue.push(RunTask(i, 0, initial_vectors[i]));
 * }
 * 
 * // Start workers
 * pool.start();
 * 
 * // Wait for completion
 * work_queue.signalDone();
 * pool.stop();
 * pool.waitAll();
 * 
 * // Collect results
 * auto best = results.getMinimum();
 * @endcode
 */
class WorkerThreadPool {
public:
    /**
     * @brief Construct worker thread pool
     * 
     * Initializes pool with references to orchestration components.
     * No worker threads are spawned until start() is called.
     * 
     * Ownership Model:
     * - device_pool: Non-owning reference (must outlive this object)
     * - work_queue: Non-owning reference (must outlive this object)
     * - result_collector: Non-owning reference (must outlive this object)
     * - configs: Non-owning reference (must outlive this object)
     * - kraus_ops: Copied (each worker needs its own copy for RunOrchestrator)
     * 
     * @param device_pool Device assignment manager (non-owning)
     * @param work_queue Task distribution queue (non-owning)
     * @param result_collector Result aggregation collector (non-owning)
     * @param configs Minimizer configurations (non-owning, shared by all workers)
     * @param kraus_ops Kraus operators for quantum channel (copied)
     * @param input_dim Input dimension for quantum state vectors
     * 
     * @throws std::invalid_argument if configs is empty
     */
    WorkerThreadPool(
        DevicePool& device_pool,
        ConcurrentQueue<RunTask>& work_queue,
        ResultCollector& result_collector,
        const std::vector<MinimizerConfig>& configs,
        const HostKrausOperators& kraus_ops,
        int input_dim
    );
    
    /**
     * @brief Destructor - ensures all threads are joined
     * 
     * Calls stop() and waitAll() if not already called.
     * Safe to call even if workers already stopped.
     */
    ~WorkerThreadPool();
    
    // Non-copyable (owns threads)
    WorkerThreadPool(const WorkerThreadPool&) = delete;
    WorkerThreadPool& operator=(const WorkerThreadPool&) = delete;
    
    // Non-movable (workers hold references to local state)
    WorkerThreadPool(WorkerThreadPool&&) = delete;
    WorkerThreadPool& operator=(WorkerThreadPool&&) = delete;
    
    /**
     * @brief Start worker threads
     * 
     * Spawns worker_count threads, each running workerLoop().
     * Initial worker count equals device_pool.numDevices().
     * 
     * @throws std::runtime_error if already started
     */
    void start();
    
    /**
     * @brief Signal shutdown to all workers
     * 
     * Sets shutdown_requested_ flag. Workers will exit after
     * completing current task and seeing empty queue or timeout.
     * 
     * This is a cooperative shutdown - does NOT forcefully kill threads.
     * Call waitAll() after stop() to wait for actual completion.
     * 
     * Idempotent - safe to call multiple times.
     */
    void stop();
    
    /**
     * @brief Wait for all worker threads to complete
     * 
     * Blocks until all threads have exited (joined).
     * 
     * @throws std::runtime_error if not started
     */
    void waitAll();
    
    // ========================================================================
    // New Dynamic Worker Management (Phase 3)
    // ========================================================================
    
    /**
     * @brief Add workers to the pool
     * 
     * Spawns count new worker threads immediately.
     * Workers begin pulling from queue right away.
     * 
     * @param count Number of workers to add
     */
    void addWorkers(int count);
    
    /**
     * @brief Remove workers from the pool gracefully
     * 
     * Signals count workers to stop after finishing current task.
     * Workers will complete their current task, then exit.
     * This is a graceful removal - no forced termination.
     * 
     * @param count Number of workers to remove
     */
    void removeWorkers(int count);
    
    /**
     * @brief Get current worker count
     * 
     * @return Total number of worker threads (including those marked for removal)
     */
    int getWorkerCount() const;
    
    /**
     * @brief Get number of workers actively executing tasks
     * 
     * @return Number of workers currently processing a task (not just waiting)
     */
    int getActiveWorkerCount() const;
    
    /**
     * @brief Remove workers that are using a specific physical device
     * 
     * Gracefully stops workers assigned to the given device.
     * Workers will complete their current task before stopping.
     * Blocks until all affected workers have joined.
     * 
     * @param physical_device_id CUDA device ID
     * @return Number of workers removed
     */
    int removeWorkersUsingDevice(int physical_device_id);
    
    /**
     * @brief Get worker IDs using a specific device
     * 
     * @param physical_device_id CUDA device ID
     * @return Vector of worker IDs assigned to this device
     */
    std::vector<int> getWorkerIDsUsingDevice(int physical_device_id) const;
    
private:
    /**
     * @brief Worker state structure
     */
    struct Worker {
        std::unique_ptr<std::thread> thread;
        std::atomic<bool> should_stop{false};
        std::atomic<bool> is_active{false};
        int worker_id;
        IComputeDevice* assigned_device = nullptr;  ///< Device assigned to this worker
        
        Worker(int id) : worker_id(id) {}
    };
    
    // Orchestration component references (non-owning)
    DevicePool& device_pool_;
    ConcurrentQueue<RunTask>& work_queue_;
    ResultCollector& result_collector_;
    const std::vector<MinimizerConfig>& configs_;
    
    // Data needed for RunOrchestrator creation
    HostKrausOperators kraus_ops_;  ///< Copied (each worker needs own copy)
    int input_dim_;                 ///< Input dimension for vectors
    
    // Thread management
    std::vector<std::unique_ptr<Worker>> worker_objects_;  ///< Worker state objects
    std::atomic<bool> shutdown_requested_;   ///< Shutdown signal
    mutable std::mutex mutex_;               ///< Protects workers_ vector
};

} // namespace entropy

#endif // WORKER_THREAD_POOL_H_
