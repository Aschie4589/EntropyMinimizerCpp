#include "minimizer/orchestration/worker_thread_pool.h"
#include "minimizer/orchestration/run_orchestrator.h"
#include "minimizer/orchestration/progress_tracker.h"
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>


#include "utilities/messaging/DEBUG_LOGGER.h"

namespace entropy {

// ============================================================================
// Constructor & Destructor
// ============================================================================

WorkerThreadPool::WorkerThreadPool(
    DevicePool& device_pool,
    ConcurrentQueue<RunTask>& work_queue,
    ResultCollector& result_collector,
    const std::vector<MinimizerConfig>& configs,
    const HostKrausOperators& kraus_ops,
    int input_dim
)
    : device_pool_(device_pool),
      work_queue_(work_queue),
      result_collector_(result_collector),
      configs_(configs),
      kraus_ops_(kraus_ops),
      input_dim_(input_dim),
      shutdown_requested_(false)
{
    if (configs.empty()) {
        throw std::invalid_argument("WorkerThreadPool: configs cannot be empty");
    }
    
    if (input_dim <= 0) {
        throw std::invalid_argument("WorkerThreadPool: input_dim must be positive");
    }
    std::cout << "WorkerThreadPool: WorkerThreadPool constructed." << std::endl;
}

WorkerThreadPool::~WorkerThreadPool() {
    // Ensure graceful shutdown
    bool has_workers = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        has_workers = !worker_objects_.empty();
    }
    
    if (has_workers) {
        stop();
        try {
            waitAll();
        } catch (const std::exception& e) {
            // Log but don't throw in destructor
            std::cerr << "WorkerThreadPool destructor: " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Lifecycle Methods
// ============================================================================

void WorkerThreadPool::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "WorkerThreadPool: Starting worker threads." << std::endl;    
    if (!worker_objects_.empty()) {
        throw std::runtime_error("WorkerThreadPool already started");
    }
    
    shutdown_requested_ = false;
    
    // Spawn one worker per device
    size_t num_devices = device_pool_.numDevices();
    if (num_devices == 0) {
        throw std::runtime_error("DevicePool has no devices - cannot start workers");
    }
    std::cout << "WorkerThreadPool: Spawning " << num_devices << " workers." << std::endl;    

    // Use addWorkers to spawn initial workers
    mutex_.unlock();
    addWorkers(static_cast<int>(num_devices));
    mutex_.lock();
}

void WorkerThreadPool::stop() {
    std::cout << "WorkerThreadPool: Stopping worker threads." << std::endl;
    shutdown_requested_ = true;
}

void WorkerThreadPool::waitAll() {
    std::vector<std::unique_ptr<Worker>> workers_to_join;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_objects_.empty()) {
            return;  // Nothing to wait for
        }
        workers_to_join = std::move(worker_objects_);
    }
    
    // Join all threads outside the lock
    for (auto& worker : workers_to_join) {
        if (worker->thread && worker->thread->joinable()) {
            worker->thread->join();
        }
    }
    
    // Update state
    std::lock_guard<std::mutex> lock(mutex_);
    worker_objects_.clear();
}

// ============================================================================
// New Dynamic Worker Management (Phase 3)
// ============================================================================

void WorkerThreadPool::addWorkers(int count) {
    if (count <= 0) {
        return;
    }
    std::cout << "WorkerThreadPool: Adding " << count << " workers." << std::endl;    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Determine starting worker ID
    int start_id = worker_objects_.size();
    
    // Spawn new workers
    for (int i = 0; i < count; ++i) {
        std::cout << "WorkerThreadPool: Spawning worker " << (start_id + i) << "." << std::endl;
        // As id, choose the first available integer (check for holes)
        int worker_id;
        for (int wid = 0; ; ++wid) {
            bool id_in_use = false;
            for (const auto& w : worker_objects_) {
                if (w->worker_id == wid) {
                    id_in_use = true;
                    break;
                }
            }
            if (!id_in_use) {
                worker_id = wid;
                break;
            }
        }

        auto worker = std::make_unique<Worker>(worker_id);
        
        // Give the worker its thread, running the worker loop
        worker->thread = std::make_unique<std::thread>([this, worker_id]() {
            std::string log_filename = "worker_" + std::to_string(worker_id) + "_log.txt";
            DEBUG_LOG("Worker " + std::to_string(worker_id) + " thread started.", log_filename);

            // ====== NEW: Acquire device at worker start ======
            IComputeDevice* device = nullptr;
            try {
                device = device_pool_.acquireDevice();
                DEBUG_LOG("Acquired device: ID=" + std::to_string(device->getDeviceID()) + 
                         ", Name=" + device->getName(), log_filename);
                
                // Store device pointer in Worker object
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    for (auto& w : worker_objects_) {
                        if (w->worker_id == worker_id) {
                            w->assigned_device = device;
                            break;
                        }
                    }
                }
            } catch (const std::exception& e) {
                DEBUG_LOG("FATAL: Failed to acquire device: " + std::string(e.what()), log_filename);
                return;  // Exit worker thread - cannot proceed without device
            }
            // ==================================================
            
            // Get worker object
            Worker* worker_ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (auto& w : worker_objects_) {
                    if (w->worker_id == worker_id) {
                        worker_ptr = w.get();
                        break;
                    }
                }
            }
            
            if (!worker_ptr) {
                DEBUG_LOG("Worker " + std::to_string(worker_id) + " could not find its Worker object, exiting.", log_filename);
                device_pool_.releaseDevice(device);
                return;
            }
            
            DEBUG_LOG("Worker " + std::to_string(worker_id) + " started.", log_filename);
            
            try {
                // Worker loop: process tasks until shutdown
                DEBUG_LOG("Entering task processing loop", log_filename);
                int iteration = 0;
                while (!worker_ptr->should_stop.load()) {
                    iteration++;
                    DEBUG_LOG("Loop iteration " + std::to_string(iteration) + ", attempting to fetch task", log_filename);
                    
                    // Try to get a task with timeout
                    auto task_opt = work_queue_.tryPopFor(std::chrono::milliseconds(100));
                    
                    if (!task_opt.has_value()) {
                        DEBUG_LOG("No task available (queue empty or timeout)", log_filename);
                        // No task available - check if we should exit
                        if (shutdown_requested_.load() && work_queue_.empty()) {
                            DEBUG_LOG("Shutdown requested and queue empty, exiting loop", log_filename);
                            break;
                        }
                        continue;
                    }
                    
                    RunTask task = std::move(*task_opt);
                    worker_ptr->is_active = true;
                    DEBUG_LOG("Got task: run_id=" + std::to_string(task.run_id) + ", config_id=" + std::to_string(task.config_id), log_filename);
                    // Log the entries of the vector (first 10 entries)
                    std::ostringstream vec_ss;
                    vec_ss << "Initial vector (first 10 entries): [";
                    for (int vi = 0; vi < std::min(10, task.initial_vector.dimension); ++vi) {
                        vec_ss << task.initial_vector.data[vi];
                        if (vi < std::min(10, task.initial_vector.dimension) - 1) {
                            vec_ss << ", ";
                        }
                    }
                    vec_ss << "]";
                    DEBUG_LOG(vec_ss.str(), log_filename);

                    // Collect and print the first 10 entries of kraus
                    std::ostringstream kraus_ss;
                    kraus_ss << "Kraus operators (first 10 entries of first operator): [";
                    if (kraus_ops_.precision == PrecisionType::DOUBLE) {
                        const auto* typed_data = reinterpret_cast<const std::complex<double>*>(kraus_ops_.data.data());
                        for (int ki = 0; ki < std::min(10, kraus_ops_.input_dim * kraus_ops_.output_dim); ++ki) {
                            kraus_ss << typed_data[ki];
                            if (ki < std::min(10, kraus_ops_.input_dim * kraus_ops_.output_dim) - 1) {
                                kraus_ss << ", ";
                            }
                        }
                    } else {
                        const auto* typed_data = reinterpret_cast<const std::complex<float>*>(kraus_ops_.data.data());
                        for (int ki = 0; ki < std::min(10, kraus_ops_.input_dim * kraus_ops_.output_dim); ++ki) {
                            kraus_ss << typed_data[ki];
                            if (ki < std::min(10, kraus_ops_.input_dim * kraus_ops_.output_dim) - 1) {
                                kraus_ss << ", ";
                            }
                        }
                    }
                    kraus_ss << "]";
                    DEBUG_LOG(kraus_ss.str(), log_filename);

                    try {
                        DEBUG_LOG("Using assigned device for worker " + std::to_string(worker_id), log_filename);
                        
                        // ====== CHANGE: Use acquired device instead of getDeviceForWorker ======
                        // Device was already acquired at worker thread start
                        
                        // Get configuration
                        DEBUG_LOG("Getting config_id=" + std::to_string(task.config_id), log_filename);
                        const MinimizerConfig& config = configs_[task.config_id];
                        DEBUG_LOG("Config retrieved", log_filename);
                        
                        // Log the configuration
                        DEBUG_LOG("Using MinimizerConfig:\n" + config.to_string(), log_filename);
                        // Also log the input dimension
                        DEBUG_LOG("Input dimension for this run: " + std::to_string(input_dim_), log_filename);
                        // Also log the first 10 entries of the kraus operator
                        DEBUG_LOG("Kraus operator precision: " + 
                            std::string(config.algorithm.precision == PrecisionType::DOUBLE ? "DOUBLE" : "FLOAT"), log_filename);
                        DEBUG_LOG("Kraus operator count: " + std::to_string(kraus_ops_.kraus_count), log_filename);
                        DEBUG_LOG("Kraus operator input_dim: " + std::to_string(kraus_ops_.input_dim), log_filename);
                        DEBUG_LOG("Kraus operator output_dim: " + std::to_string(kraus_ops_.output_dim), log_filename);

                        // And log device info
                        DEBUG_LOG("Using device: ID=" + std::to_string(device->getDeviceID()) + 
                            ", Name=" + device->getName(), log_filename);

                        // Create RunOrchestrator for this task
                        DEBUG_LOG("Creating RunOrchestrator for run_id=" + std::to_string(task.run_id), log_filename);

                        // Create progress callback if tracker is available
                        RunOrchestrator::ProgressCallback progress_cb = nullptr;
                        if (progress_tracker_) {
                            // Capture tracker by pointer (safe - tracker outlives workers)
                            ProgressTracker* tracker_ptr = progress_tracker_;
                            progress_cb = [tracker_ptr](int run_id, int iteration, double entropy) {
                                tracker_ptr->updateProgress(run_id, iteration, entropy);
                            };
                        }

                        RunOrchestrator orchestrator(
                            config,
                            kraus_ops_,
                            input_dim_,
                            *device,  // Dereference acquired device pointer
                            progress_cb  // Pass progress callback (or nullptr if no tracker)
                        );
                        DEBUG_LOG("RunOrchestrator created successfully", log_filename);
                        
                        // Execute minimization run
                        DEBUG_LOG("Starting execution of run_id=" + std::to_string(task.run_id) + ", input_dim=" + std::to_string(input_dim_), log_filename);
                        RunResult result = orchestrator.execute(task.run_id, task.initial_vector);
                        // Log the result
                        DEBUG_LOG("Run execution completed: run_id=" + std::to_string(task.run_id) + 
                            ", final_entropy=" + std::to_string(result.final_entropy) + 
                            ", iterations_taken=" + std::to_string(result.iterations_taken) + 
                            ", runtime_seconds=" + std::to_string(result.runtime_seconds), log_filename);
                        DEBUG_LOG("Execution completed for run_id=" + std::to_string(task.run_id), log_filename);
                        
                        // Collect result
                        DEBUG_LOG("Adding result for run_id=" + std::to_string(task.run_id) + " (success=" + std::string(result.isSuccess() ? "true" : "false") + ", entropy=" + std::to_string(result.final_entropy) + ")", log_filename);
                        result_collector_.addResult(task.run_id, result);
                        DEBUG_LOG("Result collected for run_id=" + std::to_string(task.run_id), log_filename);
                        
                    } catch (const std::exception& e) {
                        // Task execution failed
                        DEBUG_LOG("EXCEPTION caught during task execution: " + std::string(e.what()), log_filename);
                        RunResult error_result;
                        error_result.run_id = task.run_id;
                        error_result.error_type = RunErrorType::UNKNOWN;
                        error_result.error_message = std::string("Worker ") + 
                                                     std::to_string(worker_id) + 
                                                     " exception: " + e.what();
                        result_collector_.addResult(task.run_id, error_result);
                        std::cerr << "WorkerThreadPool: " << error_result.error_message << std::endl;
                        DEBUG_LOG("Error result collected for run_id=" + std::to_string(task.run_id), log_filename);
                    }
                    
                    worker_ptr->is_active = false;
                    DEBUG_LOG("Task completed, marked inactive for run_id=" + std::to_string(task.run_id), log_filename);
                }
                
                DEBUG_LOG("Exited task processing loop", log_filename);
            } catch (const std::exception& e) {
                DEBUG_LOG("FATAL ERROR: " + std::string(e.what()), log_filename);
            }
            
            // ====== NEW: Release device at worker end ======
            DEBUG_LOG("Releasing device before shutdown", log_filename);
            try {
                device_pool_.releaseDevice(device);
                DEBUG_LOG("Device released successfully", log_filename);
            } catch (const std::exception& e) {
                DEBUG_LOG("Error releasing device: " + std::string(e.what()), log_filename);
            }
            // ===============================================
            
            DEBUG_LOG("Worker shutting down", log_filename);
        });
        
        worker_objects_.push_back(std::move(worker));
    }
}

void WorkerThreadPool::removeWorkers(int count) {
    if (count <= 0) {
        return;
    }

    DEBUG_LOG("WorkerThreadPool: Removing " + std::to_string(count) + " workers.", "worker_pool_log.txt");
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (count >= static_cast<int>(worker_objects_.size())) {
        // Removing all workers - just call stop
        shutdown_requested_ = true;
        DEBUG_LOG("WorkerThreadPool: Shutdown requested due to removing all workers.", "worker_pool_log.txt");
        return;
    }
    
    // Signal last N workers to stop
    int stop_from = worker_objects_.size() - count;
    for (int i = stop_from; i < static_cast<int>(worker_objects_.size()); ++i) {
        DEBUG_LOG("WorkerThreadPool: Signaling worker " + std::to_string(worker_objects_[i]->worker_id) + " to stop.", "worker_pool_log.txt");
        worker_objects_[i]->should_stop = true;
    }
    
    // Wait for them to finish and remove
    std::vector<std::unique_ptr<Worker>> workers_to_remove;
    for (int i = stop_from; i < static_cast<int>(worker_objects_.size()); ++i) {
        workers_to_remove.push_back(std::move(worker_objects_[i]));
    }
    worker_objects_.erase(worker_objects_.begin() + stop_from, worker_objects_.end());
    DEBUG_LOG("WorkerThreadPool: Removed " + std::to_string(workers_to_remove.size()) + " workers from internal list.", "worker_pool_log.txt");
    
    // Join threads outside lock
    mutex_.unlock();
    for (auto& worker : workers_to_remove) {
        if (worker->thread && worker->thread->joinable()) {
            DEBUG_LOG("WorkerThreadPool: Joining worker " + std::to_string(worker->worker_id) + " thread.", "worker_pool_log.txt");
            worker->thread->join();
        }
    }
    mutex_.lock();
}

int WorkerThreadPool::getWorkerCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return worker_objects_.size();
}

int WorkerThreadPool::getActiveWorkerCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int count = 0;
    for (const auto& worker : worker_objects_) {
        if (worker->is_active.load()) {
            count++;
        }
    }
    return count;
}

int WorkerThreadPool::removeWorkersUsingDevice(int physical_device_id) {
    std::vector<std::unique_ptr<Worker>> workers_to_remove;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find workers using this device
        for (auto it = worker_objects_.begin(); it != worker_objects_.end(); ) {
            if ((*it)->assigned_device) {
                try {
                    int device_id = device_pool_.getPhysicalDeviceID((*it)->assigned_device);
                    if (device_id == physical_device_id) {
                        DEBUG_LOG("Marking worker " + std::to_string((*it)->worker_id) + 
                                 " for removal (uses device " + std::to_string(physical_device_id) + ")",
                                 "worker_pool_log.txt");
                        (*it)->should_stop = true;
                        workers_to_remove.push_back(std::move(*it));
                        it = worker_objects_.erase(it);
                    } else {
                        ++it;
                    }
                } catch (const std::exception& e) {
                    DEBUG_LOG("Error checking device for worker " + std::to_string((*it)->worker_id) + 
                             ": " + e.what(), "worker_pool_log.txt");
                    ++it;
                }
            } else {
                ++it;
            }
        }
    }
    
    // Join threads outside lock (blocks until workers finish current task)
    for (auto& worker : workers_to_remove) {
        if (worker->thread && worker->thread->joinable()) {
            DEBUG_LOG("Waiting for worker " + std::to_string(worker->worker_id) + " to finish...",
                     "worker_pool_log.txt");
            worker->thread->join();
            DEBUG_LOG("Worker " + std::to_string(worker->worker_id) + " joined successfully",
                     "worker_pool_log.txt");
        }
    }
    
    int removed_count = workers_to_remove.size();
    std::cout << "WorkerThreadPool: Removed " << removed_count 
              << " worker(s) using device " << physical_device_id << std::endl;
    
    return removed_count;
}

std::vector<int> WorkerThreadPool::getWorkerIDsUsingDevice(int physical_device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<int> worker_ids;
    for (const auto& worker : worker_objects_) {
        if (worker->assigned_device) {
            try {
                int device_id = device_pool_.getPhysicalDeviceID(worker->assigned_device);
                if (device_id == physical_device_id) {
                    worker_ids.push_back(worker->worker_id);
                }
            } catch (...) {
                // Device not found, skip
            }
        }
    }
    
    return worker_ids;
}

void WorkerThreadPool::setProgressTracker(ProgressTracker* tracker) {
    std::lock_guard<std::mutex> lock(mutex_);
    progress_tracker_ = tracker;
    std::cout << "WorkerThreadPool: Progress tracker " 
              << (tracker ? "enabled" : "disabled") << std::endl;
}

} // namespace entropy
