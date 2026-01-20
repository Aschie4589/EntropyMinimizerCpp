#include "minimizer/orchestration/minimizer_orchestrator.h"
#include "minimizer/orchestration/gpu_registry.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <set>
#include <algorithm>

namespace entropy {

// ============================================================================
// Helper Methods
// ============================================================================

HostVector MinimizerOrchestrator::generateInitialVector(int dim) {
    if (dim <= 0) {
        throw std::invalid_argument(
            "MinimizerOrchestrator::generateInitialVector: dim must be > 0, got " + 
            std::to_string(dim)
        );
    }
    
    // Create vector
    HostVector vec;
    vec.dimension = dim;
    vec.data.resize(dim);
    
    // Fill with random complex values
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < dim; ++i) {
        double real_part = dist(rng_);
        double imag_part = dist(rng_);
        vec.data[i] = std::complex<double>(real_part, imag_part);
    }
    
    // Normalize to unit norm (required for quantum states)
    double norm = 0.0;
    for (int i = 0; i < dim; ++i) {
        norm += std::norm(vec.data[i]);  // |z|^2
    }
    norm = std::sqrt(norm);
    
    if (norm < 1e-15) {
        // Extremely unlikely but handle gracefully
        throw std::runtime_error(
            "MinimizerOrchestrator::generateInitialVector: Generated zero vector"
        );
    }
    
    for (int i = 0; i < dim; ++i) {
        vec.data[i] /= norm;
    }
    
    return vec;
}

void MinimizerOrchestrator::fillWorkQueue(
    ConcurrentQueue<RunTask>& queue,
    int num_attempts,
    int input_dim
) {
    if (num_attempts <= 0) {
        throw std::invalid_argument(
            "MinimizerOrchestrator::fillWorkQueue: num_attempts must be > 0, got " + 
            std::to_string(num_attempts)
        );
    }
    
    for (int i = 0; i < num_attempts; ++i) {
        RunTask task;
        task.run_id = i;
        task.config_id = 0;  // Single config for now (future: support multiple configs)
        task.initial_vector = generateInitialVector(input_dim);
        task.priority = 0;   // Future: use for prioritization
        
        queue.push(std::move(task));
    }
}

// ============================================================================
// Main Orchestration
// ============================================================================
// Main API: findMOE with ResourceConfig
// ============================================================================

RunResult MinimizerOrchestrator::findMOE(
    const MinimizerConfig& config,
    const HostKrausOperators& kraus_ops,
    int input_dim
    ) {
    // 1. Validate configurations
    try {
        config.validate();
    } catch (const std::exception& e) {
        throw std::invalid_argument(
            std::string("MinimizerOrchestrator::findMOE: Invalid configuration: ") + e.what()
        );
    }

    if (config.multi_run.num_attempts <= 0) {
        throw std::invalid_argument(
            "MinimizerOrchestrator::findMOE: num_attempts must be > 0, got " + 
            std::to_string(config.multi_run.num_attempts)
        );
    }
    
    if (input_dim <= 0) {
        throw std::invalid_argument(
            "MinimizerOrchestrator::findMOE: input_dim must be > 0, got " + 
            std::to_string(input_dim)
        );
    }
    
    std::cout << "Validated configuration!" << std::endl;
    
    // 2. Initialize GPURegistry. This tells the minimizer what devices are available.
    GPURegistry& registry = GPURegistry::instance();
    try {
        registry.loadConfig(config.resource.gpu_config_file);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not load GPU config, using default: " << e.what() << std::endl;
    }

    std::cout << "Initialized GPURegistry with "
              << registry.getAllGPUInfo().size() << " available devices. Of them, "
                << registry.getEnabledGPUs().size() << " are enabled."
              << std::endl;
    
    std::cout << "Creating DevicePool and initializing from registry..." << std::endl;

    // 3. Create orchestration components. DevicePool is a pool of compute devices for work distribution, which are created dynamically based on the available hardware.
    DevicePool device_pool(config.resource);
    device_pool.initializeFromRegistry(registry);

    std::cout << "Initialized DevicePool with "
              << device_pool.numDevices() << " devices ("
              << device_pool.numGPUs() << " GPUs, "
              << device_pool.numCPUs() << " CPUs)."
              << std::endl;
    // Create work queue and result collector.
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    std::cout << "Created work queue and result collector." << std::endl;

    // Store pointers for callbacks
    device_pool_ = &device_pool;
    work_queue_ = &work_queue;
    result_collector_ = &result_collector;
    
    std::cout << "Stored pointers for callbacks." << std::endl;
    // Prepare configuration array (single config for now)
    std::vector<MinimizerConfig> configs = {config};
    std::cout << "About to fill work queue with tasks..." << std::endl;
    // 4. Fill work queue with tasks
    fillWorkQueue(work_queue, config.multi_run.num_attempts, input_dim);
    std::cout << "Filled work queue with "
              << config.multi_run.num_attempts << " tasks."
              << std::endl;
    std::cout << "Starting minimization with "
              << device_pool.numDevices() << " devices and "
              << config.multi_run.num_attempts << " attempts."
              << std::endl;

    // 5. Create and start worker thread pool
    WorkerThreadPool worker_pool(
        device_pool,
        work_queue,
        result_collector,
        configs,
        kraus_ops,
        input_dim
    );
    std::cout << "Starting worker thread pool..." << std::endl;
    worker_pool_ = &worker_pool;
    
    // 6. Subscribe to configuration changes (if enabled)
    if (config.resource.allow_dynamic_scaling) {
        registry.subscribeToChanges(
            [this](const std::vector<int>& enabled_gpus) {
                this->onConfigurationChanged(enabled_gpus);
            }
        );
        registry.startWatching(config.resource.poll_interval);
    }
    
    worker_pool.start();
    std::cout << "Worker thread pool started." << std::endl;
    
    // 7. Wait for all tasks to complete
    while (result_collector.numCompleted() + result_collector.numErrors() < 
           static_cast<size_t>(config.multi_run.num_attempts)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        std::cout << "Waiting for tasks to complete: "
                  << result_collector.numCompleted() << " completed, "
                  << result_collector.numErrors() << " errors, "
                  << work_queue.size() << " pending."
                  << std::endl;
        
        // Check if queue is empty and workers are idle
        if (work_queue.empty() && worker_pool.getActiveWorkerCount() == 0) {
            // All tasks done
            break;
        }
    }
    
    // 8. Stop watching for changes
    if (config.resource.allow_dynamic_scaling) {
        registry.stopWatching();
    }
    
    // 9. Signal completion and stop workers
    work_queue.signalDone();
    worker_pool.stop();
    worker_pool.waitAll();
    
    // Clear pointers
    device_pool_ = nullptr;
    worker_pool_ = nullptr;
    result_collector_ = nullptr;
    work_queue_ = nullptr;
    
    // 10. Get minimum entropy result
    try {
        RunResult best = result_collector.getMinimum();
        return best;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("MinimizerOrchestrator::findMOE: No successful runs: ") + e.what()
        );
    }
}

// ============================================================================
// Statistics and Monitoring (Phase 4)
// ============================================================================

MinimizerOrchestrator::OrchestrationStats MinimizerOrchestrator::getStats() const {
    OrchestrationStats stats;
    
    if (worker_pool_) {
        stats.total_workers = worker_pool_->getWorkerCount();
        stats.active_workers = worker_pool_->getActiveWorkerCount();
    }
    
    if (result_collector_) {
        stats.completed_runs = result_collector_->numCompleted();
        stats.failed_runs = result_collector_->numErrors();
    }
    
    if (work_queue_) {
        stats.pending_tasks = work_queue_->size();
    }
    
    if (device_pool_) {
        stats.device_status = device_pool_->getDeviceStatus();
    }
    
    return stats;
}

// ============================================================================
// Dynamic Resource Adjustment (Phase 4)
// ============================================================================

void MinimizerOrchestrator::onConfigurationChanged(const std::vector<int>& enabled_gpus) {
    std::cout << "MinimizerOrchestrator: Configuration changed, adjusting resources..." << std::endl;
    if (!device_pool_ || !worker_pool_) {
        std::cout << "MinimizerOrchestrator: Device pool or worker pool not initialized. Skipping adjustment." << std::endl;
        return;  // Not initialized yet or already cleaned up
    }
    
    try {
        std::cout << "MinimizerOrchestrator: New enabled GPUs: ";
        for (int gpu_id : enabled_gpus) {
            std::cout << gpu_id << " ";
        }
        std::cout << std::endl;
        adjustResources(enabled_gpus);

        std::cout << "MinimizerOrchestrator: Resource adjustment complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error adjusting resources: " << e.what() << std::endl;
    }
}

void MinimizerOrchestrator::adjustResources(const std::vector<int>& enabled_gpus) {
    std::cout << "MinimizerOrchestrator: ===== Resource Adjustment Started =====" << std::endl;
    
    if (!device_pool_ || !worker_pool_) {
        std::cout << "MinimizerOrchestrator: Pools not initialized, skipping." << std::endl;
        return;
    }
    
    // ========================================================================
    // STEP 1: Identify devices to remove and add
    // ========================================================================
    
    std::vector<int> current_gpus;
    for (const auto& status : device_pool_->getDeviceStatus()) {
        if (status.type == DeviceType::GPU && status.available) {
            current_gpus.push_back(status.physical_device_id);
        }
    }
    
    std::set<int> enabled_set(enabled_gpus.begin(), enabled_gpus.end());
    std::set<int> current_set(current_gpus.begin(), current_gpus.end());
    
    // Devices to remove: in current but not in enabled
    std::vector<int> devices_to_remove;
    std::set_difference(current_set.begin(), current_set.end(),
                       enabled_set.begin(), enabled_set.end(),
                       std::back_inserter(devices_to_remove));
    
    // Devices to add: in enabled but not in current
    std::vector<int> devices_to_add;
    std::set_difference(enabled_set.begin(), enabled_set.end(),
                       current_set.begin(), current_set.end(),
                       std::back_inserter(devices_to_add));
    
    std::cout << "MinimizerOrchestrator: Current GPUs: ";
    for (int id : current_gpus) std::cout << id << " ";
    std::cout << std::endl;
    
    std::cout << "MinimizerOrchestrator: Enabled GPUs: ";
    for (int id : enabled_gpus) std::cout << id << " ";
    std::cout << std::endl;
    
    std::cout << "MinimizerOrchestrator: Devices to remove: " << devices_to_remove.size();
    if (!devices_to_remove.empty()) {
        std::cout << " (";
        for (size_t i = 0; i < devices_to_remove.size(); ++i) {
            std::cout << devices_to_remove[i];
            if (i < devices_to_remove.size() - 1) std::cout << ", ";
        }
        std::cout << ")";
    }
    std::cout << std::endl;
    
    std::cout << "MinimizerOrchestrator: Devices to add: " << devices_to_add.size();
    if (!devices_to_add.empty()) {
        std::cout << " (";
        for (size_t i = 0; i < devices_to_add.size(); ++i) {
            std::cout << devices_to_add[i];
            if (i < devices_to_add.size() - 1) std::cout << ", ";
        }
        std::cout << ")";
    }
    std::cout << std::endl;
    
    // ========================================================================
    // STEP 2: Mark devices for removal (prevents new acquisitions)
    // ========================================================================
    
    for (int device_id : devices_to_remove) {
        std::cout << "MinimizerOrchestrator: Marking device " << device_id << " for removal" << std::endl;
        device_pool_->markDeviceForRemoval(device_id);
    }
    
    // ========================================================================
    // STEP 3: Gracefully remove workers using marked devices
    // ========================================================================
    
    int total_workers_removed = 0;
    for (int device_id : devices_to_remove) {
        auto worker_ids = worker_pool_->getWorkerIDsUsingDevice(device_id);
        std::cout << "MinimizerOrchestrator: Device " << device_id 
                  << " has " << worker_ids.size() << " worker(s) assigned" << std::endl;
        
        std::cout << "MinimizerOrchestrator: Removing workers using device " << device_id << std::endl;
        int removed = worker_pool_->removeWorkersUsingDevice(device_id);
        total_workers_removed += removed;
        std::cout << "MinimizerOrchestrator: Removed " << removed << " worker(s) from device " << device_id << std::endl;
    }
    
    std::cout << "MinimizerOrchestrator: Total workers removed: " << total_workers_removed << std::endl;
    
    // ========================================================================
    // STEP 4: Clean up devices (now safe, ref_count == 0)
    // ========================================================================
    
    std::cout << "MinimizerOrchestrator: Cleaning up marked devices..." << std::endl;
    device_pool_->cleanupMarkedDevices();
    
    // ========================================================================
    // STEP 5: Add new devices
    // ========================================================================
    
    for (int device_id : devices_to_add) {
        std::cout << "MinimizerOrchestrator: Adding device " << device_id << std::endl;
        try {
            device_pool_->addGPU(device_id);
            std::cout << "MinimizerOrchestrator: Successfully added device " << device_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "MinimizerOrchestrator: Failed to add device " << device_id 
                      << ": " << e.what() << std::endl;
        }
    }
    
    // ========================================================================
    // STEP 6: Add workers for new devices
    // ========================================================================
    
    int workers_to_add = devices_to_add.size();
    if (workers_to_add > 0) {
        std::cout << "MinimizerOrchestrator: Adding " << workers_to_add << " worker(s) for new devices" << std::endl;
        try {
            worker_pool_->addWorkers(workers_to_add);
            std::cout << "MinimizerOrchestrator: Successfully added " << workers_to_add << " worker(s)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "MinimizerOrchestrator: Failed to add workers: " << e.what() << std::endl;
        }
    }
    
    // ========================================================================
    // Final Status
    // ========================================================================
    
    std::cout << "MinimizerOrchestrator: ===== Resource Adjustment Complete =====" << std::endl;
    std::cout << "MinimizerOrchestrator: Total devices: " << device_pool_->numDevices() << std::endl;
    std::cout << "MinimizerOrchestrator: Total workers: " << worker_pool_->getWorkerCount() << std::endl;
    std::cout << "MinimizerOrchestrator: Active workers: " << worker_pool_->getActiveWorkerCount() << std::endl;
}

} // namespace entropy
