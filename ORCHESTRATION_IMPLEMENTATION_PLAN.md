# Orchestration Implementation Plan
## Work-Stealing Architecture with Dynamic Resource Scaling

---

## Overview

Implement a work-stealing orchestration system that:
1. Manages multiple GPU/CPU devices dynamically
2. Distributes minimization runs across devices efficiently  
3. Scales resources up/down based on external configuration
4. Provides thread-safe task queue and result collection
5. Handles failures gracefully without crashing entire system

---

## File Structure

```
include/minimizer/orchestration/
├── device_pool.h              # Manages collection of compute devices
├── concurrent_queue.h         # Thread-safe task queue with timeout
├── result_collector.h         # Thread-safe result aggregation
├── resource_monitor.h         # Polls external config for resource changes
├── worker_thread_pool.h       # Manages elastic worker threads
├── run_orchestrator.h         # Executes single minimization run
└── minimizer_orchestrator.h   # Top-level: coordinates everything

src/minimizer/orchestration/
├── CMakeLists.txt
├── device_pool.cpp
├── result_collector.cpp
├── resource_monitor.cpp
├── worker_thread_pool.cpp
├── run_orchestrator.cpp
└── minimizer_orchestrator.cpp

src/main/tests/orchestration/
├── test_concurrent_queue.cpp
├── test_device_pool.cpp
├── test_result_collector.cpp
├── test_resource_monitor.cpp
├── test_worker_pool.cpp
├── test_run_orchestrator.cpp
├── test_minimizer_orchestrator.cpp
└── test_integration_e2e.cpp
```

---

## Implementation Strategy (Bottom-Up)

### **Phase 1: Foundation Components** (No device dependencies)

These can be tested in isolation without CUDA/compute devices.

#### **Step 1.1: ConcurrentQueue (Header-only)**
**File:** `include/minimizer/orchestration/concurrent_queue.h`

**Responsibilities:**
- Thread-safe FIFO queue
- Blocking pop with timeout (for checking shutdown signals)
- Non-blocking tryPop()
- Signal completion to wake waiting threads

**Key Types:**
```cpp
template<typename T>
class ConcurrentQueue {
public:
    void push(T item);
    std::optional<T> tryPop();
    std::optional<T> tryPopFor(std::chrono::milliseconds timeout);
    void signalDone();
    size_t size() const;
    bool empty() const;
};
```

**Testing:** `test_concurrent_queue.cpp`
- Single producer, single consumer
- Multiple producers, multiple consumers
- Timeout behavior (pop should return after timeout)
- signalDone() wakes all waiting threads
- Thread safety stress test (1000 pushes from 4 threads)

---

#### **Step 1.2: ResultCollector**
**File:** `include/minimizer/orchestration/result_collector.h`

**Responsibilities:**
- Thread-safe storage of run results
- Find minimum entropy result
- Track errors per run
- Provide statistics (mean, std, success rate)

**Key Types:**
```cpp
struct RunResult {
    int run_id;
    double final_entropy;
    HostVector final_vector;
    int iterations_taken;
    double runtime_seconds;
    std::string error_message;  // Empty if successful
};

class ResultCollector {
public:
    void addResult(int run_id, const RunResult& result);
    void addError(int run_id, const std::string& error);
    RunResult getMinimum() const;
    std::vector<RunResult> getAllResults() const;
    size_t numCompleted() const;
    size_t numErrors() const;
};
```

**Testing:** `test_result_collector.cpp`
- Add results from multiple threads
- getMinimum() returns correct result
- Error tracking works
- Statistics computation (mean entropy, std dev)
- Thread safety: 100 threads adding results concurrently

---

#### **Step 1.3: ResourceMonitor**
**File:** `include/minimizer/orchestration/resource_monitor.h`

**Responsibilities:**
- Poll external YAML file for resource limit changes
- Detect changes and trigger callbacks
- Support file-based and environment variable sources
- Graceful start/stop

**Key Types:**
```cpp
struct ResourceLimits {
    int max_gpus;
    int max_cpus;
    std::chrono::milliseconds poll_interval{5000};
    std::string config_file;
};

class ResourceMonitor {
public:
    using ChangeCallback = std::function<void(const ResourceLimits&)>;
    
    explicit ResourceMonitor(const ResourceLimits& initial);
    void start(ChangeCallback on_change);
    void stop();
    ResourceLimits getCurrentLimits() const;
    
private:
    ResourceLimits pollConfig();
    bool hasChanged(const ResourceLimits& a, const ResourceLimits& b) const;
};
```

**Testing:** `test_resource_monitor.cpp`
- Poll file, detect changes (write to temp file, update it, verify callback)
- Environment variable source (set env var, verify read)
- No callback on identical config
- Callback triggered on change
- Graceful shutdown (stop() while polling)
- Multiple rapid changes handled correctly

---

### **Phase 2: Device Management**

#### **Step 2.1: DevicePool (Static)**
**File:** `include/minimizer/orchestration/device_pool.h`

**Responsibilities:**
- Initialize GPU/CPU devices based on config
- Assign one device per worker (round-robin)
- Track device usage

**Key Types:**
```cpp
enum class DeviceType { GPU, CPU };

struct DeviceInfo {
    int device_id;       // CUDA device ID, or -1 for CPU
    DeviceType type;
    bool in_use;
};

class DevicePool {
public:
    explicit DevicePool(const ResourceConfig& config);
    ~DevicePool();
    
    // Get device for worker (called once per worker thread)
    IComputeDevice& getDeviceForWorker(int worker_id);
    
    // Query
    size_t numDevices() const;
    std::vector<DeviceInfo> getDeviceInfo() const;
    
private:
    void initializeGPUs(int count);
    void initializeCPUs(int count);
    
    struct DeviceSlot {
        std::unique_ptr<IComputeDevice> device;
        DeviceInfo info;
    };
    
    std::vector<DeviceSlot> devices_;
    mutable std::mutex mutex_;
};
```

**Testing:** `test_device_pool.cpp`
- Initialize with config (2 GPUs, 1 CPU)
- getDeviceForWorker() returns different devices (round-robin)
- Device count matches config
- CPU-only mode (max_gpus=0, max_cpus=4)
- GPU-only mode (max_gpus=2, max_cpus=0)
- Invalid config (max_gpus=-2) throws exception

---

#### **Step 2.2: DevicePool (Dynamic Resizing)**
**File:** Extend `device_pool.h`

**Add Methods:**
```cpp
class DevicePool {
public:
    // Dynamic resizing
    void resize(int new_max_gpus, int new_max_cpus);
    bool shouldShutdown(int worker_id) const;
    void markDeviceForRemoval(int worker_id);
    void finalizeRemovals();  // Called after workers exit
    
    // Observability
    size_t activeDeviceCount() const;
    
private:
    void addGPUs(int count);
    void removeGPUs(int count);
    void addCPUs(int count);
    void removeCPUs(int count);
    
    std::atomic<int> pending_removals_{0};
};
```

**Testing:** `test_device_pool.cpp` (extend)
- Scale up: 2 GPUs → 4 GPUs (devices added)
- Scale down: 4 GPUs → 2 GPUs (devices marked for removal)
- shouldShutdown() returns true for removed worker IDs
- finalizeRemovals() cleans up after workers exit
- Concurrent resize and worker access (thread safety)

---

### **Phase 3: Worker Management**

#### **Step 3.1: RunOrchestrator (Single Run Execution)**
**File:** `include/minimizer/orchestration/run_orchestrator.h`

**Responsibilities:**
- Execute single minimization run from start to finish
- Own/manage: ConditionsChecker, EntropyPredictor, CheckpointManager
- Use (not own): IComputeDevice, AlgorithmManager
- Report progress via callbacks
- Handle exceptions gracefully

**Key Types:**
```cpp
struct RunTask {
    int run_id;
    int config_id;          // Index into shared config array
    HostVector initial_vector;
    int priority = 0;       // For future work prioritization
};

class RunOrchestrator {
public:
    using ProgressCallback = std::function<void(int run_id, int iteration, double entropy)>;
    
    RunOrchestrator(
        const MinimizerConfig& config,
        IComputeDevice& device,
        ProgressCallback progress_cb = nullptr
    );
    
    RunResult execute(const HostVector& initial_vector);
    
private:
    void mainLoop();
    bool checkStoppingConditions();
    void handleCheckpoint(int iteration);
    
    const MinimizerConfig& config_;
    IComputeDevice& device_;
    
    // Components (owned)
    std::unique_ptr<AlgorithmManager> algorithm_mgr_;
    std::unique_ptr<ConditionsChecker> conditions_;
    std::unique_ptr<EntropyPredictor> predictor_;
    std::unique_ptr<CheckpointManager> checkpoint_mgr_;
    
    ProgressCallback progress_cb_;
};
```

**Testing:** `test_run_orchestrator.cpp`
- Execute simple run (CPU device, 100 iters)
    - Execute simple run with more complex kraus operators (random unitary matrices), check entropy decreases, 1000 iters
- Stopping condition: max iterations reached
- Stopping condition: convergence detected
- Progress callback invoked correctly
- Exception in iteration → error result returned
- Checkpoint saving works (if enabled)

---

#### **Step 3.2: WorkerThreadPool**
**File:** `include/minimizer/orchestration/worker_thread_pool.h`

**Responsibilities:**
- Manage worker threads (one per device)
- Pull tasks from queue, execute via RunOrchestrator
- Collect results, handle errors
- Support dynamic resize (add/remove workers)
- Graceful shutdown

**Key Types:**
```cpp
class WorkerThreadPool {
public:
    WorkerThreadPool(
        DevicePool& device_pool,
        ConcurrentQueue<RunTask>& work_queue,
        ResultCollector& result_collector,
        const std::vector<MinimizerConfig>& configs
    );
    
    void start();
    void resize(size_t new_size);
    void stop();
    void waitAll();
    
    size_t activeWorkerCount() const;
    
private:
    void workerLoop(int worker_id);
    
    DevicePool& device_pool_;
    ConcurrentQueue<RunTask>& work_queue_;
    ResultCollector& result_collector_;
    const std::vector<MinimizerConfig>& configs_;
    
    std::vector<std::thread> workers_;
    std::atomic<bool> shutdown_requested_{false};
    mutable std::mutex mutex_;
};
```

**Testing:** `test_worker_pool.cpp`
- Start pool with 4 workers
- Workers process all tasks from queue
- Results collected correctly
- Graceful shutdown (stop() waits for current tasks)
- Dynamic resize: 4 → 8 workers (tasks processed faster)
- Dynamic resize: 8 → 2 workers (workers exit gracefully)
- Worker exception handling (one worker fails, others continue)

---

### **Phase 4: Top-Level Orchestration**

#### **Step 4.1: MinimizerOrchestrator (Static Resources)**
**File:** `include/minimizer/orchestration/minimizer_orchestrator.h`

**Responsibilities:**
- Top-level API: `findMOE(config)`
- Create and coordinate all components
- Manage component lifecycle
- Return minimum entropy result

**Key Types:**
```cpp
class MinimizerOrchestrator {
public:
    MinimizerOrchestrator() = default;
    
    RunResult findMOE(const MinimizerConfig& config);
    
    // Observability
    size_t getNumDevices() const;
    size_t getActiveWorkers() const;
    size_t getCompletedRuns() const;
    
private:
    HostVector generateInitialVector(const MinimizerConfig& config);
    void fillWorkQueue(const MinimizerConfig& config);
};
```

**Implementation:**
```cpp
RunResult MinimizerOrchestrator::findMOE(const MinimizerConfig& config) {
    // 1. Initialize components
    DevicePool device_pool(config.resources);
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    std::vector<MinimizerConfig> configs = {config};  // Shared config
    
    // 2. Fill work queue
    for (int i = 0; i < config.multi_run.num_attempts; ++i) {
        work_queue.push(RunTask{
            .run_id = i,
            .config_id = 0,
            .initial_vector = generateInitialVector(config)
        });
    }
    
    // 3. Start workers
    WorkerThreadPool worker_pool(device_pool, work_queue, result_collector, configs);
    worker_pool.start();
    
    // 4. Wait for completion
    worker_pool.waitAll();
    work_queue.signalDone();
    
    // 5. Return minimum
    return result_collector.getMinimum();
}
```

**Testing:** `test_minimizer_orchestrator.cpp`
- Execute findMOE() with 10 runs on 2 CPUs
- All runs complete successfully
- Correct minimum returned
- Progress tracked correctly
- Empty config (0 attempts) handled gracefully

---

#### **Step 4.2: MinimizerOrchestrator (Dynamic Resources)**
**File:** Extend `minimizer_orchestrator.h`

**Add:**
```cpp
class MinimizerOrchestrator {
public:
    RunResult findMOEWithDynamicScaling(const MinimizerConfig& config);
    
private:
    void setupResourceMonitoring(
        ResourceMonitor& monitor,
        DevicePool& device_pool,
        WorkerThreadPool& worker_pool
    );
};
```

**Implementation:**
```cpp
RunResult MinimizerOrchestrator::findMOEWithDynamicScaling(
    const MinimizerConfig& config
) {
    // Setup components (same as static)
    DevicePool device_pool(config.resources);
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    WorkerThreadPool worker_pool(/* ... */);
    
    // Setup resource monitoring
    ResourceMonitor::ResourceLimits monitor_config{
        .max_gpus = config.resources.max_gpus,
        .max_cpus = config.resources.max_cpus,
        .poll_interval = config.resources.poll_interval,
        .config_file = config.resources.config_file
    };
    
    ResourceMonitor monitor(monitor_config);
    monitor.start([&](const auto& new_limits) {
        device_pool.resize(new_limits.max_gpus, new_limits.max_cpus);
        worker_pool.resize(new_limits.max_gpus + new_limits.max_cpus);
    });
    
    // Fill queue and execute (same as static)
    fillWorkQueue(config);
    worker_pool.start();
    worker_pool.waitAll();
    
    // Cleanup
    monitor.stop();
    
    return result_collector.getMinimum();
}
```

**Testing:** `test_minimizer_orchestrator.cpp` (extend)
- Dynamic scaling: config changes mid-execution
- Scale up: 2 → 4 GPUs during execution
- Scale down: 4 → 2 GPUs during execution
- Config file updated multiple times
- Workers adapt without dropping tasks

---

### **Phase 5: Integration & End-to-End**

#### **Step 5.1: Integration Test**
**File:** `test_integration_e2e.cpp`

**Scenarios:**
1. **Basic workflow:** 100 runs, 2 GPUs, static config
2. **CPU-only:** 50 runs, 4 CPU workers
3. **Mixed devices:** 100 runs, 2 GPUs + 2 CPUs
4. **Dynamic scaling:** Start with 1 GPU, scale to 4 mid-execution
5. **Failure handling:** Inject errors in 10% of runs, verify others complete
6. **Stress test:** 10,000 runs, ensure no deadlocks/crashes
7. **Config from file:** Load config from YAML, verify all settings applied

**Assertions:**
- All runs complete (or fail gracefully)
- Minimum entropy < initial entropy
- No memory leaks (use valgrind)
- No data races (use thread sanitizer)
- Performance: >N runs/second per device

---

## Configuration Updates

### **Add to ResourceConfig:**

```cpp
struct ResourceConfig {
    // Existing fields
    int max_gpus = -1;
    int max_cpus = 1;
    
    // NEW: Dynamic scaling
    bool enable_dynamic_scaling = false;
    std::chrono::milliseconds poll_interval{5000};
    std::string config_file = "";
    
    enum class ConfigSource {
        NONE,           // Static, no monitoring
        FILE,           // YAML file
        ENVIRONMENT     // Environment variables
    };
    ConfigSource source = ConfigSource::NONE;
    
    // NEW: Worker behavior
    bool allow_worker_preemption = false;  // Can stop mid-task?
    std::chrono::seconds worker_shutdown_timeout{30};
    int min_devices = 1;  // Never go below this
    
    // NEW: Task queue
    size_t max_queue_size = 10000;
    
    void validate() const {
        if (max_gpus < -1) {
            throw std::invalid_argument("max_gpus must be >= -1");
        }
        if (max_cpus < 0) {
            throw std::invalid_argument("max_cpus must be >= 0");
        }
        if (min_devices < 1) {
            throw std::invalid_argument("min_devices must be >= 1");
        }
    }
};
```

### **YAML example:**

```yaml
minimizer:
  resources:
    max_gpus: 2
    max_cpus: 4
    enable_dynamic_scaling: true
    poll_interval: 5000
    config_file: "/tmp/minimizer_resources.yml"
    source: FILE
    allow_worker_preemption: false
    worker_shutdown_timeout: 30
    min_devices: 1
    max_queue_size: 10000
```

---

## Testing Strategy Summary

### **Unit Tests (8 files, ~40 tests total)**

| File | Tests | Focus |
|------|-------|-------|
| `test_concurrent_queue.cpp` | 6 | Thread safety, timeout, signaling |
| `test_result_collector.cpp` | 5 | Concurrent adds, min finding, stats |
| `test_resource_monitor.cpp` | 6 | File polling, env vars, callbacks |
| `test_device_pool.cpp` | 8 | Init, resize, round-robin, cleanup |
| `test_run_orchestrator.cpp` | 6 | Single run, stopping, checkpoints |
| `test_worker_pool.cpp` | 7 | Workers, resize, shutdown, errors |
| `test_minimizer_orchestrator.cpp` | 5 | End-to-end static/dynamic |
| `test_integration_e2e.cpp` | 7 | Full workflows, stress, perf |

### **Test Infrastructure Needs:**

1. **Mock devices:** CPU-only tests (no CUDA required)
2. **Temp files:** ResourceMonitor file tests
3. **Thread sanitizer:** Detect data races
4. **Valgrind:** Detect memory leaks
5. **Fixtures:** Reusable test configs, device pools

---

## Implementation Order (Step-by-Step)

### **Iteration 1: Foundation (No CUDA)**
1. ✅ ConcurrentQueue (header-only)
   - Write header
   - Write tests
   - Verify thread safety

2. ✅ ResultCollector
   - Write header + impl
   - Write tests
   - Verify thread safety

3. ✅ ResourceMonitor
   - Write header + impl
   - Write file polling logic
   - Write tests (create temp files)

**Milestone 1:** All foundation tests pass, no CUDA dependency

---

### **Iteration 2: Device Management**
4. ✅ DevicePool (static)
   - Write header + impl
   - Initialize GPU/CPU devices
   - Write tests (CPU-only mode for CI)

5. ✅ DevicePool (dynamic resize)
   - Add resize() methods
   - Write resize tests
   - Verify graceful shutdown

**Milestone 2:** DevicePool can add/remove devices dynamically

---

### **Iteration 3: Worker Execution**
6. ✅ RunOrchestrator
   - Write header + impl
   - Integrate AlgorithmManager, conditions, predictor
   - Write tests (simple CPU runs)

7. ✅ WorkerThreadPool
   - Write header + impl
   - Workers pull from queue
   - Write tests (mock RunOrchestrator)

**Milestone 3:** Workers can execute runs and collect results

---

### **Iteration 4: Top-Level Orchestration**
8. ✅ MinimizerOrchestrator (static)
   - Write findMOE()
   - Integrate all components
   - Write basic tests

9. ✅ MinimizerOrchestrator (dynamic)
   - Add findMOEWithDynamicScaling()
   - Connect ResourceMonitor
   - Write dynamic scaling tests

**Milestone 4:** Full orchestration working with dynamic scaling

---

### **Iteration 5: Polish & Testing**
10. ✅ Integration tests
    - E2E scenarios
    - Stress tests
    - Performance benchmarks

11. ✅ Documentation
    - Usage examples
    - Architecture diagrams
    - Performance tuning guide

**Milestone 5:** Production-ready, fully tested

---

## Build System Updates

### **CMakeLists.txt:**

```cmake
# src/minimizer/orchestration/CMakeLists.txt

add_library(minimizer_orchestration STATIC
    device_pool.cpp
    result_collector.cpp
    resource_monitor.cpp
    worker_thread_pool.cpp
    run_orchestrator.cpp
    minimizer_orchestrator.cpp
)

target_include_directories(minimizer_orchestration PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(minimizer_orchestration PUBLIC
    minimizer_config
    minimizer_algorithm
    minimizer_state
    minimizer_stopping
    minimizer_prediction
    minimizer_management
    compute_device
    yaml-cpp
)

target_compile_features(minimizer_orchestration PUBLIC cxx_std_17)

# Enable thread sanitizer for tests
if(ENABLE_TSAN)
    target_compile_options(minimizer_orchestration PRIVATE -fsanitize=thread)
    target_link_options(minimizer_orchestration PRIVATE -fsanitize=thread)
endif()
```

### **Test CMakeLists.txt:**

```cmake
# src/main/tests/orchestration/CMakeLists.txt

set(ORCHESTRATION_TESTS
    test_concurrent_queue.cpp
    test_result_collector.cpp
    test_resource_monitor.cpp
    test_device_pool.cpp
    test_run_orchestrator.cpp
    test_worker_pool.cpp
    test_minimizer_orchestrator.cpp
    test_integration_e2e.cpp
)

foreach(TEST_FILE ${ORCHESTRATION_TESTS})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    
    add_executable(${TEST_NAME} ${TEST_FILE})
    
    target_link_libraries(${TEST_NAME} PRIVATE
        minimizer_orchestration
        GTest::gtest
        GTest::gtest_main
    )
    
    gtest_discover_tests(${TEST_NAME})
endforeach()
```

---

## Key Design Principles

### **1. Separation of Concerns**
- **DevicePool:** Owns devices, manages lifecycle
- **WorkerThreadPool:** Owns threads, manages workers
- **RunOrchestrator:** Executes single run (stateless)
- **MinimizerOrchestrator:** Coordinates components

### **2. Thread Safety**
- Use `std::mutex` for complex state
- Use `std::atomic` for simple counters
- Use `std::shared_mutex` for read-heavy data
- Lock-free queue for performance (if needed)

### **3. Testability**
- Inject dependencies (DevicePool, Queue, Collector)
- Provide mock implementations
- CPU-only tests for CI
- Separate unit/integration tests

### **4. Graceful Degradation**
- Worker failure doesn't crash system
- Dynamic scale-down waits for tasks to finish
- Resource monitor handles file errors

### **5. Observability**
- Metrics for active workers, completed tasks
- Progress callbacks
- Error collection
- Logging at key points

---

## Error Handling Strategy

### **Worker-Level Errors:**
```cpp
try {
    auto result = run.execute(task.initial_vector);
    result_collector.addResult(task.run_id, result);
} catch (const std::exception& e) {
    result_collector.addError(task.run_id, e.what());
    // Continue with next task
}
```

### **Device-Level Errors:**
```cpp
try {
    device_pool.getDeviceForWorker(worker_id);
} catch (const CudaError& e) {
    // Log error, mark device as failed, continue with CPUs
    logger.error("GPU " + std::to_string(worker_id) + " failed: " + e.what());
    // Worker exits, pool adapts
}
```

### **System-Level Errors:**
```cpp
try {
    return orchestrator.findMOE(config);
} catch (const std::exception& e) {
    // Unrecoverable error (no devices available, etc.)
    logger.error("Orchestration failed: " + e.what());
    throw;
}
```

---

## Performance Considerations

### **Queue Contention:**
- Start with mutex-based queue
- Profile: if >5% time in queue ops, switch to lock-free
- Use `boost::lockfree::queue` or `moodycamel::ConcurrentQueue`

### **Memory Allocation:**
- Preallocate vectors where possible
- Use object pools for frequent allocations
- Move semantics for large objects (vectors)

### **CUDA Context Switching:**
- One device per thread (no context switching)
- Pin threads to CPU cores (affinity)
- Batch CUDA operations where possible

### **Load Balancing:**
- Work-stealing naturally balances heterogeneous devices
- Fast devices pull more tasks
- No manual partitioning needed

---

## Success Criteria

### **Functionality:**
- ✅ Execute 1000 runs across 4 GPUs + 4 CPUs
- ✅ Dynamic scale from 1 GPU → 4 GPUs without dropping tasks
- ✅ Handle 10% failure rate gracefully
- ✅ Return correct minimum entropy

### **Performance:**
- ✅ >90% device utilization (workers not idle)
- ✅ <1% overhead from queue/synchronization
- ✅ Linear scaling up to 8 devices

### **Reliability:**
- ✅ No deadlocks (stress test 10,000 runs)
- ✅ No memory leaks (valgrind clean)
- ✅ No data races (thread sanitizer clean)
- ✅ Graceful shutdown (<5s)

### **Maintainability:**
- ✅ All components unit tested
- ✅ Clear separation of concerns
- ✅ Documented interfaces
- ✅ Easy to mock for testing

---

## Open Questions / Future Enhancements

1. **Task prioritization:** Should some runs have higher priority?
2. **Adaptive batch sizing:** Should we batch small tasks?
3. **Fault tolerance:** Should we retry failed runs automatically?
4. **Distributed execution:** Should we support multi-node?
5. **GPU sharing:** Should multiple workers share one GPU (via streams)?

These can be addressed in future iterations after core functionality is stable.

---

## Next Steps for Agent

Start with **Iteration 1: Foundation** and implement in this order:

1. **ConcurrentQueue** (easiest, header-only, no dependencies)
2. **ResultCollector** (simple, no threading complexity in tests)
3. **ResourceMonitor** (file I/O, testable with temp files)

Each step should:
- Create header file
- Create implementation (if not header-only)
- Create comprehensive tests
- Verify all tests pass
- Commit before moving to next step

This ensures incremental progress with continuous validation.
