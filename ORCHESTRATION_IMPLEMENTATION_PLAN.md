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

### **Phase 2: Device Management** ✅ COMPLETE (36/36 tests passing)

#### **Step 2.1: DevicePool (Static)** ✅ COMPLETE
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

#### **Step 2.2: DevicePool (Dynamic Resizing)** ✅ COMPLETE
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

#### **Step 3.1: StrategyFactory & RunOrchestrator Integration** ✅ COMPLETED

##### **Step 3.1.1-3.1.4: StrategyFactory Implementation** ✅ COMPLETE

**Status:** Fully implemented, tested, and integrated into RunOrchestrator.

**Accomplishments:**
- ✅ Memory-aware factory pattern with AUTO/GENERIC/CUDA modes
- ✅ Workspace size estimation with scaling analysis
- ✅ GPU memory querying (cudaMemGetInfo)
- ✅ Fallback logic: AUTO → CUDA when memory fits, else → GenericMinimizationStrategy
- ✅ RunOrchestrator integrated with StrategyFactory::create()
- ✅ 15 comprehensive test cases all passing (100%)

**Solution:** Memory-aware factory pattern with AUTO/GENERIC/CUDA modes.

---

**Step 3.1.1: Create StrategyFactory Interface**

**File:** `include/minimizer/algorithm/strategy_factory.h`

**Key Components:**
```cpp
enum class StrategyType {
    AUTO,     // Auto-select based on device backend + memory
    GENERIC,  // Force GenericMinimizationStrategy
    CUDA      // Force CudaMinimizationStrategy (CUDA only)
};

class StrategyFactory {
public:
    // Create with AUTO selection
    static std::unique_ptr<IMinimizationStrategy> create(
        IComputeDevice& device,
        int kraus_count, int input_dim, int output_dim,
        PrecisionType precision, int num_streams = 4
    );
    
    // Create with explicit type
    static std::unique_ptr<IMinimizationStrategy> create(
        StrategyType type, IComputeDevice& device,
        int kraus_count, int input_dim, int output_dim,
        PrecisionType precision, int num_streams = 4
    );
    
    // Memory estimation (before allocation)
    static size_t estimateWorkspaceSize(
        int kraus_count, int input_dim, int output_dim,
        PrecisionType precision
    );
    
    // Check available device memory
    static size_t getAvailableMemory(IComputeDevice& device);
};
```

**Memory Estimation Formula:**
```
complex_size = sizeof(std::complex<T>)  // 8 (float) or 16 (double) bytes
real_size = sizeof(T)                    // 4 (float) or 8 (double) bytes

vecs1 = d × M × complex_size             // Step 1: {K_i|ψ⟩}
sing1 = d × M × complex_size             // Step 1: SVD U vectors
vecs2 = d² × N × complex_size            // Step 2: {K_j^H|φ_i⟩}
sing2 = N × complex_size                 // Step 2: SVD U (rank 1)
sv1 = d × real_size                      // Step 1: singular values
sv2 = d² × real_size                     // Step 2: singular values
svd1_workspace ≈ M × d × complex_size    // cuSOLVER (QR algorithm)
svd2_workspace ≈ N × d² × complex_size   // cuSOLVER (randomized)

Total ≈ 3×d×M×c + 2×d²×N×c + N×c + (d+d²)×r
```

**AUTO Selection Logic:**
1. Check device backend (CUDA vs CPU)
2. If CUDA:
   - Estimate workspace size
   - Query GPU free memory via `cudaMemGetInfo()`
   - If `free_memory ≥ 1.3 × workspace`: CudaMinimizationStrategy
   - Else: GenericMinimizationStrategy (fallback)
3. If CPU: GenericMinimizationStrategy

**Safety Margins:**
- AUTO mode: 30% buffer (1.3x)
- Explicit CUDA: 20% buffer (1.2x)
- GENERIC: No check (fallback, less memory efficient)

---

**Files:**
- `include/minimizer/algorithm/strategy_factory.h` - 209 lines, fully implemented
- `src/minimizer/algorithm/strategy_factory.cpp` - Implementation complete
- `src/main/tests/test_strategy_factory.cpp` - 15 comprehensive test cases

**Implementation Details:**
- ✅ `estimateWorkspaceSize()` - Accurate d² scaling analysis
- ✅ `getAvailableMemory()` - GPU memory querying via cudaMemGetInfo
- ✅ `create(StrategyType, device, ...)` - Type-based factory with validation
- ✅ `create(device, ...)` - DEFAULT AUTO selection for convenience
- ✅ AUTO fallback logic - CUDA→Generic on insufficient memory
- ✅ Explicit CUDA validation - 20% buffer for safety
- ✅ Error messages - Clear distinction between validation failures

**Test Results (15/15 passing):**
1. ✅ `estimateWorkspaceSize_float_vs_double` - 2x memory ratio verified
2. ✅ `estimateWorkspaceSize_scaling` - d² scaling confirmed
3. ✅ `estimateWorkspaceSize_dimensions_positive` - Dimension validation
4. ✅ `getAvailableMemory_cpu` - Returns SIZE_MAX as expected
5. ✅ `getAvailableMemory_cuda` - GPU query functional
6. ✅ `create_auto_cpu_returns_generic` - CPU defaults to Generic
7. ✅ `create_auto_cuda_sufficient` - AUTO→CUDA when memory fits (30% buffer)
8. ✅ `create_auto_cuda_insufficient` - AUTO→Generic fallback works
9. ✅ `create_explicit_generic_cpu` - Force Generic on CPU
10. ✅ `create_explicit_generic_cuda` - Force Generic on CUDA
11. ✅ `create_explicit_cuda_on_cpu_throws` - Proper error handling
12. ✅ `create_explicit_cuda_sufficient` - CUDA with memory validation
13. ✅ `create_explicit_cuda_insufficient` - CUDA OOM detection (20% buffer)
14. ✅ `create_invalid_dimensions` - Negative dimension rejection
15. ✅ `created_strategy_initializes_correctly` - Integration verification

**RunOrchestrator Integration:**
- ✅ RunOrchestrator now uses `StrategyFactory::create()` instead of hard-coded GenericMinimizationStrategy
- ✅ AUTO selection enables GPU acceleration on CUDA systems
- ✅ Fallback to CPU when GPU memory insufficient
- ✅ All 14 RunOrchestrator tests still passing

---

##### **Step 3.1.5: Configuration Integration (Phase 2)** ⏳ FUTURE

**Goal:** Allow users to control strategy selection via YAML config.

**Changes:**

1. **Extend AlgorithmConfig** (`include/minimizer/config/algorithm_config.h`):
   ```cpp
   struct AlgorithmConfig {
       double epsilon = 1e-3;
       PrecisionType precision = PrecisionType::DOUBLE;
       int device_id = 0;
       StrategyType strategy = StrategyType::AUTO;  // NEW
   };
   ```

2. **Create YAML Parser** (`src/minimizer/config/algorithm_config_parser.cpp`):
   ```cpp
   StrategyType parseStrategyType(const std::string& str) {
       if (str == "AUTO") return StrategyType::AUTO;
       if (str == "GENERIC") return StrategyType::GENERIC;
       if (str == "CUDA") return StrategyType::CUDA;
       throw std::invalid_argument("Invalid strategy: " + str);
   }
   ```

3. **Update YAML Configs** (`configs/minimizer_default.yml`):
   ```yaml
   algorithm:
     epsilon: 1.0e-3
     precision: DOUBLE
     device_id: 0
     strategy: AUTO  # NEW: AUTO | GENERIC | CUDA
   ```

4. **Update RunOrchestrator** to use `config.algorithm.strategy`:
   ```cpp
   auto strategy = StrategyFactory::create(
       config.algorithm.strategy,  // Use config field
       device_, kraus_ops.kraus_count, /* ... */
   );
   ```

**Note:** Phase 2 deferred to future work. Phase 1 provides full functionality with AUTO as default.

---

##### **Step 3.1.6: CheckpointManager Implementation** ⏳ NEXT

**Goal:** Enable checkpoint save/restore functionality in RunOrchestrator.

**Current Status:** Stubbed out due to missing `json.hpp` dependency.

**Files:**
- `include/minimizer/orchestration/checkpoint_manager.h` - Interface exists
- `src/minimizer/orchestration/checkpoint_manager.cpp` - Stub implementation
- `src/main/tests/orchestration/test_checkpoint_manager.cpp` - Basic tests

**Tasks:**
1. Integrate nlohmann/json library (already in `src/external/nlohmann/`)
2. Implement `CheckpointManager::save()` - Serialize state to JSON file
3. Implement `CheckpointManager::load()` - Deserialize from JSON file
4. Add compression support (optional, via zlib)
5. Implement checkpoint cleanup (keep last N checkpoints)
6. Update RunOrchestrator to enable checkpointing
7. Add comprehensive tests (save/load, compression, cleanup)

**Design:**
```cpp
class CheckpointManager {
public:
    void save(int iteration, double entropy, const HostVector& vector);
    CheckpointData load(const std::string& filepath);
    void cleanup();  // Remove old checkpoints
    
private:
    CheckpointConfig config_;
    std::vector<std::string> saved_files_;  // Track for cleanup
};
```

**Checkpoint File Format (JSON):**
```json
{
  "run_id": "abc-123",
  "iteration": 1000,
  "entropy": 0.123456,
  "vector": {
    "dimension": 16,
    "data_real": [0.1, 0.2, ...],
    "data_imag": [0.0, 0.1, ...]
  },
  "timestamp": "2025-12-09T10:30:00Z"
}
```

**Testing:**
- Save checkpoint, verify file exists and format correct
- Load checkpoint, verify vector restored correctly
- Compression enabled/disabled
- Cleanup keeps last N files
- Invalid checkpoint file throws exception
- RunOrchestrator integration (save every N iterations)

---

##### **RunOrchestrator Summary** ✅ COMPLETED

**Files:** 
- `include/minimizer/orchestration/types.h` - Shared types
- `include/minimizer/orchestration/run_orchestrator.h`
- `src/minimizer/orchestration/run_orchestrator.cpp`
- `src/main/tests/orchestration/test_run_orchestrator.cpp`

**Status:** Fully implemented and tested (14/14 tests passing)

**Shared Types (`include/minimizer/orchestration/types.h`):**
```cpp
struct HostVector {
    std::vector<std::complex<double>> data;
    int dimension;
};

enum class RunErrorType {
    NONE, INVALID_INPUT, DEVICE_ERROR, CHECKPOINT_FAILURE,
    TIMEOUT, ALGORITHM_FAILURE, UNKNOWN
};

struct RunTask {
    int run_id;
    int config_id;          // Index into shared config array (future multi-config)
    HostVector initial_vector;
    int priority = 0;       // For future work prioritization
};
```

**RunOrchestrator Interface:**
```cpp
class RunOrchestrator {
public:
    // Thread-safe callback signature
    // Called from worker thread - implementation MUST be thread-safe
    // Protected by try-catch - exceptions won't crash runs
    using ProgressCallback = std::function<void(int run_id, int iteration, double entropy)>;
    
    RunOrchestrator(
        const MinimizerConfig& config,
        const HostKrausOperators& kraus_ops,  // Copied by AlgorithmManager
        int input_dim,
        IComputeDevice& device,
        ProgressCallback progress_cb = nullptr
    );
    
    RunResult execute(int run_id, const HostVector& initial_vector);
    
private:
    // Owned components
    std::unique_ptr<EntropyManager> entropy_manager_;
    std::unique_ptr<AlgorithmManager> algorithm_mgr_;
    std::unique_ptr<ConditionsChecker> conditions_;
    std::unique_ptr<EntropyPredictor> predictor_;
    // CheckpointManager temporarily disabled (requires json.hpp)
};
```

**Design Decisions:**

1. **HostVector Deduplication:** 
   - Previously defined in 3 places (result_collector.h, checkpoint_manager.h, run_orchestrator.h)
   - Now in single location: `include/minimizer/orchestration/types.h`
   - Eliminates forward declaration hacks and compilation conflicts

2. **Kraus Operators Ownership:**
   - Passed to constructor but NOT stored in RunOrchestrator
   - AlgorithmManager makes internal copy (avoids expensive duplication)
   - Documented clearly in constructor comments

3. **Error Handling:**
   - RunResult now includes `RunErrorType error_type` field for structured errors
   - Multiple catch blocks categorize exceptions (invalid_argument → INVALID_INPUT, etc.)
   - Device/CUDA errors detected via string matching in exception messages
   - All tests updated to check `error_type` field

4. **Progress Callback Thread Safety:**
   - Documented that callbacks run on worker thread
   - Implementation MUST be thread-safe (atomics, mutexes, lock-free)
   - Wrapped in try-catch to prevent callback exceptions from crashing runs
   - Callback errors silently logged (in production, use proper logging)

5. **Checkpoint System:**
   - Disabled due to missing json.hpp dependency
   - Stubbed out in handleCheckpoint() with clear comments
   - Can be re-enabled when serializer is integrated
   - Consider Null Object Pattern for cleaner code (future enhancement)

**Testing:** `test_run_orchestrator.cpp` - 14/14 passing
- Constructor validation (valid/invalid epsilon, with/without callback)
- Execute simple run (success with error_type = NONE)
- Max iterations limit reached
- Convergence detection
- Invalid input dimension (error_type = INVALID_INPUT)
- Progress callback invoked from worker thread
- Progress callback receives decreasing entropy
- Multiple sequential runs
- Different initial vectors
- Target entropy threshold
- Result validation (entropy, iterations, runtime)

**Known Limitations:**
- Checkpoint functionality disabled (requires json.hpp integration) - See Step 3.1.6
- Strategy selection hard-coded to AUTO (Phase 2 config integration pending) - See Step 3.1.5
- No timeout mechanism (TIMEOUT error type reserved for future)
- Error categorization uses string matching (fragile, but acceptable for now)

---

#### **Step 3.2: WorkerThreadPool** 🔴 NEXT - PRIORITY TASK

**Status:** Not yet started. This is a CRITICAL blocking component for Phase 4.

**Why Critical:**
- Enables parallel execution of multiple runs
- All foundation infrastructure is ready (DevicePool, ConcurrentQueue, ResultCollector)
- Required before MinimizerOrchestrator can function

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

### **Phase 4: Top-Level Orchestration** 🔴 BLOCKED (Waiting for WorkerThreadPool)

#### **Step 4.1: MinimizerOrchestrator (Static Resources)** 🔴 BLOCKED
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

#### **Step 4.2: MinimizerOrchestrator (Dynamic Resources)** 🔴 BLOCKED
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

## Recent Changes (December 9, 2025)

### **Refactoring: Shared Types and Error Handling**

**Motivation:** Eliminate code duplication, improve error handling, and prepare for WorkerThreadPool implementation.

**Changes Made:**

1. **Created `include/minimizer/orchestration/types.h`** - Single source of truth for shared types:
   - `HostVector` - Previously duplicated in 3 files, now canonical definition
   - `RunErrorType` enum - Structured error categorization (NONE, INVALID_INPUT, DEVICE_ERROR, CHECKPOINT_FAILURE, TIMEOUT, ALGORITHM_FAILURE, UNKNOWN)
   - `RunTask` struct - Task specification (moved from run_orchestrator.h for sharing with WorkerThreadPool)
   - `RunResult` struct - Execution outcome (moved from result_collector.h, now includes error_type field)

2. **Updated RunOrchestrator** (`include/minimizer/orchestration/run_orchestrator.h`):
   - Removed duplicate RunTask definition (now in types.h)
   - Enhanced ProgressCallback documentation:
     - Thread safety requirements (MUST be thread-safe, runs on worker thread)
     - Exception safety (protected by try-catch, won't crash runs)
     - Performance guidelines (non-blocking, fast, use atomics/lock-free)
   - Clarified Kraus operators ownership:
     - NOT stored in RunOrchestrator (avoid expensive duplication)
     - Copied by AlgorithmManager internally
     - Documented in constructor parameters

3. **Improved Error Handling** (`src/minimizer/orchestration/run_orchestrator.cpp`):
   - Added multiple catch blocks to categorize exceptions:
     - `std::invalid_argument` → RunErrorType::INVALID_INPUT
     - `std::runtime_error` → Device/checkpoint/algorithm errors (string matching)
     - `std::exception` → RunErrorType::UNKNOWN
     - Non-standard exceptions → RunErrorType::UNKNOWN
   - Progress callback wrapped in try-catch to prevent callback exceptions from crashing runs
   - buildResult() sets error_type = NONE for successful runs

4. **Updated Tests** (`src/main/tests/orchestration/test_run_orchestrator.cpp`):
   - All tests now check `error_type` field in addition to `error_message`
   - Invalid input test verifies error_type == INVALID_INPUT
   - Success tests verify error_type == NONE
   - All 14 tests passing

5. **Cleaned Up Result Collector** (`include/minimizer/orchestration/result_collector.h`):
   - Removed duplicate HostVector definition (uses types.h)
   - Removed duplicate RunResult definition (uses types.h)

6. **Cleaned Up Checkpoint Manager** (`include/minimizer/management/checkpoint_manager.h`):
   - Removed duplicate HostVector definition (uses types.h)
   - Now includes types.h for canonical definition

**Test Results:**
- RunOrchestrator: 14/14 tests passing ✅
- ConcurrentQueue: 14/14 tests passing ✅
- ResultCollector: 22/22 tests passing ✅
- ResourceMonitor: 19/19 tests passing ✅
- DevicePool: 35/36 tests passing (1 skipped - no CUDA) ✅

**Design Principles Followed:**
- **DRY (Don't Repeat Yourself):** Single canonical definition for shared types
- **RAII:** All resources properly owned and managed via smart pointers
- **Dependency Injection:** Device and config injected, not owned
- **Separation of Concerns:** Types separate from behavior, clear ownership model
- **Exception Safety:** All exceptions caught and categorized, no resource leaks
- **Thread Safety:** Progress callbacks documented as running on worker thread

**Known Limitations:**
- Checkpoint functionality disabled (requires json.hpp dependency - future work)
- Error categorization uses string matching (fragile but acceptable for now)
- No timeout mechanism (RunErrorType::TIMEOUT reserved for future)

**Next Steps:**
- Ready to implement **Phase 3 Step 3.2: WorkerThreadPool**
- All shared types (HostVector, RunTask, RunResult, RunErrorType) now available
- RunOrchestrator fully tested and ready to be used by worker threads
- Consider implementing Null Object Pattern for CheckpointManager (deferred to future work)

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

---

## PROJECT STATUS UPDATE (December 10, 2025)

### **Overall Progress: 73% Complete (11/15 major tasks)**

**Completed Components:**
- ✅ Phase 1: Foundation (ConcurrentQueue, ResultCollector, ResourceMonitor)
- ✅ Phase 2: Device Management (DevicePool static + dynamic resize)
- ✅ Phase 3.1: Strategy Factory (memory-aware selection + RunOrchestrator)

**Test Results: 120/120 passing (100%)**
- ConcurrentQueue: 14/14 ✅
- ResultCollector: 22/22 ✅
- ResourceMonitor: 19/19 ✅
- DevicePool: 36/36 ✅
- StrategyFactory: 15/15 ✅
- RunOrchestrator: 14/14 ✅

**Next Priority (BLOCKING):**
- 🔴 Phase 3.2: **WorkerThreadPool** (4-6 hours) - Critical for parallel execution
- 🔴 Phase 4.1: **MinimizerOrchestrator** (6-8 hours) - Orchestrates all components

All dependencies ready for implementation. Infrastructure is complete and tested.
