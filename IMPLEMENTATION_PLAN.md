# Implementation Plan: Algorithm Manager and Orchestration Layer

**Date**: December 3, 2025  
**Last Updated**: December 10, 2025
**Based on**: ARCHITECTURE.md v2.0  
**Status**: 73% Complete - WorkerThreadPool Next Priority

---

## Overview

This plan implements the refactored minimizer architecture with clear separation between:
1. **MinimizationStrategy** - Stateless computation with RAII workspace ownership
2. **AlgorithmManager** - State management (Kraus, vector) with getter/setter interface
3. **RunOrchestrator** - Iteration loop, circular buffer, conditions checking
4. **MinimizerOrchestrator** - Multi-run MOE workflows

---

## Phase 1: Refactor MinimizationStrategy (Stateless + RAII) ✅ COMPLETE

**Goal**: Make `MinimizationStrategy` stateless for computation while retaining workspace ownership

### Task 1.1: Update MinimizationStrategy Interface

**File**: `include/minimizer/core/minimization_strategy.h`

**Changes**:
1. Remove `initialize()` overload that takes Kraus and initial vector
2. Add new `initialize()` with just dimensions and epsilon:
   ```cpp
   virtual void initialize(
       int kraus_count,
       int input_dim,
       int output_dim,
       double epsilon,
       PrecisionType precision
   ) = 0;
   ```
3. Change `stepOnce()` signature to accept memory references and return entropy:
   ```cpp
   virtual double stepOnce(
       IDeviceMemory* d_kraus,
       IDeviceMemory* d_current_vector
   ) = 0;
   ```
4. Remove `getEntropy()` method (no longer caching)
5. Remove `getCurrentVector()` method (AlgorithmManager will handle)
6. Rename `getMemoryRequired()` to `getWorkspaceSize()`:
   ```cpp
   virtual size_t getWorkspaceSize() const = 0;
   ```
7. Remove member variables:
   - `d_kraus_` (moved to AlgorithmManager)
   - `d_current_vector_` (moved to AlgorithmManager)
   - `current_entropy_` (no longer cached)
   - `step1_valid_`, `entropy_valid_` (stateless now)

**Keep** (algorithm-specific workspace):
- `d_vecs_1_`, `d_sing_1_`, `d_vecs_2_`, `d_sing_2_`
- `d_sv_1_`, `d_sv_2_`
- `svd_solver_1_`, `svd_solver_2_`
- `streams_`

**Estimated Time**: 2 hours

---

### Task 1.2: Update CudaMinimizationStrategy Implementation

**File**: `src/minimizer/core/cuda_minimization_strategy.cu`

**Changes**:
1. Update `initialize()` implementation:
   - Remove Kraus/vector allocation
   - Keep workspace buffer allocation
   - Keep SVD solver creation
2. Update `stepOnce()` to:
   - Accept `IDeviceMemory*` parameters
   - Use passed `d_kraus` and `d_current_vector` instead of members
   - Return computed entropy instead of caching
   - Update `d_current_vector` in-place
3. Update `getWorkspaceSize()`:
   - Remove Kraus and vector from calculation
   - Return only workspace buffer sizes
4. Remove helper method implementations for removed methods

**Estimated Time**: 3 hours

---

### Task 1.3: Update GenericMinimizationStrategy Implementation

**File**: `src/minimizer/core/generic_minimization_strategy.cpp`

**Changes**: Same as Task 1.2 but for CPU implementation

**Estimated Time**: 3 hours

---

### Task 1.4: Update Tests

**Files**: 
- `src/main/tests/minimizer/minimization_strategy_test.cpp`
- Any other strategy tests

**Changes**:
1. Update test fixtures to create separate device memory for Kraus/vector
2. Update test calls to pass memory references
3. Update assertions to check return values instead of `getEntropy()`
4. Add tests for stateless behavior (multiple calls with different inputs)

**Estimated Time**: 2 hours

**Total Phase 1**: ~10 hours

---

## Phase 2: Implement AlgorithmManager ✅ COMPLETE

**Goal**: Create state management layer with getter/setter interface

### Task 2.1: Create AlgorithmManager Header

**File**: `include/minimizer/core/algorithm_manager.h`

**Content**:
```cpp
#ifndef ALGORITHM_MANAGER_H_
#define ALGORITHM_MANAGER_H_

#include "minimizer/core/minimization_strategy.h"
#include "minimizer/core/entropy_manager.h"
#include "compute/device/IComputeDevice.h"
#include "compute/memory/IDeviceMemory.h"
#include <memory>
#include <vector>
#include <complex>

namespace entropy {

class AlgorithmManager {
public:
    /**
     * @brief Construct AlgorithmManager
     * 
     * @param device Device for computation (non-owning reference)
     * @param entropy_manager Manages entropy transformations (non-owning reference)
     * @param kraus Host-side Kraus operators
     * @param platform_config Platform configuration
     * @param opt_config Optimization configuration
     */
    AlgorithmManager(
        IComputeDevice& device,
        EntropyManager& entropy_manager,
        const HostKrausOperators& kraus,
        const PlatformConfig& platform_config,
        const OptimizationConfig& opt_config
    );
    
    ~AlgorithmManager() = default;
    
    // Initialization
    void initialize(const std::vector<std::complex<double>>& initial_vector);
    void initializeRandom();
    
    // Algorithm execution
    void stepOnce();  // Calls strategy, updates EntropyManager
    
    // State access - getters/setters
    std::vector<std::complex<double>> getCurrentVector() const;  // EXPENSIVE!
    void setCurrentVector(const std::vector<std::complex<double>>& vec);
    
    // Entropy accessors (delegate to EntropyManager)
    double getEpsilonEntropy() const;
    double getEstimatedEntropy() const;
    std::pair<double, double> getEntropyBounds() const;
    
    // Strategy management
    void setStrategy(std::unique_ptr<IMinimizationStrategy> strategy);
    
    // Query methods
    bool isInitialized() const { return initialized_; }
    PrecisionType getPrecision() const;
    
private:
    // Device memory (RAII)
    std::unique_ptr<IDeviceMemory> d_kraus_;
    std::unique_ptr<IDeviceMemory> d_current_vector_;
    
    // Strategy (owns workspace)
    std::unique_ptr<IMinimizationStrategy> strategy_;
    
    // Non-owning references
    IComputeDevice& device_;
    EntropyManager& entropy_manager_;
    
    // Configuration
    PlatformConfig platform_config_;
    OptimizationConfig opt_config_;
    
    // Problem dimensions
    int kraus_count_;
    int input_dim_;
    int output_dim_;
    PrecisionType precision_;
    
    // State
    bool initialized_;
};

} // namespace entropy

#endif // ALGORITHM_MANAGER_H_
```

**Estimated Time**: 1 hour

---

### Task 2.2: Implement AlgorithmManager

**File**: `src/minimizer/core/algorithm_manager.cpp`

**Implementation**:
1. Constructor: Store references, configuration, dimensions
2. `initialize()`:
   - Allocate `d_kraus_` and copy from host
   - Allocate `d_current_vector_` and copy from host
   - Call `strategy_->initialize()` with dimensions
3. `initializeRandom()`:
   - Generate random normalized vector on host
   - Call `initialize()` with it
4. `stepOnce()`:
   - Call `strategy_->stepOnce(d_kraus_.get(), d_current_vector_.get())`
   - Pass result to `entropy_manager_.updateEpsilonEntropy()`
5. `getCurrentVector()`:
   - Allocate host buffer
   - Copy from `d_current_vector_` to host
   - Return vector
6. `setCurrentVector()`:
   - Validate size
   - Copy from host to `d_current_vector_`
7. Entropy getters: Delegate to `entropy_manager_`

**Estimated Time**: 3 hours

---

### Task 2.3: Add to CMakeLists.txt

**File**: `src/CMakeLists.txt` (or appropriate subdirectory)

**Changes**: Add `algorithm_manager.cpp` to build

**Estimated Time**: 15 minutes

---

### Task 2.4: Create Unit Tests

**File**: `src/main/tests/minimizer/algorithm_manager_test.cpp`

**Tests**:
1. Construction and initialization
2. `stepOnce()` updates entropy
3. `getCurrentVector()` returns correct data
4. `setCurrentVector()` updates device memory
5. Entropy getters delegate correctly
6. Random initialization produces normalized vector
7. Strategy can be swapped

**Estimated Time**: 2 hours

**Total Phase 2**: ~6.25 hours

---

## Phase 3: Implement RunOrchestrator & StrategyFactory ✅ COMPLETE

**Status**: RunOrchestrator ✅ COMPLETED, StrategyFactory 🔄 IN PROGRESS

**Goal**: Complete single-run execution with memory-aware strategy selection

### Task 3.1: Implement StrategyFactory (Memory-Aware Selection)

**See:** `ORCHESTRATION_IMPLEMENTATION_PLAN.md` Phase 3 Step 3.1.1-3.1.4

**Summary:**
- Create `StrategyFactory` with AUTO/GENERIC/CUDA modes
- Implement workspace memory estimation (before allocation)
- Query GPU memory via `cudaMemGetInfo()`
- AUTO logic: Check backend → estimate → validate → select optimal strategy
- Update RunOrchestrator to use factory instead of hard-coded GenericMinimizationStrategy

**Files:**
- `include/minimizer/algorithm/strategy_factory.h`
- `src/minimizer/algorithm/strategy_factory.cpp`
- `src/main/tests/test_strategy_factory.cpp`

**Status**: Implementation in progress

**Estimated Time**: 6 hours

---

### Task 3.2: Implement CheckpointManager

**See:** `ORCHESTRATION_IMPLEMENTATION_PLAN.md` Phase 3 Step 3.1.6

**Goal**: Enable checkpoint save/restore in RunOrchestrator

**Current Status**: Stubbed out (missing json.hpp integration)

**Tasks:**
1. Integrate nlohmann/json library
2. Implement `CheckpointManager::save()` - JSON serialization
3. Implement `CheckpointManager::load()` - JSON deserialization
4. Add compression support (optional)
5. Implement checkpoint cleanup (keep last N)
6. Enable in RunOrchestrator
7. Comprehensive tests

**Files:**
- `include/minimizer/orchestration/checkpoint_manager.h` (exists)
- `src/minimizer/orchestration/checkpoint_manager.cpp` (stub)
- `src/main/tests/orchestration/test_checkpoint_manager.cpp`

**Status**: Deferred to future work

**Estimated Time**: 4 hours

---

### Task 3.3: RunOrchestrator Implementation ✅ COMPLETED

**Files:**
- `include/minimizer/orchestration/types.h` - Shared types
- `include/minimizer/orchestration/run_orchestrator.h`
- `src/minimizer/orchestration/run_orchestrator.cpp`
- `src/main/tests/orchestration/test_run_orchestrator.cpp`

**Status**: Fully implemented and tested (14/14 tests passing)

**Key Features:**
- Single-run execution with iteration loop
- Component coordination (AlgorithmManager, ConditionsChecker, EntropyPredictor)
- Progress callbacks (thread-safe)
- Structured error handling (RunErrorType enum)
- Shared types (HostVector, RunTask, RunResult)

**Known Limitations:**
- Checkpoint disabled (pending Task 3.2)
- Hard-coded GenericMinimizationStrategy (pending Task 3.1)

---

## Phase 4: Implement RunOrchestrator (LEGACY - NOW Phase 3 Task 3.3)

**Goal**: Create single-run iteration loop with circular buffer and conditions

### Task 3.1: Create RunOrchestrator Header

**File**: `include/minimizer/orchestration/run_orchestrator.h`

**Content**:
```cpp
#ifndef RUN_ORCHESTRATOR_H_
#define RUN_ORCHESTRATOR_H_

#include "minimizer/core/algorithm_manager.h"
#include "minimizer/conditions/conditions_checker.h"
#include "minimizer/core/entropy_manager.h"
#include "minimizer/core/entropy_predictor.h"
#include "minimizer/utilities/checkpoint_manager.h"
#include "utilities/messaging/message_handler.h"
#include "utilities/circular_buffer.h"
#include <string>
#include <chrono>
#include <memory>

namespace entropy {

struct RunResult {
    std::string run_id;
    double final_entropy;
    double run_minimum_entropy;  // Best found in this run
    int total_iterations;
    StopReason stop_reason;
    std::chrono::duration<double> elapsed_time;
    
    // Lightweight summary for history
    struct Summary {
        std::string run_id;
        double run_minimum_entropy;
        int total_iterations;
        StopReason stop_reason;
        double elapsed_seconds;
    };
    
    Summary summarize() const;
};

class RunOrchestrator {
public:
    RunOrchestrator(
        std::unique_ptr<AlgorithmManager> algorithm_manager,
        std::unique_ptr<ConditionsChecker> conditions_checker,
        std::unique_ptr<EntropyManager> entropy_manager,
        std::unique_ptr<EntropyPredictor> entropy_predictor,
        std::unique_ptr<CheckpointManager> checkpoint_manager,
        MessageHandler& msg_handler,
        const MinimizerConfig& config
    );
    
    // Run lifecycle
    void initializeRun(const std::vector<std::complex<double>>& start_vector);
    void initializeRunRandom();
    
    RunResult runToConvergence();
    RunResult runToTarget(double target_entropy);
    
    // Run state
    double getCurrentEntropy() const;
    double getRunMinimumEntropy() const;
    int getCurrentIteration() const;
    std::string getRunId() const;
    
    // Reset for reuse
    void reset();
    
private:
    // Components (owned)
    std::unique_ptr<AlgorithmManager> algorithm_manager_;
    std::unique_ptr<ConditionsChecker> conditions_checker_;
    std::unique_ptr<EntropyManager> entropy_manager_;
    std::unique_ptr<EntropyPredictor> entropy_predictor_;
    std::unique_ptr<CheckpointManager> checkpoint_manager_;
    
    // Per-run state
    std::string run_id_;
    int current_iteration_;
    double run_minimum_entropy_;
    
    // Circular buffer (convergence window)
    CircularBuffer<double> entropy_window_;
    
    // Non-owning references
    MessageHandler& message_handler_;
    const MinimizerConfig& config_;
    
    // Helper methods
    void performIteration();
    void saveCheckpointIfNeeded();
    std::string generateRunId();
};

} // namespace entropy

#endif // RUN_ORCHESTRATOR_H_
```

**Estimated Time**: 1 hour

---

### Task 3.2: Implement RunOrchestrator

**File**: `src/minimizer/orchestration/run_orchestrator.cpp`

**Implementation**:
1. Constructor: Store components, initialize circular buffer with config size
2. `initializeRun()`:
   - Generate new `run_id_` (UUID)
   - Reset iteration counter
   - Reset run minimum
   - Clear circular buffer
   - Call `algorithm_manager_->initialize()`
3. `runToConvergence()`:
   - Main loop (as detailed in ARCHITECTURE.md):
     - `algorithm_manager_->stepOnce()`
     - Get entropy
     - Update run minimum
     - Push to circular buffer
     - Update predictor
     - Check conditions
     - Checkpoint if needed
     - Log periodically
   - Return `RunResult`
4. `reset()`:
   - Clear run state
   - Reset components
   - Clear buffer
5. Helper methods

**Estimated Time**: 4 hours

---

### Task 3.3: Create Directory Structure

**Directories**:
- `include/minimizer/orchestration/`
- `src/minimizer/orchestration/`

**Estimated Time**: 5 minutes

---

### Task 3.4: Update CMakeLists.txt

Add `src/minimizer/orchestration/CMakeLists.txt` with `run_orchestrator.cpp`

**Estimated Time**: 15 minutes

---

### Task 3.5: Create Unit Tests

**File**: `src/main/tests/minimizer/run_orchestrator_test.cpp`

**Tests**:
1. Initialization generates unique run IDs
2. Run loop exits on max iterations
3. Run loop exits on convergence
4. Circular buffer updated correctly
5. Run minimum tracked correctly
6. Reset clears state properly
7. Checkpoints saved at correct intervals
8. EntropyPredictor updated each iteration

**Estimated Time**: 3 hours

**Total Phase 3**: ~8.33 hours

---

## Phase 4: Implement MinimizerOrchestrator

**Goal**: Multi-run MOE workflows with sequential execution

### Task 4.1: Create MinimizerOrchestrator Header

**File**: `include/minimizer/orchestration/minimizer_orchestrator.h`

**Content**:
```cpp
#ifndef MINIMIZER_ORCHESTRATOR_H_
#define MINIMIZER_ORCHESTRATOR_H_

#include "minimizer/orchestration/run_orchestrator.h"
#include "utilities/messaging/message_handler.h"
#include <vector>
#include <memory>

namespace entropy {

struct MOEResult {
    double global_MOE;
    std::vector<RunResult::Summary> run_summaries;
    int total_runs;
    std::chrono::duration<double> total_elapsed_time;
};

class MinimizerOrchestrator {
public:
    MinimizerOrchestrator(
        const HostKrausOperators& kraus,
        const MinimizerConfig& config,
        MessageHandler& msg_handler,
        IComputeDevice& device
    );
    
    // Multi-run workflows
    MOEResult findMOE();
    RunResult runSingleAttempt();
    
    // Accessors
    double getGlobalMOE() const { return global_MOE_; }
    int getTotalRuns() const { return total_runs_completed_; }
    std::vector<RunResult::Summary> getRunHistory() const { return run_history_; }
    
private:
    // Components
    std::unique_ptr<RunOrchestrator> run_orchestrator_;
    
    // Multi-run state
    double global_MOE_;
    std::vector<RunResult::Summary> run_history_;
    int total_runs_completed_;
    
    // Non-owning references
    MessageHandler& message_handler_;
    const MinimizerConfig& config_;
    const HostKrausOperators& kraus_;
    IComputeDevice& device_;
    
    // Helper methods
    void createRunOrchestrator();
};

} // namespace entropy

#endif // MINIMIZER_ORCHESTRATOR_H_
```

**Estimated Time**: 1 hour

---

### Task 4.2: Implement MinimizerOrchestrator

**File**: `src/minimizer/orchestration/minimizer_orchestrator.cpp`

**Implementation**:
1. Constructor: Initialize state, create `RunOrchestrator`
2. `findMOE()`:
   - Loop N times (from config)
   - Reset RunOrchestrator
   - Initialize with random vector
   - Run to convergence
   - Update global MOE if better
   - Store summary
   - Log progress
3. `runSingleAttempt()`:
   - Create/reset RunOrchestrator
   - Run once
   - Return result
4. `createRunOrchestrator()`:
   - Factory method to create all components

**Estimated Time**: 3 hours

---

### Task 4.3: Update CMakeLists.txt

Add `minimizer_orchestrator.cpp` to build

**Estimated Time**: 5 minutes

---

### Task 4.4: Create Unit Tests

**File**: `src/main/tests/minimizer/minimizer_orchestrator_test.cpp`

**Tests**:
1. Single run returns valid result
2. MOE workflow runs N times
3. Global MOE updated correctly
4. Run history stored as summaries
5. RunOrchestrator reused across runs
6. Best run optionally checkpointed

**Estimated Time**: 2 hours

**Total Phase 4**: ~6.08 hours

---

## Phase 5: Integration and Migration

**Goal**: Update main executables to use new architecture

### Task 5.1: Update phase1 Main

**File**: `src/main/phase1/phase1.cu`

**Changes**:
1. Remove old `EntropyMinimizer` usage
2. Create `MinimizerOrchestrator`
3. Call `findMOE()`
4. Process results

**Estimated Time**: 2 hours

---

### Task 5.2: Update MOE Main

**File**: `src/main/moe/moe_main.cpp`

**Changes**: Similar to Task 5.1

**Estimated Time**: 2 hours

---

### Task 5.3: Integration Tests

**Create**: End-to-end tests with real data

**Tests**:
1. Full MOE run completes successfully
2. Results match old implementation (within tolerance)
3. Memory usage is reasonable
4. Performance is acceptable

**Estimated Time**: 3 hours

---

### Task 5.4: Update Documentation

**Files**:
- README.md
- ARCHITECTURE.md (mark as implemented)
- Code comments

**Estimated Time**: 2 hours

**Total Phase 5**: ~9 hours

---

## Summary

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 1 | Refactor MinimizationStrategy | 10 hours |
| 2 | Implement AlgorithmManager | 6.25 hours |
| 3 | Implement RunOrchestrator | 8.33 hours |
| 4 | Implement MinimizerOrchestrator | 6.08 hours |
| 5 | Integration and Migration | 9 hours |
| **Total** | | **~39.66 hours** (~5 days) |

---

## Dependencies

### External Dependencies
- All existing infrastructure (IDevice, DeviceMemory, etc.)
- EntropyManager (already implemented)
- EntropyPredictor (already implemented)
- ConditionsChecker (already implemented)
- CircularBuffer (already implemented)
- CheckpointManager (needs implementation or use existing)
- MessageHandler (already implemented)

### Order of Implementation
1. **Must do first**: Phase 1 (Strategy refactor)
2. **Can parallelize**: Phases 2, 3, 4 (independent components)
3. **Must do last**: Phase 5 (integration)

---

## Risk Mitigation

### Risk 1: Strategy Refactor Breaks Existing Code
**Mitigation**: 
- Keep old implementation in `old/` directory temporarily
- Comprehensive unit tests before/after
- Feature flag to switch between old/new

### Risk 2: Memory Allocation Patterns Unclear
**Mitigation**:
- Write memory allocation tests early
- Document ownership clearly in code
- Use RAII everywhere (automatic cleanup)

### Risk 3: Performance Regression
**Mitigation**:
- Benchmark before/after
- Profile memory usage
- Optimize hot paths if needed

### Risk 4: Integration Complexity
**Mitigation**:
- Incremental integration (one component at a time)
- Keep old code paths working initially
- Extensive integration testing

---

## Success Criteria

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Memory usage ≤ old implementation
- [ ] Performance within 10% of old implementation
- [ ] Code coverage ≥ 80%
- [ ] Documentation complete
- [ ] No memory leaks (valgrind clean)
- [ ] CUDA memory errors resolved (cuda-memcheck clean)

---

## Next Steps

All three phases from this IMPLEMENTATION_PLAN are now complete!

### Current Status (December 10, 2025)

**Completed:**
- ✅ Phase 1: MinimizationStrategy refactoring (stateless computation)
- ✅ Phase 2: AlgorithmManager (state management with RAII)
- ✅ Phase 3: RunOrchestrator (single-run orchestration) + StrategyFactory (memory-aware selection)

**Test Results:**
- All MinimizationStrategy tests passing
- All AlgorithmManager tests passing
- All RunOrchestrator tests passing (14/14)
- All StrategyFactory tests passing (15/15)

### Next Phase: Worker Orchestration

The focus now shifts to the ORCHESTRATION_IMPLEMENTATION_PLAN.md:

**In Progress:**
- Phase 3.2: **WorkerThreadPool** (next priority - 4-6 hours)
  - Manages worker threads (one per device)
  - Pulls tasks from ConcurrentQueue
  - Executes via RunOrchestrator
  - Collects results via ResultCollector

**Blocked (waiting for WorkerThreadPool):**
- Phase 4.1: MinimizerOrchestrator (top-level orchestration - 6-8 hours)
- Phase 4.2: Dynamic resource scaling (future enhancement)

### Implementation Readiness

✅ **All dependencies ready:**
- Foundation infrastructure (120/120 tests passing)
- Device management (DevicePool - 36 tests)
- Strategy selection (StrategyFactory - 15 tests)
- Single-run execution (RunOrchestrator - 14 tests)

**Ready to implement WorkerThreadPool** - See ORCHESTRATION_IMPLEMENTATION_PLAN.md for specifications.

---

**Status**: 73% Complete - Proceeding to Phase 3.2 (WorkerThreadPool)

