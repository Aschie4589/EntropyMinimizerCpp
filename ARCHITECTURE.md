# EntropyMinimizer Architecture Design

**Version**: 2.0  
**Date**: November 25, 2025  
**Status**: Design Phase

---

## Table of Contents
1. [Overview](#overview)
2. [Core Design Principles](#core-design-principles)
3. [Component Architecture](#component-architecture)
4. [Resource Ownership Model](#resource-ownership-model)
5. [Configuration System](#configuration-system)
6. [Multi-Run vs Single-Run Strategy](#multi-run-vs-single-run-strategy)
7. [Device Management](#device-management)
8. [Performance Considerations](#performance-considerations)
9. [Open Questions](#open-questions)

---

## Overview

This document describes the architectural redesign of the EntropyMinimizer system, decomposing the monolithic "god class" into focused, testable components with clear responsibilities.

**Key Goals:**
- **Separation of Concerns**: Each component has one clear responsibility
- **Testability**: Components can be tested in isolation
- **Hardware Abstraction**: Support GPU (CUDA), CPU, and future backends (ROCm)
- **Thread Safety**: Enable multi-device parallelization
- **Clean Ownership**: Clear rules for who owns what resources
- **Performance**: Handle millions of iterations efficiently

---

## Core Design Principles

1. **Single Responsibility**: Each class does one thing well
2. **Dependency Injection**: Components receive dependencies via constructor
3. **Interface-Based Design**: Program to interfaces (IDevice, IPredictionStrategy)
4. **RAII Everywhere**: No manual memory management
5. **Immutable Configs**: Configuration objects are immutable after construction
6. **Observable State**: Clear query methods for component state
7. **Efficient State Management**: Only track essential data (no full history for millions of iterations)

---

## Component Architecture

### Architecture Layers

```
Application Layer
    └── MultiDeviceOrchestrator (optional, for parallel execution)
         └── DeviceManager

Orchestration Layer
    └── MinimizerOrchestrator (coordinates multi-run workflows like findMOE)
         └── RunOrchestrator (manages single minimization run)

Core Components (used by RunOrchestrator)
    ├── AlgorithmManager (hardware-agnostic algorithm execution)
    ├── ConditionsChecker (stopping criteria evaluation)
    ├── EntropyManager (entropy computation & transformations)
    ├── EntropyPredictor (predictive analytics)
    └── CheckpointManager (I/O and state persistence)

Strategy Layer
    ├── StrategyFactory (memory-aware strategy selection)
    ├── MinimizationStrategy (stateless algorithm implementation, owns workspace buffers)
    └── PredictionStrategy (forecasting models)

Infrastructure Layer
    ├── IDevice (hardware abstraction)
    ├── DeviceMemory<T> (RAII wrappers)
    ├── MessageHandler (logging)
    └── VectorSerializer (persistence)
```

---

## Component Details

### 1. MinimizerOrchestrator (Multi-Run Coordinator)

**Responsibility**: Coordinates multi-run workflows (e.g., `findMOE()` with N attempts)

**Interface:**
```cpp
class MinimizerOrchestrator {
public:
    MinimizerOrchestrator(
        const HostKrausOperators& kraus,
        const MinimizerConfig& config,
        MessageHandler& msg_handler,
        IDevice& device
    );
    
    // Multi-run workflows
    MOEResult findMOE();  // Runs N minimization attempts, tracks global MOE
    
    // Accessors
    double getGlobalMOE() const;
    int getTotalRuns() const;
    std::vector<RunSummary> getRunHistory() const;
    
private:
    std::unique_ptr<RunOrchestrator> run_orchestrator_;
    
    // Multi-run state
    double global_MOE_;
    std::vector<RunSummary> run_history_;  // Lightweight summaries only
    int total_runs_completed_;
    
    // Non-owning references
    MessageHandler& message_handler_;
    const MinimizerConfig& config_;
};
```

**Key Points:**
- Owns `RunOrchestrator` (creates new instance or resets for each run)
- Tracks **global** MOE across all runs
- Aggregates **lightweight summaries** from multiple runs (not full iteration history)
- Does NOT manage per-run iteration state

---

### 2. RunOrchestrator (Single-Run Manager)

**Responsibility**: Manages a single minimization run from initialization to convergence

**Interface:**
```cpp
struct RunResult {
    std::string run_id;
    double final_entropy;
    double run_minimum_entropy;  // Best found in this run
    int total_iterations;
    StopReason stop_reason;
    std::chrono::duration<double> elapsed_time;
    
    RunSummary summarize() const;  // Lightweight version for history
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
    void initializeRun();  // Random start
    void initializeRun(const HostVector& start_vector);
    
    RunResult runToConvergence();  // Run until stopping condition
    RunResult runToTarget(double target_entropy);  // Run until entropy < target
    
    // Per-iteration control
    StepResult stepOnce();  // Single iteration + convergence check
    
    // Run state (current run only)
    double getCurrentEntropy() const;
    double getRunMinimumEntropy() const;  // MOE for THIS run
    int getCurrentIteration() const;
    std::string getRunId() const;
    
    void reset();  // Prepare for new run (clears state, reuses components)
    
private:
    // Components (owned, reused across runs)
    std::unique_ptr<AlgorithmManager> algorithm_manager_;
    std::unique_ptr<ConditionsChecker> conditions_checker_;
    std::unique_ptr<EntropyManager> entropy_manager_;
    std::unique_ptr<EntropyPredictor> entropy_predictor_;
    std::unique_ptr<CheckpointManager> checkpoint_manager_;
    
    // Per-run state (cleared on reset())
    std::string run_id_;
    int current_iteration_;
    double run_minimum_entropy_;  // Best entropy found in THIS run
    
    // Circular buffer for convergence checking (fixed size N)
    CircularBuffer<double> entropy_window_;  // Last N entropies only
    
    // Non-owning references
    MessageHandler& message_handler_;
    const MinimizerConfig& config_;
    
    // Helper methods
    void performIteration();
    bool checkStoppingConditions();
};
```

**Key Points:**
- Manages **single run** state: iteration count, run ID, run-specific MOE
- Uses **circular buffer** for entropy history (fixed size, typically 20-500 entries)
- Coordinates components (AlgorithmManager, ConditionsChecker, etc.)
- **Reusable**: `reset()` clears state, reuses component instances for efficiency
- Does NOT know about global MOE across runs

**Memory Efficiency:**
- Millions of iterations → store only last N entropies in circular buffer
- Track only: current_entropy, run_minimum_entropy, iteration count
- EntropyPredictor maintains separate window (can be different size)

---

### 3. AlgorithmManager

**Responsibility**: Execute minimization algorithm step and manage problem state (Kraus operators, current vector)

**Interface:**
```cpp
class AlgorithmManager {
public:
    AlgorithmManager(
        IDevice& device,
        EntropyManager& entropy_manager,  // Non-owning reference
        const HostKrausOperators& kraus,
        const PlatformConfig& platform_config,
        const OptimizationConfig& opt_config
    );
    
    // Initialization
    void initialize(const HostVector& initial_vector);
    void initializeRandom();
    
    // Algorithm execution
    void stepOnce();  // One iteration: calls strategy, updates EntropyManager
    
    // State access - getters/setters pattern
    HostVector getCurrentVector() const;  // Device→Host copy (EXPENSIVE)
    void setCurrentVector(const HostVector& vec);  // Host→Device copy
    
    // Entropy accessors (delegate to EntropyManager)
    double getEpsilonEntropy() const;
    double getEstimatedEntropy() const;
    std::pair<double, double> getEntropyBounds() const;
    
    // Strategy management
    void setStrategy(std::unique_ptr<MinimizationStrategy> strategy);
    
private:
    // Device memory ownership (RAII)
    std::unique_ptr<IDeviceMemory> d_kraus_;           // Kraus operators on device
    std::unique_ptr<IDeviceMemory> d_current_vector_;  // Current vector on device
    
    // Strategy (owns algorithm-specific buffers)
    std::unique_ptr<MinimizationStrategy> strategy_;
    
    // Non-owning references
    IDevice& device_;
    EntropyManager& entropy_manager_;  // Owned by RunOrchestrator
    
    // Configuration
    PlatformConfig platform_config_;
    OptimizationConfig opt_config_;
};
```

**Key Points:**
- **Owns device memory** for Kraus operators and current vector via RAII
- **Does NOT own** algorithm-specific buffers (owned by MinimizationStrategy)
- **Getter/setter pattern** for state access (no return values from `stepOnce()`)
- Updates `EntropyManager` after each step (calls `updateEpsilonEntropy()`)
- Delegates entropy queries to `EntropyManager` for convenience
- Strategy pattern allows different algorithm implementations
- Minimal host-device transfers (only when explicitly requested via getters/setters)

---

### RunOrchestrator Workflow Details

The `RunOrchestrator` manages the iteration loop and coordinates all components:

```cpp
RunResult RunOrchestrator::runToConvergence() {
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        // 1. Execute one algorithm step (updates vector, computes entropy)
        algorithm_manager_->stepOnce();  // Calls strategy, updates EntropyManager
        
        // 2. Get current epsilon entropy from AlgorithmManager
        double current_entropy = algorithm_manager_->getEpsilonEntropy();
        
        // 3. Update run minimum
        if (current_entropy < run_minimum_entropy_) {
            run_minimum_entropy_ = current_entropy;
        }
        
        // 4. Push to circular buffer for convergence checking
        entropy_window_.push(current_entropy);
        
        // 5. Update predictor (separate buffer inside EntropyPredictor)
        entropy_predictor_->addDataPoint(current_entropy);
        
        // 6. Check stopping conditions
        StopReason reason = conditions_checker_->checkStoppingConditions(
            current_iteration_,
            current_entropy,
            entropy_window_  // Passed by const reference
        );
        
        if (reason != StopReason::CONTINUE) {
            // 7. Optionally save final checkpoint
            if (checkpoint_manager_->shouldCheckpoint(current_iteration_)) {
                auto vector = algorithm_manager_->getCurrentVector();
                checkpoint_manager_->saveCheckpoint(
                    run_id_, current_iteration_, vector, /* metadata */
                );
            }
            
            auto end_time = std::chrono::steady_clock::now();
            return RunResult{
                run_id_,
                current_entropy,
                run_minimum_entropy_,
                current_iteration_,
                reason,
                end_time - start_time
            };
        }
        
        // 8. Periodic checkpoint
        if (checkpoint_manager_->shouldCheckpoint(current_iteration_)) {
            auto vector = algorithm_manager_->getCurrentVector();  // EXPENSIVE!
            checkpoint_manager_->saveCheckpoint(
                run_id_, current_iteration_, vector, /* metadata */
            );
        }
        
        // 9. Periodic logging
        if (current_iteration_ % config_.logging_interval == 0) {
            message_handler_.info(
                "Iteration {}: entropy = {:.6e}",
                current_iteration_,
                current_entropy
            );
        }
        
        current_iteration_++;
    }
}
```

**Key Flow:**
- `AlgorithmManager::stepOnce()` updates `EntropyManager` internally
- `RunOrchestrator` queries entropy via `getEpsilonEntropy()` or `getEstimatedEntropy()`
- Circular buffer updated by `RunOrchestrator` after each step
- `ConditionsChecker` receives const reference to buffer (no copy)
- `EntropyPredictor` maintains its own separate buffer
- Expensive host copies (`getCurrentVector()`) only for checkpoints

---

### 4. StrategyFactory

**Responsibility**: Memory-aware strategy selection and instantiation

**Problem**: Kraus operators for large quantum systems require massive GPU memory. Strategy selection must account for:
- Device backend (CUDA vs CPU)
- Available GPU memory
- Estimated workspace requirements
- Performance trade-offs

**Solution**: Factory pattern with AUTO/GENERIC/CUDA modes.

**Interface:**
```cpp
enum class StrategyType {
    AUTO,     // Auto-select based on device + memory
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

**AUTO Selection Logic:**
1. Check device backend via `device.getBackend()`
2. If CUDA:
   - Estimate workspace: `size = estimateWorkspaceSize(d, N, M, precision)`
   - Query GPU memory: `cudaMemGetInfo(&free, &total)`
   - If `free >= 1.3 × size`: Return `CudaMinimizationStrategy` (30% buffer)
   - Else: Return `GenericMinimizationStrategy` (fallback)
3. If CPU: Return `GenericMinimizationStrategy`

**Memory Estimation Formula:**
```
complex_size = sizeof(std::complex<T>)  // 8 (float) or 16 (double)
real_size = sizeof(T)                    // 4 (float) or 8 (double)

vecs1 = d × M × complex_size             // {K_i|ψ⟩}
sing1 = d × M × complex_size             // SVD U vectors
vecs2 = d² × N × complex_size            // {K_j^H|φ_i⟩}
sing2 = N × complex_size                 // SVD U (rank 1)
sv1 = d × real_size                      // Singular values
sv2 = d² × real_size                     // Singular values
svd1_workspace ≈ M × d × complex_size    // cuSOLVER (QR)
svd2_workspace ≈ N × d² × complex_size   // cuSOLVER (randomized)

Total ≈ 3×d×M×c + 2×d²×N×c + N×c + (d+d²)×r
```

**Safety Margins:**
- AUTO: 30% buffer (1.3×) for safe operation
- Explicit CUDA: 20% buffer (1.2×) validation
- GENERIC: No check (fallback, works everywhere)

**Key Design Points:**
- **Stateless**: Pure functions, no global state
- **RAII-Compliant**: Returns `unique_ptr` for automatic cleanup
- **Fail-Safe**: AUTO never throws, always falls back to Generic
- **Explicit Control**: Power users can force specific strategy
- **Memory-First**: Estimates before allocating (critical for large systems)

**Usage Example:**
```cpp
// In RunOrchestrator constructor:
auto strategy = StrategyFactory::create(
    device_,
    kraus_ops.kraus_count,
    kraus_ops.input_dim,
    kraus_ops.output_dim,
    config.algorithm.precision
);
algorithm_mgr_->setStrategy(std::move(strategy));
```

---

### 5. MinimizationStrategy (Refactored Interface)

**Responsibility**: Stateless algorithm implementation with RAII workspace management

**Key Design Changes:**
1. **Stateless computation**: No cached entropy or vector state
2. **RAII workspace ownership**: Strategy allocates/owns algorithm-specific buffers
3. **Receives memory references**: Kraus and vector passed as parameters to `stepOnce()`
4. **Returns entropy**: Instead of caching, directly returns computed value

**Interface:**
```cpp
class IMinimizationStrategy {
public:
    virtual ~IMinimizationStrategy() = default;
    
    /**
     * @brief Initialize strategy with problem dimensions
     * 
     * Allocates algorithm-specific workspace buffers (d_vecs_1_, d_sing_1_, etc.)
     * via RAII. Does NOT receive Kraus or vector data.
     * 
     * @param kraus_count Number of Kraus operators (d)
     * @param input_dim Input dimension (N)
     * @param output_dim Output dimension (M)
     * @param epsilon Perturbation parameter
     * @param precision FLOAT or DOUBLE
     */
    virtual void initialize(
        int kraus_count,
        int input_dim,
        int output_dim,
        double epsilon,
        PrecisionType precision
    ) = 0;
    
    /**
     * @brief Execute one iteration (stateless)
     * 
     * Receives device memory references for Kraus and current vector.
     * Performs algorithm step:
     * 1. Apply Kraus operators: {K_i|ψ⟩}
     * 2. SVD and logarithmic scaling
     * 3. Second SVD to extract new |ψ⟩
     * 
     * Updates d_current_vector in-place on device.
     * Returns computed epsilon entropy.
     * 
     * @param d_kraus Device memory for Kraus operators (read-only)
     * @param d_current_vector Device memory for current vector (read-write)
     * @return Computed epsilon entropy S(Φ_ε(|ψ⟩⟨ψ|))
     */
    virtual double stepOnce(
        IDeviceMemory* d_kraus,
        IDeviceMemory* d_current_vector
    ) = 0;
    
    /**
     * @brief Query workspace memory requirements
     * 
     * Returns total device memory needed for algorithm-specific buffers
     * (vecs, singular vectors, singular values, SVD workspace).
     * 
     * Used for memory planning and strategy selection.
     */
    virtual size_t getWorkspaceSize() const = 0;
    
    /**
     * @brief Get current precision
     */
    virtual PrecisionType getPrecision() const = 0;
    
protected:
    // Strategy owns algorithm-specific buffers via RAII
    std::unique_ptr<IDeviceMemory> d_vecs_1_;   // Step 1 vectors (d×M)
    std::unique_ptr<IDeviceMemory> d_sing_1_;   // Step 1 singular vectors (d×M)
    std::unique_ptr<IDeviceMemory> d_vecs_2_;   // Step 2 vectors (d²×N)
    std::unique_ptr<IDeviceMemory> d_sing_2_;   // Step 2 singular vectors (N)
    std::unique_ptr<IDeviceMemory> d_sv_1_;     // Step 1 singular values (d)
    std::unique_ptr<IDeviceMemory> d_sv_2_;     // Step 2 singular values (d²)
    
    std::unique_ptr<ISVDSolver> svd_solver_1_;
    std::unique_ptr<ISVDSolver> svd_solver_2_;
};
```

**Memory Ownership Model:**

```
AlgorithmManager
    ├── owns d_kraus_ (Kraus operators)              [via RAII]
    ├── owns d_current_vector_ (current state)       [via RAII]
    └── owns strategy_
         └── MinimizationStrategy
              ├── owns d_vecs_1_ (workspace)         [via RAII]
              ├── owns d_sing_1_ (workspace)         [via RAII]
              ├── owns d_vecs_2_ (workspace)         [via RAII]
              ├── owns d_sing_2_ (workspace)         [via RAII]
              ├── owns d_sv_1_ (workspace)           [via RAII]
              ├── owns d_sv_2_ (workspace)           [via RAII]
              └── owns svd_solver_1_, svd_solver_2_  [via RAII]
```

**Allocation Pattern:**

*AlgorithmManager::initialize():*
```cpp
void AlgorithmManager::initialize(const HostVector& initial_vector) {
    // 1. Allocate Kraus operators on device
    size_t kraus_bytes = kraus_count_ * output_dim_ * input_dim_ * complex_size;
    d_kraus_ = device_.allocate(kraus_bytes);
    d_kraus_->copyFromHost(kraus_host_data, kraus_bytes);
    
    // 2. Allocate current vector on device
    size_t vector_bytes = input_dim_ * complex_size;
    d_current_vector_ = device_.allocate(vector_bytes);
    d_current_vector_->copyFromHost(initial_vector.data(), vector_bytes);
    
    // 3. Initialize strategy (allocates its workspace)
    strategy_->initialize(
        kraus_count_, input_dim_, output_dim_,
        opt_config_.epsilon, precision_
    );
}
```

*MinimizationStrategy::initialize():*
```cpp
void CudaMinimizationStrategy::initialize(
    int kraus_count, int input_dim, int output_dim,
    double epsilon, PrecisionType precision
) {
    // Allocate workspace buffers (strategy owns these!)
    d_vecs_1_ = device_.allocate(kraus_count * output_dim * complex_size);
    d_sing_1_ = device_.allocate(kraus_count * output_dim * complex_size);
    d_vecs_2_ = device_.allocate(kraus_count * kraus_count * input_dim * complex_size);
    d_sing_2_ = device_.allocate(input_dim * complex_size);
    d_sv_1_ = device_.allocate(kraus_count * real_size);
    d_sv_2_ = device_.allocate(kraus_count * kraus_count * real_size);
    
    // Create SVD solvers (queries workspace size internally)
    SVDSpec spec1{SVDVectors::THIN, SVDVectors::NONE, SVDAlgorithm::QR};
    svd_solver_1_ = device_.createSVDSolver(output_dim, kraus_count, spec1, precision);
    
    SVDSpec spec2{SVDVectors::THIN, SVDVectors::NONE, SVDAlgorithm::RANDOMIZED};
    spec2.rank = 1;
    svd_solver_2_ = device_.createSVDSolver(input_dim, kraus_count * kraus_count, spec2, precision);
}
```

*AlgorithmManager::stepOnce():*
```cpp
void AlgorithmManager::stepOnce() {
    // 1. Call strategy (stateless, receives memory references)
    double epsilon_entropy = strategy_->stepOnce(
        d_kraus_.get(),
        d_current_vector_.get()
    );
    
    // 2. Update EntropyManager
    entropy_manager_.updateEpsilonEntropy(epsilon_entropy);
    
    // Note: d_current_vector_ was updated in-place by strategy
}
```

**Benefits:**
- Clear ownership: AlgorithmManager owns data, Strategy owns workspace
- RAII ensures proper cleanup in all cases
- Strategy can be swapped without affecting AlgorithmManager
- Memory requirements accurately computed by strategy
- No cached state in strategy → easier to reason about

---

### 4. ConditionsChecker

**Responsibility**: Evaluate stopping conditions

**Interface:**
```cpp
enum class StopReason {
    CONTINUE,
    CONVERGED,
    MAX_ITERATIONS,
    NUMERICAL_INSTABILITY,
    TARGET_REACHED,
    PREDICTION_SATISFIED,
    USER_TERMINATED
};

class ConditionsChecker {
public:
    ConditionsChecker(const ConditionsConfig& config);
    
    // Register conditions
    void addCondition(std::unique_ptr<Condition> condition);
    
    // Main check - receives only necessary data
    StopReason checkStoppingConditions(
        int iteration,
        double current_entropy,
        const CircularBuffer<double>& recent_entropy_history
    );
    
private:
    std::vector<std::unique_ptr<Condition>> conditions_;
    ConditionsConfig config_;
};

// Condition interface
class Condition {
public:
    virtual ~Condition() = default;
    virtual StopReason check(
        int iteration,
        double current_entropy,
        const CircularBuffer<double>& history
    ) = 0;
};

// Concrete implementations
class MaxIterationsCondition : public Condition { 
    StopReason check(...) override {
        return iteration >= max_iterations_ ? StopReason::MAX_ITERATIONS 
                                             : StopReason::CONTINUE;
    }
};

class NumericalInstabilityCondition : public Condition {
    // Checks if entropy increased (delta < 0)
    StopReason check(...) override;
};

class ConvergenceCondition : public Condition {
    // Checks if avg improvement over window < tolerance
    StopReason check(...) override;
};

class TargetEntropyCondition : public Condition {
    StopReason check(...) override;
};
```

**Key Points:**
- Strategy pattern for extensible stopping conditions
- Receives **const reference** to circular buffer (no copy)
- Stateless: all state passed as arguments
- Efficient: only accesses last N entropies from circular buffer

---

### 5. EntropyManager

**Responsibility**: Entropy computation and transformations (epsilon_entropy → estimated_entropy)

**Interface:**
```cpp
class EntropyManager {
public:
    EntropyManager(double epsilon, int input_dim);
    
    void updateEpsilonEntropy(double eps_entropy);
    
    double getEpsilonEntropy() const;
    double getEstimatedEntropy() const;
    std::pair<double, double> getEntropyBounds() const;  // (lower, upper)
    
private:
    // Constants (computed once in constructor)
    double epsilon_;
    double bin_entropy_;      // H_bin(epsilon) = -ε*log(ε) - (1-ε)*log(1-ε)
    double entropy_error_;    // bin_entropy / (2*(1-epsilon))
    
    // State (updated per iteration)
    double epsilon_entropy_;      // Raw entropy from device
    double estimated_entropy_;    // After applying corrections
    double lower_bound_;
    double upper_bound_;
    
    void computeEstimation();  // Internal: apply inequalities
};
```

**Key Points:**
- Handles mathematical transformations from perturbed to actual entropy
- Caches computed values (no recomputation unless updated)
- Separate from prediction (which is handled by EntropyPredictor)
- Lightweight: only stores current entropy values, not history

**Mathematical Background:**
- Perturbed channel: Φ_ε(ρ) = (1-ε)Φ(ρ) + ε·I/d
- Entropy bounds using binary entropy function
- Error estimate: entropy_error quantifies approximation quality

---

### 6. EntropyPredictor

**Responsibility**: Predict future entropy values based on historical data

**Interface:**
```cpp
struct PredictionResult {
    double predicted_entropy;
    int predicted_iterations;
    double confidence;  // e.g., R² value
    bool valid;  // false if not enough data or poor fit
};

class EntropyPredictor {
public:
    EntropyPredictor(
        std::unique_ptr<PredictionStrategy> strategy,
        size_t window_size = 200  // Separate from convergence window
    );
    
    void addDataPoint(double entropy);
    void reset();
    
    PredictionResult predict();
    bool hasEnoughData() const;
    
private:
    std::unique_ptr<PredictionStrategy> strategy_;
    CircularBuffer<double> entropy_history_;  // Own buffer, different size than convergence
};

// Strategy interface
class PredictionStrategy {
public:
    virtual ~PredictionStrategy() = default;
    virtual PredictionResult predict(
        const CircularBuffer<double>& history
    ) = 0;
};

// Implementations
class ExponentialFittingStrategy : public PredictionStrategy {
    // Fits exponential decay model to entropy deltas
    // Model: delta[i] ≈ a * exp(b * i)
    // Uses linear regression on log(delta)
    PredictionResult predict(const CircularBuffer<double>& history) override;
};

class LinearExtrapolationStrategy : public PredictionStrategy {
    // Simple linear extrapolation
    PredictionResult predict(const CircularBuffer<double>& history) override;
};
```

**Key Points:**
- Strategy pattern for different prediction models
- Maintains **separate circular buffer** (window_size can differ from convergence window)
- Typical sizes: convergence window = 20, prediction window = 200-500
- Separate from EntropyManager (prediction vs computation)
- Only triggers when enough data available and fit quality is good (R² > threshold)

---

### 7. CheckpointManager

**Responsibility**: Persist and load state

**Interface:**
```cpp
struct CheckpointMetadata {
    std::string run_id;
    int iteration;
    double current_entropy;
    double run_minimum_entropy;
    std::chrono::system_clock::time_point timestamp;
    std::string config_hash;  // For validation
};

class CheckpointManager {
public:
    CheckpointManager(
        const CheckpointConfig& config,
        std::unique_ptr<VectorSerializer> serializer
    );
    
    void saveCheckpoint(
        const std::string& run_id,
        int iteration,
        const HostVector& vector,
        const CheckpointMetadata& metadata
    );
    
    std::pair<HostVector, CheckpointMetadata> loadCheckpoint(
        const std::string& checkpoint_path
    );
    
    // Automatic checkpoint management
    bool shouldCheckpoint(int iteration) const;
    void cleanOldCheckpoints(const std::string& run_id, int keep_last_n = 3);
    
private:
    CheckpointConfig config_;
    std::unique_ptr<VectorSerializer> serializer_;
    
    std::filesystem::path buildPath(const std::string& run_id, int iteration);
};
```

**Key Points:**
- Handles file I/O, path management, metadata
- Uses dependency injection for serialization strategy
- Atomic writes (temp file + rename pattern)
- Can clean old checkpoints (keep only last N)
- Lightweight metadata (no full iteration history)

---

## Resource Ownership Model

### Memory Ownership Rules

#### Kraus Operators
```
main() [owns host copy: std::vector or raw array]
  ↓ (const reference)
MinimizerOrchestrator
  ↓ (const reference)
RunOrchestrator
  ↓ (const reference)
AlgorithmManager
  ↓ (copies to device)
MinimizationStrategy [owns device copy via DeviceMemory<T> RAII wrapper]
```

**Rule**: Host kraus owned by application; device copies owned by MinimizationStrategy (RAII)

#### Current Vector
```
MinimizationStrategy [owns device copy via DeviceMemory<T>]
  ↑ (queries via AlgorithmManager)
RunOrchestrator [can request host copy via AlgorithmManager::getCurrentVector()]
```

**Rule**: Device vector owned by MinimizationStrategy; host copies created on-demand (expensive!)

#### Entropy State
```
MinimizationStrategy [computes entropy on device, stores in device memory]
  ↑ (queries value)
EntropyManager [caches current entropy value on host]
  ↑ (queries)
RunOrchestrator [tracks run_minimum_entropy]
```

**Rule**: Device-side computation, host-side caching in EntropyManager

#### Entropy History Buffers
```
RunOrchestrator [owns CircularBuffer<double> for convergence checking]
  ↓ (const reference passed to)
ConditionsChecker [reads, doesn't own]

EntropyPredictor [owns separate CircularBuffer<double> for prediction]
```

**Rule**: 
- **Two separate buffers** with different purposes and sizes
- Convergence buffer: small (20 entries), owned by RunOrchestrator
- Prediction buffer: larger (200-500 entries), owned by EntropyPredictor
- No full iteration history kept (would be GBs for millions of iterations!)

#### Per-Run State

| State                    | Owner                | Type              | Notes                          |
|--------------------------|----------------------|-------------------|--------------------------------|
| `run_id`                 | RunOrchestrator      | std::string       | UUID for this run              |
| `current_iteration`      | RunOrchestrator      | int               | Counter within this run        |
| `run_minimum_entropy`    | RunOrchestrator      | double            | Best found in THIS run         |
| `entropy_window`         | RunOrchestrator      | CircularBuffer    | Last N=20 for convergence      |
| `current_vector` (device)| MinimizationStrategy | DeviceMemory<T>   | Lives on GPU                   |
| `current_entropy`        | EntropyManager       | double            | Cached host value              |

#### Multi-Run State

| State                    | Owner                    | Type                       | Notes                     |
|--------------------------|--------------------------|----------------------------|---------------------------|
| `global_MOE`             | MinimizerOrchestrator    | double                     | Best across ALL runs      |
| `run_history`            | MinimizerOrchestrator    | std::vector<RunSummary>    | Lightweight summaries     |
| `total_runs_completed`   | MinimizerOrchestrator    | int                        | Run counter               |

**Memory Footprint Example** (for 1M iteration run):
- Full history: 1M × 8 bytes = 8 MB per run
- Circular buffer (N=20): 20 × 8 bytes = 160 bytes
- Prediction buffer (N=500): 500 × 8 bytes = 4 KB
- **Savings**: ~2000x reduction in memory!

---

## Configuration System

### Hierarchical Configuration

```cpp
// Base configs (POD structs, immutable after construction)
struct PlatformConfig {
    int device_id = 0;
    int num_streams = 1;
    DeviceType preferred_device = DeviceType::CUDA;
};

struct OptimizationConfig {
    double epsilon = 1e-3;
    int max_iterations = 500000;
    double convergence_tolerance = 1e-15;
    int convergence_window_size = 20;  // For ConditionsChecker
};

struct PrecisionConfig {
    bool use_adaptive_precision = true;
    bool start_with_float = true;
    double switch_threshold = 1e-6;  // Switch to double when entropy < this
};

struct ConditionsConfig {
    bool check_numerical_instability = true;
    bool check_convergence = true;
    bool enable_prediction = true;
    double prediction_tolerance = 1e-5;
    double prediction_rsquared_threshold = 0.999;
    int prediction_window_size = 200;  // For EntropyPredictor
};

struct CheckpointConfig {
    bool enabled = false;
    int interval = 1000;  // Save every N iterations
    std::string directory = "checkpoints/";
    std::string filename_pattern = "{run_id}_iter{iteration}.dat";
    int keep_last_n = 3;  // Auto-cleanup old checkpoints
};

struct MultiRunConfig {
    int num_attempts = 100;
    bool aggregate_checkpoints = false;  // Save only best run?
    bool stop_on_target = false;  // Stop all runs if one reaches target?
    double target_entropy = -1.0;  // If stop_on_target enabled
};

// Aggregate config
struct MinimizerConfig {
    PlatformConfig platform;
    OptimizationConfig optimization;
    PrecisionConfig precision;
    ConditionsConfig conditions;
    CheckpointConfig checkpoint;
    MultiRunConfig multi_run;
};
```

### ConfigBuilder (Fluent API)

```cpp
class ConfigBuilder {
public:
    // Factory methods
    static ConfigBuilder createDefault();
    static ConfigBuilder createForPhase1();
    static ConfigBuilder fromYAML(const std::string& path);
    
    // Fluent setters (return *this for chaining)
    ConfigBuilder& withEpsilon(double eps);
    ConfigBuilder& withMaxIterations(int iters);
    ConfigBuilder& withDevice(int device_id);
    ConfigBuilder& enableCheckpointing(int interval);
    ConfigBuilder& withNumAttempts(int attempts);
    ConfigBuilder& withConvergenceWindow(int size);
    ConfigBuilder& withPredictionWindow(int size);
    
    // Build (returns immutable config)
    MinimizerConfig build() const;
    
private:
    MinimizerConfig config_;
};
```

### YAMLConfigLoader (Serialization)

```cpp
class YAMLConfigLoader {
public:
    // Load complete config from YAML
    static MinimizerConfig load(const std::string& yaml_path);
    
    // Save config to YAML
    static void save(const MinimizerConfig& config, const std::string& yaml_path);
    
    // Update existing config from YAML (merge)
    static void update(MinimizerConfig& config, const std::string& yaml_path);
};
```

### Usage Patterns

```cpp
// Pattern 1: Programmatic
auto config = ConfigBuilder::createDefault()
    .withEpsilon(1e-3)
    .withMaxIterations(100000)
    .enableCheckpointing(1000)
    .withPredictionWindow(500)
    .build();

// Pattern 2: YAML-based
auto config = YAMLConfigLoader::load("configs/phase1.yml");

// Pattern 3: YAML + overrides
auto config = ConfigBuilder::fromYAML("configs/defaults.yml")
    .withDevice(2)  // Override device
    .withNumAttempts(50)  // Override number of runs
    .build();
```

### Example YAML Config

```yaml
# configs/phase1.yml
platform:
  device_id: 0
  num_streams: 1
  preferred_device: cuda

optimization:
  epsilon: 0.001
  max_iterations: 500000
  convergence_tolerance: 1.0e-15
  convergence_window_size: 20

precision:
  use_adaptive_precision: true
  start_with_float: true
  switch_threshold: 1.0e-6

conditions:
  check_numerical_instability: true
  check_convergence: true
  enable_prediction: true
  prediction_tolerance: 1.0e-5
  prediction_rsquared_threshold: 0.999
  prediction_window_size: 200

checkpoint:
  enabled: true
  interval: 1000
  directory: "checkpoints/"
  filename_pattern: "{run_id}_iter{iteration}.dat"
  keep_last_n: 3

multi_run:
  num_attempts: 100
  aggregate_checkpoints: false
  stop_on_target: false
```

---

## Multi-Run vs Single-Run Strategy

### The Problem

Current EntropyMinimizer has two main workflows:
1. **`runMinimization()`**: Single run to convergence
2. **`findMOE()`**: Multiple runs (N attempts), track global minimum

Attributes like `current_iteration`, `current_entropy` are **per-run**, while `MOE` can be **per-run** or **global**.

Each run can have **millions** of iterations, making full history storage infeasible.

### The Solution: Two-Level Orchestration

#### Level 1: RunOrchestrator (Single Run)
- Manages **one** minimization run
- Tracks: `run_id`, `current_iteration`, `run_minimum_entropy`
- **Circular buffer** for last N entropies (convergence checking)
- Methods: `runToConvergence()`, `runToTarget()`
- Stateful but **reusable** via `reset()`

#### Level 2: MinimizerOrchestrator (Multi-Run)
- Manages **multiple** runs
- Tracks: `global_MOE`, `run_history[]`, `total_runs_completed`
- Stores only **lightweight summaries** per run (not full iteration data)
- Methods: `findMOE()` (runs N times)
- Owns or creates RunOrchestrator instances

### Attribute Allocation

| Attribute               | Old Location      | New Location           | Scope      | Memory/Run    |
|-------------------------|-------------------|------------------------|------------|---------------|
| `kraus_operators`       | EntropyMinimizer  | Strategy (device copy) | Global     | ~MBs (once)   |
| `current_vector`        | Minimizer         | Strategy (device)      | Per-run    | ~KBs          |
| `current_entropy`       | Minimizer         | EntropyManager         | Per-run    | 8 bytes       |
| `current_iteration`     | EntropyMinimizer  | RunOrchestrator        | Per-run    | 4 bytes       |
| `run_id`                | EntropyMinimizer  | RunOrchestrator        | Per-run    | ~40 bytes     |
| `run_minimum_entropy`   | —                 | RunOrchestrator        | Per-run    | 8 bytes       |
| `global_MOE`            | EntropyMinimizer  | MinimizerOrchestrator  | Multi-run  | 8 bytes       |
| `entropy_window`        | EntropyMinimizer  | RunOrchestrator        | Per-run    | 160 bytes (N=20) |
| `prediction_buffer`     | EntropyEstimator  | EntropyPredictor       | Per-run    | 4KB (N=500)   |
| `minimization_attempts` | Config            | MultiRunConfig         | Config     | 4 bytes       |
| **Total per-run state** | —                 | —                      | —          | **~5 KB** ✓   |

### Workflow Examples

#### Single Run
```cpp
// User code
MinimizerOrchestrator orchestrator(kraus, config, msg_handler, device);
RunResult result = orchestrator.runSingleAttempt();

// Internally in MinimizerOrchestrator::runSingleAttempt()
run_orchestrator_->initializeRun();
RunResult result = run_orchestrator_->runToConvergence();
return result;  // Contains: final_entropy, iterations, run_id, stop_reason
```

#### Multi-Run (findMOE)
```cpp
// User code
MOEResult result = orchestrator.findMOE();

// Internally in MinimizerOrchestrator::findMOE()
global_MOE_ = std::numeric_limits<double>::max();

for (int i = 0; i < config_.multi_run.num_attempts; ++i) {
    // Reuse RunOrchestrator (efficient!)
    run_orchestrator_->reset();  // Clears state, keeps components
    run_orchestrator_->initializeRun();
    
    RunResult run_result = run_orchestrator_->runToConvergence();
    
    // Update global state
    if (run_result.run_minimum_entropy < global_MOE_) {
        global_MOE_ = run_result.run_minimum_entropy;
        
        // Optional: checkpoint best run
        if (config_.multi_run.aggregate_checkpoints) {
            saveGlobalBest(run_result);
        }
    }
    
    // Store lightweight summary (not full iteration data!)
    run_history_.push_back(run_result.summarize());
    
    total_runs_completed_++;
}

return MOEResult{global_MOE_, run_history_, total_runs_completed_};
```

### RunOrchestrator::reset() Behavior

```cpp
void RunOrchestrator::reset() {
    // Clear per-run state
    run_id_ = "";  // Will be regenerated on next initializeRun()
    current_iteration_ = 0;
    run_minimum_entropy_ = std::numeric_limits<double>::max();
    
    // Clear buffers
    entropy_window_.clear();
    
    // Reset components (they clear internal state but stay allocated)
    entropy_manager_->reset();
    entropy_predictor_->reset();
    conditions_checker_->reset();
    
    // AlgorithmManager: will reinitialize with new random vector
    // CheckpointManager: stateless, no reset needed
}
```

### Benefits of This Design

1. **Clear Separation**: Run-level vs experiment-level concerns
2. **Reusability**: RunOrchestrator can be reset and reused (avoid allocations)
3. **Memory Efficiency**: Only ~5KB per-run state, no full iteration history
4. **Testability**: Can test single-run logic independently
5. **Extensibility**: Easy to add new multi-run strategies (e.g., adaptive sampling)
6. **No Ambiguity**: `current_iteration` is always "iteration in current run"

### RunSummary (Lightweight)

```cpp
struct RunSummary {
    std::string run_id;
    double run_minimum_entropy;
    int total_iterations;
    StopReason stop_reason;
    double elapsed_seconds;
    
    // No full iteration history!
    // Total size: ~60 bytes
};

struct MOEResult {
    double global_MOE;
    std::vector<RunSummary> run_summaries;  // 100 runs × 60 bytes = 6 KB
    int total_runs;
};
```

---

## Device Management

### Single-Device Scenario

```cpp
// Application creates device
DeviceManager device_manager;
auto device = device_manager.createDevice(0);  // GPU 0

// Create orchestrator
MessageHandler msg_handler;
auto config = ConfigBuilder::createDefault().build();

MinimizerOrchestrator orchestrator(kraus, config, msg_handler, *device);
MOEResult result = orchestrator.findMOE();
```

### Multi-Device Parallelization

For truly parallel execution across multiple GPUs:

```cpp
class MultiDeviceOrchestrator {
public:
    MultiDeviceOrchestrator(
        const HostKrausOperators& kraus,
        const MinimizerConfig& config,
        const std::vector<int>& device_ids,
        MessageHandler& msg_handler
    );
    
    // Parallel findMOE across devices
    MOEResult findMOEParallel();
    
private:
    DeviceManager device_manager_;
    std::vector<std::unique_ptr<IDevice>> devices_;
    std::vector<std::thread> worker_threads_;
    
    // One MinimizerOrchestrator per device
    std::vector<std::unique_ptr<MinimizerOrchestrator>> orchestrators_;
    
    // Thread-safe result aggregation
    std::mutex result_mutex_;
    double global_MOE_;
    std::vector<RunSummary> all_run_summaries_;
};
```

**Pattern**: 
- 100 attempts, 4 GPUs → 25 attempts per GPU
- Each thread gets:
  - Own `MinimizerOrchestrator` instance
  - Own `IDevice` reference (from pool)
  - Subset of total work
- At end: aggregate results (find global minimum across all devices)

**Thread Safety**: No shared state between orchestrators (except final aggregation with mutex)

---

## Performance Considerations

### Memory Management

#### Iteration History Strategy

**Problem**: Millions of iterations × 8 bytes/entropy = GBs of memory

**Solution**: Circular buffers with fixed sizes

```cpp
template<typename T>
class CircularBuffer {
public:
    CircularBuffer(size_t capacity);
    
    void push(T value);  // Overwrites oldest if full
    T get(size_t index) const;  // Relative to current position
    T newest() const;
    T oldest() const;
    
    double average() const;
    double sum() const;
    
    size_t size() const;
    size_t capacity() const;
    bool full() const;
    
private:
    std::vector<T> buffer_;
    size_t head_;
    size_t count_;
    size_t capacity_;
};
```

**Usage**:
- Convergence checking: CircularBuffer<double>(20)
- Prediction: CircularBuffer<double>(200-500)
- Total memory: ~4 KB instead of ~8 MB per run

#### Device-Host Transfer Minimization

**Expensive Operations** (minimize):
- `getCurrentVector()`: Device→Host copy (~100KB-1MB)
- `setCurrentVector()`: Host→Device copy
- `getEntropy()`: Usually cached, but initial query may sync

**Cheap Operations**:
- Algorithm iteration (stays on device)
- Entropy computation (stays on device)
- Circular buffer updates (host-side, tiny)

**Strategy**:
- Only transfer vector for checkpoints or at end of run
- Cache entropy value on host (update per iteration)
- Keep algorithm state on device throughout run

### Iteration Performance

**Target**: ~1000 iterations/second (depends on system size)

**Bottlenecks**:
1. Algorithm step (GPU kernel) - can't avoid
2. Entropy computation (GPU kernel) - can't avoid
3. Circular buffer update (host) - negligible
4. Convergence checking (host) - negligible
5. Logging (if every iteration) - **major bottleneck!**

**Optimizations**:
- Log only every N iterations (e.g., N=100)
- Use async logging (separate thread)
- Minimize device synchronization
- Batch operations where possible

### Checkpoint Performance

**Problem**: Saving checkpoint = sync + host copy + file I/O

**Strategy**:
- Checkpoint infrequently (every 1000-10000 iterations)
- Use async I/O if possible
- Atomic writes (temp file + rename)
- Auto-cleanup old checkpoints

---

## Open Questions

### 1. Checkpoint Aggregation in Multi-Run
**Question**: In `findMOE()` with 100 runs, do we:
- A) Save checkpoint for each run separately?
- B) Only save checkpoint for the best run so far?
- C) Save aggregate statistics periodically?

**Current Thinking**: Option B - only checkpoint when global_MOE improves
**Rationale**: Saves disk space, user typically only cares about best result

### 2. RunOrchestrator Lifecycle
**Question**: Should MinimizerOrchestrator:
- A) Own one RunOrchestrator and call `reset()` between runs?
- B) Create fresh RunOrchestrator for each run?

**Current Thinking**: Option A for efficiency (reuse components)
**Rationale**: Avoid repeated allocations (important for 100+ runs)

### 3. Precision Switching
**Question**: Should precision switching (float→double) be:
- A) Inside MinimizationStrategy (current approach)?
- B) Managed by a separate PrecisionManager?
- C) Handled by AlgorithmManager?

**Current Thinking**: Option A, but exposed via AlgorithmManager interface
**Rationale**: Strategy knows internal state, can decide when to switch

### 4. Entropy History Ownership
**Question**: EntropyPredictor and ConditionsChecker both need entropy history:
- A) Each maintains own buffer?
- B) Share reference to RunOrchestrator's buffer?
- C) Separate HistoryManager component?

**Current Thinking**: Option A (each has own circular buffer)
**Rationale**: Different window sizes, different update frequencies, small memory cost

### 5. Logging Frequency
**Question**: For runs with millions of iterations, log:
- A) Every iteration?
- B) Every N iterations (N=100-1000)?
- C) Adaptive (more frequent early, less frequent later)?

**Current Thinking**: Option B with configurable N
**Rationale**: Balance between visibility and performance

---

## Implementation Priority

### Phase 1: Core Components (Week 1-2)
1. Implement `CircularBuffer<T>` utility
2. Implement `EntropyManager`
3. Implement `ConditionsChecker` + concrete Condition classes
4. Implement `EntropyPredictor` + ExponentialFittingStrategy

### Phase 2: Orchestration (Week 2-3)
1. Implement `RunOrchestrator`
2. Implement `MinimizerOrchestrator`
3. Implement `CheckpointManager`

### Phase 3: Algorithm Integration (Week 3-4)
1. Implement `AlgorithmManager`
2. Refactor existing `MinimizationStrategy` to fit new interface
3. Integration testing

### Phase 4: Configuration & I/O (Week 4-5)
1. Implement `ConfigBuilder`
2. Implement `YAMLConfigLoader`
3. Update main() to use new architecture

### Phase 5: Multi-Device Support (Week 5-6)
1. Implement `DeviceManager`
2. Implement `MultiDeviceOrchestrator`
3. Thread safety testing

---

## Migration Notes

### From Current EntropyMinimizer

```cpp
// OLD (god class)
EntropyMinimizer minimizer(kraus, d, N, M, &config);
minimizer.initializeRun();
minimizer.findMOE();
double final_moe = minimizer.MOE;

// NEW (decomposed)
auto config = ConfigBuilder::createDefault()
    .withEpsilon(1e-3)
    .withNumAttempts(100)
    .build();

DeviceManager device_manager;
auto device = device_manager.createDevice(0);
MessageHandler msg_handler;

MinimizerOrchestrator orchestrator(kraus, config, msg_handler, *device);
MOEResult result = orchestrator.findMOE();

std::cout << "Global MOE: " << result.global_MOE << std::endl;
std::cout << "Total runs: " << result.total_runs << std::endl;
```

### Key Differences
- Configuration is immutable struct (not pointer)
- Device is explicitly created and passed
- MessageHandler is dependency-injected
- Return values instead of side effects where possible
- Clear ownership and lifecycle management
- Memory-efficient circular buffers instead of full history

---

## Appendix: Class Dependency Graph

```
main()
  └── creates
       ├── DeviceManager
       ├── MessageHandler
       └── MinimizerOrchestrator
            ├── owns RunOrchestrator
            │    ├── owns AlgorithmManager
            │    │    └── references IDevice
            │    ├── owns ConditionsChecker
            │    │    └── owns vector<Condition*>
            │    ├── owns EntropyManager
            │    ├── owns EntropyPredictor
            │    │    └── owns PredictionStrategy
            │    └── owns CheckpointManager
            │         └── owns VectorSerializer
            └── references MessageHandler
```

---

**End of Architecture Document**
