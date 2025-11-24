# EntropyMinimizerCpp Refactoring Plan

## Overview
Complete refactoring strategy to address critical code quality issues:
- Manual memory management → RAII
- CUDA lock-in → Platform abstraction
- God class → Focused components
- No tests → Comprehensive testing

## Phase 1: Foundation - RAII & Error Handling (Week 1-2)

### 1.1 RAII Wrappers for CUDA Resources
**Priority**: 🔴 CRITICAL
**Dependency**: None
**Effort**: Medium

Create wrappers:
- `CudaDeviceMemory<T>` - replaces cudaMalloc/cudaFree
- `CudaStream` - replaces raw cudaStream_t
- `CudaSolverHandle` - replaces cusolverDnHandle_t
- `CudaBlasHandle` - replaces cublasHandle_t
- `CurandStateArray` - replaces raw curandState*

### 1.2 Standardize Error Handling
**Priority**: 🔴 CRITICAL
**Dependency**: None
**Effort**: Low-Medium

- Create `CUDA_CHECK(call)` macro
- All CUDA errors → throw std::runtime_error
- Consistent exception handling throughout

## Phase 2: Compute Abstraction Layer (Week 2-4)

### 2.1 Core Interfaces
Create interface hierarchy:
- `IComputeDevice` - Top-level device interface
- `IDeviceMemory<T>` - Generic device memory
- `IComputeStream` - Async execution
- `ILinearAlgebra<T>` - QR, matmul, etc.
- `IRandomGenerator<T>` - RNG operations

### 2.2 CUDA Backend Implementation
Implement CUDA-specific classes using Phase 1 RAII wrappers

### 2.3 CPU Fallback Backend
Implement CPU backend using Eigen/LAPACKE for testing without GPU

### 2.4 Device Factory
Create factory for auto-detection and device creation

## Phase 3: Decompose EntropyMinimizer God Class (Week 4-6)

Extract components:
- `MinimizationOrchestrator` - Workflow coordination
- `PrecisionManager` - Float/double switching
- `ConvergenceMonitor` - Stopping criteria
- `CheckpointManager` - Save/load state
- `IterationTracker` - Progress tracking
- `SignalHandler` - SIGTERM/SIGINT handling
- `CudaMinimizerFactory` - Create minimizer instances

## Phase 4: Configuration & Testability (Week 6-7)

### 4.1 Unified Configuration System
Replace scattered config with runtime YAML-based configuration

### 4.2 Add Unit Tests
Create comprehensive test suite using Google Test

## Phase 5: Advanced Improvements (Week 7+)

- Type safety improvements
- Better async/concurrent processing
- ROCm backend (AMD GPU support)

## Implementation Strategy

**Iterative Approach**:
1. Week 1-2: RAII Foundation
2. Week 3-4: Start Abstraction
3. Week 5-6: Break Up God Class
4. Week 7+: Test & Refine

**Key Principles**:
- Small steps
- Keep it working
- Test as you go
- Parallel development where possible
- Document decisions