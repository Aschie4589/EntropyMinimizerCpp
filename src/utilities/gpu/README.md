# GPU Resource Manager

Dynamic GPU allocation and management for multi-GPU systems.

## Features

- **Runtime GPU enable/disable** via config file or API
- **Memory-aware allocation** - checks free memory before assigning
- **Thread-safe** - safe for concurrent GPU requests
- **Auto-config reload** - monitors config file for changes (default: every 10s)
- **Load balancing** - distributes jobs across available GPUs
- **Blocking wait** - `waitForGPU()` blocks until GPU becomes available

## Quick Start

### Basic Usage

```cpp
#include "utilities/gpu/gpu_resource_manager.h"

GPUResourceManager mgr;

// Acquire a GPU
int gpu_id = mgr.acquireGPU();
if (gpu_id != -1) {
    cudaSetDevice(gpu_id);
    // ... do work ...
    mgr.releaseGPU(gpu_id);
}
```

### With Config File

```cpp
// Use config file for dynamic control
GPUResourceManager mgr("configs/gpu_config.txt");

// Config file is checked every 10 seconds
// Edit configs/gpu_config.txt while program runs to enable/disable GPUs
```

### Runtime Control

```cpp
GPUResourceManager mgr;

// Manually disable GPU (e.g., another user needs it)
mgr.disableGPU(0);

// Later, re-enable it
mgr.enableGPU(0);

// Check status
std::cout << mgr.getStatusString();
```

### Multi-threaded Usage

```cpp
GPUResourceManager mgr;

// Multiple threads can safely request GPUs
std::vector<std::thread> workers;
for (int i = 0; i < 10; i++) {
    workers.emplace_back([&mgr, i]() {
        int gpu = mgr.waitForGPU();  // Blocks until available
        cudaSetDevice(gpu);
        // ... work ...
        mgr.releaseGPU(gpu);
    });
}
```

## Config File Format

`configs/gpu_config.txt`:
```
# Comment lines start with #
GPU 0: enabled
GPU 1: disabled    # Reserved for other user
GPU 2: enabled
GPU 3: enabled
```

Valid statuses: `enabled`, `disabled` (case-insensitive)

## API Reference

### Constructor
```cpp
GPUResourceManager(
    const std::string& config_file = "",      // Path to config file
    size_t min_free_memory = 5GB,             // Minimum free memory required
    int check_interval = 10                   // Config reload interval (seconds)
);
```

### Acquisition/Release
```cpp
int acquireGPU(size_t required_memory = 0);  // Returns GPU ID or -1
void releaseGPU(int gpu_id);
int waitForGPU(int timeout_seconds = 0);     // Blocking wait
```

### Runtime Control
```cpp
void enableGPU(int gpu_id);
void disableGPU(int gpu_id, bool force = false);
void reloadConfig();                          // Force config reload
```

### Status/Info
```cpp
int getAvailableGPUCount();
bool isGPUAvailable(int gpu_id);
std::string getStatusString();                // Detailed status report
```

## Use Cases

### 1. Shared Cluster Environment
Users can dynamically reserve GPUs without restarting jobs:
```bash
# User A is running phase1
# User B needs GPU 0 urgently

# Edit configs/gpu_config.txt:
GPU 0: disabled

# Within 10 seconds, phase1 stops using GPU 0
# (current job on GPU 0 finishes, but no new jobs assigned)
```

### 2. Adaptive Resource Allocation
```cpp
// Check available GPUs and adapt batch size
int num_gpus = mgr.getAvailableGPUCount();
int batch_size = num_gpus * CHANNELS_PER_GPU;

// Process adaptively
for (int i = 0; i < total_channels; i += batch_size) {
    // ... parallel processing ...
}
```

### 3. Memory-Sensitive Tasks
```cpp
// Require specific memory amount
size_t required = 10ULL * 1024 * 1024 * 1024;  // 10GB
int gpu = mgr.acquireGPU(required);
```

## Files

- `gpu_resource_manager.h` - Header file
- `gpu_resource_manager.cu` - Implementation
- `example_usage.cu` - Example program
- `../../configs/gpu_config.txt` - Example config file

## Placement Rationale

Located in `utilities/gpu/` because:
- **Not config**: Active resource management, not just configuration
- **GPU-specific**: Uses CUDA APIs for memory queries
- **Utility role**: Supporting infrastructure for main compute code
- **Reusable**: Can be used across different phases/executables

Alternative locations considered:
- `src/utilities/config/` ❌ - Not purely configuration
- `src/runtime/` ❌ - No such category exists
- `src/gpu/` ❌ - Too top-level for a utility
- `src/utilities/resource/` ⚠️ - Could work, but GPU-specific nature fits better

## Building

The library is automatically built as part of the utilities module:
```bash
cd build
cmake ..
make gpu_utilities
```

## Example Program

Build and run the example:
```bash
cd build
make
./src/utilities/gpu/example_usage
```
