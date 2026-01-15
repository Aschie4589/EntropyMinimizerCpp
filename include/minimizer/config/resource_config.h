#ifndef RESOURCE_CONFIG_H_
#define RESOURCE_CONFIG_H_

#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <sstream>

namespace entropy {

/**
 * @brief Configuration for dynamic resource allocation
 * 
 * Replaces simple ResourceLimits with rich configuration for:
 * - User-specified GPU/CPU counts
 * - GPU selection policies
 * - Dynamic scaling behavior
 * - Configuration file paths
 * - Fallback strategies
 * 
 * This structure is backward compatible with ResourceLimits through
 * default values and factory methods.
 * 
 * Design:
 * - Immutable after construction (pass by const reference)
 * - Validated on construction
 * - Extensible for future policies
 * 
 * Usage:
 * @code
 * ResourceConfig config;
 * config.desired_gpus = 2;
 * config.desired_cpus = 4;
 * config.gpu_policy = ResourceConfig::GPUSelectionPolicy::MOST_FREE_MEMORY;
 * config.allow_dynamic_scaling = true;
 * config.validate();
 * 
 * MinimizerOrchestrator orchestrator;
 * auto result = orchestrator.findMOE(minimizer_config, kraus_ops, input_dim, config);
 * @endcode
 */
struct ResourceConfig {
    // =========================================================================
    // Resource Requests
    // =========================================================================
    
    /**
     * @brief Number of GPUs to use
     * 
     * - 0: No GPUs (CPU only, unless fallback_to_cpu=false)
     * - N: Request N GPUs from DeviceRegistry
     * 
     * Actual allocation depends on:
     * - Available GPUs in gpu_config.yaml
     * - GPU selection policy
     * - Memory requirements
     */
    int desired_gpus = 0;
    
    /**
     * @brief Number of CPU workers to use
     * 
     * - 0: No CPU workers
     * - N: Create N CPU device instances
     * 
     * Note: CPU workers are always created as requested
     * (no dynamic selection like GPUs)
     */
    int desired_cpus = 1;
    
    // =========================================================================
    // GPU Selection Policy
    // =========================================================================
    
    /**
     * @brief Policy for selecting which GPUs to use
     */
    enum class GPUSelectionPolicy {
        /**
         * @brief Pick GPUs with most free memory (default)
         * 
         * Queries cudaMemGetInfo and sorts by free memory descending.
         * Best for memory-intensive workloads.
         */
        MOST_FREE_MEMORY,
        
        /**
         * @brief Pick GPUs with lowest utilization
         * 
         * Requires NVML integration (future enhancement).
         * Falls back to MOST_FREE_MEMORY if NVML unavailable.
         */
        LEAST_UTILIZED,
        
        /**
         * @brief Distribute evenly across all enabled GPUs
         * 
         * Round-robin assignment from enabled GPU list.
         * Good for balanced multi-GPU systems.
         */
        ROUND_ROBIN,
        
        /**
         * @brief Use specific GPUs in order of preference
         * 
         * Requires preferred_gpu_ids to be populated.
         * Falls back to MOST_FREE_MEMORY if preferred GPUs unavailable.
         */
        PREFER_SPECIFIC
    };
    
    /**
     * @brief GPU selection policy to use
     */
    GPUSelectionPolicy gpu_policy = GPUSelectionPolicy::MOST_FREE_MEMORY;
    
    /**
     * @brief Preferred GPU order (for PREFER_SPECIFIC policy)
     * 
     * Example: {2, 0, 1} = prefer GPU 2, then 0, then 1
     */
    std::vector<int> preferred_gpu_ids;
    
    // =========================================================================
    // Memory Requirements
    // =========================================================================
    
    /**
     * @brief Minimum free memory per GPU (bytes)
     * 
     * GPUs with less free memory are not selected.
     * Set to 0 for no requirement.
     * 
     * Example: 1024*1024*1024 = 1GB minimum
     */
    size_t min_gpu_memory = 0;
    
    // =========================================================================
    // Dynamic Behavior
    // =========================================================================
    
    /**
     * @brief Enable runtime device adjustment
     * 
     * If true:
     * - DeviceRegistry watches gpu_config.yaml for changes
     * - MinimizerOrchestrator adjusts workers dynamically
     * - Workers are added/removed as GPUs become available/unavailable
     * 
     * If false:
     * - Static device allocation at startup
     * - No runtime adjustments
     * - Lower overhead
     */
    bool allow_dynamic_scaling = true;
    
    /**
     * @brief Polling interval for config file changes
     * 
     * Only used if allow_dynamic_scaling=true.
     * Lower values = faster response, higher I/O overhead.
     */
    std::chrono::milliseconds poll_interval{1000};
    
    // =========================================================================
    // Configuration Sources
    // =========================================================================
    
    /**
     * @brief Path to GPU configuration file
     * 
     * Default: "configs/gpu_config.yaml"
     * Format: see gpu_config.yaml documentation
     */
    std::string gpu_config_file = "configs/gpu_config.yaml";
    
    // =========================================================================
    // Fallback Behavior
    // =========================================================================
    
    /**
     * @brief Use CPU if no GPUs available
     * 
     * If true and desired_gpus > 0 but no GPUs available:
     * - Create at least one CPU worker
     * - Execution continues (slower)
     * 
     * If false and no GPUs available:
     * - Throw exception
     */
    bool fallback_to_cpu = true;
    
    // =========================================================================
    // Validation
    // =========================================================================
    
    /**
     * @brief Validate configuration
     * 
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (desired_gpus < 0) {
            throw std::invalid_argument("ResourceConfig::validate: desired_gpus must be >= 0");
        }
        if (desired_cpus < 0) {
            throw std::invalid_argument("ResourceConfig::validate: desired_cpus must be >= 0");
        }
        if (desired_gpus == 0 && desired_cpus == 0 && !fallback_to_cpu) {
            throw std::invalid_argument("ResourceConfig::validate: At least one device required (or enable fallback_to_cpu)");
        }
        if (min_gpu_memory < 0) {
            throw std::invalid_argument("ResourceConfig::validate: min_gpu_memory must be >= 0");
        }
        if (poll_interval.count() <= 0) {
            throw std::invalid_argument("ResourceConfig::validate: poll_interval must be > 0");
        }
        if (gpu_config_file.empty()) {
            throw std::invalid_argument("ResourceConfig::validate: gpu_config_file cannot be empty");
        }
    }
    
    /**
     * @brief Convert to string for debugging
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "ResourceConfig{"
            << "desired_gpus=" << desired_gpus
            << ", desired_cpus=" << desired_cpus
            << ", gpu_policy=" << static_cast<int>(gpu_policy);
        
        if (!preferred_gpu_ids.empty()) {
            oss << ", preferred_gpu_ids=[";
            for (size_t i = 0; i < preferred_gpu_ids.size(); ++i) {
                if (i > 0) oss << ",";
                oss << preferred_gpu_ids[i];
            }
            oss << "]";
        }
        
        oss << ", min_gpu_memory=" << (min_gpu_memory / (1024*1024)) << "MB"
            << ", allow_dynamic=" << allow_dynamic_scaling
            << ", poll_interval=" << poll_interval.count() << "ms"
            << ", fallback_to_cpu=" << fallback_to_cpu
            << ", gpu_config_file='" << gpu_config_file << "'}";
        
        return oss.str();
    }
    
    // =========================================================================
    // Factory Methods
    // =========================================================================
    
    /**
     * @brief Create default configuration (backward compatible)
     * 
     * Matches old ResourceLimits behavior:
     * - 1 GPU + 1 CPU
     * - No dynamic scaling
     * - CPU fallback enabled
     */
    static ResourceConfig createDefault() {
        ResourceConfig config;
        config.desired_gpus = 1;
        config.desired_cpus = 1;
        config.allow_dynamic_scaling = false;  // Static by default for compatibility
        config.fallback_to_cpu = true;
        return config;
    }
    
    /**
     * @brief Create GPU-only configuration
     * 
     * @param num_gpus Number of GPUs to use
     * @param allow_dynamic Enable dynamic scaling
     */
    static ResourceConfig createGPUOnly(int num_gpus, bool allow_dynamic = false) {
        ResourceConfig config;
        config.desired_gpus = num_gpus;
        config.desired_cpus = 0;
        config.allow_dynamic_scaling = allow_dynamic;
        config.fallback_to_cpu = true;
        return config;
    }
    
    /**
     * @brief Create CPU-only configuration
     * 
     * @param num_cpus Number of CPU workers
     */
    static ResourceConfig createCPUOnly(int num_cpus) {
        ResourceConfig config;
        config.desired_gpus = 0;
        config.desired_cpus = num_cpus;
        config.allow_dynamic_scaling = false;
        config.fallback_to_cpu = true;
        return config;
    }
    
    /**
     * @brief Create dynamic configuration with specific GPUs
     * 
     * @param preferred_gpus List of preferred GPU device IDs
     * @param num_cpus Number of CPU workers
     */
    static ResourceConfig createWithPreferredGPUs(
        const std::vector<int>& preferred_gpus,
        int num_cpus = 0
    ) {
        ResourceConfig config;
        config.desired_gpus = preferred_gpus.size();
        config.desired_cpus = num_cpus;
        config.gpu_policy = GPUSelectionPolicy::PREFER_SPECIFIC;
        config.preferred_gpu_ids = preferred_gpus;
        config.allow_dynamic_scaling = false;
        config.fallback_to_cpu = true;
        return config;
    }
};

} // namespace entropy

#endif // RESOURCE_CONFIG_H_
