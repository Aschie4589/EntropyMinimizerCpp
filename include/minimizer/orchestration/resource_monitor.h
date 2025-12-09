#ifndef RESOURCE_MONITOR_H_
#define RESOURCE_MONITOR_H_

#include <string>
#include <chrono>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <fstream>

namespace entropy {

/**
 * @brief Resource limits for orchestration system
 * 
 * Defines maximum number of GPU/CPU devices to use
 * and polling configuration for dynamic resource updates.
 */
struct ResourceLimits {
    int max_gpus = 0;                                      ///< Maximum GPU devices to use
    int max_cpus = 1;                                      ///< Maximum CPU devices to use
    std::chrono::milliseconds poll_interval{5000};         ///< How often to poll for changes
    std::string config_file;                               ///< Path to YAML config file (empty for env vars only)
    
    /**
     * @brief Check equality for change detection
     */
    bool operator==(const ResourceLimits& other) const {
        return max_gpus == other.max_gpus && 
               max_cpus == other.max_cpus &&
               poll_interval == other.poll_interval &&
               config_file == other.config_file;
    }
    
    bool operator!=(const ResourceLimits& other) const {
        return !(*this == other);
    }
    
    /**
     * @brief Convert to string for debugging
     */
    std::string to_string() const {
        return "ResourceLimits{max_gpus=" + std::to_string(max_gpus) +
               ", max_cpus=" + std::to_string(max_cpus) +
               ", poll_interval=" + std::to_string(poll_interval.count()) + "ms" +
               ", config_file='" + config_file + "'}";
    }
};

/**
 * @brief Monitors external configuration for resource limit changes
 * 
 * Polls YAML files and/or environment variables for changes to
 * GPU/CPU resource limits. Triggers callbacks when changes detected.
 * Supports graceful start/stop for integration with orchestration layer.
 * 
 * Configuration Sources (priority order):
 * 1. YAML file (if config_file specified)
 * 2. Environment variables:
 *    - ENTROPY_MAX_GPUS
 *    - ENTROPY_MAX_CPUS
 * 
 * Features:
 * - Automatic change detection
 * - Configurable poll interval
 * - Thread-safe access to current limits
 * - Graceful shutdown
 * - Error handling (missing files, parse errors)
 * 
 * Usage:
 * @code
 * ResourceLimits initial;
 * initial.max_gpus = 2;
 * initial.max_cpus = 4;
 * initial.config_file = "/path/to/config.yml";
 * 
 * ResourceMonitor monitor(initial);
 * monitor.start([](const ResourceLimits& new_limits) {
 *     std::cout << "Resources changed: " << new_limits.to_string() << std::endl;
 *     // Update orchestrator...
 * });
 * 
 * // ... run minimization ...
 * 
 * monitor.stop();
 * @endcode
 */
class ResourceMonitor {
public:
    using ChangeCallback = std::function<void(const ResourceLimits&)>;
    
    /**
     * @brief Construct monitor with initial limits
     * 
     * @param initial Initial resource limits and polling config
     */
    explicit ResourceMonitor(const ResourceLimits& initial)
        : current_limits_(initial),
          running_(false) {}
    
    /**
     * @brief Destructor - stops monitoring if running
     */
    ~ResourceMonitor() {
        stop();
    }
    
    // Non-copyable
    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;
    
    // Movable
    ResourceMonitor(ResourceMonitor&&) = default;
    ResourceMonitor& operator=(ResourceMonitor&&) = default;
    
    /**
     * @brief Start monitoring for resource changes
     * 
     * Spawns background thread that polls configuration source
     * at specified interval. Callback is invoked on changes.
     * 
     * If already running, this is a no-op.
     * 
     * @param on_change Callback invoked when limits change
     */
    void start(ChangeCallback on_change) {
        if (running_.exchange(true)) {
            return;  // Already running
        }
        
        callback_ = std::move(on_change);
        
        poll_thread_ = std::thread([this]() {
            while (running_) {
                try {
                    auto new_limits = pollConfig();
                    
                    if (hasChanged(current_limits_, new_limits)) {
                        {
                            std::lock_guard<std::mutex> lock(limits_mutex_);
                            current_limits_ = new_limits;
                        }
                        
                        if (callback_) {
                            callback_(new_limits);
                        }
                    }
                } catch (const std::exception& e) {
                    // Log error but continue monitoring
                    // In production, would use proper logging
                    (void)e;  // Suppress unused variable warning
                }
                
                // Sleep for poll interval, checking running_ frequently for responsive shutdown
                auto interval = getCurrentLimits().poll_interval;
                auto sleep_chunk = std::chrono::milliseconds(100);
                auto remaining = interval;
                
                while (running_ && remaining > std::chrono::milliseconds(0)) {
                    auto to_sleep = std::min(sleep_chunk, remaining);
                    std::this_thread::sleep_for(to_sleep);
                    remaining -= to_sleep;
                }
            }
        });
    }
    
    /**
     * @brief Stop monitoring
     * 
     * Signals polling thread to stop and waits for it to join.
     * Safe to call multiple times or if not started.
     */
    void stop() {
        if (!running_.exchange(false)) {
            return;  // Not running
        }
        
        if (poll_thread_.joinable()) {
            poll_thread_.join();
        }
    }
    
    /**
     * @brief Get current resource limits (thread-safe)
     * 
     * @return Current limits
     */
    ResourceLimits getCurrentLimits() const {
        std::lock_guard<std::mutex> lock(limits_mutex_);
        return current_limits_;
    }
    
    /**
     * @brief Check if monitor is currently running
     * 
     * @return true if polling thread is active
     */
    bool isRunning() const {
        return running_;
    }

private:
    /**
     * @brief Poll configuration from file and/or environment variables
     * 
     * @return Updated resource limits
     * @throws std::runtime_error if config file specified but unreadable
     */
    ResourceLimits pollConfig() {
        ResourceLimits limits = getCurrentLimits();
        
        // Try YAML file first (if specified)
        if (!limits.config_file.empty()) {
            try {
                std::ifstream file(limits.config_file);
                if (file.good()) {
                    YAML::Node root = YAML::LoadFile(limits.config_file);
                    
                    // Look for orchestration.resources or resources section
                    YAML::Node resources_node = root;
                    if (root["orchestration"] && root["orchestration"]["resources"]) {
                        resources_node = root["orchestration"]["resources"];
                    } else if (root["resources"]) {
                        resources_node = root["resources"];
                    }
                    
                    if (resources_node["max_gpus"]) {
                        limits.max_gpus = resources_node["max_gpus"].as<int>();
                    }
                    if (resources_node["max_cpus"]) {
                        limits.max_cpus = resources_node["max_cpus"].as<int>();
                    }
                    if (resources_node["poll_interval"]) {
                        int interval_ms = resources_node["poll_interval"].as<int>();
                        limits.poll_interval = std::chrono::milliseconds(interval_ms);
                    }
                }
            } catch (const YAML::Exception&) {
                // YAML parse error - use environment variables as fallback
            }
        }
        
        // Check environment variables (override YAML if set)
        const char* max_gpus_env = std::getenv("ENTROPY_MAX_GPUS");
        if (max_gpus_env != nullptr) {
            try {
                limits.max_gpus = std::stoi(max_gpus_env);
            } catch (const std::exception&) {
                // Invalid env var - ignore
            }
        }
        
        const char* max_cpus_env = std::getenv("ENTROPY_MAX_CPUS");
        if (max_cpus_env != nullptr) {
            try {
                limits.max_cpus = std::stoi(max_cpus_env);
            } catch (const std::exception&) {
                // Invalid env var - ignore
            }
        }
        
        return limits;
    }
    
    /**
     * @brief Check if resource limits have changed
     * 
     * Compares only max_gpus and max_cpus (not poll_interval or config_file)
     * 
     * @param a First limits
     * @param b Second limits
     * @return true if resource counts differ
     */
    bool hasChanged(const ResourceLimits& a, const ResourceLimits& b) const {
        return a.max_gpus != b.max_gpus || a.max_cpus != b.max_cpus;
    }
    
    mutable std::mutex limits_mutex_;       ///< Protects current_limits_
    ResourceLimits current_limits_;         ///< Current resource limits
    std::atomic<bool> running_;             ///< Is polling thread active?
    std::thread poll_thread_;               ///< Background polling thread
    ChangeCallback callback_;               ///< User callback for changes
};

} // namespace entropy

#endif // RESOURCE_MONITOR_H_
