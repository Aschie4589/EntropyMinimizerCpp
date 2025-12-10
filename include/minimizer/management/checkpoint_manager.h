#ifndef CHECKPOINT_MANAGER_H
#define CHECKPOINT_MANAGER_H

#include "minimizer/config/checkpoint_config.h"
#include "minimizer/orchestration/types.h"
#include <string>
#include <chrono>
#include <memory>
#include <filesystem>
#include <functional>

namespace entropy {

// Forward declaration
class VectorSerializer;

/**
 * @brief Metadata stored with each checkpoint
 */
struct CheckpointMetadata {
    std::string run_id;
    int iteration;
    double current_entropy;
    double run_minimum_entropy;
    std::chrono::system_clock::time_point timestamp;
    std::string config_hash;  // For validation (optional)
    
    CheckpointMetadata() 
        : iteration(0), 
          current_entropy(0.0), 
          run_minimum_entropy(0.0),
          timestamp(std::chrono::system_clock::now()) {}
};

/**
 * @brief Manages checkpoint saving, loading, and cleanup
 * 
 * Responsibilities:
 * - Save quantum state vectors to disk with metadata
 * - Load checkpoints for run resumption
 * - Automatic cleanup of old checkpoints
 * - Atomic file writes (temp + rename)
 * - Path management and directory creation
 * 
 * Thread-safety: Not thread-safe. Each RunOrchestrator should have its own instance.
 */
class CheckpointManager {
public:
    /**
     * @brief Construct checkpoint manager
     * @param config Checkpoint configuration
     * @param serializer Vector serialization strategy (injected dependency)
     */
    CheckpointManager(
        const CheckpointConfig& config,
        std::unique_ptr<VectorSerializer> serializer
    );
    
    ~CheckpointManager();
    
    // Disable copy (owns unique resources)
    CheckpointManager(const CheckpointManager&) = delete;
    CheckpointManager& operator=(const CheckpointManager&) = delete;
    
    // Enable move
    CheckpointManager(CheckpointManager&&) noexcept;
    CheckpointManager& operator=(CheckpointManager&&) noexcept;
    
    /**
     * @brief Save checkpoint to disk
     * @param run_id Unique identifier for this run
     * @param iteration Current iteration number
     * @param vector Current quantum state vector
     * @param metadata Additional metadata (entropy values, timestamps, etc.)
     * @throws std::runtime_error if save fails
     */
    void saveCheckpoint(
        const std::string& run_id,
        int iteration,
        const HostVector& vector,
        const CheckpointMetadata& metadata
    );
    
    /**
     * @brief Load checkpoint from disk
     * @param checkpoint_path Full path to checkpoint file
     * @return Pair of (vector, metadata)
     * @throws std::runtime_error if load fails or file doesn't exist
     */
    std::pair<HostVector, CheckpointMetadata> loadCheckpoint(
        const std::string& checkpoint_path
    );
    
    /**
     * @brief Check if checkpoint should be saved at this iteration
     * @param iteration Current iteration number
     * @return true if checkpoint should be saved (based on interval)
     */
    bool shouldCheckpoint(int iteration) const;
    
    /**
     * @brief Clean old checkpoints for a run, keeping only the last N
     * @param run_id Run identifier to clean checkpoints for
     * @param keep_last_n Number of recent checkpoints to keep (0 = keep all)
     * 
     * Note: If keep_last_n is 0 or not provided, uses config.keep_last_n
     */
    void cleanOldCheckpoints(const std::string& run_id, int keep_last_n = -1);
    
    /**
     * @brief Get the checkpoint directory path
     * @return Filesystem path to checkpoint directory
     */
    std::filesystem::path getCheckpointDirectory() const;
    
    /**
     * @brief List all checkpoints for a given run
     * @param run_id Run identifier
     * @return Vector of checkpoint file paths, sorted by iteration
     */
    std::vector<std::filesystem::path> listCheckpoints(const std::string& run_id) const;
    
    /**
     * @brief Check if checkpointing is enabled
     * @return true if enabled
     */
    bool isEnabled() const { return config_.enabled; }
    
private:
    /**
     * @brief Build full checkpoint file path
     * @param run_id Run identifier
     * @param iteration Iteration number
     * @return Full filesystem path
     */
    std::filesystem::path buildPath(const std::string& run_id, int iteration) const;
    
    /**
     * @brief Ensure checkpoint directory exists, create if needed
     * @throws std::runtime_error if directory creation fails
     */
    void ensureDirectoryExists();
    
    /**
     * @brief Perform atomic write (write to temp, then rename)
     * @param final_path Target file path
     * @param write_func Function that performs the actual write
     */
    void atomicWrite(
        const std::filesystem::path& final_path,
        std::function<void(const std::string&)> write_func
    );
    
    /**
     * @brief Extract iteration number from checkpoint filename
     * @param path Checkpoint file path
     * @return Iteration number, or -1 if not found
     */
    int extractIterationFromPath(const std::filesystem::path& path) const;
    
    CheckpointConfig config_;
    std::unique_ptr<VectorSerializer> serializer_;
    bool directory_checked_;  // Cache flag to avoid repeated checks
};

} // namespace entropy

#endif // CHECKPOINT_MANAGER_H
