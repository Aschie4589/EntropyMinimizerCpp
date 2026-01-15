#include "minimizer/management/checkpoint_manager.h"
#include "utilities/serializer/serializer.h"
#include "utilities/uuid/uuid.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <ctime>
#include <regex>

namespace entropy {

// ============================================================================
// Constructor / Destructor
// ============================================================================

CheckpointManager::CheckpointManager(
    const CheckpointConfig& config,
    std::unique_ptr<utils::VectorSerializer> serializer
)
    : config_(config),
      serializer_(std::move(serializer)),
      directory_checked_(false)
{
    if (!serializer_) {
        throw std::invalid_argument("CheckpointManager: serializer cannot be null");
    }
    
    // Create directory on construction if enabled
    if (config_.enabled) {
        ensureDirectoryExists();
    }
}

CheckpointManager::~CheckpointManager() = default;

CheckpointManager::CheckpointManager(CheckpointManager&&) noexcept = default;

CheckpointManager& CheckpointManager::operator=(CheckpointManager&&) noexcept = default;

// ============================================================================
// Public Methods
// ============================================================================

void CheckpointManager::saveCheckpoint(
    const std::string& run_id,
    int iteration,
    const HostVector& vector,
    const CheckpointMetadata& metadata
)
{
    if (!config_.enabled) {
        return;  // Silently skip if disabled
    }
    
    if (run_id.empty()) {
        throw std::invalid_argument("CheckpointManager::saveCheckpoint: run_id cannot be empty");
    }
    
    if (iteration < 0) {
        throw std::invalid_argument("CheckpointManager::saveCheckpoint: iteration must be non-negative");
    }
    
    if (vector.data.empty()) {
        throw std::invalid_argument("CheckpointManager::saveCheckpoint: vector cannot be empty");
    }
    
    ensureDirectoryExists();
    
    auto checkpoint_path = buildPath(run_id, iteration);
    
    // Create metadata JSON for serialization
    std::string description = "Checkpoint: run_id=" + run_id + 
                             ", iteration=" + std::to_string(iteration) +
                             ", entropy=" + std::to_string(metadata.current_entropy) +
                             ", min_entropy=" + std::to_string(metadata.run_minimum_entropy);
    
    // Use atomic write pattern (write to temp, then rename)
    atomicWrite(checkpoint_path, [&](const std::string& temp_path) {
        // Serialize vector with metadata
        utils::VectorSerializer::serialize(
            "vector",                   // type (lowercase as expected by serializer)
            temp_path,                  // filename
            vector.data,                // vector data
            description,                // description
            vector.dimension,           // d
            1,                          // N (not used for single vector)
            0                           // M (not used for single vector)
        );
    });
    
    // Auto-cleanup old checkpoints if configured
    if (config_.keep_last_n > 0) {
        cleanOldCheckpoints(run_id, config_.keep_last_n);
    }
}

std::pair<HostVector, CheckpointMetadata> CheckpointManager::loadCheckpoint(
    const std::string& checkpoint_path
)
{
    if (!std::filesystem::exists(checkpoint_path)) {
        throw std::runtime_error("CheckpointManager::loadCheckpoint: file does not exist: " + checkpoint_path);
    }
    
    // Use serializer to load data
    utils::DeserializedData data = serializer_->deserialize(checkpoint_path);
    
    // Create HostVector
    HostVector vector;
    vector.data = data.vectorData;
    vector.dimension = data.d;
    
    // Parse metadata from description or JSON
    CheckpointMetadata metadata;
    
    // Try to extract metadata from description string
    // Format: "Checkpoint: run_id=XXX, iteration=YYY, entropy=ZZZ, min_entropy=WWW"
    std::string desc = data.description;
    
    // Extract run_id
    size_t run_id_pos = desc.find("run_id=");
    if (run_id_pos != std::string::npos) {
        size_t start = run_id_pos + 7;
        size_t end = desc.find(",", start);
        if (end == std::string::npos) end = desc.find(" ", start);
        if (end != std::string::npos) {
            metadata.run_id = desc.substr(start, end - start);
        }
    }
    
    // Extract iteration
    size_t iter_pos = desc.find("iteration=");
    if (iter_pos != std::string::npos) {
        size_t start = iter_pos + 10;
        size_t end = desc.find(",", start);
        if (end != std::string::npos) {
            metadata.iteration = std::stoi(desc.substr(start, end - start));
        }
    }
    
    // Extract current entropy
    size_t entropy_pos = desc.find("entropy=");
    if (entropy_pos != std::string::npos) {
        // Skip "min_entropy=" which also contains "entropy="
        if (desc.substr(entropy_pos - 4, 4) != "min_") {
            size_t start = entropy_pos + 8;
            size_t end = desc.find(",", start);
            if (end != std::string::npos) {
                metadata.current_entropy = std::stod(desc.substr(start, end - start));
            }
        }
    }
    
    // Extract min entropy
    size_t min_entropy_pos = desc.find("min_entropy=");
    if (min_entropy_pos != std::string::npos) {
        size_t start = min_entropy_pos + 12;
        metadata.run_minimum_entropy = std::stod(desc.substr(start));
    }
    
    // Set timestamp to file modification time
    auto file_time = std::filesystem::last_write_time(checkpoint_path);
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        file_time - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now()
    );
    metadata.timestamp = sctp;
    
    return {vector, metadata};
}

bool CheckpointManager::shouldCheckpoint(int iteration) const
{
    if (!config_.enabled) {
        return false;
    }
    
    if (config_.interval <= 0) {
        return false;  // Invalid interval, never checkpoint
    }
    
    // Checkpoint at multiples of interval (including iteration 0 if desired)
    return (iteration % config_.interval) == 0;
}

void CheckpointManager::cleanOldCheckpoints(const std::string& run_id, int keep_last_n)
{
    if (keep_last_n < 0) {
        keep_last_n = config_.keep_last_n;
    }
    
    if (keep_last_n <= 0) {
        return;  // Keep all checkpoints
    }
    
    auto checkpoints = listCheckpoints(run_id);
    
    // Sort by iteration (ascending)
    std::sort(checkpoints.begin(), checkpoints.end(), [this](const auto& a, const auto& b) {
        return extractIterationFromPath(a) < extractIterationFromPath(b);
    });
    
    // Delete all but the last N
    if (checkpoints.size() > static_cast<size_t>(keep_last_n)) {
        size_t num_to_delete = checkpoints.size() - keep_last_n;
        for (size_t i = 0; i < num_to_delete; ++i) {
            try {
                std::filesystem::remove(checkpoints[i]);
            } catch (const std::exception& e) {
                // Log error but continue cleanup
                // In production, would use proper logging
                // For now, silently continue
            }
        }
    }
}

std::filesystem::path CheckpointManager::getCheckpointDirectory() const
{
    return std::filesystem::path(config_.directory);
}

std::vector<std::filesystem::path> CheckpointManager::listCheckpoints(const std::string& run_id) const
{
    std::vector<std::filesystem::path> checkpoints;
    
    auto dir = getCheckpointDirectory();
    if (!std::filesystem::exists(dir)) {
        return checkpoints;  // Empty vector
    }
    
    // Iterate through directory and find matching checkpoints
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            
            // Check if filename contains run_id
            if (filename.find(run_id) != std::string::npos) {
                checkpoints.push_back(entry.path());
            }
        }
    }
    
    return checkpoints;
}

// ============================================================================
// Private Methods
// ============================================================================

std::filesystem::path CheckpointManager::buildPath(const std::string& run_id, int iteration) const
{
    // Replace placeholders in filename pattern
    std::string filename = config_.filename_pattern;
    
    // Replace {run_id} - find full placeholder and replace with actual run_id
    size_t pos = filename.find("{run_id}");
    if (pos != std::string::npos) {
        filename.replace(pos, std::string("{run_id}").length(), run_id);
    }
    
    // Replace {iteration} with zero-padded number (8 digits)
    pos = filename.find("{iteration}");
    if (pos != std::string::npos) {
        std::ostringstream oss;
        oss << std::setw(8) << std::setfill('0') << iteration;
        filename.replace(pos, std::string("{iteration}").length(), oss.str());
    }
    
    return std::filesystem::path(config_.directory) / filename;
}

void CheckpointManager::ensureDirectoryExists()
{
    if (directory_checked_) {
        return;  // Already verified
    }
    
    auto dir = getCheckpointDirectory();
    
    if (!std::filesystem::exists(dir)) {
        try {
            std::filesystem::create_directories(dir);
        } catch (const std::exception& e) {
            throw std::runtime_error("CheckpointManager: failed to create directory: " + 
                                   dir.string() + " - " + e.what());
        }
    }
    
    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error("CheckpointManager: path exists but is not a directory: " + 
                               dir.string());
    }
    
    directory_checked_ = true;
}

void CheckpointManager::atomicWrite(
    const std::filesystem::path& final_path,
    std::function<void(const std::string&)> write_func
)
{
    // Create temporary file path (same directory, with .tmp suffix)
    auto temp_path = final_path;
    temp_path += ".tmp";
    
    try {
        // Write to temporary file
        write_func(temp_path.string());
        
        // Atomic rename (on POSIX systems, rename is atomic)
        std::filesystem::rename(temp_path, final_path);
        
    } catch (const std::exception& e) {
        // Clean up temp file if it exists
        if (std::filesystem::exists(temp_path)) {
            std::filesystem::remove(temp_path);
        }
        throw std::runtime_error("CheckpointManager::atomicWrite failed: " + std::string(e.what()));
    }
}

int CheckpointManager::extractIterationFromPath(const std::filesystem::path& path) const
{
    std::string filename = path.filename().string();
    
    // Try to extract number using regex
    // Pattern: find 8-digit number (from our zero-padded format)
    std::regex iter_regex(R"(iter(\d{8}))");
    std::smatch match;
    
    if (std::regex_search(filename, match, iter_regex) && match.size() > 1) {
        return std::stoi(match[1].str());
    }
    
    // Fallback: try to find any number sequence
    std::regex num_regex(R"(\d+)");
    if (std::regex_search(filename, match, num_regex)) {
        return std::stoi(match[0].str());
    }
    
    return -1;  // Not found
}

} // namespace entropy
