#include <gtest/gtest.h>
#include "minimizer/management/checkpoint_manager.h"
#include "utilities/serializer/serializer.h"

#include <filesystem>
#include <fstream>
#include <complex>
#include <vector>

using namespace entropy;
namespace fs = std::filesystem;

// ============================================================================
// Test Fixtures
// ============================================================================

class CheckpointManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test directory
        test_dir_ = fs::temp_directory_path() / "checkpoint_test";
        
        // Clean up any existing test directory
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
        
        // Create fresh test directory
        fs::create_directories(test_dir_);
        
        // Configure for testing
        config_.enabled = true;
        config_.directory = test_dir_.string();
        config_.interval = 100;
        config_.filename_pattern = "{run_id}_iter{iteration}.dat";
        config_.keep_last_n = 3;
        
        // Create test vector
        test_vector_.data = {
            {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}, {0.0, 0.0}, {0.25, 0.75}
        };
        test_vector_.dimension = 5;
        
        // Create test metadata
        test_metadata_.run_id = "test-run-123";
        test_metadata_.iteration = 1000;
        test_metadata_.current_entropy = 1.234;
        test_metadata_.run_minimum_entropy = 0.987;
        test_metadata_.timestamp = std::chrono::system_clock::now();
        test_metadata_.config_hash = "abc123";
    }
    
    void TearDown() override {
        // Clean up test directory
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }
    
    // Helper to create manager
    std::unique_ptr<CheckpointManager> createManager() {
        return std::make_unique<CheckpointManager>(
            config_,
            std::make_unique<VectorSerializer>()
        );
    }
    
    fs::path test_dir_;
    CheckpointConfig config_;
    HostVector test_vector_;
    CheckpointMetadata test_metadata_;
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(CheckpointManagerTest, ConstructorValid) {
    EXPECT_NO_THROW(createManager());
}

TEST_F(CheckpointManagerTest, ConstructorNullSerializer) {
    EXPECT_THROW(
        CheckpointManager manager(config_, nullptr),
        std::invalid_argument
    );
}

TEST_F(CheckpointManagerTest, ConstructorCreatesDirectory) {
    auto manager = createManager();
    
    EXPECT_TRUE(fs::exists(test_dir_));
    EXPECT_TRUE(fs::is_directory(test_dir_));
}

TEST_F(CheckpointManagerTest, ConstructorDisabledDoesNotCreateDirectory) {
    // Remove the directory we created in SetUp
    fs::remove_all(test_dir_);
    
    config_.enabled = false;
    auto manager = createManager();
    
    // Directory should not be created when disabled
    EXPECT_FALSE(fs::exists(test_dir_));
}

// ============================================================================
// Save Checkpoint Tests
// ============================================================================

TEST_F(CheckpointManagerTest, SaveCheckpointValid) {
    auto manager = createManager();
    
    EXPECT_NO_THROW(
        manager->saveCheckpoint("run-001", 500, test_vector_, test_metadata_)
    );
    
    // Check file exists
    auto checkpoints = manager->listCheckpoints("run-001");
    EXPECT_EQ(checkpoints.size(), 1);
}

TEST_F(CheckpointManagerTest, SaveCheckpointDisabled) {
    config_.enabled = false;
    auto manager = createManager();
    
    // Should not throw, but should not create file
    EXPECT_NO_THROW(
        manager->saveCheckpoint("run-001", 500, test_vector_, test_metadata_)
    );
    
    // No checkpoints should be created
    // Note: directory won't exist since disabled
}

TEST_F(CheckpointManagerTest, SaveCheckpointEmptyRunId) {
    auto manager = createManager();
    
    EXPECT_THROW(
        manager->saveCheckpoint("", 500, test_vector_, test_metadata_),
        std::invalid_argument
    );
}

TEST_F(CheckpointManagerTest, SaveCheckpointNegativeIteration) {
    auto manager = createManager();
    
    EXPECT_THROW(
        manager->saveCheckpoint("run-001", -1, test_vector_, test_metadata_),
        std::invalid_argument
    );
}

TEST_F(CheckpointManagerTest, SaveCheckpointEmptyVector) {
    auto manager = createManager();
    
    HostVector empty_vector;
    empty_vector.dimension = 0;
    
    EXPECT_THROW(
        manager->saveCheckpoint("run-001", 500, empty_vector, test_metadata_),
        std::invalid_argument
    );
}

TEST_F(CheckpointManagerTest, SaveMultipleCheckpoints) {
    config_.keep_last_n = 0;  // Disable auto-cleanup for this test
    auto manager = createManager();
    
    // Save 5 checkpoints
    for (int i = 0; i < 5; ++i) {
        manager->saveCheckpoint("run-001", i * 100, test_vector_, test_metadata_);
    }
    
    auto checkpoints = manager->listCheckpoints("run-001");
    EXPECT_EQ(checkpoints.size(), 5);
}

TEST_F(CheckpointManagerTest, SaveCheckpointCreatesValidFile) {
    auto manager = createManager();
    
    manager->saveCheckpoint("run-001", 500, test_vector_, test_metadata_);
    
    auto checkpoints = manager->listCheckpoints("run-001");
    ASSERT_EQ(checkpoints.size(), 1);
    
    // Check file is not empty
    auto file_size = fs::file_size(checkpoints[0]);
    EXPECT_GT(file_size, 0);
}

// ============================================================================
// Load Checkpoint Tests
// ============================================================================

TEST_F(CheckpointManagerTest, LoadCheckpointValid) {
    auto manager = createManager();
    
    // Save checkpoint
    manager->saveCheckpoint("run-001", 500, test_vector_, test_metadata_);
    
    // Get checkpoint path
    auto checkpoints = manager->listCheckpoints("run-001");
    ASSERT_EQ(checkpoints.size(), 1);
    
    // Load checkpoint
    auto [loaded_vector, loaded_metadata] = manager->loadCheckpoint(checkpoints[0].string());
    
    // Verify vector data
    ASSERT_EQ(loaded_vector.data.size(), test_vector_.data.size());
    for (size_t i = 0; i < loaded_vector.data.size(); ++i) {
        EXPECT_NEAR(loaded_vector.data[i].real(), test_vector_.data[i].real(), 1e-10);
        EXPECT_NEAR(loaded_vector.data[i].imag(), test_vector_.data[i].imag(), 1e-10);
    }
    EXPECT_EQ(loaded_vector.dimension, test_vector_.dimension);
}

TEST_F(CheckpointManagerTest, LoadCheckpointNonexistent) {
    auto manager = createManager();
    
    EXPECT_THROW(
        manager->loadCheckpoint("/nonexistent/path/checkpoint.dat"),
        std::runtime_error
    );
}

TEST_F(CheckpointManagerTest, LoadCheckpointMetadata) {
    auto manager = createManager();
    
    // Save checkpoint with specific metadata
    test_metadata_.run_id = "metadata-test-run";
    test_metadata_.iteration = 12345;
    test_metadata_.current_entropy = 2.718;
    test_metadata_.run_minimum_entropy = 1.414;
    
    manager->saveCheckpoint("metadata-test-run", 12345, test_vector_, test_metadata_);
    
    // Load and verify metadata
    auto checkpoints = manager->listCheckpoints("metadata-test-run");
    ASSERT_EQ(checkpoints.size(), 1);
    
    auto [loaded_vector, loaded_metadata] = manager->loadCheckpoint(checkpoints[0].string());
    
    EXPECT_EQ(loaded_metadata.run_id, "metadata-test-run");
    EXPECT_EQ(loaded_metadata.iteration, 12345);
    EXPECT_NEAR(loaded_metadata.current_entropy, 2.718, 1e-10);
    EXPECT_NEAR(loaded_metadata.run_minimum_entropy, 1.414, 1e-10);
}

// ============================================================================
// Should Checkpoint Tests
// ============================================================================

TEST_F(CheckpointManagerTest, ShouldCheckpointAtInterval) {
    auto manager = createManager();
    
    config_.interval = 100;
    
    EXPECT_TRUE(manager->shouldCheckpoint(0));
    EXPECT_FALSE(manager->shouldCheckpoint(50));
    EXPECT_TRUE(manager->shouldCheckpoint(100));
    EXPECT_FALSE(manager->shouldCheckpoint(150));
    EXPECT_TRUE(manager->shouldCheckpoint(200));
    EXPECT_TRUE(manager->shouldCheckpoint(1000));
}

TEST_F(CheckpointManagerTest, ShouldCheckpointDisabled) {
    config_.enabled = false;
    auto manager = createManager();
    
    EXPECT_FALSE(manager->shouldCheckpoint(0));
    EXPECT_FALSE(manager->shouldCheckpoint(100));
    EXPECT_FALSE(manager->shouldCheckpoint(1000));
}

TEST_F(CheckpointManagerTest, ShouldCheckpointInvalidInterval) {
    config_.interval = 0;
    auto manager = createManager();
    
    EXPECT_FALSE(manager->shouldCheckpoint(0));
    EXPECT_FALSE(manager->shouldCheckpoint(100));
    
    config_.interval = -1;
    manager = createManager();
    
    EXPECT_FALSE(manager->shouldCheckpoint(0));
    EXPECT_FALSE(manager->shouldCheckpoint(100));
}

// ============================================================================
// Clean Old Checkpoints Tests
// ============================================================================

TEST_F(CheckpointManagerTest, CleanOldCheckpointsKeepLastN) {
    config_.keep_last_n = 0;  // Initially disable auto-cleanup
    auto manager = createManager();
    
    // Create 10 checkpoints
    for (int i = 0; i < 10; ++i) {
        manager->saveCheckpoint("run-cleanup", i * 100, test_vector_, test_metadata_);
    }
    
    auto checkpoints_before = manager->listCheckpoints("run-cleanup");
    EXPECT_EQ(checkpoints_before.size(), 10);
    
    // Clean, keeping only last 3
    manager->cleanOldCheckpoints("run-cleanup", 3);
    
    auto checkpoints_after = manager->listCheckpoints("run-cleanup");
    EXPECT_EQ(checkpoints_after.size(), 3);
}

TEST_F(CheckpointManagerTest, CleanOldCheckpointsUsesConfigDefault) {
    config_.keep_last_n = 2;
    auto manager = createManager();
    
    // Create 5 checkpoints
    for (int i = 0; i < 5; ++i) {
        manager->saveCheckpoint("run-default", i * 100, test_vector_, test_metadata_);
    }
    
    // Clean using default from config
    manager->cleanOldCheckpoints("run-default");
    
    auto checkpoints = manager->listCheckpoints("run-default");
    EXPECT_EQ(checkpoints.size(), 2);
}

TEST_F(CheckpointManagerTest, CleanOldCheckpointsKeepAll) {
    config_.keep_last_n = 0;  // Initially disable auto-cleanup
    auto manager = createManager();
    
    // Create 5 checkpoints
    for (int i = 0; i < 5; ++i) {
        manager->saveCheckpoint("run-keepall", i * 100, test_vector_, test_metadata_);
    }
    
    // Clean with keep_last_n = 0 (keep all)
    manager->cleanOldCheckpoints("run-keepall", 0);
    
    auto checkpoints = manager->listCheckpoints("run-keepall");
    EXPECT_EQ(checkpoints.size(), 5);
}

TEST_F(CheckpointManagerTest, CleanOldCheckpointsKeepsNewest) {
    auto manager = createManager();
    
    // Create checkpoints with increasing iterations
    std::vector<int> iterations = {100, 200, 300, 400, 500};
    for (int iter : iterations) {
        manager->saveCheckpoint("run-newest", iter, test_vector_, test_metadata_);
    }
    
    // Clean, keeping only last 2
    manager->cleanOldCheckpoints("run-newest", 2);
    
    auto checkpoints = manager->listCheckpoints("run-newest");
    ASSERT_EQ(checkpoints.size(), 2);
    
    // Verify the remaining checkpoints are the newest (400 and 500)
    // We need to check the filenames contain the highest iteration numbers
    std::vector<std::string> filenames;
    for (const auto& cp : checkpoints) {
        filenames.push_back(cp.filename().string());
    }
    
    // Both filenames should contain higher iteration numbers
    bool found_400_or_500_1 = filenames[0].find("00000400") != std::string::npos || 
                               filenames[0].find("00000500") != std::string::npos;
    bool found_400_or_500_2 = filenames[1].find("00000400") != std::string::npos || 
                               filenames[1].find("00000500") != std::string::npos;
    
    EXPECT_TRUE(found_400_or_500_1);
    EXPECT_TRUE(found_400_or_500_2);
}

TEST_F(CheckpointManagerTest, AutoCleanupOnSave) {
    config_.keep_last_n = 3;
    auto manager = createManager();
    
    // Save 5 checkpoints - auto-cleanup should trigger on each save
    for (int i = 0; i < 5; ++i) {
        manager->saveCheckpoint("run-auto", i * 100, test_vector_, test_metadata_);
        
        // After each save, should keep at most 3
        auto checkpoints = manager->listCheckpoints("run-auto");
        EXPECT_LE(checkpoints.size(), 3);
    }
    
    // Final state should have exactly 3
    auto final_checkpoints = manager->listCheckpoints("run-auto");
    EXPECT_EQ(final_checkpoints.size(), 3);
}

// ============================================================================
// List Checkpoints Tests
// ============================================================================

TEST_F(CheckpointManagerTest, ListCheckpointsEmpty) {
    auto manager = createManager();
    
    auto checkpoints = manager->listCheckpoints("nonexistent-run");
    EXPECT_TRUE(checkpoints.empty());
}

TEST_F(CheckpointManagerTest, ListCheckpointsMultipleRuns) {
    auto manager = createManager();
    
    // Create checkpoints for different runs
    manager->saveCheckpoint("run-A", 100, test_vector_, test_metadata_);
    manager->saveCheckpoint("run-A", 200, test_vector_, test_metadata_);
    manager->saveCheckpoint("run-B", 100, test_vector_, test_metadata_);
    manager->saveCheckpoint("run-C", 100, test_vector_, test_metadata_);
    
    // List checkpoints for each run
    auto checkpoints_A = manager->listCheckpoints("run-A");
    auto checkpoints_B = manager->listCheckpoints("run-B");
    auto checkpoints_C = manager->listCheckpoints("run-C");
    
    EXPECT_EQ(checkpoints_A.size(), 2);
    EXPECT_EQ(checkpoints_B.size(), 1);
    EXPECT_EQ(checkpoints_C.size(), 1);
}

// ============================================================================
// Utility Method Tests
// ============================================================================

TEST_F(CheckpointManagerTest, GetCheckpointDirectory) {
    auto manager = createManager();
    
    auto dir = manager->getCheckpointDirectory();
    EXPECT_EQ(dir.string(), test_dir_.string());
}

TEST_F(CheckpointManagerTest, IsEnabled) {
    config_.enabled = true;
    auto manager1 = createManager();
    EXPECT_TRUE(manager1->isEnabled());
    
    config_.enabled = false;
    auto manager2 = createManager();
    EXPECT_FALSE(manager2->isEnabled());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(CheckpointManagerTest, SaveAndLoadLargeVector) {
    auto manager = createManager();
    
    // Create large vector (1000 elements)
    HostVector large_vector;
    large_vector.dimension = 1000;
    for (int i = 0; i < 1000; ++i) {
        double real = static_cast<double>(i) / 1000.0;
        double imag = static_cast<double>(1000 - i) / 1000.0;
        large_vector.data.emplace_back(real, imag);
    }
    
    manager->saveCheckpoint("run-large", 0, large_vector, test_metadata_);
    
    auto checkpoints = manager->listCheckpoints("run-large");
    ASSERT_EQ(checkpoints.size(), 1);
    
    auto [loaded_vector, loaded_metadata] = manager->loadCheckpoint(checkpoints[0].string());
    
    ASSERT_EQ(loaded_vector.data.size(), large_vector.data.size());
    for (size_t i = 0; i < loaded_vector.data.size(); ++i) {
        EXPECT_NEAR(loaded_vector.data[i].real(), large_vector.data[i].real(), 1e-10);
        EXPECT_NEAR(loaded_vector.data[i].imag(), large_vector.data[i].imag(), 1e-10);
    }
}

TEST_F(CheckpointManagerTest, OverwriteCheckpointSameIteration) {
    auto manager = createManager();
    
    // Save checkpoint
    manager->saveCheckpoint("run-overwrite", 100, test_vector_, test_metadata_);
    
    // Modify vector
    test_vector_.data[0] = {99.0, 99.0};
    
    // Save again with same iteration (should overwrite)
    manager->saveCheckpoint("run-overwrite", 100, test_vector_, test_metadata_);
    
    // Should still have only 1 checkpoint
    auto checkpoints = manager->listCheckpoints("run-overwrite");
    EXPECT_EQ(checkpoints.size(), 1);
    
    // Loaded data should have new value
    auto [loaded_vector, loaded_metadata] = manager->loadCheckpoint(checkpoints[0].string());
    EXPECT_NEAR(loaded_vector.data[0].real(), 99.0, 1e-10);
    EXPECT_NEAR(loaded_vector.data[0].imag(), 99.0, 1e-10);
}

TEST_F(CheckpointManagerTest, CustomFilenamePattern) {
    config_.filename_pattern = "checkpoint_{run_id}_{iteration}.ckpt";
    auto manager = createManager();
    
    manager->saveCheckpoint("custom-run", 777, test_vector_, test_metadata_);
    
    auto checkpoints = manager->listCheckpoints("custom-run");
    ASSERT_EQ(checkpoints.size(), 1);
    
    std::string filename = checkpoints[0].filename().string();
    EXPECT_NE(filename.find("custom-run"), std::string::npos);
    EXPECT_NE(filename.find("00000777"), std::string::npos);
    EXPECT_NE(filename.find(".ckpt"), std::string::npos);
}

TEST_F(CheckpointManagerTest, VariableLengthRunId) {
    auto manager = createManager();
    
    // Test with different run_id lengths
    std::vector<std::string> run_ids = {
        "a",                                    // 1 char
        "short",                                // 5 chars  
        "medium-length-id",                     // 16 chars
        "very-long-run-identifier-with-uuid",   // 35 chars
        "123e4567-e89b-12d3-a456-426614174000"  // UUID format (36 chars)
    };
    
    for (const auto& run_id : run_ids) {
        manager->saveCheckpoint(run_id, 100, test_vector_, test_metadata_);
        
        auto checkpoints = manager->listCheckpoints(run_id);
        ASSERT_EQ(checkpoints.size(), 1) << "Failed for run_id: " << run_id;
        
        std::string filename = checkpoints[0].filename().string();
        EXPECT_NE(filename.find(run_id), std::string::npos) 
            << "run_id '" << run_id << "' not found in filename: " << filename;
        EXPECT_NE(filename.find("00000100"), std::string::npos)
            << "Iteration not found in filename: " << filename;
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(CheckpointManagerTest, CompleteWorkflow) {
    config_.keep_last_n = 0;  // Disable auto-cleanup for this test
    auto manager = createManager();
    
    std::string run_id = "integration-run";
    
    // Simulate minimization run with periodic checkpoints
    for (int iter = 0; iter <= 1000; iter += 100) {
        if (manager->shouldCheckpoint(iter)) {
            // Update metadata
            test_metadata_.iteration = iter;
            test_metadata_.current_entropy = 10.0 - iter * 0.001;  // Decreasing
            test_metadata_.run_minimum_entropy = test_metadata_.current_entropy;
            
            manager->saveCheckpoint(run_id, iter, test_vector_, test_metadata_);
        }
    }
    
    // Should have created checkpoints at 0, 100, 200, ..., 1000
    auto checkpoints = manager->listCheckpoints(run_id);
    EXPECT_EQ(checkpoints.size(), 11);
    
    // Sort checkpoints by iteration number to get the latest
    std::sort(checkpoints.begin(), checkpoints.end(), [&manager](const auto& a, const auto& b) {
        return a.filename() < b.filename();  // Filenames are zero-padded so alphabetical = numerical
    });
    
    // Load final checkpoint
    auto final_checkpoint = checkpoints.back();
    auto [final_vector, final_metadata] = manager->loadCheckpoint(final_checkpoint.string());
    
    // Verify final state
    EXPECT_EQ(final_metadata.iteration, 1000);
    EXPECT_NEAR(final_metadata.current_entropy, 9.0, 1e-10);
}
