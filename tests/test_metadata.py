"""
Unit tests for metadata enrichment features.
Tests git repository tracking, random seed generation,
hardware fingerprinting, and experiment tag functionality.
"""

import os
import json
import tempfile
import unittest
import shutil
from unittest.mock import patch, MagicMock

from utils.metadata import (
    ExperimentMetadata,
    GitInfo,
    HardwareInfo,
    SoftwareInfo,
    get_git_info,
    get_hardware_info,
    get_software_info,
    create_experiment_metadata,
    generate_episode_seed,
    update_experiment_metadata,
    save_metadata_to_file,
    load_metadata_from_file,
    MetadataManager
)


class TestMetadataEnrichment(unittest.TestCase):
    """Test cases for metadata enrichment functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.metadata_dir = os.path.join(self.test_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_create_experiment_metadata(self):
        """Test that experiment metadata is created with all required fields."""
        metadata = create_experiment_metadata(
            name="Test Experiment",
            description="Test Description",
            tags=["test", "metadata"],
            custom_metadata={"test_key": "test_value"},
            seed=12345
        )
        
        # Check basic fields
        self.assertEqual(metadata.name, "Test Experiment")
        self.assertEqual(metadata.description, "Test Description")
        self.assertEqual(metadata.tags, {"test", "metadata"})
        self.assertEqual(metadata.custom_metadata["test_key"], "test_value")
        self.assertEqual(metadata.base_random_seed, 12345)
        
        # Check that automatic fields are populated
        self.assertIsNotNone(metadata.experiment_id)
        self.assertIsNotNone(metadata.created_at)
        self.assertIsNotNone(metadata.updated_at)
        
        # Check that git info was attempted to be retrieved
        self.assertIsNotNone(metadata.git_info)
        
        # Check that hardware info was retrieved
        self.assertIsNotNone(metadata.hardware_info)
        self.assertIsNotNone(metadata.hardware_info.platform)
        self.assertIsNotNone(metadata.hardware_info.cpu_count)
        
        # Check that software info was retrieved
        self.assertIsNotNone(metadata.software_info)
        self.assertIsNotNone(metadata.software_info.python_version)
        self.assertTrue(len(metadata.software_info.packages) > 0)

    def test_episode_seed_generation(self):
        """Test that episode seeds are deterministic based on base seed."""
        metadata = create_experiment_metadata(
            name="Test Experiment",
            seed=12345
        )
        
        # Generate seeds for the same episodes multiple times
        seed1_first = generate_episode_seed(metadata, 1)
        seed1_second = generate_episode_seed(metadata, 1)
        seed2 = generate_episode_seed(metadata, 2)
        
        # The same episode ID should always generate the same seed from the same base seed
        self.assertEqual(seed1_first, seed1_second)
        
        # Different episode IDs should generate different seeds
        self.assertNotEqual(seed1_first, seed2)

    @patch("utils.metadata.get_git_info")
    def test_git_tracking(self, mock_get_git_info):
        """Test that git repository information is tracked correctly."""
        # Mock git info
        mock_git = GitInfo(
            commit_hash="abcdef1234567890",
            branch="main",
            is_dirty=False,
            commit_timestamp="2023-09-15T12:00:00",
            commit_message="Test commit",
            remote_url="https://github.com/user/repo.git"
        )
        mock_get_git_info.return_value = mock_git
        
        # Create metadata with mocked git info
        metadata = create_experiment_metadata(name="Git Test")
        
        # Check that git info was correctly captured
        self.assertEqual(metadata.git_info.commit_hash, "abcdef1234567890")
        self.assertEqual(metadata.git_info.branch, "main")
        self.assertEqual(metadata.git_info.is_dirty, False)
        self.assertEqual(metadata.git_info.commit_timestamp, "2023-09-15T12:00:00")
        self.assertEqual(metadata.git_info.commit_message, "Test commit")
        self.assertEqual(metadata.git_info.remote_url, "https://github.com/user/repo.git")

    def test_metadata_serialization(self):
        """Test that metadata can be serialized to JSON and deserialized correctly."""
        metadata = create_experiment_metadata(
            name="Serialization Test",
            description="Testing serialization",
            tags=["test", "serialization"],
            custom_metadata={"key1": "value1", "key2": 123},
            seed=42
        )
        
        # Convert to JSON
        json_data = metadata.to_json()
        
        # Load JSON back into object
        loaded_metadata = ExperimentMetadata.from_json(json_data)
        
        # Check that values match
        self.assertEqual(loaded_metadata.name, "Serialization Test")
        self.assertEqual(loaded_metadata.description, "Testing serialization")
        self.assertEqual(loaded_metadata.tags, {"test", "serialization"})
        self.assertEqual(loaded_metadata.custom_metadata["key1"], "value1")
        self.assertEqual(loaded_metadata.custom_metadata["key2"], 123)
        self.assertEqual(loaded_metadata.base_random_seed, 42)
        self.assertEqual(loaded_metadata.experiment_id, metadata.experiment_id)

    def test_metadata_file_operations(self):
        """Test saving and loading metadata to/from files."""
        metadata = create_experiment_metadata(
            name="File Test",
            description="Testing file operations",
            tags=["test", "file"],
            seed=42
        )
        
        # Save to file
        file_path = os.path.join(self.metadata_dir, "test_metadata.json")
        success = save_metadata_to_file(metadata, file_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(file_path))
        
        # Load from file
        loaded_metadata = load_metadata_from_file(file_path)
        self.assertIsNotNone(loaded_metadata)
        self.assertEqual(loaded_metadata.name, "File Test")
        self.assertEqual(loaded_metadata.description, "Testing file operations")
        self.assertEqual(loaded_metadata.tags, {"test", "file"})
        self.assertEqual(loaded_metadata.base_random_seed, 42)

    def test_metadata_manager(self):
        """Test the metadata manager functionality."""
        # Initialize manager
        manager = MetadataManager(storage_dir=self.metadata_dir)
        
        # Create a new experiment
        metadata = manager.create_experiment(
            name="Manager Test",
            description="Testing manager",
            tags=["test", "manager"],
            custom_metadata={"key": "value"},
            seed=42
        )
        
        # Check that metadata was created correctly
        self.assertEqual(metadata.name, "Manager Test")
        self.assertEqual(metadata.description, "Testing manager")
        self.assertEqual(metadata.tags, {"test", "manager"})
        self.assertEqual(metadata.custom_metadata["key"], "value")
        
        # Check that a file was created
        file_path = os.path.join(self.metadata_dir, f"{metadata.experiment_id}.json")
        self.assertTrue(os.path.exists(file_path))
        
        # Update metadata
        updated_metadata = manager.update_metadata(
            tags=["updated"],
            custom_data={"new_key": "new_value"}
        )
        
        # Check that updates were applied
        self.assertIn("updated", updated_metadata.tags)
        self.assertEqual(updated_metadata.custom_metadata["new_key"], "new_value")
        
        # List experiments
        experiment_ids = manager.list_experiments()
        self.assertIn(metadata.experiment_id, experiment_ids)
        
        # Load experiment
        loaded_metadata = manager.load_experiment(metadata.experiment_id)
        self.assertIsNotNone(loaded_metadata)
        self.assertEqual(loaded_metadata.name, "Manager Test")
        
        # Search experiments
        results = manager.search_experiments(tags=["updated"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].experiment_id, metadata.experiment_id)


if __name__ == "__main__":
    unittest.main()