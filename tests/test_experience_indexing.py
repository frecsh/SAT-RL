"""
Test the experience indexing functionality.

This module contains tests for the ExperienceIndex and PrioritizedReplayBuffer classes,
verifying their functionality for fast access and prioritized experience replay.
"""

import unittest
import os
import json
import numpy as np
import tempfile
import shutil
import pickle
from pathlib import Path

# Import the modules to test
from src.utils.experience_indexing import (
    ExperienceIndex, PrioritizedReplayBuffer, create_index_for_directory
)

class TestExperienceIndex(unittest.TestCase):
    """Test the ExperienceIndex class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test JSON Lines file
        self.test_jsonl_path = os.path.join(self.test_dir, "test_transitions.jsonl")
        self.create_test_jsonl_file()
        
        # Create a test NPZ file
        self.test_npz_path = os.path.join(self.test_dir, "test_transitions.npz")
        self.create_test_npz_file()
        
        # Paths for index files
        self.test_jsonl_idx_path = os.path.join(self.test_dir, "test_transitions.idx")
        self.test_npz_idx_path = os.path.join(self.test_dir, "test_transitions_npz.idx")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_test_jsonl_file(self):
        """Create a test JSON Lines file with sample transitions."""
        transitions = [
            {
                "state": [1, 0, 1],
                "action": 0,
                "reward": 0.5,
                "next_state": [1, 1, 1],
                "done": False
            },
            {
                "state": [1, 1, 1],
                "action": 1,
                "reward": 1.0,
                "next_state": [0, 1, 1],
                "done": False
            },
            {
                "state": [0, 1, 1],
                "action": 2,
                "reward": -0.5,
                "next_state": [0, 0, 1],
                "done": True
            }
        ]
        
        with open(self.test_jsonl_path, 'w') as f:
            for transition in transitions:
                f.write(json.dumps(transition) + "\n")
    
    def create_test_npz_file(self):
        """Create a test NPZ file with sample transitions."""
        states = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 1]])
        actions = np.array([0, 1, 2])
        rewards = np.array([0.5, 1.0, -0.5])
        next_states = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        dones = np.array([False, False, True])
        
        np.savez(
            self.test_npz_path,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )
    
    def test_create_index_jsonl(self):
        """Test index creation for JSON Lines file."""
        # Create the index
        index = ExperienceIndex(self.test_jsonl_path)
        
        # Check that the index was created
        self.assertTrue(os.path.exists(self.test_jsonl_path.replace(".jsonl", ".idx")))
        
        # Check the index properties
        self.assertEqual(index.transitions_count, 3)
        self.assertEqual(len(index.offsets), 3)
        self.assertEqual(len(index.weights), 3)
        
        # Check that each weight is initialized to 1.0
        for weight in index.weights:
            self.assertEqual(weight, 1.0)
    
    def test_load_index_jsonl(self):
        """Test loading an existing index for a JSON Lines file."""
        # First create an index
        index1 = ExperienceIndex(self.test_jsonl_path)
        
        # Now load the index
        index2 = ExperienceIndex(self.test_jsonl_path, create_if_missing=False)
        
        # Check that the indices match
        self.assertEqual(index1.transitions_count, index2.transitions_count)
        self.assertEqual(index1.offsets, index2.offsets)
        np.testing.assert_array_equal(index1.weights, index2.weights)
    
    def test_get_transition_jsonl(self):
        """Test getting a transition from a JSON Lines file."""
        # Create the index
        index = ExperienceIndex(self.test_jsonl_path)
        
        # Get a transition
        transition = index.get_transition(1)
        
        # Check that the transition is correct
        self.assertEqual(transition["action"], 1)
        self.assertEqual(transition["reward"], 1.0)
        self.assertEqual(transition["state"], [1, 1, 1])
    
    def test_sample_transitions_jsonl(self):
        """Test sampling transitions from a JSON Lines file."""
        # Create the index
        index = ExperienceIndex(self.test_jsonl_path)
        
        # Sample transitions
        transitions = index.sample(2)
        
        # Check that we got the right number of transitions
        self.assertEqual(len(transitions), 2)
        
        # Check that each transition has the expected fields
        for transition in transitions:
            self.assertIn("state", transition)
            self.assertIn("action", transition)
            self.assertIn("reward", transition)
            self.assertIn("next_state", transition)
            self.assertIn("done", transition)
    
    def test_update_weights_jsonl(self):
        """Test updating weights in an index for a JSON Lines file."""
        # Create the index
        index = ExperienceIndex(self.test_jsonl_path)
        
        # Update the weights
        new_weights = [2.0, 3.0, 4.0]
        index.update_weights([0, 1, 2], new_weights)
        
        # Check that the weights were updated
        np.testing.assert_array_equal(index.weights, new_weights)
        self.assertEqual(index._weight_sum, sum(new_weights))
        
        # Sample with prioritization
        transitions = index.sample(2, prioritized=True)
        
        # Check that we got the right number of transitions
        self.assertEqual(len(transitions), 2)
    
    def test_create_index_npz(self):
        """Test index creation for NPZ file."""
        # Create the index
        index = ExperienceIndex(self.test_npz_path, self.test_npz_idx_path)
        
        # Check that the index was created
        self.assertTrue(os.path.exists(self.test_npz_idx_path))
        
        # Check the index properties
        self.assertEqual(index.transitions_count, 3)
        self.assertEqual(len(index.offsets), 3)
    
    def test_get_transition_npz(self):
        """Test getting a transition from an NPZ file."""
        # Create the index
        index = ExperienceIndex(self.test_npz_path, self.test_npz_idx_path)
        
        # Get a transition
        transition = index.get_transition(1)
        
        # Check that the transition has the expected fields
        self.assertIn("states", transition)
        self.assertIn("actions", transition)
        self.assertIn("rewards", transition)
        
        # Check some values
        self.assertEqual(transition["actions"], 1)
        self.assertEqual(transition["rewards"], 1.0)
    
    def test_cache_performance(self):
        """Test cache performance for repeated access."""
        # Create the index with a small cache
        index = ExperienceIndex(self.test_jsonl_path, cache_size=2)
        
        # Access the same transition multiple times
        for _ in range(5):
            transition = index.get_transition(0)
        
        # Check cache statistics
        cache_stats = index.get_cache_stats()
        self.assertEqual(cache_stats["hits"], 4)
        self.assertEqual(cache_stats["misses"], 1)
        
        # Access different transitions to fill the cache
        index.get_transition(1)
        index.get_transition(2)
        
        # Access the first transition again (should be a miss since cache is full)
        index.get_transition(0)
        
        # Check updated cache statistics
        cache_stats = index.get_cache_stats()
        self.assertEqual(cache_stats["misses"], 4)  # 1 + 1 + 1 + 1 = 4
    
    def test_index_errors(self):
        """Test error handling in the ExperienceIndex class."""
        # Create the index
        index = ExperienceIndex(self.test_jsonl_path)
        
        # Test index out of bounds
        with self.assertRaises(IndexError):
            index.get_transition(-1)
        
        with self.assertRaises(IndexError):
            index.get_transition(100)
        
        # Test bad weights update
        with self.assertRaises(ValueError):
            index.update_weights([0, 1], [1.0])  # Mismatched lengths


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Test the PrioritizedReplayBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add some transitions
        for i in range(10):
            transition = {
                "state": [i % 2, (i+1) % 2],
                "action": i % 3,
                "reward": i / 10.0,
                "next_state": [(i+1) % 2, (i+2) % 2],
                "done": i == 9
            }
            self.buffer.add(transition)
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        # Check initial size
        self.assertEqual(len(self.buffer), 10)
        
        # Check capacity
        self.assertEqual(self.buffer.capacity, 100)
    
    def test_buffer_add(self):
        """Test adding transitions to the buffer."""
        # Add one more transition
        transition = {
            "state": [1, 0],
            "action": 0,
            "reward": 1.0,
            "next_state": [0, 1],
            "done": True
        }
        self.buffer.add(transition)
        
        # Check the new size
        self.assertEqual(len(self.buffer), 11)
        
        # Check that the transition was added
        self.assertEqual(self.buffer.transitions[10]["reward"], 1.0)
    
    def test_buffer_sampling(self):
        """Test sampling from the buffer."""
        # Sample from the buffer
        transitions, indices, weights = self.buffer.sample(5)
        
        # Check that we got the right number of transitions
        self.assertEqual(len(transitions), 5)
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(weights), 5)
        
        # Check that all weights are 1.0 (since all priorities are the same)
        for weight in weights:
            self.assertEqual(weight, 1.0)
    
    def test_update_priorities(self):
        """Test updating priorities in the buffer."""
        # Sample from the buffer
        transitions, indices, weights = self.buffer.sample(5)
        
        # Update the priorities
        new_priorities = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.buffer.update_priorities(indices, new_priorities)
        
        # Sample again (with the updated priorities)
        transitions2, indices2, weights2 = self.buffer.sample(5)
        
        # The weights should now be different
        self.assertFalse(np.allclose(weights, weights2))
    
    def test_buffer_overflow(self):
        """Test buffer behavior when capacity is reached."""
        # Fill the buffer beyond capacity
        for i in range(100):
            transition = {
                "state": [0, 0],
                "action": 0,
                "reward": 0.0,
                "next_state": [0, 0],
                "done": False
            }
            self.buffer.add(transition)
        
        # Check that size equals capacity
        self.assertEqual(len(self.buffer), 100)
        
        # Add one more to trigger overflow
        transition = {
            "state": [1, 1],
            "action": 1,
            "reward": 1.0,
            "next_state": [1, 1],
            "done": True
        }
        self.buffer.add(transition)
        
        # Size should still be capacity
        self.assertEqual(len(self.buffer), 100)
        
        # The oldest transition should have been replaced
        # After adding 10 (setUp) + 100 (loop) + 1 (final) = 111 items,
        # the position in a buffer of capacity 100 should be 111 % 100 = 11.
        self.assertEqual(self.buffer.position, 11)
    
    def test_sampling_errors(self):
        """Test error handling in sampling."""
        # Try to sample more than we have
        with self.assertRaises(ValueError):
            self.buffer.sample(20)


class TestDirectoryIndexing(unittest.TestCase):
    """Test the create_index_for_directory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create multiple test files
        self.create_test_files()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_test_files(self):
        """Create multiple test files in the test directory."""
        # Create a JSON Lines file
        jsonl_path = os.path.join(self.test_dir, "transitions1.jsonl")
        with open(jsonl_path, 'w') as f:
            for i in range(3):
                transition = {
                    "state": [i % 2, (i+1) % 2],
                    "action": i,
                    "reward": float(i)
                }
                f.write(json.dumps(transition) + "\n")
        
        # Create another JSON Lines file in a subdirectory
        subdir = os.path.join(self.test_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)
        jsonl_path2 = os.path.join(subdir, "transitions2.jsonl")
        with open(jsonl_path2, 'w') as f:
            for i in range(3):
                transition = {
                    "state": [i % 2, (i+1) % 2],
                    "action": i,
                    "reward": float(i)
                }
                f.write(json.dumps(transition) + "\n")
        
        # Create an NPZ file
        npz_path = os.path.join(self.test_dir, "transitions.npz")
        states = np.array([[1, 0], [0, 1], [1, 1]])
        actions = np.array([0, 1, 2])
        np.savez(npz_path, states=states, actions=actions)
    
    def test_create_indices_recursive(self):
        """Test creating indices for all files in a directory (recursively)."""
        # Create indices for all files
        indices = create_index_for_directory(self.test_dir, recursive=True)
        
        # Check that we have the expected number of indices
        self.assertEqual(len(indices), 3)
        
        # Check that each index file was created
        for filepath in indices:
            idx_path = filepath.replace(".jsonl", ".idx").replace(".npz", ".idx")
            self.assertTrue(os.path.exists(idx_path))
    
    def test_create_indices_nonrecursive(self):
        """Test creating indices for files in a directory (non-recursively)."""
        # Create indices for files in the top directory only
        indices = create_index_for_directory(self.test_dir, recursive=False)
        
        # Check that we have the expected number of indices (2, not 3)
        self.assertEqual(len(indices), 2)
    
    def test_create_indices_with_pattern(self):
        """Test creating indices for files matching a specific pattern."""
        # Create indices only for JSON Lines files
        indices = create_index_for_directory(
            self.test_dir,
            file_patterns=["*.jsonl"],
            recursive=True
        )
        
        # Check that we have only the JSONL indices
        self.assertEqual(len(indices), 2)
        for filepath in indices:
            self.assertTrue(filepath.endswith(".jsonl"))


if __name__ == "__main__":
    unittest.main()