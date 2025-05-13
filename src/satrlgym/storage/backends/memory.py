"""
Memory-based storage backends.

This module provides in-memory storage backends for experience data.
"""

import json
import os
import pickle
from collections import defaultdict
from typing import Any

import numpy as np

from src.satrlgym.storage.storage_base import StorageBase


class MemoryExperienceStorage(StorageBase):
    """
    Simple in-memory storage for experience data.

    This class stores transitions in RAM using Python data structures.
    """

    def __init__(self, max_size: int = 1000000):
        """
        Initialize memory storage.

        Args:
            max_size: Maximum number of transitions to store
        """
        self.max_size = max_size
        self.transitions = []
        self.position = 0

    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a single transition to storage.

        Args:
            transition: Dictionary with experience data
        """
        if len(self.transitions) < self.max_size:
            self.transitions.append(transition)
        else:
            # Overwrite old transitions when full
            self.transitions[self.position] = transition
            self.position = (self.position + 1) % self.max_size

    def add_batch(self, batch: dict[str, Any]) -> None:
        """
        Add a batch of transitions to storage.

        Args:
            batch: Dictionary of batched experience data
        """
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            transition = {k: v[i] for k, v in batch.items()}
            self.add_transition(transition)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of batched experience data
        """
        if len(self.transitions) == 0:
            return {}

        # Sample indices with replacement
        indices = np.random.randint(0, len(self.transitions), size=batch_size)

        # Create batch
        batch = defaultdict(list)
        for idx in indices:
            for k, v in self.transitions[idx].items():
                batch[k].append(v)

        # Convert to arrays
        for k, v in batch.items():
            if isinstance(v[0], (int, float, bool, np.ndarray)):
                batch[k] = np.array(v)

        return dict(batch)

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.transitions = []
        self.position = 0

    def save(self, path: str) -> None:
        """
        Save the storage to disk.

        Args:
            path: Path where to save the storage
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.transitions, f)

    def load(self, path: str) -> None:
        """
        Load storage from disk.

        Args:
            path: Path from where to load the storage
        """
        with open(path, "rb") as f:
            self.transitions = pickle.load(f)
        self.position = len(self.transitions) % self.max_size

    def close(self) -> None:
        """Close the storage and free resources."""

    @property
    def size(self) -> int:
        """
        Get the number of transitions in storage.

        Returns:
            Number of stored transitions
        """
        return len(self.transitions)


class MemoryMappedExperienceStorage(StorageBase):
    """
    Memory-mapped storage for experience data.

    This class uses memory-mapped files for efficient storage and retrieval
    of large amounts of experience data.
    """

    def __init__(self, data_path: str, max_size: int = 1000000):
        """
        Initialize memory-mapped storage.

        Args:
            data_path: Path for storing memory-mapped files
            max_size: Maximum number of transitions to store
        """
        self.data_path = data_path
        self.max_size = max_size
        self.position = 0
        self.count = 0
        self.transitions = []

        # Create directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)

        # Path to index file
        self.index_path = os.path.join(data_path, "index.json")

        # Load index if it exists
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self._save_index()

    def _save_index(self):
        """Save index to disk."""
        index_data = {
            "count": self.count,
            "position": self.position,
            "transitions": self.transitions,
        }

        with open(self.index_path, "w") as f:
            json.dump(index_data, f, indent=2)

    def _load_index(self):
        """Load index from disk."""
        with open(self.index_path) as f:
            index_data = json.load(f)

        self.count = index_data.get("count", 0)
        self.position = index_data.get("position", 0)
        self.transitions = index_data.get("transitions", [])

    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a single transition to storage.

        Args:
            transition: Dictionary with experience data
        """
        # Record transition metadata
        transition_info = {
            "keys": list(transition.keys()),
            "position": len(self.transitions),
        }
        self.transitions.append(transition_info)

        # Save the actual data
        file_path = os.path.join(
            self.data_path, f"transition_{len(self.transitions)-1}.pkl"
        )
        with open(file_path, "wb") as f:
            pickle.dump(transition, f)

        # Update count and position
        if self.count < self.max_size:
            self.count += 1
        self.position = (self.position + 1) % self.max_size

        # Save index
        self._save_index()

    def add_batch(self, batch: dict[str, Any]) -> None:
        """
        Add a batch of transitions to storage.

        Args:
            batch: Dictionary of batched experience data
        """
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            transition = {k: v[i] for k, v in batch.items()}
            self.add_transition(transition)

    def write_batch(self, batch: dict[str, Any]) -> None:
        """
        Write a batch of transitions efficiently.

        This is an alias for add_batch but could be optimized later.

        Args:
            batch: Dictionary of batched experience data
        """
        self.add_batch(batch)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of batched experience data
        """
        if not self.transitions:
            return {}

        # Sample indices with replacement
        indices = np.random.randint(0, len(self.transitions), size=batch_size)

        # Create batch
        batch = defaultdict(list)
        for idx in indices:
            file_path = os.path.join(self.data_path, f"transition_{idx}.pkl")
            with open(file_path, "rb") as f:
                transition = pickle.load(f)

            for k, v in transition.items():
                batch[k].append(v)

        # Convert to arrays
        for k, v in batch.items():
            if isinstance(v[0], (int, float, bool, np.ndarray)):
                batch[k] = np.array(v)

        return dict(batch)

    def clear(self) -> None:
        """Clear all stored transitions."""
        # Remove all transition files
        for i in range(len(self.transitions)):
            file_path = os.path.join(self.data_path, f"transition_{i}.pkl")
            if os.path.exists(file_path):
                os.remove(file_path)

        # Reset state
        self.transitions = []
        self.position = 0
        self.count = 0

        # Save index
        self._save_index()

    def save(self, path: str) -> None:
        """
        Save the storage to disk.

        Args:
            path: Path where to save the storage
        """
        # Just save the index, transitions are already saved
        self._save_index()

    def load(self, path: str) -> None:
        """
        Load storage from disk.

        Args:
            path: Path from where to load the storage
        """
        # Load the index
        self._load_index()

    def close(self) -> None:
        """Close the storage and free resources."""
        self._save_index()

    @property
    def size(self) -> int:
        """
        Get the number of transitions in storage.

        Returns:
            Number of stored transitions
        """
        return len(self.transitions)

    def __del__(self):
        """Clean up when object is deleted."""
        self.close()
