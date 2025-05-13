"""
File-based storage backend.

This module provides a file-based storage backend for experience data.
"""

import json
import os
import pickle
from collections import defaultdict
from typing import Any

import numpy as np

from src.satrlgym.storage.storage_base import StorageBase


class FileExperienceStorage(StorageBase):
    """
    File-based storage for experience data.

    This class stores transitions in files on disk.
    """

    def __init__(self, data_path: str, max_size: int = 1000000):
        """
        Initialize file storage.

        Args:
            data_path: Directory path for storing files
            max_size: Maximum number of transitions to store
        """
        self.data_path = data_path
        self.max_size = max_size
        self.position = 0
        self.count = 0

        # Create directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)

        # Path to metadata file
        self.metadata_path = os.path.join(data_path, "metadata.json")

        # Load metadata if it exists
        if os.path.exists(self.metadata_path):
            self._load_metadata()
        else:
            self._save_metadata()

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata = {
            "count": self.count,
            "position": self.position,
            "max_size": self.max_size,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self):
        """Load metadata from disk."""
        with open(self.metadata_path) as f:
            metadata = json.load(f)

        self.count = metadata.get("count", 0)
        self.position = metadata.get("position", 0)
        self.max_size = metadata.get("max_size", self.max_size)

    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a single transition to storage.

        Args:
            transition: Dictionary with experience data
        """
        # Create the transition file
        file_path = os.path.join(self.data_path, f"transition_{self.position}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(transition, f)

        # Update count and position
        if self.count < self.max_size:
            self.count += 1
        self.position = (self.position + 1) % self.max_size

        # Save metadata
        self._save_metadata()

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
        if self.count == 0:
            return {}

        # Sample indices with replacement
        indices = np.random.randint(0, self.count, size=batch_size)

        # Create batch
        batch = defaultdict(list)
        for idx in indices:
            # Map to actual file index (might be wrapped around)
            if self.count < self.max_size:
                file_idx = idx
            else:
                file_idx = (self.position + idx) % self.max_size

            file_path = os.path.join(self.data_path, f"transition_{file_idx}.pkl")
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
        for i in range(self.max_size):
            file_path = os.path.join(self.data_path, f"transition_{i}.pkl")
            if os.path.exists(file_path):
                os.remove(file_path)

        # Reset state
        self.position = 0
        self.count = 0

        # Save metadata
        self._save_metadata()

    def save(self, path: str) -> None:
        """
        Save the storage to disk.

        Args:
            path: Path where to save the storage
        """
        # Just save the metadata, transitions are already saved
        self._save_metadata()

    def load(self, path: str) -> None:
        """
        Load storage from disk.

        Args:
            path: Path from where to load the storage
        """
        # Load the metadata
        self._load_metadata()

    def close(self) -> None:
        """Close the storage and free resources."""
        self._save_metadata()

    @property
    def size(self) -> int:
        """
        Get the number of transitions in storage.

        Returns:
            Number of stored transitions
        """
        return self.count
