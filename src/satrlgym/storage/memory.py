"""
Memory-based implementation of experience storage.

This module provides a simple storage backend that keeps all
experiences in memory for fast access.
"""

import json
import pickle
from typing import Any

import numpy as np
from experience.storage.base import ExperienceStorage


class InMemoryExperienceStorage(ExperienceStorage):
    """
    Stores experiences in memory using a circular buffer approach.

    This is the simplest storage backend, but has limitations on how
    much data can be stored based on available system memory.
    """

    def __init__(self, max_size: int = 100000) -> None:
        """
        Initialize in-memory experience storage.

        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.memory = {}  # Dictionary mapping from field name to list of values
        self.current_size = 0
        self.next_idx = 0

    def add(self, experience_batch: dict[str, Any]) -> None:
        """
        Add experiences to storage.

        Args:
            experience_batch: Dictionary with experience data
        """
        batch_size = len(next(iter(experience_batch.values())))

        # Initialize memory if needed
        if not self.memory:
            self.memory = {k: [] for k in experience_batch.keys()}

        # Process each experience in the batch
        for i in range(batch_size):
            # Extract single experience from batch
            experience = {k: v[i] for k, v in experience_batch.items()}

            # Handle circular buffer logic
            if self.current_size >= self.max_size:
                # Replace old item
                idx = self.next_idx
                for k in self.memory:
                    self.memory[k][idx] = experience[k]
            else:
                # Append new item
                for k in self.memory:
                    self.memory[k].append(experience[k])

            # Update counters
            self.next_idx = (self.next_idx + 1) % self.max_size
            self.current_size = min(self.current_size + 1, self.max_size)

    def get(self, indices: list[int] | None = None) -> dict[str, Any]:
        """
        Retrieve experiences from storage.

        Args:
            indices: Optional list of indices to retrieve. If None, retrieves all data.

        Returns:
            Dictionary of retrieved experience data
        """
        if self.current_size == 0:
            return {}

        # Default to all indices if none provided
        if indices is None:
            indices = list(range(self.current_size))

        # Retrieve items at indices
        result = {}
        for key, values in self.memory.items():
            result[key] = np.array([values[i] for i in indices if i < len(values)])

        return result

    def sample(self, batch_size: int) -> dict[str, Any]:
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary of sampled experience data
        """
        if self.current_size == 0:
            raise ValueError("Cannot sample from empty storage")

        indices = np.random.randint(0, self.current_size, size=batch_size)
        return self.get(indices.tolist())

    def get_size(self) -> int:
        """
        Get the number of experiences in storage.

        Returns:
            Number of stored experiences
        """
        return self.current_size

    def save(self, path: str) -> None:
        """
        Save experiences to a file.

        Args:
            path: Path to save the file
        """
        # Save data as pickle
        with open(f"{path}_data.pkl", "wb") as f:
            pickle.dump(self.memory, f)

        # Save metadata as JSON
        metadata = {
            "current_size": self.current_size,
            "max_size": self.max_size,
            "next_idx": self.next_idx,
        }

        with open(f"{path}_metadata.json", "w") as f:
            json.dump(metadata, f)

    def load(self, path: str) -> None:
        """
        Load experiences from a file.

        Args:
            path: Path to load the file from
        """
        import os

        # Load data
        data_path = f"{path}_data.pkl"
        meta_path = f"{path}_metadata.json"

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "rb") as f:
            self.memory = pickle.load(f)

        # Load metadata if available
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
                self.current_size = metadata.get("current_size", 0)
                self.max_size = metadata.get("max_size", self.max_size)
                self.next_idx = metadata.get("next_idx", 0)
        else:
            # Estimate size from the data
            if self.memory:
                first_key = next(iter(self.memory))
                self.current_size = len(self.memory[first_key])
                self.max_size = max(self.max_size, self.current_size)
                self.next_idx = self.current_size % self.max_size

    def clear(self) -> None:
        """
        Clear all experiences from storage.
        """
        self.memory = {}
        self.current_size = 0
        self.next_idx = 0
