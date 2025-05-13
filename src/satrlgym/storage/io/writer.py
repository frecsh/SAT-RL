"""
Experience data writer.

This module provides classes for writing experience data to various storage backends.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.file import FileExperienceStorage

# Import storage backends
from ..backends.hdf5 import HDF5ExperienceStorage
from ..backends.memory import MemoryMappedExperienceStorage
from ..backends.parquet import ParquetExperienceStorage
from ..factory import create_storage

# Set up logging
logger = logging.getLogger(__name__)


class ExperienceWriter:
    """
    Writer for experience data to various storage backends.

    This class provides a unified interface for writing experience data
    to different storage backends, handling backend-specific details.
    """

    def __init__(
        self,
        data_path: str | Path,
        storage_type: str = "hdf5",
        buffer_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize the writer.

        Args:
            data_path: Path where data will be stored
            storage_type: Type of storage backend ('hdf5', 'parquet', 'memory', 'file')
            buffer_size: Number of transitions to buffer before writing to disk
            **kwargs: Additional backend-specific parameters
        """
        self.data_path = Path(data_path)
        self.storage_type = storage_type
        self.buffer_size = buffer_size

        # Create storage backend
        storage_kwargs = {"buffer_size": buffer_size, **kwargs}

        # Create appropriate storage backend
        if storage_type == "hdf5":
            self.storage = HDF5ExperienceStorage(data_path, **storage_kwargs)
        elif storage_type == "parquet":
            self.storage = ParquetExperienceStorage(data_path, **storage_kwargs)
        elif storage_type == "memory":
            self.storage = MemoryMappedExperienceStorage(data_path, **storage_kwargs)
        elif storage_type == "file":
            self.storage = FileExperienceStorage(data_path, **storage_kwargs)
        else:
            # In future, this will use the registry mechanism
            try:
                self.storage = create_storage(storage_type, data_path, **storage_kwargs)
            except ValueError:
                raise ValueError(f"Unsupported storage type: {storage_type}")

    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a single transition to storage.

        Args:
            transition: Dictionary containing transition data
        """
        self.storage.add_transition(transition)

    def write(self, transition: dict[str, Any]) -> None:
        """
        Write a single transition.

        Args:
            transition: Dictionary containing transition data
        """
        self.add_transition(transition)

    def add_batch(self, batch: dict[str, Any]) -> None:
        """
        Add a batch of transitions to storage.

        Args:
            batch: Dictionary containing batched data where each value
                  is a list or array with the same first dimension
        """
        self.storage.write_batch(batch)

    def add_experience(
        self,
        states: list | np.ndarray,
        actions: list | np.ndarray,
        rewards: list | np.ndarray,
        next_states: list | np.ndarray,
        dones: list | np.ndarray,
        **kwargs,
    ) -> None:
        """
        Add standard RL experience tuples to storage.

        This is a convenience method for the common (s, a, r, s', done) format.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags (True/False or 0/1)
            **kwargs: Additional data to store with each transition
        """
        # Convert to numpy arrays if needed
        if not isinstance(states, np.ndarray):
            states = np.array(states)
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        if not isinstance(next_states, np.ndarray):
            next_states = np.array(next_states)
        if not isinstance(dones, np.ndarray):
            dones = np.array(dones)

        # Create batch dictionary
        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            **kwargs,
        }

        # Add batch to storage
        self.add_batch(batch)

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Add metadata to the experience storage.

        Args:
            metadata: Dictionary containing metadata
        """
        # Store metadata in the storage backend if supported
        if hasattr(self.storage, "add_metadata"):
            self.storage.add_metadata(metadata)
        elif hasattr(self.storage, "metadata"):
            # If the backend has a metadata attribute but no add method
            if not hasattr(self.storage, "_metadata"):
                self.storage._metadata = {}
            self.storage._metadata.update(metadata)
        else:
            # At minimum, log the metadata
            logger.info(f"Adding metadata to storage: {metadata}")
            # Create a metadata attribute if it doesn't exist
            if not hasattr(self.storage, "_metadata"):
                self.storage._metadata = {}
            self.storage._metadata.update(metadata)

    def close(self) -> None:
        """Close the storage and save any remaining data."""
        if hasattr(self, "storage"):
            self.storage.close()

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()

    def __del__(self):
        """Cleanup when object is deleted."""
        self.close()
