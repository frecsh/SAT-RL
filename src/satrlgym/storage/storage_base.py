"""
Base class for experience storage backends.

This module defines the interface for experience storage backends.
"""

from abc import ABC, abstractmethod
from typing import Any


class StorageBase(ABC):
    """
    Base class for experience storage backends.

    Storage backends are used to save and retrieve transitions
    (state, action, reward, next_state, done) for reinforcement learning.
    """

    @abstractmethod
    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a single transition to storage.

        Args:
            transition: Dictionary containing experience data
        """

    @abstractmethod
    def add_batch(self, batch: dict[str, Any]) -> None:
        """
        Add a batch of transitions to storage.

        Args:
            batch: Dictionary of batched experience data
        """

    @abstractmethod
    def sample(self, batch_size: int) -> dict[str, Any]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of batched experience data
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored transitions."""

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the storage to disk.

        Args:
            path: Path where to save the storage
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load storage from disk.

        Args:
            path: Path from where to load the storage
        """

    @abstractmethod
    def close(self) -> None:
        """Close the storage and free resources."""

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Get the number of transitions in storage.

        Returns:
            Number of stored transitions
        """

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close storage."""
        self.close()
