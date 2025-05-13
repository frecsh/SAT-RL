"""
Base classes for storage implementations.

This module defines the abstract base classes for storage implementations.
"""

import abc
from typing import Any


class StorageBase(abc.ABC):
    """Abstract base class for all storage backends."""

    def __init__(self):
        """Initialize the storage base class."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the storage and perform cleanup."""

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close storage."""
        self.close()


class ExperienceStorage(StorageBase):
    """
    Abstract base class for experience storage backends.

    This class defines the interface for storing and retrieving
    experience transitions.
    """

    def __init__(self):
        """Initialize the experience storage."""
        super().__init__()
        self.data_path = None

    @abc.abstractmethod
    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a transition to the storage.

        Args:
            transition: Dictionary containing transition data
        """

    @abc.abstractmethod
    def write_batch(self, batch: dict[str, Any]) -> None:
        """
        Write a batch of data to storage.

        Args:
            batch: Dictionary with arrays of transition components
        """
