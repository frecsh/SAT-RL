"""
NPZ Storage Adapter for experience replay data.

This module provides storage and retrieval of experience data in NumPy's .npz format.
"""

import os
from typing import Any

import numpy as np

from .base import StorageBase


class NPZStorage(StorageBase):
    """Storage adapter for NumPy's .npz compressed file format."""

    def __init__(self, path: str):
        """
        Initialize the NPZ storage adapter.

        Args:
            path: Path to the NPZ file
        """
        super().__init__()
        self.path = path
        self.metadata = {}

    def write_batch(self, batch: dict[str, Any]) -> None:
        """
        Write a batch of experience data to the NPZ file.

        Args:
            batch: Dictionary of arrays to write
        """
        # Create a copy of the batch to avoid modifying the original
        data_to_save = dict(batch)

        # Add metadata if it exists
        if hasattr(self, "metadata") and self.metadata:
            data_to_save["__metadata__"] = np.array([str(self.metadata)])

        # Save to NPZ file
        np.savez_compressed(self.path, **data_to_save)

    def read_batch(
        self, batch_size: int | None = None, offset: int = 0
    ) -> dict[str, Any]:
        """
        Read a batch of experience data from the NPZ file.

        Args:
            batch_size: Number of transitions to read (None for all)
            offset: Starting index for reading

        Returns:
            Dictionary containing the batch data
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"NPZ file not found: {self.path}")

        with np.load(self.path, allow_pickle=True) as data:
            # Load all fields
            result = {}
            for key in data.files:
                if key == "__metadata__":
                    # Load metadata separately
                    try:
                        self.metadata = eval(str(data[key][0]))
                    except BaseException:
                        self.metadata = {"note": "Failed to parse metadata"}
                    continue

                # Load the field data
                field_data = data[key]

                # Apply batch_size and offset if requested
                if batch_size is not None and offset is not None:
                    if len(field_data.shape) > 0:  # Only slice if not scalar
                        end_idx = min(offset + batch_size, len(field_data))
                        field_data = field_data[offset:end_idx]

                result[key] = field_data

        return result

    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata associated with the dataset.

        Returns:
            Dictionary containing metadata
        """
        # If metadata hasn't been loaded yet, try to load it
        if not hasattr(self, "metadata") or not self.metadata:
            try:
                with np.load(self.path, allow_pickle=True) as data:
                    if "__metadata__" in data.files:
                        self.metadata = eval(str(data["__metadata__"][0]))
            except BaseException:
                self.metadata = {}

        return self.metadata

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Set metadata to be stored with the dataset.

        Args:
            metadata: Dictionary containing metadata to store
        """
        self.metadata = metadata

        # If the file already exists, update it with the new metadata
        if os.path.exists(self.path):
            # Load existing data
            batch = self.read_batch()
            # Write back with new metadata
            self.write_batch(batch)

    def append_batch(self, batch: dict[str, Any]) -> None:
        """
        Append a batch to existing NPZ file.

        Args:
            batch: Dictionary of arrays to append
        """
        # Check if file already exists
        if os.path.exists(self.path):
            # Load existing data
            existing_batch = self.read_batch()

            # Merge batches
            merged_batch = {}
            for key in set(list(existing_batch.keys()) + list(batch.keys())):
                if key in existing_batch and key in batch:
                    merged_batch[key] = np.concatenate(
                        [existing_batch[key], batch[key]], axis=0
                    )
                elif key in existing_batch:
                    merged_batch[key] = existing_batch[key]
                else:
                    merged_batch[key] = batch[key]

            # Write merged batch
            self.write_batch(merged_batch)
        else:
            # Just write the batch directly
            self.write_batch(batch)
