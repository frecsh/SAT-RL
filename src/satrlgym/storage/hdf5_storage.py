"""
HDF5 Storage Adapter for experience replay data.

This module provides storage and retrieval of experience data in HDF5 format.
"""

import json
import os
from typing import Any

import h5py
import numpy as np

from .base import StorageBase


class HDF5Storage(StorageBase):
    """Storage adapter for HDF5 file format."""

    def __init__(self, path: str, compression: str | None = "gzip"):
        """
        Initialize the HDF5 storage adapter.

        Args:
            path: Path to the HDF5 file
            compression: Compression filter to use (None, 'gzip', 'lzf', or 'szip')
        """
        super().__init__()
        self.path = path
        self.compression = compression
        self.metadata = {}

    def write_batch(self, batch: dict[str, Any]) -> None:
        """
        Write a batch of experience data to the HDF5 file.

        Args:
            batch: Dictionary of arrays to write
        """
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(self.path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        # Create a copy to avoid modifying the original
        data_to_save = {}
        for key, value in batch.items():
            # Convert lists to numpy arrays
            if isinstance(value, list):
                data_to_save[key] = np.array(value)
            else:
                data_to_save[key] = value

        # Write to HDF5 file
        with h5py.File(self.path, "w") as f:
            # Add each field as a dataset
            for key, value in data_to_save.items():
                kwargs = {}
                if self.compression and isinstance(value, np.ndarray):
                    kwargs["compression"] = self.compression

                f.create_dataset(key, data=value, **kwargs)

            # Store metadata as JSON if present
            if hasattr(self, "metadata") and self.metadata:
                try:
                    metadata_json = json.dumps(self.metadata)
                    f.attrs["metadata"] = metadata_json
                except Exception as e:
                    print(f"Failed to store metadata: {e}")

    def read_batch(
        self, batch_size: int | None = None, offset: int = 0
    ) -> dict[str, Any]:
        """
        Read a batch of experience data from the HDF5 file.

        Args:
            batch_size: Number of transitions to read (None for all)
            offset: Starting index for reading

        Returns:
            Dictionary containing the batch data
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")

        result = {}

        with h5py.File(self.path, "r") as f:
            # Load metadata if present
            if "metadata" in f.attrs:
                try:
                    metadata_json = f.attrs["metadata"]
                    self.metadata = json.loads(metadata_json)
                except BaseException:
                    self.metadata = {"note": "Failed to parse metadata"}

            # Load datasets
            for key in f.keys():
                dataset = f[key]

                # Apply batch_size and offset if requested
                if batch_size is not None and offset < len(dataset):
                    end_idx = min(offset + batch_size, len(dataset))
                    result[key] = dataset[offset:end_idx]
                else:
                    result[key] = dataset[...]  # Load entire dataset

        return result

    def append_batch(self, batch: dict[str, Any]) -> None:
        """
        Append a batch to existing HDF5 file.

        Args:
            batch: Dictionary of arrays to append
        """
        # Check if file exists
        if not os.path.exists(self.path):
            # If file doesn't exist, just write the batch
            self.write_batch(batch)
            return

        # Load existing data
        existing_data = self.read_batch()

        # Create a copy to avoid modifying the original
        batch_copy = {}
        for key, value in batch.items():
            # Convert lists to numpy arrays
            if isinstance(value, list):
                batch_copy[key] = np.array(value)
            else:
                batch_copy[key] = value

        # Merge batches
        merged_batch = {}
        for key in set(list(existing_data.keys()) + list(batch_copy.keys())):
            if key in existing_data and key in batch_copy:
                # Concatenate arrays
                merged_batch[key] = np.concatenate(
                    [existing_data[key], batch_copy[key]], axis=0
                )
            elif key in existing_data:
                merged_batch[key] = existing_data[key]
            else:
                merged_batch[key] = batch_copy[key]

        # Write merged batch
        self.write_batch(merged_batch)

    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata associated with the dataset.

        Returns:
            Dictionary containing metadata
        """
        # If metadata hasn't been loaded yet, read file header
        if not hasattr(self, "metadata") or not self.metadata:
            if os.path.exists(self.path):
                try:
                    with h5py.File(self.path, "r") as f:
                        if "metadata" in f.attrs:
                            metadata_json = f.attrs["metadata"]
                            self.metadata = json.loads(metadata_json)
                except BaseException:
                    self.metadata = {}
            else:
                self.metadata = {}

        return self.metadata

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Set metadata to be stored with the dataset.

        Args:
            metadata: Dictionary containing metadata to store
        """
        self.metadata = metadata

        # If the file already exists, update metadata
        if os.path.exists(self.path):
            with h5py.File(self.path, "r+") as f:
                try:
                    metadata_json = json.dumps(metadata)
                    f.attrs["metadata"] = metadata_json
                except Exception as e:
                    print(f"Failed to update metadata: {e}")
