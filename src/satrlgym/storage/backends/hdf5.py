"""
HDF5-based experience storage backend.

This module implements an experience storage backend using HDF5 storage format,
which is suitable for structured data with hierarchical organization.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from satrlgym.storage.base import ExperienceStorage


class HDF5ExperienceStorage(ExperienceStorage):
    """
    Store transitions in HDF5 format.

    HDF5 (Hierarchical Data Format) is suitable for storing structured data
    with hierarchical organization. This backend is good for large datasets
    with complex nested structures.
    """

    def __init__(
        self,
        data_path: str | Path,
        compression: str | None = "gzip",
        chunk_size: int = 1000,
        **kwargs: Any,
    ):
        """
        Initialize HDF5ExperienceStorage.

        Args:
            data_path: Path to store the HDF5 file
            compression: Compression algorithm to use (None for no compression)
            chunk_size: Size of chunks for HDF5 datasets
            **kwargs: Additional arguments
        """
        # Initialize the base class with just self
        super().__init__()

        # Set data_path manually
        self.data_path = Path(data_path)
        self.compression = compression
        self.chunk_size = chunk_size
        self.file = None
        self.is_open = False
        self.group_counter = 0
        self.transition_counter = 0
        self.buffer = []  # Buffer for storing transitions
        self.buffer_count = 0  # Added buffer_count attribute

        # Open the file
        self._open_file()

    def _open_file(self):
        """Open the HDF5 file and initialize datasets as needed."""
        if self.is_open:
            return

        # Create or open the file
        file_exists = os.path.exists(self.data_path)
        self.file = h5py.File(self.data_path, "a")
        self.is_open = True

        # Initialize counter for sequential storage
        if "transition_count" not in self.file.attrs:
            self.file.attrs["transition_count"] = 0

        # Store metadata
        if not file_exists and hasattr(self, "metadata"):
            metadata_str = json.dumps(self.metadata)
            self.file.attrs["metadata"] = metadata_str

    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a single transition to the storage.

        Args:
            transition: A dictionary containing transition data
        """
        self.buffer.append(transition)
        self.buffer_count += 1

        # Write to disk if buffer is full
        if self.buffer_count >= self.chunk_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered transitions to disk."""
        if not self.buffer:
            return

        # Make sure file is open
        self._open_file()

        # Get current position in the file
        start_idx = self.file.attrs["transition_count"]

        # Process buffer into groups by structure
        grouped_data = {}
        for transition in self.buffer:
            # Use structure as key for grouping
            structure_key = tuple(sorted(transition.keys()))
            if structure_key not in grouped_data:
                grouped_data[structure_key] = []
            grouped_data[structure_key].append(transition)

        # Write each group to its own dataset
        for structure_key, transitions in grouped_data.items():
            group_name = f"group_{hash(structure_key) % 10000}"

            # Create group if it doesn't exist
            if group_name not in self.file:
                self.file.create_group(group_name)

                # Store structure info
                self.file[group_name].attrs["keys"] = json.dumps(list(structure_key))

            # Get group reference
            group = self.file[group_name]

            # Process each key in the structure
            for key in structure_key:
                # Extract the values for this key from all transitions
                values = [t.get(key) for t in transitions]

                # Convert to numpy array
                if isinstance(values[0], (np.ndarray, list)):
                    values = np.array(values)
                    # If object dtype, serialize to avoid HDF5 errors
                    if values.dtype == np.dtype("O"):
                        serialized_values = []
                        for val in values:
                            if isinstance(val, np.ndarray) and val.ndim > 1:
                                # Pickle multi-dimensional arrays
                                serialized_values.append(pickle.dumps(val))
                            else:
                                # For simpler objects
                                serialized_values.append(pickle.dumps(val))
                        values = np.array(serialized_values, dtype=np.dtype("S"))
                else:
                    # Handle scalar values
                    values = np.array(values)
                    # If object dtype, serialize
                    if values.dtype == np.dtype("O"):
                        values = np.array(
                            [pickle.dumps(v) for v in values], dtype=np.dtype("S")
                        )

                # Determine dataset path
                dataset_path = f"{key}_{start_idx}"

                # Create or extend dataset
                group.create_dataset(
                    dataset_path,
                    data=values,
                    compression=self.compression,
                    compression_opts=None if self.compression != "gzip" else 4,
                )

                # Store index range in dataset attributes
                group[dataset_path].attrs["start_idx"] = start_idx
                group[dataset_path].attrs["end_idx"] = start_idx + len(transitions)

        # Update transition count
        self.file.attrs["transition_count"] += len(self.buffer)

        # Clear buffer
        self.buffer = []
        self.buffer_count = 0

        # Ensure data is written to disk
        self.file.flush()

    def write_batch(self, batch_data: dict[str, Any]) -> None:
        """
        Write a batch of experience data to storage.

        Args:
            batch_data: Dictionary containing batch data to be written
        """
        # Convert batch data to individual transitions
        batch_size = None
        for key, value in batch_data.items():
            if batch_size is None:
                if isinstance(value, np.ndarray):
                    batch_size = value.shape[0]
                elif isinstance(value, list):
                    batch_size = len(value)

            if not isinstance(value, (np.ndarray, list)):
                # Broadcast scalar to array
                batch_data[key] = np.full(batch_size, value)

        # Create individual transitions and add to buffer
        for i in range(batch_size):
            transition = {key: value[i] for key, value in batch_data.items()}
            self.add_transition(transition)

    def close(self) -> None:
        """Close the storage and save any remaining data."""
        if self.buffer:
            self._flush_buffer()

        if self.file and self.is_open:
            self.file.close()
            self.file = None
            self.is_open = False

    def __del__(self):
        """Ensure file is closed when object is deleted."""
        self.close()

    def _deserialize_if_needed(self, data):
        """
        Deserialize data if it was serialized during storage.

        Args:
            data: Data to check and potentially deserialize

        Returns:
            Deserialized data if serialized, otherwise original data
        """
        if isinstance(data, np.ndarray) and data.dtype.kind == "S":
            # This might be serialized data
            try:
                return np.array([pickle.loads(item) for item in data])
            except Exception:
                # If deserialization fails, return as is
                return data
        return data

    def read_transition(self, index: int) -> dict[str, Any]:
        """
        Read a single transition from storage.

        Args:
            index: Index of transition to read

        Returns:
            Dictionary containing transition data
        """
        self._open_file()

        if not self.is_open:
            raise ValueError("Storage is not open")

        if index >= self.file.attrs["transition_count"]:
            raise IndexError(f"Index {index} out of range")

        transition = {}

        # Search through all groups for fields with this index
        for group_name in self.file:
            if not group_name.startswith("group_"):
                continue

            group = self.file[group_name]

            # Check each dataset in the group
            for dataset_name in group:
                dataset = group[dataset_name]

                # Check if this dataset covers our index
                if "start_idx" in dataset.attrs and "end_idx" in dataset.attrs:
                    start = dataset.attrs["start_idx"]
                    end = dataset.attrs["end_idx"]

                    if start <= index < end:
                        # Get the real field name (strip index)
                        field_name = dataset_name.split("_")[0]

                        # Extract value for this index
                        value = dataset[index - start]

                        # Deserialize if needed
                        value = self._deserialize_if_needed(value)

                        transition[field_name] = value

        return transition

    def read_batch(self, start_idx: int, batch_size: int) -> dict[str, Any]:
        """
        Read a batch of transitions.

        Args:
            start_idx: Starting index
            batch_size: Number of transitions to read

        Returns:
            Dictionary with arrays of transitions
        """
        self._open_file()

        if not self.is_open:
            raise ValueError("Storage is not open")

        total_count = self.file.attrs["transition_count"]
        if start_idx >= total_count:
            raise IndexError(f"Start index {start_idx} out of range")

        # Adjust batch_size if necessary
        batch_size = min(batch_size, total_count - start_idx)

        batch = {}

        # Search through all groups for fields with this index range
        for group_name in self.file:
            if not group_name.startswith("group_"):
                continue

            group = self.file[group_name]

            # Check each dataset in the group
            for dataset_name in group:
                dataset = group[dataset_name]

                # Check if this dataset overlaps our index range
                if "start_idx" in dataset.attrs and "end_idx" in dataset.attrs:
                    ds_start = dataset.attrs["start_idx"]
                    ds_end = dataset.attrs["end_idx"]

                    if ds_start <= start_idx + batch_size - 1 and start_idx < ds_end:
                        # Calculate overlap range
                        overlap_start = max(start_idx, ds_start)
                        overlap_end = min(start_idx + batch_size, ds_end)

                        # Get the real field name (strip index)
                        field_name = dataset_name.split("_")[0]

                        # Extract values for this index range
                        values = dataset[
                            overlap_start - ds_start : overlap_end - ds_start
                        ]

                        # Deserialize if needed
                        values = self._deserialize_if_needed(values)

                        # Create padding if needed for partial overlap
                        if (
                            overlap_start > start_idx
                            or overlap_end < start_idx + batch_size
                        ):
                            if field_name not in batch:
                                batch[field_name] = np.zeros(
                                    batch_size, dtype=values.dtype
                                )
                            batch[field_name][
                                overlap_start - start_idx : overlap_end - start_idx
                            ] = values
                        else:
                            batch[field_name] = values

        return batch
