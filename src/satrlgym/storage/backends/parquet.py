"""
Parquet-based experience storage backend.

This module implements an experience storage backend using the Apache Parquet format,
which provides efficient columnar storage with compression.
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from satrlgym.storage.base import ExperienceStorage
from satrlgym.storage.utils.serialization import encode_numpy_objects


class ParquetExperienceStorage(ExperienceStorage):
    """
    Store transitions in Parquet files.

    Parquet is a columnar storage format that's efficient for storing
    tabular data. This backend is suitable for large datasets that
    need efficient storage and querying capabilities.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 1000,
        compression: str = "snappy",
        **kwargs: Any,
    ):
        """
        Initialize ParquetExperienceStorage.

        Args:
            data_path: Path to store the Parquet files
            batch_size: Number of transitions to buffer before writing
            compression: Compression algorithm to use
            **kwargs: Additional arguments
        """
        # Initialize the base class with just self
        super().__init__()

        # Set data_path manually
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.compression = compression
        self.buffer = []
        self.batch_counter = 0
        self.is_open = True

        # Create the directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)

    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a transition to the storage.

        Args:
            transition: Dictionary containing transition data
        """
        if not self.is_open:
            raise ValueError("Cannot add transition to closed storage")

        self.buffer.append(transition)

        # Flush buffer if it reaches the batch size
        if len(self.buffer) >= self.batch_size:
            self._flush_buffer()

    def write_batch(
        self, batch: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Write a batch of data to storage.

        Args:
            batch: Dictionary with arrays of transition components
            metadata: Optional metadata to store with the batch
        """
        if not self.is_open:
            raise ValueError("Cannot write batch to closed storage")

        # Convert batch to individual transitions and add to buffer
        batch_size = self._get_batch_size(batch)
        for i in range(batch_size):
            transition = {}
            for key, value in batch.items():
                # Get the i-th item from each array
                if isinstance(value, np.ndarray):
                    transition[key] = value[i]
                elif isinstance(value, list) and len(value) > i:
                    transition[key] = value[i]
                else:
                    # Handle scalar values
                    transition[key] = value

            # Add metadata if provided
            if metadata:
                for meta_key, meta_value in metadata.items():
                    if meta_key not in transition:
                        transition[f"_meta_{meta_key}"] = meta_value

            self.buffer.append(transition)

        # Flush buffer if it reaches the batch size
        if len(self.buffer) >= self.batch_size:
            self._flush_buffer()

    def _get_batch_size(self, batch: dict[str, Any]) -> int:
        """
        Get the size of a batch by examining the first array.

        Args:
            batch: Dictionary with arrays of transition components

        Returns:
            Size of the batch
        """
        # Find the first array-like value
        for value in batch.values():
            if hasattr(value, "__len__") and not isinstance(value, (str, dict)):
                return len(value)

        # If no array is found, return 0
        return 0

    def _flush_buffer(self) -> None:
        """Flush the buffer to Parquet files."""
        if not self.buffer:
            return

        # Ensure the directory exists
        os.makedirs(self.data_path, exist_ok=True)

        try:
            # Convert buffer to columns for Parquet
            columns = {}
            for transition in self.buffer:
                # Encode numpy arrays and complex objects
                encoded_transition = encode_numpy_objects(transition)

                # Add fields to columns
                for key, value in encoded_transition.items():
                    if key not in columns:
                        columns[key] = []
                    columns[key].append(value)

            # Pre-process data to handle multi-dimensional arrays
            processed_columns = {}

            for name, values in columns.items():
                # Check for multi-dimensional arrays or complex objects
                has_multidim = any(
                    isinstance(v, np.ndarray) and v.ndim > 1
                    for v in values
                    if isinstance(v, np.ndarray)
                )
                has_complex_objects = any(
                    isinstance(v, (dict, list, tuple)) for v in values
                )

                if has_multidim or has_complex_objects:
                    # Serialize complex types
                    serialized_values = []
                    for val in values:
                        # Handle different types that need serialization
                        serialized_values.append(pickle.dumps(val))
                    processed_columns[name] = serialized_values

                    # Add a metadata field to indicate this is serialized
                    if name not in processed_columns:
                        processed_columns[name + "_metadata"] = [
                            "serialized:pickle"
                        ] * len(values)
                else:
                    # Keep simple types as is
                    processed_columns[name] = values

            # Convert columns to PyArrow arrays
            arrays = []
            fields = []

            for name, values in processed_columns.items():
                try:
                    # Try standard conversion for simple types
                    arrays.append(pa.array(values))
                    fields.append(pa.field(name, arrays[-1].type))
                except (TypeError, pa.ArrowInvalid) as e:
                    if "serialized" in name or any(
                        isinstance(v, bytes) for v in values
                    ):
                        # For serialized data, use binary type
                        arrays.append(pa.array(values, type=pa.binary()))
                        fields.append(pa.field(name, pa.binary()))
                    else:
                        warnings.warn(
                            f"Could not convert column {name} to Arrow array: {e}"
                        )
                        # Skip this column
                        if len(arrays) > len(fields):
                            arrays.pop()

            # Create table and write to Parquet
            if arrays:
                table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
                file_path = self.data_path / f"batch_{self.batch_counter:06d}.parquet"

                # Ensure the parent directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write the table
                pq.write_table(table, file_path, compression=self.compression)
                self.batch_counter += 1

            # Clear buffer
            self.buffer.clear()

        except Exception as e:
            warnings.warn(f"Error writing Parquet file: {e}")
            # Don't clear buffer on error so data isn't lost

    def close(self) -> None:
        """Close the storage and flush any remaining data."""
        if self.is_open:
            self._flush_buffer()
            self.is_open = False

    def __enter__(self) -> "ParquetExperienceStorage":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and close storage."""
        self.close()
