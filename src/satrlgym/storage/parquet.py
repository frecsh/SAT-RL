"""
Parquet-based implementation of experience storage.

This module provides a storage backend using Apache Parquet format,
which is efficient for columnar data storage and retrieval.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

from .base import ExperienceStorage


class ParquetExperienceStorage(ExperienceStorage):
    """
    Stores experiences in Parquet files for efficient columnar access.
    """

    def __init__(self, data_path: str | Path, **kwargs):
        """
        Initialize Parquet experience storage.

        Args:
            data_path: Path to store data
            **kwargs: Additional arguments
        """
        super().__init__(data_path, **kwargs)

        if not PARQUET_AVAILABLE:
            raise ImportError(
                "Parquet storage requires pandas and pyarrow. "
                "Please install them with: pip install pandas pyarrow"
            )

        # Make sure the parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)

        # Make sure it has .parquet extension
        if not str(data_path).endswith(".parquet"):
            self.data_path = Path(str(data_path) + ".parquet")

        self.metadata = {
            "format_version": "1.0",
            "episode_count": 0,
            "transition_count": 0,
        }

    def add_transition(self, transition: dict[str, Any]) -> None:
        """
        Add a single transition to storage.

        Args:
            transition: Dictionary with transition data
        """
        # Create DataFrame for single transition
        df = pd.DataFrame([transition])

        # Create or append to parquet file
        if os.path.exists(self.data_path):
            # Read existing file to append
            existing_df = pd.read_parquet(self.data_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(self.data_path, engine="pyarrow")
        else:
            # Create new file
            df.to_parquet(self.data_path, engine="pyarrow")

        # Update metadata
        self.metadata["transition_count"] += 1
        if transition.get("done", False):
            self.metadata["episode_count"] += 1

    def write_batch(self, batch_data: dict[str, Any]) -> None:
        """
        Write a batch of transitions to storage.
        This method takes transition data in columnar format and adds it to storage.

        Args:
            batch_data: Dictionary with transition data in columnar format
                       (each key maps to a list/array of values)
        """
        # Verify that all arrays have the same length
        first_key = next(iter(batch_data))
        batch_size = len(batch_data[first_key])

        for key, values in batch_data.items():
            if len(values) != batch_size:
                raise ValueError(
                    f"All arrays must have the same length. {key} has length {len(values)}, expected {batch_size}"
                )

        # Convert NumPy arrays to lists for easier pandas handling
        processed_data = {}
        for key, values in batch_data.items():
            if isinstance(values, np.ndarray):
                if values.ndim > 1:
                    # Store multidimensional arrays as lists of lists
                    processed_data[key] = [arr.tolist() for arr in values]
                else:
                    # Store 1D arrays as is
                    processed_data[key] = values.tolist()
            else:
                processed_data[key] = values

        # Create DataFrame from batch data
        df = pd.DataFrame(processed_data)

        # Create or append to parquet file
        if os.path.exists(self.data_path):
            # Read existing file to append
            existing_df = pd.read_parquet(self.data_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(self.data_path, engine="pyarrow")
        else:
            # Create new file
            df.to_parquet(self.data_path, engine="pyarrow")

        # Update metadata
        self.metadata["transition_count"] += batch_size
        if "dones" in batch_data:
            self.metadata["episode_count"] += sum(
                1 for done in batch_data["dones"] if done
            )

    def get_all(self) -> dict[str, Any]:
        """
        Get all transitions from storage.

        Returns:
            Dictionary with transition data in columnar format
        """
        if not os.path.exists(self.data_path):
            return {}

        # Read all data
        df = pd.read_parquet(self.data_path)

        # Convert to columnar format (dictionary of arrays)
        result = {}
        for column in df.columns:
            # Convert back to numpy arrays
            if isinstance(df[column].iloc[0], list):
                # Handle multi-dimensional arrays stored as lists
                result[column] = np.array(df[column].tolist())
            else:
                # Handle scalar columns
                result[column] = df[column].to_numpy()

        return result

    def get_batch(self, indices: list[int]) -> dict[str, Any]:
        """
        Get specific transitions by indices.

        Args:
            indices: List of indices to retrieve

        Returns:
            Dictionary with transition data in columnar format
        """
        if not os.path.exists(self.data_path) or not indices:
            return {}

        # Read only the specified rows
        df = pd.read_parquet(self.data_path)

        # Filter to requested indices
        if max(indices) >= len(df):
            valid_indices = [idx for idx in indices if idx < len(df)]
            if not valid_indices:
                return {}
            df = df.iloc[valid_indices]
        else:
            df = df.iloc[indices]

        # Convert to columnar format
        result = {}
        for column in df.columns:
            # Convert back to numpy arrays
            if isinstance(df[column].iloc[0], list):
                # Handle multi-dimensional arrays stored as lists
                result[column] = np.array(df[column].tolist())
            else:
                # Handle scalar columns
                result[column] = df[column].to_numpy()

        return result

    def close(self) -> None:
        """Close storage and save any remaining data."""
        # Nothing to do for Parquet storage, as data is written immediately
