"""
HDF5 implementation of experience storage.
"""

from typing import Any

import h5py
import numpy as np

from .base import ExperienceStorage


class HDF5ExperienceStorage(ExperienceStorage):
    """
    Experience storage implementation using HDF5 files.

    This class provides storage for experience data in HDF5 format,
    which is efficient for large datasets and supports compression.
    """

    def __init__(
        self,
        path: str = None,
        compression: str = "gzip",
        compression_level: int = 4,
        max_size: int = None,
    ):
        """
        Initialize HDF5 experience storage.

        Args:
            path: Path to HDF5 file. If None, uses in-memory storage.
            compression: Compression algorithm ("gzip", "lzf", None)
            compression_level: Compression level (1-9, higher=more compression)
            max_size: Maximum number of experiences to store (None=unlimited)
        """
        self.path = path
        self.compression = compression
        self.compression_level = compression_level
        self.max_size = max_size
        self._file = None
        self._size = 0

        if path is not None:
            self._file = h5py.File(path, "a")
        else:
            self._file = h5py.File("memory:", "a")

    def add(self, experience_batch: dict[str, Any]) -> None:
        """Add experiences to storage."""
        if self._size == 0:
            # Create datasets for the first batch
            for key, value in experience_batch.items():
                shape = list(value.shape)
                dtype = value.dtype
                maxshape = [None] + shape[1:]

                # Create dataset with compression if enabled
                if self.compression:
                    self._file.create_dataset(
                        key,
                        shape=shape,
                        maxshape=maxshape,
                        dtype=dtype,
                        compression=self.compression,
                        compression_opts=self.compression_level,
                    )
                else:
                    self._file.create_dataset(
                        key, shape=shape, maxshape=maxshape, dtype=dtype
                    )

                # Store the data
                self._file[key][:] = value

            self._size = experience_batch[list(experience_batch.keys())[0]].shape[0]
        else:
            # Resize datasets and append data
            for key, value in experience_batch.items():
                batch_size = value.shape[0]
                dataset = self._file[key]
                current_size = dataset.shape[0]
                new_size = current_size + batch_size

                # Check if we need to enforce max_size
                if self.max_size is not None and new_size > self.max_size:
                    # Remove oldest entries to make room
                    overflow = new_size - self.max_size
                    # Shift existing data left
                    dataset[:-overflow] = dataset[overflow:]
                    # Add new data at the end
                    dataset[-batch_size:] = value
                    self._size = self.max_size
                else:
                    # Simply extend the dataset
                    dataset.resize((new_size,) + dataset.shape[1:])
                    dataset[current_size:new_size] = value
                    self._size = new_size

    def get(self, indices: list[int] | None = None) -> dict[str, Any]:
        """Retrieve experiences from storage."""
        result = {}

        if self._size == 0:
            return result

        if indices is None:
            # Get all data
            for key in self._file.keys():
                result[key] = self._file[key][:]
        else:
            # Get specified indices
            for key in self._file.keys():
                result[key] = self._file[key][indices]

        return result

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a random batch of experiences."""
        if self._size == 0:
            return {}

        indices = np.random.choice(
            self._size, size=min(batch_size, self._size), replace=False
        )
        return self.get(indices)

    def get_size(self) -> int:
        """Get the number of experiences in storage."""
        return self._size

    def save(self, path: str) -> None:
        """Save experiences to a file."""
        if self.path == path:
            self._file.flush()
            return

        with h5py.File(path, "w") as f:
            for key in self._file.keys():
                if self.compression:
                    f.create_dataset(
                        key,
                        data=self._file[key][:],
                        compression=self.compression,
                        compression_opts=self.compression_level,
                    )
                else:
                    f.create_dataset(key, data=self._file[key][:])

    def load(self, path: str) -> None:
        """Load experiences from a file."""
        if self._file is not None:
            self._file.close()

        self.path = path
        self._file = h5py.File(path, "a")

        # Determine size from the first dataset
        if len(self._file.keys()) > 0:
            first_key = list(self._file.keys())[0]
            self._size = self._file[first_key].shape[0]
        else:
            self._size = 0

    def clear(self) -> None:
        """Clear all experiences from storage."""
        if self._file is not None:
            for key in list(self._file.keys()):
                del self._file[key]
            self._size = 0

    def __del__(self):
        """Clean up resources when the object is deleted."""
        if self._file is not None:
            self._file.close()
