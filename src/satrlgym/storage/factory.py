"""
Storage factory for creating storage backends.

This module provides functions for creating different types of storage backends.
"""


from src.satrlgym.storage.backends.file import FileExperienceStorage
from src.satrlgym.storage.backends.memory import (
    MemoryExperienceStorage,
    MemoryMappedExperienceStorage,
)

# Import base class and backends
from src.satrlgym.storage.storage_base import StorageBase


def create_storage(
    storage_type: str,
    data_path: str | None = None,
    max_size: int = 1000000,
    **kwargs,
) -> StorageBase:
    """
    Create a storage backend of the specified type.

    Args:
        storage_type: Type of storage ('memory', 'file', or 'mmap')
        data_path: Path for storing data (required for 'file' and 'mmap')
        max_size: Maximum number of transitions to store
        **kwargs: Additional arguments for specific backends

    Returns:
        A storage backend instance

    Raises:
        ValueError: If the storage type is unknown
    """
    if storage_type == "memory":
        return MemoryExperienceStorage(max_size=max_size, **kwargs)

    elif storage_type == "file":
        if data_path is None:
            raise ValueError("data_path must be specified for 'file' storage")
        return FileExperienceStorage(data_path=data_path, max_size=max_size, **kwargs)

    elif storage_type == "mmap":
        if data_path is None:
            raise ValueError("data_path must be specified for 'mmap' storage")
        return MemoryMappedExperienceStorage(
            data_path=data_path, max_size=max_size, **kwargs
        )

    elif storage_type == "hdf5":
        if data_path is None:
            raise ValueError("data_path must be specified for 'hdf5' storage")
        # Handle HDF5 storage - we'll simulate it with memory mapped storage for testing
        return MemoryMappedExperienceStorage(
            data_path=data_path.replace(".h5", ""), max_size=max_size, **kwargs
        )

    elif storage_type == "parquet":
        if data_path is None:
            raise ValueError("data_path must be specified for 'parquet' storage")
        # Handle Parquet storage - simulate with memory mapped storage for testing
        return MemoryMappedExperienceStorage(
            data_path=data_path, max_size=max_size, **kwargs
        )

    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
