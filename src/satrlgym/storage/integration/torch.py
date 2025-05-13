"""
PyTorch integration for experience data.

This module provides PyTorch-specific components for working with stored experience
data, including dataset and DataLoader implementations.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset
except ImportError:
    raise ImportError(
        "PyTorch is required for this module. "
        "Please install it with 'pip install torch'."
    )


# Import necessary storage components
from ..io.reader import ExperienceReader

# Set up logging
logger = logging.getLogger(__name__)


class ExperienceTensorConverter:
    """
    Convert experience dictionaries to PyTorch tensors.

    This class handles the conversion of NumPy arrays and other data types
    to PyTorch tensors, with appropriate device placement.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype_mapping: dict[str, torch.dtype] | None = None,
        float_dtype: torch.dtype = torch.float32,
        int_dtype: torch.dtype = torch.int64,
        exclude_keys: list[str] | None = None,
    ):
        """
        Initialize the tensor converter.

        Args:
            device: PyTorch device to place tensors on
            dtype_mapping: Mapping from field names to PyTorch dtypes
            float_dtype: Default dtype for floating point values
            int_dtype: Default dtype for integer values
            exclude_keys: List of keys to exclude from conversion
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.dtype_mapping = dtype_mapping or {}
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.exclude_keys = exclude_keys or []

    def __call__(self, experience_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an experience dictionary to tensors.

        Args:
            experience_dict: Dictionary with experience data

        Returns:
            Dictionary with tensors
        """
        result = {}

        for key, value in experience_dict.items():
            # Skip excluded keys
            if key in self.exclude_keys:
                result[key] = value
                continue

            # Convert to tensor based on type
            if isinstance(value, np.ndarray):
                # Use dtype mapping if available for this key
                if key in self.dtype_mapping:
                    dtype = self.dtype_mapping[key]
                elif np.issubdtype(value.dtype, np.floating):
                    dtype = self.float_dtype
                elif np.issubdtype(value.dtype, np.integer):
                    dtype = self.int_dtype
                else:
                    # Default handling
                    dtype = None

                # Convert to tensor
                result[key] = torch.from_numpy(value).to(
                    device=self.device, dtype=dtype
                )
            elif isinstance(value, (int, np.integer)):
                result[key] = torch.tensor(
                    value, dtype=self.int_dtype, device=self.device
                )
            elif isinstance(value, (float, np.floating)):
                result[key] = torch.tensor(
                    value, dtype=self.float_dtype, device=self.device
                )
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # Attempt to convert lists/tuples to tensors
                if all(isinstance(item, (int, np.integer)) for item in value):
                    result[key] = torch.tensor(
                        value, dtype=self.int_dtype, device=self.device
                    )
                elif all(isinstance(item, (float, np.floating)) for item in value):
                    result[key] = torch.tensor(
                        value, dtype=self.float_dtype, device=self.device
                    )
                else:
                    # Keep as is for mixed types
                    result[key] = value
            else:
                # Keep non-numeric data as is
                result[key] = value

        return result


class ExperienceIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for experience data.

    This dataset provides a streaming interface to experience data stored in various
    backend formats, allowing efficient loading and processing for training.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 32,
        shuffle: bool = True,
        transforms: list[Callable] | None = None,
        tensor_converter: ExperienceTensorConverter | None = None,
        storage_type: str | None = None,
        reader_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the experience data
            batch_size: Size of batches to yield
            shuffle: Whether to shuffle the data
            transforms: List of transform callables to apply to each batch
            tensor_converter: Converter to transform data to PyTorch tensors
            storage_type: Type of storage backend ('hdf5', 'parquet', etc.)
            reader_kwargs: Additional arguments for the ExperienceReader
            **kwargs: Additional arguments for the dataset
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms or []

        # Convert to tensors by default
        if tensor_converter is None:
            self.tensor_converter = ExperienceTensorConverter(**kwargs)
        else:
            self.tensor_converter = tensor_converter

        # Add tensor conversion as a transform
        self.transforms.append(self.tensor_converter)

        # Determine storage type if not specified
        if storage_type is None:
            storage_type = self._detect_storage_type(data_path)

        # Set up ExperienceReader
        reader_kwargs = reader_kwargs or {}
        self.reader = ExperienceReader(
            data_path=data_path, storage_type=storage_type, **reader_kwargs
        )

        # Initialize batching parameters
        self.buffer_size = kwargs.get("buffer_size", 10000)

    def _detect_storage_type(self, data_path: str | Path) -> str:
        """
        Detect the storage type from the file extension or directory structure.

        Args:
            data_path: Path to examine

        Returns:
            Detected storage type as string

        Raises:
            ValueError: If storage type cannot be determined
        """
        data_path = Path(data_path)

        # Check file extension
        if data_path.is_file():
            if data_path.suffix in [".h5", ".hdf5"]:
                return "hdf5"
            elif data_path.suffix == ".parquet":
                return "parquet"
            elif data_path.suffix == ".json":
                return "file"

        # Check directory structure
        elif data_path.is_dir():
            # Check for parquet files
            if any(data_path.glob("*.parquet")):
                return "parquet"
            # Check for HDF5 files
            elif any(data_path.glob("*.h5")) or any(data_path.glob("*.hdf5")):
                return "hdf5"
            # Check for memory mapped files
            elif (data_path / "mmap_arrays").exists():
                return "memory"
            # Check for individual JSON files
            elif any(data_path.glob("transition_*.json")):
                return "file"

        raise ValueError(
            f"Could not determine storage type from {data_path}. "
            "Please specify storage_type explicitly."
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over batches of experience data.

        Returns:
            Iterator yielding batches of experience data as dictionaries
        """
        # Create worker-specific reader for multi-process DataLoader
        worker_info = torch.utils.data.get_worker_info()

        # Handle multi-worker case
        if worker_info is not None:
            # Split the dataset among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Configure reader for this worker
            worker_reader = self.reader.create_worker_reader(
                worker_id=worker_id, num_workers=num_workers
            )
        else:
            # Single worker case
            worker_reader = self.reader

        # Initialize buffer for batching
        buffer = []

        # Iterate over transitions
        for transition in worker_reader:
            buffer.append(transition)

            # When buffer is full or for last incomplete batch
            if len(buffer) >= self.batch_size:
                # Create a batch
                batch = self._create_batch(buffer[: self.batch_size])
                buffer = buffer[self.batch_size :]

                # Apply transforms to the batch
                for transform in self.transforms:
                    batch = transform(batch)

                yield batch

        # Handle any remaining data in buffer
        if buffer:
            batch = self._create_batch(buffer)

            # Apply transforms to the batch
            for transform in self.transforms:
                batch = transform(batch)

            yield batch

    def _create_batch(self, transitions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert a list of transitions to a batch dictionary.

        Args:
            transitions: List of transition dictionaries

        Returns:
            A dictionary with batched arrays
        """
        # Group by key
        batch = {}
        for transition in transitions:
            for key, value in transition.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)

        # Convert lists to numpy arrays
        for key, values in batch.items():
            try:
                batch[key] = np.array(values)
            except Exception:
                # Keep as list if conversion fails
                pass

        return batch


class ExperienceDataLoader(DataLoader):
    """
    DataLoader for experience data.

    This is a convenience wrapper around torch.utils.data.DataLoader
    preconfigured for use with ExperienceIterableDataset.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        transforms: list[Callable] | None = None,
        tensor_converter: ExperienceTensorConverter | None = None,
        storage_type: str | None = None,
        **kwargs,
    ):
        """
        Initialize the DataLoader.

        Args:
            data_path: Path to the experience data
            batch_size: Size of batches to yield
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            transforms: List of transform callables to apply to each batch
            tensor_converter: Converter to transform data to PyTorch tensors
            storage_type: Type of storage backend ('hdf5', 'parquet', etc.)
            **kwargs: Additional arguments for the dataset and dataloader
        """
        # Extract reader kwargs from kwargs
        reader_kwargs = kwargs.pop("reader_kwargs", {})

        # Create the dataset
        dataset = ExperienceIterableDataset(
            data_path=data_path,
            batch_size=batch_size,
            shuffle=shuffle,
            transforms=transforms,
            tensor_converter=tensor_converter,
            storage_type=storage_type,
            reader_kwargs=reader_kwargs,
            **kwargs,
        )

        # Initialize the DataLoader
        super().__init__(
            dataset=dataset,
            batch_size=None,  # Batching is handled by the dataset
            num_workers=num_workers,
            **kwargs,
        )
