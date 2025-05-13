"""PyTorch integration code for experience data.

Provides utilities for using experience data with PyTorch.
"""

import logging
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from satrlgym.storage import create_storage

# Set up logging
logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, IterableDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExperienceMapDataset(Dataset):
    """Map-style PyTorch Dataset for random access to experience data.

    This dataset provides random access to stored experience data, suitable
    for training models that don't require sequential data access.
    """

    def __init__(
        self,
        storage_path: str,
        storage_type: str = "parquet",
        fields: list[str] | None = None,
        transform: Callable | None = None,
        compression_config: dict[str, Any] | None = None,
    ):
        """Initialize a map-style dataset for experience data.

        Args:
            storage_path: Path to the storage file
            storage_type: Type of storage ('parquet', 'hdf5', or 'mmap')
            fields: Optional list of field names to load (None for all)
            transform: Optional transform function to apply to loaded data
            compression_config: Optional compression configuration
        """
        self.storage = create_storage(storage_type, storage_path, compression_config)
        self.fields = fields
        self.transform = transform
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata and index information."""
        # Load a sample batch to determine dataset structure
        sample = self.storage.read_batch(self.fields)
        if not sample:
            raise ValueError(f"No data found in storage at {self.storage.path}")

        # Calculate dataset size
        first_key = next(iter(sample.keys()))
        self.size = len(sample[first_key])

        # Check all arrays have same first dimension
        for key, value in sample.items():
            if len(value) != self.size:
                warnings.warn(
                    f"Inconsistent array sizes in storage: {key} has size {len(value)}"
                )
                self.size = min(self.size, len(value))

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item by index.

        Args:
            idx: Index of the item to get

        Returns:
            Dictionary of field name to tensor
        """
        batch = self.storage.read_batch(self.fields)
        if not batch:
            raise IndexError(f"No data found for index {idx}")

        # Extract the item at the given index from each array
        item = {}
        for k, v in batch.items():
            # Handle scalar values (like int64)
            if isinstance(v[idx], np.ndarray):
                item[k] = torch.from_numpy(v[idx])
            else:
                # Convert scalar to numpy array then to tensor
                item[k] = torch.tensor(v[idx])

        if self.transform:
            item = self.transform(item)

        return item


class ExperienceIterableDataset(IterableDataset):
    """Iterable PyTorch Dataset for streaming experience data.

    This dataset provides sequential access to stored experience data,
    suitable for training models that can process data in a streaming fashion.
    """

    def __init__(
        self,
        storage_path: str,
        storage_type: str = "parquet",
        fields: list[str] | None = None,
        transform: Callable | None = None,
        compression_config: dict[str, Any] | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        """Initialize an iterable dataset for experience data.

        Args:
            storage_path: Path to the storage file
            storage_type: Type of storage ('parquet', 'hdf5', or 'mmap')
            fields: Optional list of field names to load (None for all)
            transform: Optional transform function to apply to loaded data
            compression_config: Optional compression configuration
            batch_size: Size of batches to read from storage
            shuffle: Whether to shuffle the data
        """
        self.storage = create_storage(storage_type, storage_path, compression_config)
        self.fields = fields
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Create an iterator over the dataset.

        Returns:
            Iterator yielding dictionary of field name to tensor
        """
        # Read all data once
        batch = self.storage.read_batch(self.fields)
        if not batch:
            raise StopIteration("No data found in storage")

        # Create dataset from the batch
        dataset = []
        for i in range(len(next(iter(batch.values())))):
            item = {k: v[i] for k, v in batch.items()}
            dataset.append(item)

        # Shuffle if needed
        if self.shuffle:
            indices = torch.randperm(len(dataset)).tolist()
        else:
            indices = range(len(dataset))

        # Yield items
        for idx in indices:
            item = {}
            for k, v in dataset[idx].items():
                # Handle scalar values (like int64)
                if isinstance(v, np.ndarray):
                    item[k] = torch.from_numpy(v)
                else:
                    # Convert scalar to tensor
                    item[k] = torch.tensor(v)

            if self.transform:
                item = self.transform(item)

            yield item


def collate_experience(batch, pad_sequences=False, pad_value=0, time_dim=1):
    """Collate experience data into a batch.

    This function handles various data types and structures commonly found
    in experience data, including sequences of different lengths.

    Args:
        batch: List of dictionaries containing experience data
        pad_sequences: Whether to pad sequences to the same length
        pad_value: Value to use for padding
        time_dim: Dimension representing time in sequence data (typically 1)

    Returns:
        Dictionary of batched tensors
    """
    if not batch:
        return {}

    # Get all keys from the first item
    keys = batch[0].keys()

    result = {}
    for key in keys:
        items = [b[key] for b in batch]

        # Handle different types
        if isinstance(items[0], torch.Tensor):
            # For tensors, check if they're all the same shape
            shapes = [i.shape for i in items]

            # If all tensors have the same shape, stack them
            if len(set(shapes)) == 1:
                result[key] = torch.stack(items)
            else:
                # Different shapes - handle padding if needed
                if pad_sequences and len(shapes[0]) > 1:
                    # Find max sequence length
                    max_len = max([s[time_dim] for s in shapes])

                    # Pad sequences to the same length
                    padded_items = []
                    for item in items:
                        if item.shape[time_dim] < max_len:
                            # Create padding dimensions
                            pad_dims = [(0, 0)] * len(item.shape)
                            pad_dims[time_dim] = (0, max_len - item.shape[time_dim])

                            # Flatten the pad_dims for torch.nn.functional.pad
                            flat_pad_dims = []
                            for d in reversed(pad_dims):
                                flat_pad_dims.extend(d)

                            # Pad the tensor
                            padded = torch.nn.functional.pad(
                                item, pad=tuple(flat_pad_dims), value=pad_value
                            )
                            padded_items.append(padded)
                        else:
                            padded_items.append(item)

                    try:
                        result[key] = torch.stack(padded_items)
                    except RuntimeError as e:
                        # If stacking still fails, return as list
                        logger.warning(f"Failed to stack padded tensors for {key}: {e}")
                        result[key] = padded_items
                else:
                    # Without padding, just keep as a list
                    result[key] = items
        elif isinstance(items[0], (int, float, bool)):
            # For scalars, convert to tensor
            result[key] = torch.tensor(items)
        elif isinstance(items[0], np.ndarray):
            # For numpy arrays, convert to tensors first
            if all(i.shape == items[0].shape for i in items):
                # Same shape - stack after conversion
                result[key] = torch.stack([torch.from_numpy(i) for i in items])
            else:
                # Different shapes - handle as with tensors
                if pad_sequences:
                    tensors = [torch.from_numpy(i) for i in items]
                    # Reuse tensor padding logic
                    result[key] = collate_experience(
                        [{key: t} for t in tensors],
                        pad_sequences=True,
                        pad_value=pad_value,
                        time_dim=time_dim,
                    )[key]
                else:
                    result[key] = [torch.from_numpy(i) for i in items]
        else:
            # For other types, keep as a list
            result[key] = items

    return result


def create_experience_dataloader(
    storage_path: str,
    storage_type: str = "parquet",
    fields: list[str] | None = None,
    transform: Callable | None = None,
    compression_config: dict[str, Any] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    stream: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader for experience data.

    Args:
        storage_path: Path to the storage file
        storage_type: Type of storage ('parquet', 'hdf5', or 'mmap')
        fields: Optional list of field names to load (None for all)
        transform: Optional transform function to apply to loaded data
        compression_config: Optional compression configuration
        batch_size: Size of batches to yield
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        stream: If True, use an IterableDataset, otherwise use a MapDataset

    Returns:
        PyTorch DataLoader
    """
    if stream:
        dataset = ExperienceIterableDataset(
            storage_path=storage_path,
            storage_type=storage_type,
            fields=fields,
            transform=transform,
            compression_config=compression_config,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        dataset = ExperienceMapDataset(
            storage_path=storage_path,
            storage_type=storage_type,
            fields=fields,
            transform=transform,
            compression_config=compression_config,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=None if stream else batch_size,  # Set to None for IterableDataset
        shuffle=(
            False if stream else shuffle
        ),  # Shuffling is handled by the IterableDataset
        num_workers=num_workers,
        collate_fn=None if stream else collate_experience,
    )


class ExperienceDataset(Dataset):
    """Simple PyTorch Dataset for experience data in memory."""

    def __init__(
        self,
        experience_data: dict[str, np.ndarray],
        device: str | torch.device = "cpu",
        transform_fn: callable | None = None,
    ):
        """Initialize dataset with experience data.

        Args:
            experience_data: Dictionary of experience data arrays
            device: Device to load tensors to
            transform_fn: Optional function to transform items
        """
        self.data = {}
        for key, value in experience_data.items():
            self.data[key] = torch.tensor(value, device=device)

        self.transform_fn = transform_fn

        # Get length from the first array
        self.length = len(next(iter(self.data.values())))

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> dict:
        """Get a single item by index."""
        item = {k: v[idx] for k, v in self.data.items()}
        if self.transform_fn:
            item = self.transform_fn(item)
        return item


def create_torch_dataset(
    experience_data: dict[str, np.ndarray],
    device: str | torch.device = "cpu",
    transform_fn: callable | None = None,
) -> "ExperienceDataset":
    """Create a PyTorch dataset from experience data.

    Args:
        experience_data: Dictionary of experience data arrays
        device: Device to load tensors to
        transform_fn: Optional function to transform items

    Returns:
        PyTorch dataset
    """
    if not TORCH_AVAILABLE:
        warnings.warn(
            "PyTorch is not available. Please install torch to use this function."
        )
        return None

    return ExperienceDataset(experience_data, device, transform_fn)
