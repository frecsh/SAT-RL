"""
PyTorch integration for experience replay data.

This module provides PyTorch Dataset implementations for efficient
training with experience replay data.
"""

import random
from collections.abc import Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Create placeholder classes to avoid errors
    class IterableDataset:
        pass

    class Dataset:
        pass


from experience.storage.base import ExperienceStorage


class ExperienceIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming experience data from storage.

    This dataset provides an iterable interface for PyTorch DataLoader,
    allowing efficient streaming of experience data during training.
    """

    def __init__(
        self,
        storage: ExperienceStorage,
        batch_size: int = 32,
        transforms: dict[str, Callable] | None = None,
        device: str = "cpu",
        infinite: bool = True,
        shuffle: bool = True,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            storage: Experience storage instance to read data from
            batch_size: Size of batches to yield
            transforms: Optional dictionary of transform functions keyed by field name
            device: PyTorch device to load tensors to ('cpu', 'cuda', etc.)
            infinite: Whether to iterate infinitely
            shuffle: Whether to shuffle the data on each iteration
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ExperienceIterableDataset. "
                "Please install it with: pip install torch"
            )

        self.storage = storage
        self.batch_size = batch_size
        self.transforms = transforms or {}
        self.device = torch.device(device)
        self.infinite = infinite
        self.shuffle = shuffle

    def __iter__(self):
        """
        Create an iterator over the experience data.

        Returns:
            Iterator yielding batches of experience data as PyTorch tensors
        """
        worker_info = torch.utils.data.get_worker_info()

        # Get the total size of the dataset
        dataset_size = self.storage.get_size()

        if dataset_size == 0:
            # Empty dataset
            return

        # Handle multi-worker DataLoader
        if worker_info is not None:
            # Split workload among workers
            per_worker = int(np.ceil(dataset_size / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, dataset_size)
            range_size = end_idx - start_idx
        else:
            # Single worker case
            start_idx = 0
            range_size = dataset_size

        # Create indices for iteration
        indices = list(range(start_idx, start_idx + range_size))

        # Main iteration loop
        iteration = 0
        while True:
            # Shuffle indices if requested
            if self.shuffle:
                random.shuffle(indices)

            # Yield batches
            for i in range(0, len(indices), self.batch_size):
                # Get batch indices
                batch_indices = indices[i : i + self.batch_size]

                # Get batch data from storage
                batch = self.storage.get(batch_indices)

                # Apply transforms
                transformed_batch = {}
                for key, value in batch.items():
                    # Apply transform if one exists for this key
                    if key in self.transforms:
                        value = self.transforms[key](value)

                    # Convert to PyTorch tensor
                    try:
                        if isinstance(value, np.ndarray):
                            tensor = torch.from_numpy(value).to(self.device)
                        elif isinstance(value, (list, tuple)) and all(
                            isinstance(x, (int, float)) for x in value
                        ):
                            tensor = torch.tensor(value, device=self.device)
                        else:
                            # For complex data types, just pass as-is
                            tensor = value
                    except BaseException:
                        # If conversion fails, pass as-is
                        tensor = value

                    transformed_batch[key] = tensor

                yield transformed_batch

            iteration += 1

            # Break if not infinite
            if not self.infinite:
                break


class ExperienceBatchDataset(Dataset):
    """
    PyTorch Dataset for batch access to experience data.

    Unlike the IterableDataset, this provides indexed access to each
    individual experience entry, useful for non-sequential training.
    """

    def __init__(
        self,
        storage: ExperienceStorage,
        transforms: dict[str, Callable] | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the dataset.

        Args:
            storage: Experience storage instance to read data from
            transforms: Optional dictionary of transform functions keyed by field name
            device: PyTorch device to load tensors to ('cpu', 'cuda', etc.)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ExperienceBatchDataset. "
                "Please install it with: pip install torch"
            )

        self.storage = storage
        self.transforms = transforms or {}
        self.device = torch.device(device)
        self.size = storage.get_size()

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
            Number of experience entries
        """
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single experience entry.

        Args:
            idx: Index of the experience to retrieve

        Returns:
            Dictionary of tensors representing the experience
        """
        # Get the item from storage
        item = self.storage.get([idx])

        # Apply transforms and convert to tensors
        result = {}
        for key, value in item.items():
            # Apply transform if one exists for this key
            if key in self.transforms:
                value = self.transforms[key](value[0])
            else:
                value = value[0]  # Remove batch dimension

            # Convert to PyTorch tensor
            try:
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value).to(self.device)
                elif isinstance(value, (list, tuple)) and all(
                    isinstance(x, (int, float)) for x in value
                ):
                    tensor = torch.tensor(value, device=self.device)
                else:
                    # For complex data types, just pass as-is
                    tensor = value
            except BaseException:
                # If conversion fails, pass as-is
                tensor = value

            result[key] = tensor

        return result
