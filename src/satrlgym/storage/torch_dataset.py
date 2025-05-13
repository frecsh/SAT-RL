"""
PyTorch dataset implementations for experience replay data.
"""

from collections.abc import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from .base import ExperienceStorage


class ExperienceIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming experience data.

    This dataset is suitable for large datasets that don't fit in memory
    or when you need to iterate through the data only once.
    """

    def __init__(
        self, storage: ExperienceStorage, batch_size: int = 32, shuffle: bool = True
    ):
        """
        Initialize the PyTorch IterableDataset.

        Args:
            storage: Experience storage instance
            batch_size: Number of experiences per batch
            shuffle: Whether to shuffle the data
        """
        self.storage = storage
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = storage.get_size()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Iterate through the experience data.

        Yields:
            Batches of experience data as PyTorch tensors
        """
        if self.shuffle:
            # Generate shuffled indices for the entire dataset
            indices = torch.randperm(self.size).tolist()
        else:
            indices = range(self.size)

        # Yield data in batches
        for start_idx in range(0, self.size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]

            # Get batch from storage
            batch_data = self.storage.get(batch_indices)

            # Convert numpy arrays to PyTorch tensors
            torch_batch = {
                k: torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }

            yield torch_batch


class ExperienceDataset(Dataset):
    """
    PyTorch Dataset for experience data.

    This dataset is suitable when you need random access to your data
    and when the dataset can fit in memory.
    """

    def __init__(self, storage: ExperienceStorage):
        """
        Initialize the PyTorch Dataset.

        Args:
            storage: Experience storage instance
        """
        self.storage = storage
        self.size = storage.get_size()

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Dictionary of PyTorch tensors
        """
        # Get the single item from storage
        item_data = self.storage.get([idx])

        # Convert numpy arrays to PyTorch tensors and remove batch dimension
        torch_item = {
            k: torch.from_numpy(v[0]) if not isinstance(v[0], torch.Tensor) else v[0]
            for k, v in item_data.items()
        }

        return torch_item
