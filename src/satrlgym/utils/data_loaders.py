import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .type_system import TypeConverter, TypeSpecification

logger = logging.getLogger(__name__)


class ExperienceTorchDataset(Dataset):
    """
    PyTorch Dataset for loading experience data with automatic type conversion.
    Provides typed tensors for deep learning models with flexible configuration.
    """

    def __init__(
        self,
        data_path: str | Path,
        type_specs: dict[str, TypeSpecification] | None = None,
        transform: Callable | None = None,
        fields: list[str] | None = None,
        max_size: int | None = None,
    ):
        """
        Initialize the dataset with automatic type conversion.

        Args:
            data_path: Path to the dataset
            type_specs: Dictionary mapping field names to type specifications
            transform: Optional transform function to apply to each sample
            fields: List of fields to load (if None, load all)
            max_size: Maximum number of transitions to load (if None, load all)
        """
        try:
            from .storage import ExperienceReader
        except ImportError:
            logger.error("Could not import ExperienceReader")
            raise

        self.data_path = Path(data_path)
        self.transform = transform
        self.fields = fields
        self.type_specs = type_specs or {}
        self.type_converter = TypeConverter(type_specs)

        # Load data
        logger.info(f"Loading data from {data_path} into typed PyTorch dataset")

        # Cache data in memory for fast random access
        self.data = []

        with ExperienceReader(data_path) as reader:
            for i, batch in enumerate(reader.iter_batches(1000)):
                for transition in batch:
                    # Filter fields if needed
                    if fields is not None:
                        transition = {
                            k: v for k, v in transition.items() if k in fields
                        }

                    # Convert types according to specifications
                    typed_transition = self.type_converter.convert(transition)

                    self.data.append(typed_transition)

                    if max_size is not None and len(self.data) >= max_size:
                        break

                if max_size is not None and len(self.data) >= max_size:
                    break

        logger.info(f"Loaded {len(self.data)} typed transitions")

    def __len__(self) -> int:
        """Return the number of transitions."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a transition by index with proper typing.

        Args:
            idx: Index of the transition

        Returns:
            Transition data as a dictionary with typed torch tensors
        """
        transition = self.data[idx]

        # Convert numpy arrays to torch tensors
        torch_transition = {}
        for key, value in transition.items():
            if isinstance(value, np.ndarray):
                torch_transition[key] = torch.from_numpy(value).contiguous()
            else:
                torch_transition[key] = value

        # Apply transform if provided
        if self.transform is not None:
            torch_transition = self.transform(torch_transition)

        return torch_transition
