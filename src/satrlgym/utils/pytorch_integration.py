"""
PyTorch integration utilities for RL datasets.
Provides Dataset classes for loading data in PyTorch.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .type_system import TypeConverter, TypeSpecification

logger = logging.getLogger(__name__)


class RLMapDataset(Dataset):
    """
    PyTorch map-style Dataset for random access to RL data.
    Best used for smaller datasets or when random sampling is needed.
    """

    def __init__(
        self,
        data_path: str | Path,
        transform: Callable | None = None,
        fields: list[str] | None = None,
        max_size: int | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the dataset
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

        # Load data
        logger.info(f"Loading data from {data_path} into PyTorch map-style dataset")

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

                    self.data.append(transition)

                    if max_size is not None and len(self.data) >= max_size:
                        break

                if max_size is not None and len(self.data) >= max_size:
                    break

        logger.info(f"Loaded {len(self.data)} transitions")

    def __len__(self) -> int:
        """Return the number of transitions."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a transition by index.

        Args:
            idx: Index of the transition

        Returns:
            Transition data as a dictionary
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


class RLIterableDataset(IterableDataset):
    """
    PyTorch iterable Dataset for streaming RL data.
    Best used for large datasets that don't fit in memory.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 100,
        transform: Callable | None = None,
        fields: list[str] | None = None,
        shuffle: bool = True,
        max_size: int | None = None,
        repeat: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the dataset
            batch_size: Number of transitions to load at once
            transform: Optional transform function to apply to each sample
            fields: List of fields to load (if None, load all)
            shuffle: Whether to shuffle the batches
            max_size: Maximum number of transitions to return (if None, return all)
            repeat: Whether to repeat the dataset when it's exhausted
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.transform = transform
        self.fields = fields
        self.shuffle = shuffle
        self.max_size = max_size
        self.repeat = repeat

        logger.info(f"Initializing PyTorch iterable dataset for {data_path}")

    def __iter__(self):
        """Return an iterator over the dataset."""
        try:
            from .storage import ExperienceReader
        except ImportError:
            logger.error("Could not import ExperienceReader")
            raise

        # Create worker-specific random seed
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = (
            np.random.randint(0, 2**32) if worker_info is None else worker_info.id
        )
        rng = np.random.RandomState(worker_seed)

        logger.debug(f"Starting iterator (worker_seed={worker_seed})")

        # Keep track of returned transitions
        transitions_returned = 0

        while True:  # Loop will break when dataset is exhausted or max_size is reached
            with ExperienceReader(self.data_path) as reader:
                # Get list of all batches
                if self.shuffle:
                    # This might be memory-intensive for very large datasets
                    # A more scalable approach would be to divide the dataset into chunks
                    try:
                        batch_indices = list(range(reader.get_batch_count()))
                        rng.shuffle(batch_indices)

                        for idx in batch_indices:
                            if (
                                self.max_size is not None
                                and transitions_returned >= self.max_size
                            ):
                                return

                            batch = reader.get_batch(idx)

                            # Filter fields if needed
                            if self.fields is not None:
                                batch = [
                                    {
                                        k: v
                                        for k, v in transition.items()
                                        if k in self.fields
                                    }
                                    for transition in batch
                                ]

                            for transition in batch:
                                # Convert numpy arrays to torch tensors
                                torch_transition = {}
                                for key, value in transition.items():
                                    if isinstance(value, np.ndarray):
                                        torch_transition[key] = torch.from_numpy(
                                            value.copy()
                                        ).contiguous()
                                    else:
                                        torch_transition[key] = value

                                # Apply transform if provided
                                if self.transform is not None:
                                    torch_transition = self.transform(torch_transition)

                                yield torch_transition
                                transitions_returned += 1

                                if (
                                    self.max_size is not None
                                    and transitions_returned >= self.max_size
                                ):
                                    return
                    except Exception as e:
                        logger.warning(f"Shuffling by indices failed: {e}")
                        logger.warning("Falling back to sequential reading")
                        # Reset reader
                        reader.close()
                        reader = ExperienceReader(self.data_path)

                # Sequential reading (if shuffle failed or was disabled)
                for batch in reader.iter_batches(self.batch_size):
                    if (
                        self.max_size is not None
                        and transitions_returned >= self.max_size
                    ):
                        return

                    # Filter fields if needed
                    if self.fields is not None:
                        batch = [
                            {k: v for k, v in transition.items() if k in self.fields}
                            for transition in batch
                        ]

                    # Optionally shuffle batch
                    if self.shuffle:
                        rng.shuffle(batch)

                    for transition in batch:
                        # Convert numpy arrays to torch tensors
                        torch_transition = {}
                        for key, value in transition.items():
                            if isinstance(value, np.ndarray):
                                torch_transition[key] = torch.from_numpy(
                                    value.copy()
                                ).contiguous()
                            else:
                                torch_transition[key] = value

                        # Apply transform if provided
                        if self.transform is not None:
                            torch_transition = self.transform(torch_transition)

                        yield torch_transition
                        transitions_returned += 1

                        if (
                            self.max_size is not None
                            and transitions_returned >= self.max_size
                        ):
                            return

            if not self.repeat:
                break


class NStepTransitionDataset(Dataset):
    """
    Dataset for n-step transitions with customizable return calculation.
    Computes n-step returns on-the-fly from single-step transitions.
    """

    def __init__(
        self,
        data_path: str | Path,
        n_step: int = 1,
        gamma: float = 0.99,
        compute_returns: bool = True,
        max_size: int | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the dataset
            n_step: Number of steps to look ahead for returns
            gamma: Discount factor
            compute_returns: Whether to compute n-step returns
            max_size: Maximum number of transitions to load (if None, load all)
        """
        try:
            from .storage import ExperienceReader
        except ImportError:
            logger.error("Could not import ExperienceReader")
            raise

        self.data_path = Path(data_path)
        self.n_step = n_step
        self.gamma = gamma
        self.compute_returns = compute_returns

        # Load data
        logger.info(
            f"Loading data from {data_path} into n-step transition dataset (n={n_step})"
        )

        # Load all transitions
        self.transitions = []

        with ExperienceReader(data_path) as reader:
            # Group by episode
            episodes = []
            current_episode = []

            for batch in reader.iter_batches(1000):
                for transition in batch:
                    # Check if this is a new episode
                    if current_episode and transition.get("done", False):
                        episodes.append(current_episode)
                        current_episode = []

                    current_episode.append(transition)

                    if (
                        max_size is not None
                        and sum(len(ep) for ep in episodes) + len(current_episode)
                        >= max_size
                    ):
                        break

                if (
                    max_size is not None
                    and sum(len(ep) for ep in episodes) + len(current_episode)
                    >= max_size
                ):
                    break

            # Add the last episode if not empty
            if current_episode:
                episodes.append(current_episode)

            # Process episodes and create n-step transitions
            for episode in episodes:
                episode_len = len(episode)

                for i in range(episode_len):
                    if i + n_step >= episode_len:
                        # If we're near the end of the episode, use what's left
                        next_idx = episode_len - 1
                    else:
                        next_idx = i + n_step

                    # Compute n-step return
                    if compute_returns:
                        n_step_return = 0
                        for j in range(i, next_idx + 1):
                            # Apply gamma discount
                            n_step_return += (self.gamma ** (j - i)) * episode[j].get(
                                "reward", 0
                            )

                    # Create n-step transition
                    n_step_transition = {
                        "observation": episode[i].get("observation"),
                        "action": episode[i].get("action"),
                        "next_observation": episode[next_idx].get(
                            "next_observation", episode[next_idx].get("observation")
                        ),
                        "done": episode[next_idx].get("done", False),
                    }

                    if compute_returns:
                        n_step_transition["n_step_return"] = n_step_return

                    # Add to transitions
                    self.transitions.append(n_step_transition)

                    if max_size is not None and len(self.transitions) >= max_size:
                        break

                if max_size is not None and len(self.transitions) >= max_size:
                    break

        logger.info(f"Created {len(self.transitions)} n-step transitions")

    def __len__(self) -> int:
        """Return the number of n-step transitions."""
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get an n-step transition by index.

        Args:
            idx: Index of the transition

        Returns:
            N-step transition data as a dictionary with torch tensors
        """
        transition = self.transitions[idx]

        # Convert numpy arrays to torch tensors
        torch_transition = {}
        for key, value in transition.items():
            if isinstance(value, np.ndarray):
                torch_transition[key] = torch.from_numpy(value).contiguous()
            elif isinstance(value, (float, int, bool)):
                torch_transition[key] = torch.tensor(value)
            else:
                torch_transition[key] = value

        return torch_transition


# Common transforms
class Standardize:
    """Transform to standardize numerical observations to zero mean and unit variance."""

    def __init__(
        self,
        mean: float | np.ndarray | torch.Tensor,
        std: float | np.ndarray | torch.Tensor,
        fields: list[str] = ["observation", "next_observation"],
    ):
        """
        Initialize the transform.

        Args:
            mean: Mean value(s) for standardization
            std: Standard deviation value(s) for standardization
            fields: List of fields to apply standardization to
        """
        if isinstance(mean, np.ndarray):
            self.mean = torch.from_numpy(mean).float()
        elif isinstance(mean, (int, float)):
            self.mean = torch.tensor(mean, dtype=torch.float32)
        else:
            self.mean = mean.float()

        if isinstance(std, np.ndarray):
            self.std = torch.from_numpy(std).float()
        elif isinstance(std, (int, float)):
            self.std = torch.tensor(std, dtype=torch.float32)
        else:
            self.std = std.float()

        self.fields = fields

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Apply standardization to the sample.

        Args:
            sample: Dictionary containing transition data

        Returns:
            Standardized sample
        """
        result = sample.copy()

        for field in self.fields:
            if field in result and isinstance(result[field], torch.Tensor):
                result[field] = (result[field] - self.mean) / (self.std + 1e-8)

        return result


class RewardScaling:
    """Transform to scale rewards."""

    def __init__(self, scale: float = 1.0, reward_field: str = "reward"):
        """
        Initialize the transform.

        Args:
            scale: Scaling factor for rewards
            reward_field: Field name for rewards
        """
        self.scale = scale
        self.reward_field = reward_field

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Apply reward scaling to the sample.

        Args:
            sample: Dictionary containing transition data

        Returns:
            Sample with scaled rewards
        """
        result = sample.copy()

        if self.reward_field in result:
            if isinstance(result[self.reward_field], torch.Tensor):
                result[self.reward_field] = result[self.reward_field] * self.scale
            else:
                result[self.reward_field] = result[self.reward_field] * self.scale

        return result


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
        self.type_converter = TypeConverter()

        # Load data
        logger.info(f"Loading data from {data_path} into typed PyTorch dataset")

        # Cache data in memory for fast random access
        self.data = []

        # Create reader without using context manager
        reader = ExperienceReader(data_path)
        try:
            for i, batch in enumerate(reader.iter_batches(1000)):
                for transition in batch:
                    # Filter fields if needed
                    if fields is not None:
                        transition = {
                            k: v for k, v in transition.items() if k in fields
                        }

                    # Convert types according to specifications
                    if type_specs:
                        # Apply type conversions manually if specs are provided
                        typed_transition = {}
                        for key, value in transition.items():
                            if key in self.type_specs:
                                # Apply type conversion based on specification
                                spec = self.type_specs[key]
                                if hasattr(
                                    self.type_converter, f"convert_to_{spec.type}"
                                ):
                                    convert_method = getattr(
                                        self.type_converter, f"convert_to_{spec.type}"
                                    )
                                    typed_transition[key] = convert_method(value)
                                else:
                                    typed_transition[key] = value
                            else:
                                typed_transition[key] = value
                    else:
                        typed_transition = transition

                    self.data.append(typed_transition)

                    if max_size is not None and len(self.data) >= max_size:
                        break

                if max_size is not None and len(self.data) >= max_size:
                    break
        finally:
            # Make sure to close the reader if it has a close method
            if hasattr(reader, "close"):
                reader.close()

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
                # Create a contiguous copy to avoid read-only memory issues
                torch_transition[key] = torch.from_numpy(value.copy()).contiguous()
            elif (
                isinstance(value, list)
                and key == "observation"
                and all(isinstance(x, (int, float)) for x in value)
            ):
                # Convert observation lists to tensors
                torch_transition[key] = torch.tensor(value, dtype=torch.float32)
            else:
                torch_transition[key] = value

        # Apply transform if provided
        if self.transform is not None:
            torch_transition = self.transform(torch_transition)

        return torch_transition

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader to stack tensors properly.

        Args:
            batch: List of individual samples

        Returns:
            Properly batched data as a dictionary of tensors
        """
        collated = {}

        # Get all keys from the batch
        keys = batch[0].keys()

        for key in keys:
            # Stack/batch all tensors, handle other types appropriately
            batch_values = [sample[key] for sample in batch]

            if all(isinstance(x, torch.Tensor) for x in batch_values):
                # If all are tensors, stack them
                try:
                    collated[key] = torch.stack(batch_values)
                except BaseException:
                    # If tensors have different shapes (e.g., different sequence lengths)
                    # Keep them as a list of tensors
                    collated[key] = batch_values
            elif all(isinstance(x, np.ndarray) for x in batch_values):
                # If they're numpy arrays, convert to tensors and stack
                try:
                    tensors = [
                        torch.from_numpy(x.copy()).contiguous() for x in batch_values
                    ]
                    collated[key] = torch.stack(tensors)
                except BaseException:
                    # If arrays have different shapes
                    collated[key] = [
                        torch.from_numpy(x.copy()).contiguous() for x in batch_values
                    ]
            elif (
                all(isinstance(x, list) for x in batch_values) and key == "observation"
            ):
                # Special handling for observation lists
                try:
                    tensors = [
                        torch.tensor(x, dtype=torch.float32) for x in batch_values
                    ]
                    collated[key] = torch.stack(tensors)
                except BaseException:
                    # If lists have different lengths
                    collated[key] = tensors
            else:
                # For other types, keep as a list
                collated[key] = batch_values

        return collated
