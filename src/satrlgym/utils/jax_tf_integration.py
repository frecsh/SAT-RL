"""
JAX and TensorFlow integration utilities for RL datasets.
Provides data loading and processing tools compatible with JAX and TensorFlow.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try importing TF and JAX, but don't fail if they're not available
try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.debug("TensorFlow not available")
    TENSORFLOW_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    logger.debug("JAX not available")
    JAX_AVAILABLE = False


class TFDatasetCreator:
    """
    Creates TensorFlow datasets from our experience data format.
    """

    @staticmethod
    def create_tf_dataset(
        data_path: str | Path,
        batch_size: int = 32,
        shuffle_buffer_size: int = 10000,
        fields: list[str] | None = None,
        prefetch_size: int = tf.data.AUTOTUNE,
        filter_fn: Callable | None = None,
        map_fn: Callable | None = None,
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from experience data.

        Args:
            data_path: Path to the experience data
            batch_size: Batch size for the dataset
            shuffle_buffer_size: Buffer size for shuffling
            fields: List of fields to include in the dataset
            prefetch_size: Number of batches to prefetch
            filter_fn: Function to filter transitions
            map_fn: Function to transform transitions

        Returns:
            TensorFlow dataset
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for this functionality")

        try:
            from .storage import ExperienceReader
        except ImportError:
            logger.error("Could not import ExperienceReader")
            raise

        logger.info(f"Creating TensorFlow dataset from {data_path}")

        # Define a generator function to yield transitions
        def generator():
            with ExperienceReader(data_path) as reader:
                for batch in reader.iter_batches(
                    100
                ):  # Read in larger batches for efficiency
                    for transition in batch:
                        # Filter fields if needed
                        if fields is not None:
                            transition = {
                                k: v for k, v in transition.items() if k in fields
                            }

                        yield transition

        # Define output types and shapes
        # We need a sample transition to determine types and shapes
        with ExperienceReader(data_path) as reader:
            sample_batch = next(reader.iter_batches(1))
            if not sample_batch:
                raise ValueError("Empty dataset")

            sample_transition = sample_batch[0]
            if fields is not None:
                sample_transition = {
                    k: v for k, v in sample_transition.items() if k in fields
                }

        # Define output types and shapes for each field
        output_types = {}
        output_shapes = {}

        for key, value in sample_transition.items():
            if isinstance(value, np.ndarray):
                output_types[key] = tf.as_dtype(value.dtype)
                output_shapes[key] = tf.TensorShape(value.shape)
            elif isinstance(value, (int, np.integer)):
                output_types[key] = tf.int64
                output_shapes[key] = tf.TensorShape([])
            elif isinstance(value, (float, np.floating)):
                output_types[key] = tf.float32
                output_shapes[key] = tf.TensorShape([])
            elif isinstance(value, (bool, np.bool_)):
                output_types[key] = tf.bool
                output_shapes[key] = tf.TensorShape([])
            else:
                # Unsupported type, try to convert to string
                output_types[key] = tf.string
                output_shapes[key] = tf.TensorShape([])

        # Create the dataset
        dataset = tf.data.Dataset.from_generator(
            generator, output_types=output_types, output_shapes=output_shapes
        )

        # Apply filter function if provided
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)

        # Apply map function if provided
        if map_fn is not None:
            dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle, batch, and prefetch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)

        return dataset

    @staticmethod
    def create_infinite_dataset(
        data_path: str | Path,
        batch_size: int = 32,
        shuffle_buffer_size: int = 10000,
        fields: list[str] | None = None,
        prefetch_size: int = tf.data.AUTOTUNE,
        filter_fn: Callable | None = None,
        map_fn: Callable | None = None,
    ) -> tf.data.Dataset:
        """
        Create an infinite TensorFlow dataset that repeats forever.

        Args:
            data_path: Path to the experience data
            batch_size: Batch size for the dataset
            shuffle_buffer_size: Buffer size for shuffling
            fields: List of fields to include in the dataset
            prefetch_size: Number of batches to prefetch
            filter_fn: Function to filter transitions
            map_fn: Function to transform transitions

        Returns:
            Infinite TensorFlow dataset
        """
        dataset = TFDatasetCreator.create_tf_dataset(
            data_path=data_path,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            fields=fields,
            prefetch_size=prefetch_size,
            filter_fn=filter_fn,
            map_fn=map_fn,
        )

        # Make the dataset repeat infinitely
        dataset = dataset.repeat()

        return dataset

    @staticmethod
    def create_episode_dataset(
        data_path: str | Path,
        batch_size: int = 1,
        shuffle: bool = True,
        episode_field: str = "episode_id",
        prefetch_size: int = tf.data.AUTOTUNE,
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset where each element is a complete episode.

        Args:
            data_path: Path to the experience data
            batch_size: Number of episodes per batch
            shuffle: Whether to shuffle episodes
            episode_field: Field to group episodes by
            prefetch_size: Number of batches to prefetch

        Returns:
            TensorFlow dataset of episodes
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for this functionality")

        try:
            from .storage import ExperienceReader
        except ImportError:
            logger.error("Could not import ExperienceReader")
            raise

        logger.info(f"Creating episode-based TensorFlow dataset from {data_path}")

        # Group transitions by episode
        episodes = []

        with ExperienceReader(data_path) as reader:
            current_episode_id = None
            current_episode = []

            for batch in reader.iter_batches(
                100
            ):  # Read in larger batches for efficiency
                for transition in batch:
                    episode_id = transition.get(episode_field, 0)

                    if current_episode_id is None:
                        current_episode_id = episode_id

                    if episode_id != current_episode_id and current_episode:
                        episodes.append(current_episode)
                        current_episode = []
                        current_episode_id = episode_id

                    current_episode.append(transition)

            # Add the last episode
            if current_episode:
                episodes.append(current_episode)

        if not episodes:
            raise ValueError("No episodes found in dataset")

        logger.info(f"Found {len(episodes)} episodes")

        # Shuffle episodes if requested
        if shuffle:
            np.random.shuffle(episodes)

        # Define a generator function to yield episodes
        def generator():
            for episode in episodes:
                yield {
                    "observations": np.array([t.get("observation") for t in episode]),
                    "actions": np.array([t.get("action") for t in episode]),
                    "rewards": np.array([t.get("reward", 0.0) for t in episode]),
                    "dones": np.array([t.get("done", False) for t in episode]),
                    "episode_length": len(episode),
                }

        # Get sample episode data for shapes and types
        sample_episode = episodes[0]
        sample_data = {
            "observations": np.array([t.get("observation") for t in sample_episode]),
            "actions": np.array([t.get("action") for t in sample_episode]),
            "rewards": np.array([t.get("reward", 0.0) for t in sample_episode]),
            "dones": np.array([t.get("done", False) for t in sample_episode]),
            "episode_length": len(sample_episode),
        }

        # Define output types and shapes
        output_types = {}
        output_shapes = {}

        for key, value in sample_data.items():
            if isinstance(value, np.ndarray):
                output_types[key] = tf.as_dtype(value.dtype)
                # Use None for the time dimension to handle variable-length episodes
                output_shapes[key] = tf.TensorShape([None] + list(value.shape[1:]))
            else:
                output_types[key] = tf.int32
                output_shapes[key] = tf.TensorShape([])

        # Create the dataset
        dataset = tf.data.Dataset.from_generator(
            generator, output_types=output_types, output_shapes=output_shapes
        )

        # Batch episodes
        if batch_size > 1:
            dataset = dataset.batch(batch_size)

        # Prefetch
        dataset = dataset.prefetch(prefetch_size)

        return dataset


class JAXDataLoader:
    """
    Data loader for JAX-based reinforcement learning algorithms.
    Provides utilities for loading and processing experience data with JAX.
    """

    @staticmethod
    def load_transitions(
        data_path: str | Path,
        batch_size: int = 32,
        shuffle: bool = True,
        fields: list[str] | None = None,
        max_transitions: int | None = None,
        use_generator: bool = False,
    ) -> dict[str, jnp.ndarray] | Iterator:
        """
        Load transitions from experience data as JAX arrays.

        Args:
            data_path: Path to the experience data
            batch_size: Size of batches to load
            shuffle: Whether to shuffle the data
            fields: Optional list of fields to load
            max_transitions: Maximum number of transitions to load
            use_generator: If True, return a generator that yields batches

        Returns:
            If use_generator is False, return a dictionary of JAX arrays
            If use_generator is True, return a generator that yields batches
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for this functionality")

        try:
            from .storage import ExperienceReader
        except ImportError:
            logger.error("Could not import ExperienceReader")
            raise

        logger.info(f"Loading transitions from {data_path}")

        # If using generator, return a function that yields batches
        if use_generator:

            def batch_generator():
                # Don't use context manager since ExperienceReader doesn't support it
                reader = ExperienceReader(data_path)
                try:
                    # Load batches
                    for batch in reader.iter_batches(batch_size):
                        # Filter fields if requested
                        if fields is not None:
                            batch = [
                                {k: v for k, v in transition.items() if k in fields}
                                for transition in batch
                            ]

                        # Convert to JAX arrays
                        jax_batch = {}
                        for key in batch[0].keys():
                            jax_batch[key] = jnp.array(
                                [transition[key] for transition in batch]
                            )

                        yield jax_batch

                        # Stop if max_transitions reached
                        if (
                            max_transitions is not None
                            and len(batch) * reader.batch_count >= max_transitions
                        ):
                            break
                finally:
                    if hasattr(reader, "close"):
                        reader.close()

            return batch_generator()
        else:
            # Load all data at once
            data = {}
            total_count = 0

            # Don't use context manager since ExperienceReader doesn't support it
            reader = ExperienceReader(data_path)
            try:
                for batch in reader.iter_batches(batch_size):
                    # Filter fields if requested
                    if fields is not None:
                        batch = [
                            {k: v for k, v in transition.items() if k in fields}
                            for transition in batch
                        ]

                    # Initialize data dictionary with first batch
                    if not data:
                        data = {key: [] for key in batch[0].keys()}

                    # Accumulate data
                    for transition in batch:
                        for key, value in transition.items():
                            data[key].append(value)
                        total_count += 1

                        # Stop if max_transitions reached
                        if (
                            max_transitions is not None
                            and total_count >= max_transitions
                        ):
                            break

                    if max_transitions is not None and total_count >= max_transitions:
                        break
            finally:
                if hasattr(reader, "close"):
                    reader.close()

            # Convert to JAX arrays
            jax_data = {}
            for key, values in data.items():
                jax_data[key] = jnp.array(values)

            # Shuffle if requested
            if shuffle and len(jax_data) > 0:
                idx = jnp.arange(len(next(iter(jax_data.values()))))
                idx = jax.random.permutation(jax.random.PRNGKey(0), idx)
                for key in jax_data:
                    jax_data[key] = jax_data[key][idx]

            return jax_data

    @staticmethod
    def create_shuffled_batches(
        data_path: str | Path,
        batch_size: int = 32,
        buffer_size: int = 10000,
        fields: list[str] | None = None,
        seed: int | None = None,
    ) -> Callable:
        """
        Create a function that returns shuffled batches from experience data.

        Args:
            data_path: Path to the experience data
            batch_size: Batch size for loading
            buffer_size: Size of the buffer for shuffling
            fields: List of fields to include
            seed: Random seed for shuffling

        Returns:
            Function that returns a batch of transitions as JAX arrays when called
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for this functionality")

        try:
            from .storage import ExperienceReader
        except ImportError:
            logger.error("Could not import ExperienceReader")
            raise

        logger.info(f"Creating shuffled batch function for {data_path}")

        # Load data into buffer
        buffer = []

        with ExperienceReader(data_path) as reader:
            for batch in reader.iter_batches(100):
                for transition in batch:
                    # Filter fields if needed
                    if fields is not None:
                        transition = {
                            k: v for k, v in transition.items() if k in fields
                        }

                    buffer.append(transition)

                    if len(buffer) >= buffer_size:
                        break

                if len(buffer) >= buffer_size:
                    break

        # Initialize random state
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        # Create batch function
        def get_batch():
            # Sample indices without replacement
            indices = rng.choice(len(buffer), batch_size, replace=False)
            batch = [buffer[idx] for idx in indices]
            return JAXDataLoader._convert_batch_to_jax(batch)

        return get_batch

    @staticmethod
    def load_episodes(
        data_path: str | Path,
        episode_field: str = "episode_id",
        max_episodes: int | None = None,
        shuffle: bool = True,
    ) -> list[dict[str, jnp.ndarray]]:
        """
        Load complete episodes from experience data.

        Args:
            data_path: Path to the experience data
            episode_field: Field to group episodes by
            max_episodes: Maximum number of episodes to load
            shuffle: Whether to shuffle episodes

        Returns:
            List of episodes, where each episode is a dictionary of JAX arrays
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for this functionality")

        try:
            from .storage import ExperienceReader
        except ImportError:
            logger.error("Could not import ExperienceReader")
            raise

        logger.info(f"Loading episodes from {data_path}")

        # Group transitions by episode
        episodes_dict = {}

        # Don't use context manager since ExperienceReader doesn't support it
        reader = ExperienceReader(data_path)
        try:
            for batch in reader.iter_batches(100):
                for transition in batch:
                    episode_id = transition.get(episode_field, 0)

                    if episode_id not in episodes_dict:
                        episodes_dict[episode_id] = []

                    episodes_dict[episode_id].append(transition)

                    if max_episodes is not None and len(episodes_dict) >= max_episodes:
                        break

                if max_episodes is not None and len(episodes_dict) >= max_episodes:
                    break
        finally:
            if hasattr(reader, "close"):
                reader.close()

        # Convert episodes to list and optionally shuffle
        episode_ids = list(episodes_dict.keys())
        if shuffle:
            np.random.shuffle(episode_ids)

        episodes = []
        for episode_id in episode_ids:
            episode_transitions = episodes_dict[episode_id]

            # Convert episode to batch format
            episode_batch = {
                "observations": np.array(
                    [t.get("observation") for t in episode_transitions]
                ),
                "actions": np.array([t.get("action") for t in episode_transitions]),
                "rewards": np.array(
                    [t.get("reward", 0.0) for t in episode_transitions]
                ),
                "dones": np.array([t.get("done", False) for t in episode_transitions]),
            }

            # Convert to JAX arrays
            jax_episode = {}
            for key, value in episode_batch.items():
                jax_episode[key] = jnp.array(value)

            episodes.append(jax_episode)

        logger.info(f"Loaded {len(episodes)} episodes")

        return episodes

    @staticmethod
    def _convert_batch_to_jax(batch: list[dict[str, Any]]) -> dict[str, jnp.ndarray]:
        """
        Convert a batch of transitions to JAX arrays.

        Args:
            batch: List of transition dictionaries

        Returns:
            Dictionary with JAX arrays for each field
        """
        if not batch:
            return {}

        # Organize data by field
        fields = {}
        for transition in batch:
            for key, value in transition.items():
                if key not in fields:
                    fields[key] = []
                fields[key].append(value)

        # Convert each field to JAX array
        jax_batch = {}
        for key, values in fields.items():
            try:
                # Handle different types
                if isinstance(values[0], np.ndarray):
                    # Stack arrays
                    jax_batch[key] = jnp.array(np.stack(values))
                elif isinstance(values[0], (int, float, bool)):
                    # Create array from scalars
                    jax_batch[key] = jnp.array(values)
                else:
                    # Skip non-numeric fields
                    logger.debug(f"Skipping non-numeric field: {key}")
            except Exception as e:
                logger.warning(f"Failed to convert field {key} to JAX array: {e}")

        return jax_batch


# Common preprocessing utilities
def create_standardizer(
    data_path: str | Path,
    fields: list[str] = ["observation"],
    epsilon: float = 1e-8,
) -> tuple[Callable, dict[str, dict[str, np.ndarray]]]:
    """
    Create a standardization function for normalizing data.

    Args:
        data_path: Path to the experience data
        fields: Fields to compute statistics for
        epsilon: Small value to add to standard deviation for numerical stability

    Returns:
        Tuple of (standardize_fn, stats_dict)
    """
    try:
        from .storage import ExperienceReader
    except ImportError:
        logger.error("Could not import ExperienceReader")
        raise

    logger.info(f"Computing statistics for standardization from {data_path}")

    # Compute statistics
    field_data = {field: [] for field in fields}

    # Don't use context manager since ExperienceReader doesn't support it
    reader = ExperienceReader(data_path)
    try:
        for batch in reader.iter_batches(1000):
            for transition in batch:
                for field in fields:
                    if field in transition:
                        field_data[field].append(transition[field])
    finally:
        if hasattr(reader, "close"):
            reader.close()

    stats = {}
    for field, data in field_data.items():
        if data:
            try:
                # Convert to numpy arrays and stack
                if isinstance(data[0], np.ndarray):
                    stacked_data = np.stack(data)
                    mean = np.mean(stacked_data, axis=0)
                    std = np.std(stacked_data, axis=0)
                else:
                    mean = np.mean(data)
                    std = np.std(data)

                stats[field] = {"mean": mean, "std": std + epsilon}
            except Exception as e:
                logger.warning(f"Failed to compute statistics for field {field}: {e}")

    # Define standardization function
    if TENSORFLOW_AVAILABLE:

        def tf_standardize(batch):
            result = dict(batch)
            for field, field_stats in stats.items():
                if field in result:
                    result[field] = (result[field] - field_stats["mean"]) / field_stats[
                        "std"
                    ]
            return result

    if JAX_AVAILABLE:

        def jax_standardize(batch):
            result = dict(batch)
            for field, field_stats in stats.items():
                if field in result:
                    result[field] = (
                        result[field] - jnp.array(field_stats["mean"])
                    ) / jnp.array(field_stats["std"])
            return result

    # Return the appropriate function based on what's available
    if TENSORFLOW_AVAILABLE:
        logger.info(f"Created TensorFlow standardization function")
        return tf_standardize, stats
    elif JAX_AVAILABLE:
        logger.info(f"Created JAX standardization function")
        return jax_standardize, stats
    else:
        raise ImportError("Neither TensorFlow nor JAX is available")


def compute_n_step_returns(
    rewards: np.ndarray, done_flags: np.ndarray, gamma: float = 0.99, n_steps: int = 1
) -> np.ndarray:
    """
    Compute n-step returns from rewards and done flags.

    Args:
        rewards: Array of rewards [batch_size, time_steps]
        done_flags: Array of done flags [batch_size, time_steps]
        gamma: Discount factor
        n_steps: Number of steps to look ahead

    Returns:
        Array of n-step returns [batch_size, time_steps]
    """
    batch_size, time_steps = rewards.shape
    returns = np.zeros_like(rewards)

    for b in range(batch_size):
        for t in range(time_steps):
            # Calculate n-step return
            n_step_return = 0
            for i in range(n_steps):
                if t + i < time_steps:
                    # Apply discount
                    n_step_return += (gamma**i) * rewards[b, t + i]

                    # Stop if episode ends
                    if t + i < time_steps - 1 and done_flags[b, t + i]:
                        break

            returns[b, t] = n_step_return

    return returns
