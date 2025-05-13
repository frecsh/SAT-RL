"""
Integration with TensorFlow and JAX for RL experience data.
Provides utilities for loading experience data into TensorFlow and JAX.
"""

import logging
import os
from collections.abc import Iterator
from typing import Any

import numpy as np

from satrlgym.storage import create_storage

logger = logging.getLogger(__name__)

# Try to import TensorFlow
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

# Try to import JAX
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None

# Use the updated storage module path


def create_tf_dataset(
    storage_path: str,
    storage_type: str = "parquet",
    fields: list[str] | None = None,
    compression_config: dict[str, Any] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Create a TensorFlow Dataset from experience data.

    Args:
        storage_path: Path to the storage file
        storage_type: Type of storage ('parquet', 'hdf5', or 'mmap')
        fields: Optional list of field names to load (None for all)
        compression_config: Optional compression configuration
        batch_size: Size of batches to yield
        shuffle: Whether to shuffle the data
        shuffle_buffer_size: Size of the shuffle buffer
        prefetch: Number of batches to prefetch

    Returns:
        TensorFlow Dataset
    """
    # Create storage and load data
    storage = create_storage(storage_type, storage_path, compression_config)
    data = storage.read_batch(fields)
    if not data:
        raise ValueError(f"No data found in storage at {storage_path}")

    # Convert numpy arrays to tensors
    tensors = {k: tf.convert_to_tensor(v) for k, v in data.items()}

    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices(tensors)

    # Apply shuffling if needed
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(shuffle_buffer_size, len(next(iter(data.values()))))
        )

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Apply prefetching for performance
    dataset = dataset.prefetch(prefetch)

    return dataset


class ExperienceGenerator:
    """Data generator for experience data."""

    def __init__(
        self,
        storage_path: str,
        storage_type: str = "parquet",
        fields: list[str] | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        parse_fn: callable | None = None,
        buffer_size: int = 1000,
    ):
        """Initialize the generator.

        Args:
            storage_path: Path to the storage file or directory
            storage_type: Storage type ('parquet', 'zarr', etc.)
            fields: Optional list of field names to load
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            parse_fn: Optional function to parse each data item
            buffer_size: Buffer size for shuffling
        """
        self.storage_path = storage_path
        self.storage_type = storage_type
        self.fields = fields
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.parse_fn = parse_fn
        self.buffer_size = buffer_size

    def get_output_signature(self) -> dict[str, tf.TensorSpec]:
        """Get the output signature for the generator.

        Returns:
            Dictionary of field name to tensor spec
        """
        # Create storage
        storage = create_storage(self.storage_type, self.storage_path)

        # Load a sample to determine data types and shapes
        sample = storage.read_batch(fields=self.fields)
        if not sample:
            raise ValueError(f"No data found in storage at {self.storage_path}")

        # Create output signature
        output_signature = {}
        for key, value in sample.items():
            # First dimension is the batch dimension
            shape = (None,) + value.shape[1:]
            dtype = tf.as_dtype(value.dtype)
            output_signature[key] = tf.TensorSpec(shape=shape, dtype=dtype)

        return output_signature

    def __call__(self) -> Iterator[dict[str, np.ndarray]]:
        """Generate batches of data.

        Yields:
            Batched experience data
        """
        # Create storage
        storage = create_storage(self.storage_type, self.storage_path)

        # Get total size
        total_items = storage.get_size()

        # Read all data - debug the size
        all_data = storage.read_batch(fields=self.fields)

        # Verify data size matches expected size
        actual_size = len(next(iter(all_data.values()))) if all_data else 0
        if actual_size != total_items:
            # Log warning and adjust
            print(
                f"Warning: ExperienceGenerator expected {total_items} items but got {actual_size}"
            )
            total_items = actual_size

        # Create indices array with actual size
        indices = np.arange(total_items)

        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(indices)

        # Debug output
        num_batches = (total_items + self.batch_size - 1) // self.batch_size
        print(
            f"ExperienceGenerator creating {num_batches} batches from {total_items} items with batch size {self.batch_size}"
        )

        # Generate batches by slicing the data
        for i in range(0, total_items, self.batch_size):
            batch_indices = indices[i : min(i + self.batch_size, total_items)]

            # Create a batch with just the indices we want
            batch = {}
            for key, values in all_data.items():
                if len(values) > 0:
                    # Make sure batch_indices doesn't exceed the array bounds
                    valid_indices = batch_indices[batch_indices < len(values)]
                    if len(valid_indices) > 0:
                        batch[key] = values[valid_indices]
                    else:
                        batch[key] = np.array([])
                else:
                    batch[key] = np.array([])

            if self.parse_fn:
                batch = self.parse_fn(batch)

            # Skip empty batches
            if all(len(arr) > 0 for arr in batch.values()):
                yield batch


def create_tf_generator_dataset(
    storage_path: str,
    storage_type: str = "parquet",
    fields: list[str] | None = None,
    compression_config: dict[str, Any] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Create a TensorFlow Dataset using a generator.

    Args:
        storage_path: Path to the storage file
        storage_type: Type of storage ('parquet', 'hdf5', or 'mmap')
        fields: Optional list of field names to load (None for all)
        compression_config: Optional compression configuration
        batch_size: Size of batches to yield
        shuffle: Whether to shuffle the data
        prefetch: Number of batches to prefetch

    Returns:
        TensorFlow Dataset
    """
    generator = ExperienceGenerator(
        storage_path=storage_path,
        storage_type=storage_type,
        fields=fields,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    output_signature = generator.get_output_signature()

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    )

    # Apply prefetching for performance
    dataset = dataset.prefetch(prefetch)

    return dataset


def load_experience_for_jax(
    storage_path: str,
    storage_type: str = "parquet",
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Load experience data as JAX arrays.

    Args:
        storage_path: Path to the storage file or directory
        storage_type: Storage type ('parquet', 'zarr', etc.)
        fields: Optional list of field names to load

    Returns:
        Dictionary of field name to JAX array

    Raises:
        ImportError: If JAX is not available
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Please install jax package.")

    storage = create_storage(storage_type, storage_path)
    data = storage.read_all(fields=fields)

    # Convert numpy arrays to JAX arrays
    jax_data = {}
    for k, v in data.items():
        jax_data[k] = jnp.array(v)

    return jax_data


class ExperienceTFDataset:
    """TensorFlow dataset wrapper for experience data."""

    def __init__(
        self,
        data_path: str | os.PathLike,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
        prefetch_buffer: int = tf.data.AUTOTUNE,
        filter_fn: callable | None = None,
        transform_fn: callable | None = None,
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to the experience data
            batch_size: Batch size
            shuffle_buffer: Size of shuffle buffer
            prefetch_buffer: Size of prefetch buffer
            filter_fn: Optional function to filter items
            transform_fn: Optional function to transform items
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.filter_fn = filter_fn
        self.transform_fn = transform_fn

    def _load_data(self):
        """Load data from the specified path."""
        storage = create_storage("parquet", self.data_path)
        return storage.read_all()

    def _prepare_dataset(self):
        """Prepare the TensorFlow dataset."""
        data = self._load_data()
        tensors = {k: tf.convert_to_tensor(v) for k, v in data.items()}
        dataset = tf.data.Dataset.from_tensor_slices(tensors)

        if self.filter_fn:
            dataset = dataset.filter(self.filter_fn)

        if self.transform_fn:
            dataset = dataset.map(self.transform_fn)

        dataset = (
            dataset.shuffle(self.shuffle_buffer)
            .batch(self.batch_size)
            .prefetch(self.prefetch_buffer)
        )
        return dataset

    def __iter__(self) -> Iterator:
        """Iterate through the dataset."""
        dataset = self._prepare_dataset()
        yield from dataset


def convert_to_tf_dataset(
    experience_data: dict[str, np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer: int = 1000,
    prefetch_buffer: int = tf.data.AUTOTUNE,
    filter_fn: callable | None = None,
    transform_fn: callable | None = None,
) -> tf.data.Dataset:
    """Convert experience data to a TensorFlow dataset.

    Args:
        experience_data: Dictionary of experience data arrays
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        shuffle_buffer: Size of shuffle buffer
        prefetch_buffer: Size of prefetch buffer
        filter_fn: Optional function to filter items
        transform_fn: Optional function to transform items

    Returns:
        TensorFlow dataset
    """
    tensors = {k: tf.convert_to_tensor(v) for k, v in experience_data.items()}
    dataset = tf.data.Dataset.from_tensor_slices(tensors)

    if filter_fn:
        dataset = dataset.filter(filter_fn)

    if transform_fn:
        dataset = dataset.map(transform_fn)

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(batch_size).prefetch(prefetch_buffer)
    return dataset


def convert_to_jax(batch: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert a batch of numpy arrays to JAX arrays.

    Args:
        batch: Dictionary of numpy arrays

    Returns:
        Dictionary of JAX arrays
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not available. Please install JAX to use this function."
        )

    return {k: jnp.array(v) for k, v in batch.items()}
