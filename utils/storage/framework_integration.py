"""
Framework integration utilities for connecting Arrow/HDF5 storage backends with
machine learning frameworks. Provides dataset implementations for PyTorch and TensorFlow,
and memory-mapped file support for large datasets.
"""

import os
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator, Callable

# Import our type system
from ..type_system import TypeSpecification, ExperienceSchema, DType, TypeConverter

# Import storage backends
from .arrow_backend import StorageBackend, ArrowStorageBackend, ArrowDatasetIterator
from .hdf5_backend import HDF5StorageBackend

# Set up logging
logger = logging.getLogger(__name__)


# PyTorch Integration
try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not found. PyTorch integration will not be available.")
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints to work
    class Dataset:
        pass
    class IterableDataset:
        pass


# TensorFlow Integration
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not found. TensorFlow integration will not be available.")
    TENSORFLOW_AVAILABLE = False


class MemoryMappedStorageManager:
    """
    Manager for memory-mapped file access to experience data.
    Enables efficient random access to large datasets without loading everything into RAM.
    """
    
    def __init__(self, backend: StorageBackend, max_cache_size: int = 1000):
        """
        Initialize the memory mapping manager.
        
        Args:
            backend: The storage backend
            max_cache_size: Maximum number of experiences to cache in memory
        """
        self.backend = backend
        self.max_cache_size = max_cache_size
        self.cache = {}  # Map from index to experience
        self.access_history = []  # Track access for LRU cache
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an experience by index using memory-mapping.
        
        Args:
            idx: Experience index
            
        Returns:
            Experience dictionary
        """
        # Check if in cache
        if idx in self.cache:
            # Update access history
            self.access_history.remove(idx)
            self.access_history.append(idx)
            return self.cache[idx]
        
        # Load from backend
        experience = self.backend.read_batch(indices=[idx])[0]
        
        # Add to cache
        self.cache[idx] = experience
        self.access_history.append(idx)
        
        # Evict least recently used if cache is full
        if len(self.cache) > self.max_cache_size:
            evict_idx = self.access_history.pop(0)
            if evict_idx in self.cache:
                del self.cache[evict_idx]
        
        return experience
    
    def get_size(self) -> int:
        """Get the total number of experiences."""
        return self.backend.get_size()
    
    def refresh_cache(self):
        """Clear the cache to ensure fresh reads."""
        self.cache.clear()
        self.access_history.clear()


if TORCH_AVAILABLE:
    class ExperienceMapDataset(Dataset):
        """
        PyTorch dataset for random access to experience data.
        Uses memory-mapping for efficient access to large datasets.
        """
        
        def __init__(self, backend: StorageBackend, transform=None, 
                     max_cache_size: int = 1000):
            """
            Initialize the dataset.
            
            Args:
                backend: Storage backend
                transform: Optional transform function
                max_cache_size: Maximum number of experiences to cache
            """
            self.mmap = MemoryMappedStorageManager(backend, max_cache_size)
            self.transform = transform
        
        def __len__(self) -> int:
            """Get dataset size."""
            return self.mmap.get_size()
        
        def __getitem__(self, idx: int) -> Dict[str, Any]:
            """
            Get an experience by index.
            
            Args:
                idx: Experience index
                
            Returns:
                Experience dictionary (transformed if a transform is specified)
            """
            experience = self.mmap[idx]
            
            if self.transform:
                experience = self.transform(experience)
            
            return experience
    
    
    class ExperienceIterableDataset(IterableDataset):
        """
        PyTorch iterable dataset for streaming access to experience data.
        Useful for very large datasets that don't fit in memory.
        """
        
        def __init__(self, backend: StorageBackend, batch_size: int = 32,
                     shuffle: bool = True, transform=None):
            """
            Initialize the dataset.
            
            Args:
                backend: Storage backend
                batch_size: Batch size for iteration
                shuffle: Whether to shuffle the data
                transform: Optional transform function
            """
            self.backend = backend
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.transform = transform
        
        def __iter__(self):
            """Get iterator over experiences."""
            # If backend has built-in iterator, use it
            if hasattr(self.backend, 'get_iterator'):
                iterator = self.backend.get_iterator(
                    batch_size=self.batch_size, 
                    shuffle=self.shuffle
                )
                
                for batch in iterator:
                    if self.transform:
                        batch = [self.transform(exp) for exp in batch]
                    yield batch
            else:
                # Fall back to reading batches manually
                total_size = self.backend.get_size()
                indices = list(range(total_size))
                
                if self.shuffle:
                    import random
                    random.shuffle(indices)
                
                for i in range(0, total_size, self.batch_size):
                    batch_indices = indices[i:i+self.batch_size]
                    batch = self.backend.read_batch(indices=batch_indices)
                    
                    if self.transform:
                        batch = [self.transform(exp) for exp in batch]
                    
                    yield batch


# TensorFlow Integration
if TENSORFLOW_AVAILABLE:
    class ExperienceTensorflowDataset:
        """
        TensorFlow dataset adapter for experience data.
        Provides efficient conversion to tf.data.Dataset.
        """
        
        def __init__(self, backend: StorageBackend, transform=None):
            """
            Initialize the dataset adapter.
            
            Args:
                backend: Storage backend
                transform: Optional transform function
            """
            self.backend = backend
            self.transform = transform
        
        def create_tf_dataset(self, batch_size: int = 32, shuffle: bool = True,
                              shuffle_buffer: int = 10000, prefetch: int = 1) -> tf.data.Dataset:
            """
            Create a TensorFlow dataset.
            
            Args:
                batch_size: Batch size
                shuffle: Whether to shuffle the data
                shuffle_buffer: Size of shuffle buffer
                prefetch: Number of batches to prefetch
                
            Returns:
                tf.data.Dataset: TensorFlow dataset
            """
            # Get total size
            total_size = self.backend.get_size()
            
            # Create generator function
            def generator():
                indices = list(range(total_size))
                if shuffle:
                    import random
                    random.shuffle(indices)
                
                for idx in indices:
                    exp = self.backend.read_batch(indices=[idx])[0]
                    if self.transform:
                        exp = self.transform(exp)
                    yield exp
            
            # Determine output types and shapes
            output_types = {}
            output_shapes = {}
            
            # Get a sample experience to determine types/shapes
            sample = self.backend.read_batch(batch_size=1)[0]
            if self.transform:
                sample = self.transform(sample)
            
            for field_name, value in sample.items():
                output_types[field_name], output_shapes[field_name] = self._get_tf_type_shape(value)
            
            # Create dataset
            dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=output_types,
                output_shapes=output_shapes
            )
            
            # Apply operations
            if shuffle:
                dataset = dataset.shuffle(shuffle_buffer)
            
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(prefetch)
            
            return dataset
        
        @staticmethod
        def _get_tf_type_shape(value) -> Tuple[tf.DType, tf.TensorShape]:
            """
            Determine TensorFlow type and shape for a value.
            
            Args:
                value: Value to inspect
                
            Returns:
                Tuple of (tf.DType, tf.TensorShape)
            """
            if isinstance(value, dict):
                types = {}
                shapes = {}
                for k, v in value.items():
                    types[k], shapes[k] = ExperienceTensorflowDataset._get_tf_type_shape(v)
                return types, shapes
            
            elif isinstance(value, np.ndarray):
                dtype = tf.as_dtype(value.dtype)
                shape = tf.TensorShape([None] * len(value.shape))
                return dtype, shape
            
            elif isinstance(value, (int, np.integer)):
                return tf.int64, tf.TensorShape([])
            
            elif isinstance(value, (float, np.floating)):
                return tf.float32, tf.TensorShape([])
            
            elif isinstance(value, (bool, np.bool_)):
                return tf.bool, tf.TensorShape([])
            
            elif isinstance(value, str):
                return tf.string, tf.TensorShape([])
            
            else:
                logger.warning(f"Unknown type: {type(value)}, defaulting to float32")
                return tf.float32, tf.TensorShape([])
        
        def to_tf_example(self, experience: Dict[str, Any]) -> tf.train.Example:
            """
            Convert an experience to TensorFlow Example format.
            Useful for TFRecord serialization.
            
            Args:
                experience: Experience dictionary
                
            Returns:
                tf.train.Example: TensorFlow Example
            """
            feature = {}
            
            for field_name, value in experience.items():
                field_spec = self.backend.schema.field_specs.get(field_name)
                feature.update(self._value_to_tf_feature(field_name, value, field_spec))
            
            return tf.train.Example(features=tf.train.Features(feature=feature))
        
        def _value_to_tf_feature(self, name: str, value: Any, 
                                spec: Optional[TypeSpecification] = None) -> Dict[str, tf.train.Feature]:
            """
            Convert a value to TensorFlow Feature.
            
            Args:
                name: Field name
                value: Field value
                spec: Type specification
                
            Returns:
                Dict mapping field name to Feature
            """
            features = {}
            
            if isinstance(value, dict):
                # Handle nested dictionaries
                for k, v in value.items():
                    nested_spec = None
                    if spec and spec.nested_spec:
                        nested_spec = spec.nested_spec.get(k)
                    
                    sub_features = self._value_to_tf_feature(f"{name}/{k}", v, nested_spec)
                    features.update(sub_features)
            
            elif isinstance(value, np.ndarray):
                # Convert to appropriate feature type
                if np.issubdtype(value.dtype, np.floating):
                    features[name] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=value.flatten())
                    )
                elif np.issubdtype(value.dtype, np.integer):
                    features[name] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=value.flatten().astype(np.int64))
                    )
                elif np.issubdtype(value.dtype, np.bool_):
                    features[name] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=value.flatten().astype(np.int64))
                    )
                else:
                    features[name] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value.tobytes()])
                    )
                
                # Store shape as a separate feature
                features[f"{name}_shape"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(value.shape))
                )
            
            elif isinstance(value, (int, np.integer)):
                features[name] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[value])
                )
            
            elif isinstance(value, (float, np.floating)):
                features[name] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[value])
                )
            
            elif isinstance(value, (bool, np.bool_)):
                features[name] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(value)])
                )
            
            elif isinstance(value, str):
                features[name] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[value.encode()])
                )
            
            elif isinstance(value, bytes):
                features[name] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[value])
                )
            
            else:
                logger.warning(f"Unsupported type for TFExample: {type(value)}")
            
            return features


# Utility functions for framework conversion

def batch_to_torch(batch: List[Dict[str, Any]], device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Convert a batch of experiences to PyTorch tensors.
    
    Args:
        batch: List of experience dictionaries
        device: PyTorch device
        
    Returns:
        Dict mapping field names to tensors
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")
    
    result = {}
    
    # Group by field
    for field_name in batch[0].keys():
        values = [exp[field_name] for exp in batch]
        
        # Convert to tensor based on type
        if isinstance(values[0], np.ndarray):
            # Stack numpy arrays
            stacked = np.stack(values)
            result[field_name] = torch.from_numpy(stacked).to(device)
        
        elif isinstance(values[0], (int, float, bool)):
            # Convert primitive types
            result[field_name] = torch.tensor(values).to(device)
        
        elif isinstance(values[0], dict):
            # Convert nested dictionaries
            nested_result = {}
            for k in values[0].keys():
                nested_values = [exp[k] for exp in values]
                
                if isinstance(nested_values[0], np.ndarray):
                    stacked = np.stack(nested_values)
                    nested_result[k] = torch.from_numpy(stacked).to(device)
                else:
                    nested_result[k] = torch.tensor(nested_values).to(device)
            
            result[field_name] = nested_result
    
    return result


def batch_to_tensorflow(batch: List[Dict[str, Any]]) -> Dict[str, tf.Tensor]:
    """
    Convert a batch of experiences to TensorFlow tensors.
    
    Args:
        batch: List of experience dictionaries
        
    Returns:
        Dict mapping field names to tensors
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available")
    
    result = {}
    
    # Group by field
    for field_name in batch[0].keys():
        values = [exp[field_name] for exp in batch]
        
        # Convert to tensor based on type
        if isinstance(values[0], np.ndarray):
            # Stack numpy arrays
            stacked = np.stack(values)
            result[field_name] = tf.convert_to_tensor(stacked)
        
        elif isinstance(values[0], (int, float, bool)):
            # Convert primitive types
            result[field_name] = tf.convert_to_tensor(values)
        
        elif isinstance(values[0], dict):
            # Convert nested dictionaries
            nested_result = {}
            for k in values[0].keys():
                nested_values = [exp[k] for exp in values]
                
                if isinstance(nested_values[0], np.ndarray):
                    stacked = np.stack(nested_values)
                    nested_result[k] = tf.convert_to_tensor(stacked)
                else:
                    nested_result[k] = tf.convert_to_tensor(nested_values)
            
            result[field_name] = nested_result
    
    return result


# Common processing functions

def normalize_observations(batch: Dict[str, Any], stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Normalize observations using running statistics.
    
    Args:
        batch: Batch of experiences
        stats: Dictionary with 'mean' and 'std' for each field
        
    Returns:
        Batch with normalized observations
    """
    if 'observation' in batch:
        obs = batch['observation']
        
        # Handle dict observations
        if isinstance(obs, dict):
            for key, value in obs.items():
                if key in stats and isinstance(value, (np.ndarray, torch.Tensor, tf.Tensor)):
                    mean = stats[key]['mean']
                    std = stats[key]['std'] + 1e-6  # Avoid division by zero
                    
                    if isinstance(value, np.ndarray):
                        batch['observation'][key] = (value - mean) / std
                    elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                        batch['observation'][key] = (value - torch.tensor(mean, device=value.device)) / \
                                                   torch.tensor(std, device=value.device)
                    elif TENSORFLOW_AVAILABLE and isinstance(value, tf.Tensor):
                        batch['observation'][key] = (value - mean) / std
        
        # Handle array observations
        elif isinstance(obs, (np.ndarray, torch.Tensor, tf.Tensor)):
            if 'observation' in stats:
                mean = stats['observation']['mean']
                std = stats['observation']['std'] + 1e-6
                
                if isinstance(obs, np.ndarray):
                    batch['observation'] = (obs - mean) / std
                elif TORCH_AVAILABLE and isinstance(obs, torch.Tensor):
                    batch['observation'] = (obs - torch.tensor(mean, device=obs.device)) / \
                                          torch.tensor(std, device=obs.device)
                elif TENSORFLOW_AVAILABLE and isinstance(obs, tf.Tensor):
                    batch['observation'] = (obs - mean) / std
    
    return batch


def calculate_n_step_returns(batch: List[Dict[str, Any]], gamma: float = 0.99, n_steps: int = 1) -> List[Dict[str, Any]]:
    """
    Calculate n-step returns for a batch of experiences.
    
    Args:
        batch: List of experience dictionaries
        gamma: Discount factor
        n_steps: Number of steps for return calculation
        
    Returns:
        Batch with n-step returns added
    """
    # Must have consecutive experiences from the same episode
    for i in range(len(batch) - n_steps):
        returns = 0
        for j in range(n_steps):
            if i + j < len(batch):
                returns += (gamma ** j) * batch[i + j]['reward']
        
        batch[i]['n_step_return'] = returns
    
    # For the last n experiences, use as many steps as available
    for i in range(len(batch) - n_steps, len(batch)):
        returns = 0
        steps = len(batch) - i
        for j in range(steps):
            returns += (gamma ** j) * batch[i + j]['reward']
        
        batch[i]['n_step_return'] = returns
    
    return batch


def compute_advantages(batch: List[Dict[str, Any]], value_func: Callable, 
                       gamma: float = 0.99, lambda_: float = 0.95) -> List[Dict[str, Any]]:
    """
    Compute GAE (Generalized Advantage Estimation) for a batch of experiences.
    
    Args:
        batch: List of experience dictionaries
        value_func: Function that computes value estimates for states
        gamma: Discount factor
        lambda_: GAE parameter
        
    Returns:
        Batch with advantages added
    """
    # Calculate values for all states
    values = []
    next_values = []
    
    for exp in batch:
        values.append(value_func(exp['observation']))
        next_values.append(value_func(exp['next_observation']))
    
    # Convert to numpy arrays for easier calculation
    values = np.array(values)
    next_values = np.array(next_values)
    rewards = np.array([exp['reward'] for exp in batch])
    dones = np.array([exp['done'] for exp in batch])
    
    # Calculate deltas: r + gamma * V(s') - V(s)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    
    # Calculate advantages using GAE
    advantages = np.zeros_like(deltas)
    gae = 0
    for t in reversed(range(len(batch))):
        gae = deltas[t] + gamma * lambda_ * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # Add to batch
    for i, adv in enumerate(advantages):
        batch[i]['advantage'] = adv
        batch[i]['value'] = values[i]
    
    return batch


# Memory-mapped file utilities

def create_mmap_array(filename: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    Create a memory-mapped numpy array file.
    
    Args:
        filename: File path
        shape: Array shape
        dtype: Data type
        
    Returns:
        Memory-mapped numpy array
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create memory-mapped file
    return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)


def load_mmap_array(filename: str, shape: tuple, dtype: np.dtype, mode: str = 'r+') -> np.ndarray:
    """
    Load a memory-mapped numpy array file.
    
    Args:
        filename: File path
        shape: Array shape
        dtype: Data type
        mode: File mode ('r' for read-only, 'r+' for read-write)
        
    Returns:
        Memory-mapped numpy array
    """
    return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)


# Utility functions for handling large datasets

def is_large_dataset(backend: StorageBackend, threshold_gb: float = 1.0) -> bool:
    """
    Check if a dataset is large based on estimated size.
    
    Args:
        backend: Storage backend
        threshold_gb: Size threshold in gigabytes
        
    Returns:
        bool: Whether the dataset is considered large
    """
    # Get dataset size
    size = backend.get_size()
    
    # Estimate memory usage based on a sample
    if size == 0:
        return False
    
    sample = backend.read_batch(batch_size=1)[0]
    
    # Estimate bytes per sample
    bytes_per_sample = _estimate_experience_size(sample)
    
    # Calculate total size in GB
    total_size_gb = (bytes_per_sample * size) / (1024**3)
    
    return total_size_gb > threshold_gb


def _estimate_experience_size(experience: Dict[str, Any]) -> int:
    """
    Estimate the memory size of an experience in bytes.
    
    Args:
        experience: Experience dictionary
        
    Returns:
        int: Estimated size in bytes
    """
    size = 0
    
    for key, value in experience.items():
        if isinstance(value, np.ndarray):
            size += value.nbytes
        elif isinstance(value, dict):
            size += _estimate_experience_size(value)
        else:
            # Rough estimate for scalars
            size += 8
    
    # Add overhead for dictionary structure
    size += len(experience) * 32
    
    return size