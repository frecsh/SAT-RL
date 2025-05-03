"""
Experience indexing utilities for fast access and prioritized experience replay.

This module provides functionality for:
1. Generating index files with transition offsets
2. O(1) random sampling using indices
3. Support for prioritized experience replay weights
"""

import os
import json
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO

class ExperienceIndex:
    """
    Index for fast access to experience data in large files.
    
    This class creates and manages index files that allow O(1) random access
    to transitions stored in larger experience files, without needing to load
    the entire file into memory.
    """
    
    def __init__(self, 
                 source_file: Union[str, Path],
                 index_file: Optional[Union[str, Path]] = None,
                 create_if_missing: bool = True,
                 cache_size: int = 1000):
        """
        Initialize the experience index.
        
        Args:
            source_file: Path to the source experience file
            index_file: Path to the index file (will be auto-generated if None)
            create_if_missing: Whether to create the index if it doesn't exist
            cache_size: Number of transitions to cache in memory
        """
        self.source_file = Path(source_file)
        
        # Auto-generate index file path if not provided
        if index_file is None:
            self.index_file = self.source_file.with_suffix('.idx')
        else:
            self.index_file = Path(index_file)
        
        self.transitions_count = 0
        self.offsets = []  # List of file offsets for each transition
        self.weights = np.array([], dtype=np.float32)  # Priority weights if used
        self._weight_sum = 0.0  # Sum of all weights for weighted sampling
        
        # Cache for recently accessed transitions
        self._cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Stats tracking
        self.stats = {
            "created_at": time.time(),
            "last_updated": time.time(),
            "access_count": 0,
            "sampling_count": 0
        }
        
        # Load or create the index
        if os.path.exists(self.index_file):
            self._load_index()
        elif create_if_missing:
            self._create_index()
        else:
            raise FileNotFoundError(f"Index file {self.index_file} does not exist")
    
    def _create_index(self) -> None:
        """Create an index for the source file."""
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"Source file {self.source_file} does not exist")
        
        # Different handling based on file format
        if self.source_file.suffix in ['.jsonl', '.json']:
            self._create_index_json()
        elif self.source_file.suffix == '.npz':
            self._create_index_npz()
        elif self.source_file.suffix in ['.h5', '.hdf5']:
            self._create_index_hdf5()
        elif self.source_file.suffix == '.parquet':
            self._create_index_parquet()
        else:
            raise ValueError(f"Unsupported file format: {self.source_file.suffix}")
        
        # Save the created index
        self._save_index()
    
    def _create_index_json(self) -> None:
        """Create index for JSON Lines file."""
        offsets = []
        
        with open(self.source_file, 'rb') as f:
            offset = 0
            # Read the file line by line and record the offset of each line
            while line := f.readline():
                offsets.append(offset)
                offset += len(line)
        
        self.offsets = offsets
        self.transitions_count = len(offsets)
        
        # Initialize uniform weights
        self.weights = np.ones(self.transitions_count, dtype=np.float32)
        self._weight_sum = float(self.transitions_count)
    
    def _create_index_npz(self) -> None:
        """Create index for NPZ file."""
        try:
            import numpy as np
            data = np.load(self.source_file, allow_pickle=True)
            
            # For NPZ files, we don't need byte offsets since we load arrays directly
            # Instead, we'll just use indices
            if 'transitions' in data:
                # If transitions are stored as a single array
                self.transitions_count = len(data['transitions'])
            else:
                # If transitions are stored as separate arrays
                # Try to find a field that would indicate the count
                for key in data.keys():
                    if isinstance(data[key], np.ndarray):
                        self.transitions_count = len(data[key])
                        break
            
            # Generate sequential indices
            self.offsets = list(range(self.transitions_count))
            
            # Initialize uniform weights
            self.weights = np.ones(self.transitions_count, dtype=np.float32)
            self._weight_sum = float(self.transitions_count)
            
        except Exception as e:
            raise ValueError(f"Error creating index for NPZ file: {e}")
    
    def _create_index_hdf5(self) -> None:
        """Create index for HDF5 file."""
        try:
            import h5py
            with h5py.File(self.source_file, 'r') as f:
                # Identify the main dataset
                main_dataset = None
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        if len(f[key].shape) > 0:  # Skip scalar datasets
                            main_dataset = key
                            break
                
                if main_dataset is None:
                    raise ValueError("Could not find a valid dataset in the HDF5 file")
                
                # For HDF5, we can use direct indices
                self.transitions_count = f[main_dataset].shape[0]
                self.offsets = list(range(self.transitions_count))
                
                # Initialize uniform weights
                self.weights = np.ones(self.transitions_count, dtype=np.float32)
                self._weight_sum = float(self.transitions_count)
                
        except ImportError:
            raise ImportError("h5py is required for HDF5 file indexing")
        except Exception as e:
            raise ValueError(f"Error creating index for HDF5 file: {e}")
    
    def _create_index_parquet(self) -> None:
        """Create index for Parquet file."""
        try:
            import pyarrow.parquet as pq
            # Open the parquet file and get the number of rows
            parquet_file = pq.ParquetFile(self.source_file)
            self.transitions_count = parquet_file.metadata.num_rows
            
            # For parquet, we can use row indices directly
            self.offsets = list(range(self.transitions_count))
            
            # Initialize uniform weights
            self.weights = np.ones(self.transitions_count, dtype=np.float32)
            self._weight_sum = float(self.transitions_count)
            
        except ImportError:
            raise ImportError("pyarrow is required for Parquet file indexing")
        except Exception as e:
            raise ValueError(f"Error creating index for Parquet file: {e}")
    
    def _load_index(self) -> None:
        """Load an existing index file."""
        try:
            with open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.transitions_count = index_data['transitions_count']
            self.offsets = index_data['offsets']
            self.weights = index_data['weights']
            self._weight_sum = float(np.sum(self.weights))
            self.stats = index_data.get('stats', {
                "created_at": time.time(),
                "last_updated": time.time(),
                "access_count": 0,
                "sampling_count": 0
            })
        except Exception as e:
            raise ValueError(f"Error loading index file: {e}")
    
    def _save_index(self) -> None:
        """Save the index to disk."""
        # Update stats
        self.stats["last_updated"] = time.time()
        
        # Create the index data structure
        index_data = {
            'source_file': str(self.source_file),
            'transitions_count': self.transitions_count,
            'offsets': self.offsets,
            'weights': self.weights,
            'stats': self.stats
        }
        
        # Save to disk
        with open(self.index_file, 'wb') as f:
            pickle.dump(index_data, f)
    
    def get_transition(self, index: int) -> Dict[str, Any]:
        """
        Get a specific transition by index.
        
        Args:
            index: The index of the transition to retrieve
            
        Returns:
            The transition data as a dictionary
        """
        if index < 0 or index >= self.transitions_count:
            raise IndexError(f"Index {index} out of bounds (0-{self.transitions_count-1})")
        
        # Check cache first
        if index in self._cache:
            self._cache_hits += 1
            return self._cache[index]
        
        self._cache_misses += 1
        self.stats["access_count"] += 1
        
        # Access the transition based on file format
        if self.source_file.suffix in ['.jsonl', '.json']:
            transition = self._get_transition_json(index)
        elif self.source_file.suffix == '.npz':
            transition = self._get_transition_npz(index)
        elif self.source_file.suffix in ['.h5', '.hdf5']:
            transition = self._get_transition_hdf5(index)
        elif self.source_file.suffix == '.parquet':
            transition = self._get_transition_parquet(index)
        else:
            raise ValueError(f"Unsupported file format: {self.source_file.suffix}")
        
        # Add to cache
        self._update_cache(index, transition)
        
        return transition
    
    def _get_transition_json(self, index: int) -> Dict[str, Any]:
        """Get a transition from a JSON Lines file."""
        offset = self.offsets[index]
        
        with open(self.source_file, 'r') as f:
            f.seek(offset)
            line = f.readline()
            return json.loads(line)
    
    def _get_transition_npz(self, index: int) -> Dict[str, Any]:
        """Get a transition from an NPZ file."""
        try:
            import numpy as np
            data = np.load(self.source_file, allow_pickle=True)
            
            # Different handling based on how data is stored
            if 'transitions' in data:
                # Direct transitions array
                transition_data = data['transitions'][index]
                # Convert numpy types to native Python types
                return self._convert_numpy_types(transition_data)
            else:
                # Look for standard RL transition components
                transition = {}
                
                # Expected keys in RL datasets
                expected_keys = ["states", "actions", "rewards", "next_states", "dones"]
                
                # Check for expected keys first
                for key in expected_keys:
                    if key in data and isinstance(data[key], np.ndarray) and len(data[key]) > index:
                        transition[key] = self._convert_numpy_item(data[key][index])
                
                # If we didn't find the expected keys, try all available keys
                if not transition:
                    for key in data.keys():
                        if isinstance(data[key], np.ndarray) and len(data[key]) > index:
                            transition[key] = self._convert_numpy_item(data[key][index])
                
                return transition
                
        except Exception as e:
            raise ValueError(f"Error accessing NPZ transition: {e}")
    
    def _convert_numpy_item(self, item) -> Any:
        """Convert a NumPy item to a native Python type."""
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (np.integer, np.int32, np.int64)):
            return int(item)
        elif isinstance(item, (np.floating, np.float32, np.float64)):
            return float(item)
        elif isinstance(item, np.bool_):
            return bool(item)
        else:
            return item
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Recursively convert NumPy types in an object to native Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _get_transition_hdf5(self, index: int) -> Dict[str, Any]:
        """Get a transition from an HDF5 file."""
        try:
            import h5py
            with h5py.File(self.source_file, 'r') as f:
                transition = {}
                
                # Retrieve all datasets for this index
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and len(f[key].shape) > 0:
                        if index < f[key].shape[0]:
                            transition[key] = f[key][index]
                
                return transition
                
        except ImportError:
            raise ImportError("h5py is required for HDF5 file access")
        except Exception as e:
            raise ValueError(f"Error accessing HDF5 transition: {e}")
    
    def _get_transition_parquet(self, index: int) -> Dict[str, Any]:
        """Get a transition from a Parquet file."""
        try:
            import pyarrow.parquet as pq
            # Read just the specific row from the parquet file
            table = pq.read_table(self.source_file, row_groups=[index // 1000])  # Assuming default row group size
            row_in_group = index % 1000
            
            # Convert to dictionary
            df_row = table.to_pandas().iloc[row_in_group:row_in_group+1]
            if df_row.empty:
                raise IndexError(f"Row {index} not found in parquet file")
                
            return df_row.iloc[0].to_dict()
            
        except ImportError:
            raise ImportError("pyarrow is required for Parquet file access")
        except Exception as e:
            raise ValueError(f"Error accessing Parquet transition: {e}")
    
    def _update_cache(self, index: int, transition: Dict[str, Any]) -> None:
        """Update the cache with a new transition using LRU strategy."""
        # If cache is full, implement proper LRU eviction
        if len(self._cache) >= self._cache_size:
            # Find least recently used items based on access pattern
            # We'll remove the 10% oldest items when cache is full
            num_to_remove = max(1, self._cache_size // 10)
            
            # Get the items in insertion order (approximation of LRU)
            keys_to_remove = list(self._cache.keys())[:num_to_remove]
            
            # Remove them from cache
            for key in keys_to_remove:
                del self._cache[key]
                
        # Add new item to cache - we'll use a simple ordered dict for LRU
        # Remove the item if it's already in cache to update its position
        if index in self._cache:
            del self._cache[index]
            
        # Add the item to the end (most recently used position)
        self._cache[index] = transition
    
    def sample(self, count: int = 1, prioritized: bool = False) -> List[Dict[str, Any]]:
        """
        Sample transitions randomly, with optional prioritization based on weights.
        
        Args:
            count: Number of transitions to sample
            prioritized: Whether to use prioritized sampling based on weights
            
        Returns:
            List of sampled transitions
        """
        if count > self.transitions_count:
            raise ValueError(f"Cannot sample {count} from {self.transitions_count} transitions")
        
        self.stats["sampling_count"] += count
        
        if prioritized:
            # Weighted sampling using the priority weights
            p = self.weights / self._weight_sum
            indices = np.random.choice(
                self.transitions_count, size=count, replace=True, p=p
            )
        else:
            # Uniform random sampling
            indices = np.random.randint(0, self.transitions_count, size=count)
        
        # Retrieve the sampled transitions
        return [self.get_transition(int(idx)) for idx in indices]
    
    def update_weights(self, indices: List[int], weights: List[float]) -> None:
        """
        Update the priority weights for specific transitions.
        
        Args:
            indices: Indices of the transitions to update
            weights: New weights for the transitions
        """
        if len(indices) != len(weights):
            raise ValueError("indices and weights must have the same length")
    
        # Update the weights
        for idx, weight in zip(indices, weights):
            if idx < 0 or idx >= self.transitions_count:
                raise IndexError(f"Index {idx} out of bounds")
                
            self.weights[idx] = float(weight)
        
        # Recalculate total sum to avoid accumulated floating point errors
        self._weight_sum = float(np.sum(self.weights))
        
        # Save the updated index
        self._save_index()
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache performance."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_cache_size": self._cache_size
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index usage."""
        stats = self.stats.copy()
        stats["cache"] = self.get_cache_stats()
        return stats


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer with efficient sampling capabilities.
    
    This class implements a replay buffer with prioritized sampling based on
    TD errors or other importance metrics.
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 alpha: float = 0.6, 
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 0.01):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions in the buffer
            alpha: Prioritization exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling correction exponent
            beta_increment: Increment to beta after each sampling
            epsilon: Small constant added to priorities to ensure non-zero probability
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_beta = 1.0
        
        # Experience storage
        self.transitions = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # Position to add the next experience
        self._size = 0  # Current size of the buffer
        # Add default priority for new experiences
        self.default_priority = epsilon
    
    def add(self, experience: Dict[str, Any], priority: Optional[float] = None):
        """
        Add an experience to the buffer with an optional priority.
        
        Args:
            experience: The experience data to store
            priority: Priority value (higher = more important)
        """
        # If priority is not provided, use default
        if priority is None:
            priority = self.default_priority
        
        # If buffer is full, overwrite oldest entries
        if self._size >= self.capacity:
            # Get the index to overwrite
            idx = self.position % self.capacity
            
            # Update the priority
            self.priorities[idx] = priority
            
            # Update the experience
            self.transitions[idx] = experience
        else:
            # Add the new experience to the buffer
            idx = self.position % self.capacity  # Use modulo even when adding
            if len(self.transitions) <= idx:
                self.transitions.append(experience)
            else:
                self.transitions[idx] = experience
            self.priorities[idx] = priority
            self._size += 1
        
        # Move position and wrap around when we hit capacity
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], List[int], np.ndarray]:
        """
        Sample a batch of transitions based on their priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (transitions, indices, importance_sampling_weights)
        """
        if self._size < batch_size:
            raise ValueError(f"Not enough transitions in buffer ({self._size} < {batch_size})")
        
        # Get active priorities
        active_priorities = self.priorities[:self._size]
        
        # Check if we have any valid priorities
        if np.sum(active_priorities) == 0 or not np.any(active_priorities > 0):
            # Fall back to uniform sampling if all priorities are zero
            indices = np.random.choice(self._size, batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            # Calculate sampling probabilities with numerical stability
            priorities = np.power(active_priorities, self.alpha)
            sum_priorities = np.sum(priorities)
            
            # Ensure we don't have NaN values due to division by zero
            if sum_priorities <= 0:
                probabilities = np.ones_like(priorities) / len(priorities)
            else:
                probabilities = priorities / sum_priorities
            
            # Sample transitions based on probabilities
            indices = np.random.choice(self._size, batch_size, replace=False, p=probabilities)
            
            # Calculate importance-sampling (IS) weights
            # IS weights correct for the bias introduced by prioritized sampling
            max_weight = (self._size * np.min(probabilities)) ** (-self.beta)
            weights = (self._size * probabilities[indices]) ** (-self.beta)
            
            # Avoid division by zero and scale weights
            if max_weight > 0:
                weights = weights / max_weight  # Normalize by max weight
            else:
                weights = np.ones_like(weights)
                
            # Clip weights for numerical stability
            weights = np.clip(weights, 0.01, 100.0)
        
        # Get the transitions
        sampled_transitions = [self.transitions[idx] for idx in indices]
        
        # Increment beta for next sampling (annealing)
        self.beta = min(self.beta + self.beta_increment, self.max_beta)
        
        return sampled_transitions, indices.tolist(), weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update the priorities of specific transitions.
        
        Args:
            indices: Indices of the transitions to update
            priorities: New priorities for the transitions
        """
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length")
            
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self._size:
                # Add epsilon to prevent zero probability
                # Also clip priorities to ensure numerical stability
                self.priorities[idx] = max(priority, 1e-8) + self.epsilon
            else:
                # Raise error for invalid indices
                raise IndexError(f"Index {idx} is out of range (0-{self._size-1})")
                
        # If we're using a max priority, update it
        if hasattr(self, 'max_priority'):
            self.max_priority = max(self.max_priority, max(priorities) + self.epsilon)
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if enough samples are available."""
        return self._size >= batch_size
    
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return self._size


def create_index_for_directory(
    directory: Union[str, Path], 
    recursive: bool = True,
    file_patterns: List[str] = ['*.jsonl', '*.npz', '*.h5', '*.hdf5', '*.parquet'],
    overwrite: bool = False
) -> Dict[str, ExperienceIndex]:
    """
    Create indices for all supported experience files in a directory.
    
    Args:
        directory: Directory to scan for experience files
        recursive: Whether to scan subdirectories
        file_patterns: Patterns to match experience files
        overwrite: Whether to overwrite existing index files
        
    Returns:
        Dictionary mapping file paths to ExperienceIndex objects
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise ValueError(f"{directory} is not a valid directory")
    
    # Find all matching files
    all_files = []
    for pattern in file_patterns:
        if recursive:
            all_files.extend(directory_path.glob(f"**/{pattern}"))
        else:
            all_files.extend(directory_path.glob(pattern))
    
    # Create indices
    indices = {}
    for filepath in all_files:
        index_path = filepath.with_suffix('.idx')
        
        # Skip if index exists and we're not overwriting
        if index_path.exists() and not overwrite:
            try:
                # Try to load the existing index
                index = ExperienceIndex(filepath, index_path)
                indices[str(filepath)] = index
                print(f"Loaded existing index for {filepath}")
            except Exception as e:
                print(f"Error loading index for {filepath}: {e}")
                if overwrite:
                    print(f"Will re-create index")
                else:
                    print(f"Skipping this file")
                    continue
        else:
            # Create new index
            try:
                index = ExperienceIndex(filepath, index_path, create_if_missing=True)
                indices[str(filepath)] = index
                print(f"Created new index for {filepath}")
            except Exception as e:
                print(f"Error creating index for {filepath}: {e}")
    
    return indices