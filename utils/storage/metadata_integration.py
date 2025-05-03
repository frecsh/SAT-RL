"""
Metadata integration with storage backends.
Connects the metadata system with experience storage backends to enable
perfect reproducibility and searchable experience data.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple

# Import our metadata system
from ..metadata import (
    ExperimentMetadata, 
    create_experiment_metadata, 
    update_experiment_metadata,
    generate_episode_seed,
    metadata_manager
)

# Import our storage backends
from .arrow_backend import StorageBackend, ArrowStorageBackend
from .hdf5_backend import HDF5StorageBackend

# Set up logging
logger = logging.getLogger(__name__)


class MetadataStorageAdapter:
    """
    Adapter for connecting metadata with storage backends.
    Adds metadata to experience storage and handles per-episode seeds.
    """
    
    def __init__(
        self, 
        backend: StorageBackend,
        experiment_name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        metadata_dir: str = "metadata"
    ):
        """
        Initialize the metadata storage adapter.
        
        Args:
            backend: Storage backend for experiences
            experiment_name: Name of the experiment
            description: Description of the experiment
            tags: List of searchable tags
            custom_metadata: Additional custom metadata
            seed: Random seed for reproducibility
            metadata_dir: Directory to store metadata files
        """
        self.backend = backend
        
        # Ensure metadata directory exists
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Create or load metadata
        self.metadata = create_experiment_metadata(
            name=experiment_name,
            description=description,
            tags=tags,
            custom_metadata=custom_metadata,
            seed=seed
        )
        
        # Add experiment ID to backend metadata
        self._update_backend_metadata()
        
        # Save metadata to file
        self.metadata_file = os.path.join(metadata_dir, f"{self.metadata.experiment_id}.json")
        self._save_metadata()
        
        # Track episode count
        self.episode_counter = 0
    
    def _update_backend_metadata(self):
        """Update the storage backend's metadata with experiment information."""
        # Get existing metadata from backend
        backend_meta = getattr(self.backend, "metadata", {}) or {}
        
        # Add experiment ID and basic info
        backend_meta["experiment_id"] = self.metadata.experiment_id
        backend_meta["experiment_name"] = self.metadata.name
        backend_meta["base_random_seed"] = self.metadata.base_random_seed
        
        # Add git information if available
        if self.metadata.git_info and self.metadata.git_info.commit_hash:
            backend_meta["git_commit_hash"] = self.metadata.git_info.commit_hash
            backend_meta["git_branch"] = self.metadata.git_info.branch
            backend_meta["git_is_dirty"] = str(self.metadata.git_info.is_dirty)
        
        # Update the backend's metadata
        if hasattr(self.backend, "metadata"):
            self.backend.metadata.update(backend_meta)
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                f.write(self.metadata.to_json())
        except Exception as e:
            logger.error(f"Error saving metadata to file: {e}")
    
    def start_episode(self) -> int:
        """
        Start a new episode and get a deterministic seed.
        
        Returns:
            int: Random seed for the episode
        """
        # Get a deterministic seed for this episode
        episode_seed = generate_episode_seed(self.metadata, self.episode_counter)
        
        # Track that we've started a new episode
        self.episode_counter += 1
        
        return episode_seed
    
    def write_batch(self, experiences: List[Dict[str, Any]]) -> bool:
        """
        Write a batch of experiences with metadata.
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            bool: True if write was successful
        """
        # Add metadata to experiences if not already present
        for exp in experiences:
            if "metadata" not in exp:
                exp["metadata"] = {}
            
            # Add episode seed if available
            if "episode_id" in exp:
                episode_id = exp["episode_id"]
                if episode_id in self.metadata.episode_seeds:
                    exp["metadata"]["episode_seed"] = self.metadata.episode_seeds[episode_id]
            
            # Add experiment ID
            exp["metadata"]["experiment_id"] = self.metadata.experiment_id
        
        # Write experiences to backend
        return self.backend.write_batch(experiences)
    
    def add_tags(self, tags: List[str]):
        """
        Add tags to the experiment metadata.
        
        Args:
            tags: List of tags to add
        """
        if not tags:
            return
        
        # Add tags to metadata
        self.metadata.tags.update(tags)
        
        # Save updated metadata
        self._save_metadata()
    
    def add_custom_metadata(self, custom_data: Dict[str, Any]):
        """
        Add custom metadata to the experiment.
        
        Args:
            custom_data: Custom metadata to add
        """
        if not custom_data:
            return
        
        # Add custom data to metadata
        self.metadata.custom_metadata.update(custom_data)
        
        # Save updated metadata
        self._save_metadata()
    
    def update_metadata(self):
        """Update dynamic fields in the metadata."""
        # Update timestamp and git info
        self.metadata = update_experiment_metadata(self.metadata)
        
        # Save updated metadata
        self._save_metadata()
    
    def close(self):
        """Close the storage backend and finalize metadata."""
        # Update metadata one last time
        self.update_metadata()
        
        # Close the backend
        self.backend.close()


class MetadataSearchEngine:
    """
    Search engine for finding experience data based on metadata.
    Enables searching across multiple storage backends by experiment attributes.
    """
    
    def __init__(self, metadata_dir: str = "metadata", storage_dirs: Optional[List[str]] = None):
        """
        Initialize the metadata search engine.
        
        Args:
            metadata_dir: Directory containing metadata files
            storage_dirs: List of directories containing storage backends
        """
        self.metadata_dir = metadata_dir
        self.storage_dirs = storage_dirs or []
        self.storage_cache = {}  # Maps experiment_id to storage backend
    
    def list_experiments(self) -> List[str]:
        """
        List all available experiment IDs.
        
        Returns:
            List of experiment IDs
        """
        experiment_ids = []
        
        # Check if directory exists
        if not os.path.exists(self.metadata_dir):
            return experiment_ids
        
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith(".json"):
                experiment_id = filename.replace(".json", "")
                experiment_ids.append(experiment_id)
                
        return experiment_ids
    
    def load_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """
        Load metadata for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            ExperimentMetadata if found, None otherwise
        """
        filepath = os.path.join(self.metadata_dir, f"{experiment_id}.json")
        
        try:
            with open(filepath, 'r') as f:
                return ExperimentMetadata.from_json(f.read())
        except Exception as e:
            logger.error(f"Error loading metadata from {filepath}: {e}")
            return None
    
    def search_experiments(self, 
                          tags: Optional[List[str]] = None,
                          match_all_tags: bool = False,
                          name_contains: Optional[str] = None,
                          git_hash: Optional[str] = None,
                          description_contains: Optional[str] = None,
                          custom_metadata_filter: Optional[Dict[str, Any]] = None) -> List[ExperimentMetadata]:
        """
        Search for experiments matching criteria.
        
        Args:
            tags: List of tags to search for
            match_all_tags: If True, all tags must match
            name_contains: String that must be in the experiment name
            git_hash: Git commit hash to match
            description_contains: String that must be in the description
            custom_metadata_filter: Dictionary of custom metadata to match
            
        Returns:
            List of matching experiment metadata
        """
        results = []
        
        for experiment_id in self.list_experiments():
            metadata = self.load_metadata(experiment_id)
            
            if not metadata:
                continue
                
            # Check tags
            if tags:
                tag_set = set(tags)
                if match_all_tags:
                    if not tag_set.issubset(metadata.tags):
                        continue
                else:
                    if not tag_set.intersection(metadata.tags):
                        continue
            
            # Check name
            if name_contains and name_contains.lower() not in metadata.name.lower():
                continue
                
            # Check git hash
            if git_hash and metadata.git_info.commit_hash != git_hash:
                continue
                
            # Check description
            if description_contains and description_contains.lower() not in metadata.description.lower():
                continue
                
            # Check custom metadata
            if custom_metadata_filter:
                match = True
                for key, value in custom_metadata_filter.items():
                    if key not in metadata.custom_metadata or metadata.custom_metadata[key] != value:
                        match = False
                        break
                
                if not match:
                    continue
                
            results.append(metadata)
            
        return results
    
    def _find_storage_for_experiment(self, experiment_id: str) -> Optional[StorageBackend]:
        """
        Find the storage backend for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            StorageBackend if found, None otherwise
        """
        # Check cache first
        if experiment_id in self.storage_cache:
            return self.storage_cache[experiment_id]
        
        # Load metadata to get information about the storage
        metadata = self.load_metadata(experiment_id)
        if not metadata:
            return None
        
        # Search storage directories for matching backend
        for storage_dir in self.storage_dirs:
            if not os.path.exists(storage_dir):
                continue
                
            # Check Arrow files
            arrow_dir = os.path.join(storage_dir, experiment_id)
            if os.path.exists(arrow_dir):
                try:
                    # Try to load as Arrow storage
                    from .arrow_backend import load_arrow_storage
                    backend = load_arrow_storage(arrow_dir)
                    self.storage_cache[experiment_id] = backend
                    return backend
                except Exception as e:
                    logger.debug(f"Failed to load Arrow storage for {experiment_id}: {e}")
            
            # Check HDF5 files
            hdf5_file = os.path.join(storage_dir, f"{experiment_id}.h5")
            if os.path.exists(hdf5_file):
                try:
                    # Try to load as HDF5 storage
                    from .hdf5_backend import load_hdf5_storage
                    backend = load_hdf5_storage(hdf5_file)
                    self.storage_cache[experiment_id] = backend
                    return backend
                except Exception as e:
                    logger.debug(f"Failed to load HDF5 storage for {experiment_id}: {e}")
        
        # Not found
        return None
    
    def get_experiences(self, experiment_id: str, indices: Optional[List[int]] = None, 
                      batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Get experiences from an experiment.
        
        Args:
            experiment_id: Experiment ID
            indices: Optional list of specific indices to read
            batch_size: Number of experiences to read if indices not specified
            
        Returns:
            List of experience dictionaries
        """
        # Find storage backend
        backend = self._find_storage_for_experiment(experiment_id)
        if not backend:
            logger.warning(f"No storage found for experiment {experiment_id}")
            return []
        
        # Read experiences
        return backend.read_batch(batch_size=batch_size, indices=indices)
    
    def reproduce_episode(self, experiment_id: str, episode_id: int) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
        """
        Get the seed and experiences for an episode to reproduce it.
        
        Args:
            experiment_id: Experiment ID
            episode_id: Episode ID
            
        Returns:
            Tuple of (episode_seed, experiences) if found, None otherwise
        """
        # Load metadata
        metadata = self.load_metadata(experiment_id)
        if not metadata or episode_id not in metadata.episode_seeds:
            logger.warning(f"No seed found for episode {episode_id} in experiment {experiment_id}")
            return None
        
        # Get seed
        seed = metadata.episode_seeds[episode_id]
        
        # Find storage backend
        backend = self._find_storage_for_experiment(experiment_id)
        if not backend:
            logger.warning(f"No storage found for experiment {experiment_id}")
            return seed, []
        
        # Read experiences for this episode
        # This assumes experiences have "episode_id" field
        all_experiences = backend.read_batch(batch_size=1000)  # Read a large batch
        episode_experiences = [exp for exp in all_experiences if exp.get("episode_id") == episode_id]
        
        return seed, episode_experiences


# Convenience functions

def create_storage_with_metadata(
    backend_type: str,
    schema,
    path: str,
    experiment_name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    metadata_dir: str = "metadata",
    **backend_kwargs
) -> MetadataStorageAdapter:
    """
    Create a storage backend with metadata tracking.
    
    Args:
        backend_type: Type of backend ("arrow" or "hdf5")
        schema: Experience schema
        path: Path for storage
        experiment_name: Name of the experiment
        description: Description of the experiment
        tags: List of searchable tags
        custom_metadata: Additional custom metadata
        seed: Random seed for reproducibility
        metadata_dir: Directory to store metadata files
        **backend_kwargs: Additional arguments for the backend
        
    Returns:
        MetadataStorageAdapter: The storage backend with metadata
    """
    # Create the appropriate backend
    if backend_type.lower() == "arrow":
        from .arrow_backend import create_arrow_storage
        backend = create_arrow_storage(schema, path, **backend_kwargs)
    elif backend_type.lower() == "hdf5":
        from .hdf5_backend import create_hdf5_storage
        backend = create_hdf5_storage(schema, path, **backend_kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    # Create metadata adapter
    return MetadataStorageAdapter(
        backend=backend,
        experiment_name=experiment_name,
        description=description,
        tags=tags,
        custom_metadata=custom_metadata,
        seed=seed,
        metadata_dir=metadata_dir
    )


def load_storage_with_metadata(
    experiment_id: str,
    storage_dirs: Optional[List[str]] = None,
    metadata_dir: str = "metadata"
) -> Optional[Tuple[MetadataStorageAdapter, ExperimentMetadata]]:
    """
    Load a storage backend with its metadata.
    
    Args:
        experiment_id: Experiment ID
        storage_dirs: List of directories to search for storage backends
        metadata_dir: Directory containing metadata files
        
    Returns:
        Tuple of (MetadataStorageAdapter, ExperimentMetadata) if found, None otherwise
    """
    # Create search engine
    search_engine = MetadataSearchEngine(metadata_dir, storage_dirs)
    
    # Load metadata
    metadata = search_engine.load_metadata(experiment_id)
    if not metadata:
        return None
    
    # Find storage backend
    backend = search_engine._find_storage_for_experiment(experiment_id)
    if not backend:
        return None
    
    # Create metadata adapter
    adapter = MetadataStorageAdapter(
        backend=backend,
        experiment_name=metadata.name,
        description=metadata.description,
        tags=list(metadata.tags),
        custom_metadata=metadata.custom_metadata,
        seed=metadata.base_random_seed,
        metadata_dir=metadata_dir
    )
    
    # Override the metadata with the loaded one
    adapter.metadata = metadata
    
    return adapter, metadata