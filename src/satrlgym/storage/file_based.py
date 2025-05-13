"""
File-based implementation of experience storage.

This module provides a storage backend that saves experiences to disk
for persistent storage and to handle larger datasets than memory-based solutions.
"""

import json
import os
import pickle
import shutil
from typing import Any

import numpy as np
from experience.storage.base import ExperienceStorage


class FileExperienceStorage(ExperienceStorage):
    """
    Stores experiences in files on disk with a memory cache.

    This storage backend provides a compromise between in-memory storage
    (which is fast but limited in size) and fully disk-based storage
    (which can handle large datasets but is slower). It maintains a
    memory cache of the most recently accessed items for fast retrieval.
    """

    def __init__(
        self,
        directory: str,
        cache_size: int = 10000,
        chunk_size: int = 1000,
        clear_on_init: bool = False,
    ) -> None:
        """
        Initialize file-based experience storage.

        Args:
            directory: Directory to store the experience files
            cache_size: Size of in-memory cache for recent experiences
            chunk_size: Number of experiences per chunk file
            clear_on_init: Whether to clear existing data on initialization
        """
        self.directory = directory
        self.cache_size = cache_size
        self.chunk_size = chunk_size

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Clear existing data if requested
        if clear_on_init and os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith(".pkl") or file.endswith(".json"):
                    os.remove(os.path.join(directory, file))

        # Initialize metadata
        self.metadata_path = os.path.join(directory, "metadata.json")
        self.metadata = self._load_or_create_metadata()

        # Initialize cache
        self.cache = {}  # Maps from index to experience data
        self.cache_indices = []  # LRU list for cache management

    def _load_or_create_metadata(self) -> dict[str, Any]:
        """
        Load existing metadata or create new if none exists.

        Returns:
            Dictionary containing storage metadata
        """
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path) as f:
                return json.load(f)
        else:
            # Default metadata
            metadata = {
                "size": 0,
                "schema": {},
                "chunks": [],  # List of chunk filenames
                "chunk_sizes": [],  # Number of experiences in each chunk
                "current_chunk": 0,
            }
            self._save_metadata(metadata)
            return metadata

    def _save_metadata(self, metadata: dict[str, Any] | None = None) -> None:
        """
        Save metadata to disk.

        Args:
            metadata: Metadata to save (uses self.metadata if None)
        """
        if metadata is None:
            metadata = self.metadata

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)

    def _get_chunk_path(self, chunk_idx: int) -> str:
        """
        Get the path to a chunk file.

        Args:
            chunk_idx: Index of the chunk

        Returns:
            Path to the chunk file
        """
        return os.path.join(self.directory, f"chunk_{chunk_idx:06d}.pkl")

    def _load_chunk(self, chunk_idx: int) -> dict[str, Any]:
        """
        Load a chunk of experiences from disk.

        Args:
            chunk_idx: Index of the chunk to load

        Returns:
            Dictionary containing the chunk data
        """
        chunk_path = self._get_chunk_path(chunk_idx)
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        with open(chunk_path, "rb") as f:
            return pickle.load(f)

    def _save_chunk(self, chunk_idx: int, chunk_data: dict[str, Any]) -> None:
        """
        Save a chunk of experiences to disk.

        Args:
            chunk_idx: Index of the chunk to save
            chunk_data: Dictionary containing the chunk data
        """
        chunk_path = self._get_chunk_path(chunk_idx)

        with open(chunk_path, "wb") as f:
            pickle.dump(chunk_data, f)

    def _get_chunk_and_offset(self, idx: int) -> tuple[int, int]:
        """
        Convert a global index to chunk index and offset.

        Args:
            idx: Global index of the experience

        Returns:
            Tuple of (chunk_index, offset_within_chunk)
        """
        if idx >= self.metadata["size"] or idx < 0:
            raise IndexError(
                f"Index {idx} out of range (0 to {self.metadata['size']-1})"
            )

        # Navigate through chunks to find the right one
        offset = idx
        for i, chunk_size in enumerate(self.metadata["chunk_sizes"]):
            if offset < chunk_size:
                return i, offset
            offset -= chunk_size

        # Should never reach here if metadata is consistent
        raise RuntimeError("Inconsistent metadata - could not locate index in chunks")

    def _update_cache(self, idx: int, data: dict[str, Any]) -> None:
        """
        Update the memory cache with new data.

        Args:
            idx: Index of the experience
            data: Experience data to cache
        """
        # Remove from cache if already present
        if idx in self.cache:
            self.cache_indices.remove(idx)
        # Add to cache
        self.cache[idx] = data
        self.cache_indices.append(idx)

        # Maintain cache size limit
        while len(self.cache_indices) > self.cache_size:
            oldest_idx = self.cache_indices.pop(0)
            del self.cache[oldest_idx]

    def add(self, experience_batch: dict[str, Any]) -> None:
        """
        Add experiences to storage.

        Args:
            experience_batch: Dictionary with experience data
        """
        # Initialize schema if this is the first addition
        if not self.metadata["schema"]:
            self.metadata["schema"] = {
                key: str(type(value[0]).__name__)
                for key, value in experience_batch.items()
            }

        batch_size = len(next(iter(experience_batch.values())))

        # Process each experience in the batch
        for i in range(batch_size):
            # Get current chunk info
            current_chunk_idx = self.metadata["current_chunk"]

            # Load current chunk or create new
            if current_chunk_idx < len(self.metadata["chunks"]):
                if (
                    len(self.metadata["chunks"]) > current_chunk_idx
                    and self.metadata["chunk_sizes"][current_chunk_idx]
                    >= self.chunk_size
                ):
                    # Current chunk is full, move to the next one
                    current_chunk_idx += 1
                    if current_chunk_idx == len(self.metadata["chunks"]):
                        # Need to create a new chunk
                        self.metadata["chunks"].append(
                            self._get_chunk_path(current_chunk_idx)
                        )
                        self.metadata["chunk_sizes"].append(0)
                    self.metadata["current_chunk"] = current_chunk_idx

                chunk_data = self._load_chunk(current_chunk_idx)
            else:
                # Create first chunk
                chunk_data = {k: [] for k in experience_batch.keys()}
                self.metadata["chunks"].append(self._get_chunk_path(0))
                self.metadata["chunk_sizes"].append(0)

            # Extract single experience from batch
            experience = {k: v[i] for k, v in experience_batch.items()}

            # Add to chunk data
            for key, value in experience.items():
                if key not in chunk_data:
                    chunk_data[key] = []
                chunk_data[key].append(value)

            # Update metadata
            self.metadata["chunk_sizes"][current_chunk_idx] += 1
            self.metadata["size"] += 1

            # Save updated chunk
            self._save_chunk(current_chunk_idx, chunk_data)

        # Save updated metadata
        self._save_metadata()

    def get(self, indices: list[int] | None = None) -> dict[str, Any]:
        """
        Retrieve experiences from storage.

        Args:
            indices: Optional list of indices to retrieve. If None, retrieves all data.

        Returns:
            Dictionary of retrieved experience data
        """
        if self.metadata["size"] == 0:
            return {}

        # Default to all indices if none provided
        if indices is None:
            indices = list(range(self.metadata["size"]))

        # Group indices by chunk for efficient retrieval
        chunk_indices = {}
        for idx in indices:
            if idx in self.cache:
                continue  # Skip indices that are already cached

            try:
                chunk_idx, offset = self._get_chunk_and_offset(idx)
                if chunk_idx not in chunk_indices:
                    chunk_indices[chunk_idx] = []
                chunk_indices[chunk_idx].append((idx, offset))
            except IndexError:
                # Skip invalid indices
                pass

        # Retrieve items from each chunk
        for chunk_idx, idx_offsets in chunk_indices.items():
            chunk_data = self._load_chunk(chunk_idx)

            # Update cache for each retrieved item
            for global_idx, chunk_offset in idx_offsets:
                experience = {k: v[chunk_offset] for k, v in chunk_data.items()}
                self._update_cache(global_idx, experience)

        # Collect results from cache
        result = {}
        for idx in indices:
            if idx in self.cache:
                for key, value in self.cache[idx].items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)

        # Convert lists to numpy arrays
        for key in result:
            result[key] = np.array(result[key])

        return result

    def sample(self, batch_size: int) -> dict[str, Any]:
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary of sampled experience data
        """
        if self.metadata["size"] == 0:
            raise ValueError("Cannot sample from empty storage")

        indices = np.random.randint(0, self.metadata["size"], size=batch_size)
        return self.get(indices.tolist())

    def get_size(self) -> int:
        """
        Get the number of experiences in storage.

        Returns:
            Number of stored experiences
        """
        return self.metadata["size"]

    def save(self, path: str) -> None:
        """
        Save experiences to a file.

        For file-based storage, this creates a backup of the entire storage.

        Args:
            path: Path to save the backup
        """
        # Create backup directory
        backup_dir = path + "_backup"
        os.makedirs(backup_dir, exist_ok=True)

        # Copy metadata
        shutil.copy(self.metadata_path, os.path.join(backup_dir, "metadata.json"))

        # Copy all chunks
        for chunk_path in self.metadata["chunks"]:
            if os.path.exists(chunk_path):
                shutil.copy(
                    chunk_path, os.path.join(backup_dir, os.path.basename(chunk_path))
                )

    def load(self, path: str) -> None:
        """
        Load experiences from a file.

        For file-based storage, this restores from a backup.

        Args:
            path: Path to load the backup from
        """
        # Check if backup exists
        backup_dir = path + "_backup"
        if not os.path.exists(backup_dir):
            raise FileNotFoundError(f"Backup directory not found: {backup_dir}")

        # Clear current data
        self.clear()

        # Copy metadata
        backup_metadata_path = os.path.join(backup_dir, "metadata.json")
        if os.path.exists(backup_metadata_path):
            shutil.copy(backup_metadata_path, self.metadata_path)
            self.metadata = self._load_or_create_metadata()

        # Copy all chunks
        for chunk_path in self.metadata["chunks"]:
            backup_chunk_path = os.path.join(backup_dir, os.path.basename(chunk_path))
            if os.path.exists(backup_chunk_path):
                shutil.copy(backup_chunk_path, chunk_path)

    def clear(self) -> None:
        """
        Clear all experiences from storage.
        """
        # Remove all chunk files
        for chunk_path in self.metadata["chunks"]:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

        # Reset metadata
        self.metadata = {
            "size": 0,
            "schema": self.metadata.get("schema", {}),  # Preserve schema
            "chunks": [],
            "chunk_sizes": [],
            "current_chunk": 0,
        }
        self._save_metadata()

        # Clear cache
        self.cache = {}
        self.cache_indices = []
