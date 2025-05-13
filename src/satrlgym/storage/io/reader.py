"""
Experience data reader.

This module provides classes for reading experience data from various storage backends.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

# Import storage backends

# Set up logging
logger = logging.getLogger(__name__)


class ExperienceReader:
    """
    Reader for experience data from various storage backends.

    This class provides a unified interface for reading experience data
    from different storage backends, handling backend-specific details.
    """

    def __init__(
        self,
        data_path: str | Path,
        storage_type: str | None = None,
        batch_size: int = 1000,
        shuffle: bool = False,
        **kwargs,
    ):
        """
        Initialize the reader.

        Args:
            data_path: Path to the experience data
            storage_type: Type of storage backend ('hdf5', 'parquet', 'memory', 'file')
            batch_size: Size of batches to read at once
            shuffle: Whether to shuffle the data during reading
            **kwargs: Additional backend-specific parameters
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Auto-detect storage type if not specified
        if storage_type is None:
            storage_type = self._detect_storage_type(data_path)

        self.storage_type = storage_type
        self._reader = self._create_reader(**kwargs)

        # Helper attributes for iteration
        self._current_batch = None
        self._batch_index = 0
        self._exhausted = False

    def _detect_storage_type(self, data_path: str | Path) -> str:
        """
        Detect the storage type from the file extension or directory structure.

        Args:
            data_path: Path to examine

        Returns:
            Detected storage type as string

        Raises:
            ValueError: If storage type cannot be determined
        """
        data_path = Path(data_path)

        # Check file extension
        if data_path.is_file():
            if data_path.suffix in [".h5", ".hdf5"]:
                return "hdf5"
            elif data_path.suffix == ".parquet":
                return "parquet"
            elif data_path.suffix == ".json":
                return "file"
            elif data_path.suffix == ".experience":
                # Default to parquet for .experience files
                return "parquet"

        # Check directory structure
        elif data_path.is_dir():
            # Check for parquet files
            if any(data_path.glob("*.parquet")):
                return "parquet"
            # Check for HDF5 files
            elif any(data_path.glob("*.h5")) or any(data_path.glob("*.hdf5")):
                return "hdf5"
            # Check for memory mapped files
            elif (data_path / "mmap_arrays").exists():
                return "memory"
            # Check for individual JSON files
            elif any(data_path.glob("transition_*.json")):
                return "file"
            # Check for .experience files
            elif any(data_path.glob("*.experience")):
                return "parquet"

        raise ValueError(
            f"Could not determine storage type from {data_path}. "
            "Please specify storage_type explicitly."
        )

    def _create_reader(self, **kwargs):
        """
        Create a reader for the specified storage type.

        Args:
            **kwargs: Backend-specific parameters

        Returns:
            A reader object for the specified storage type

        Raises:
            ValueError: If the storage type is unsupported
        """
        if self.storage_type == "hdf5":
            return HDF5Reader(self.data_path, self.batch_size, self.shuffle, **kwargs)
        elif self.storage_type == "parquet":
            return ParquetReader(
                self.data_path, self.batch_size, self.shuffle, **kwargs
            )
        elif self.storage_type == "memory":
            return MemoryMappedReader(
                self.data_path, self.batch_size, self.shuffle, **kwargs
            )
        elif self.storage_type == "file":
            return FileReader(self.data_path, self.batch_size, self.shuffle, **kwargs)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over transitions.

        Returns:
            Iterator yielding transitions as dictionaries
        """
        return self._reader.__iter__()

    def create_worker_reader(
        self, worker_id: int, num_workers: int
    ) -> "ExperienceReader":
        """
        Create a worker-specific reader for parallel processing.

        Args:
            worker_id: Worker ID (0-based)
            num_workers: Total number of workers

        Returns:
            A reader configured for the specific worker
        """
        # Create a new reader with the same parameters
        worker_reader = ExperienceReader(
            data_path=self.data_path,
            storage_type=self.storage_type,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        # Configure worker-specific parameters
        worker_reader._reader.configure_for_worker(worker_id, num_workers)

        return worker_reader

    def read_batch(self, batch_size: int | None = None) -> dict[str, np.ndarray]:
        """
        Read a batch of transitions.

        Args:
            batch_size: Size of the batch to read (overrides self.batch_size)

        Returns:
            A dictionary with batched arrays
        """
        # Use specified batch size or default
        batch_size = batch_size or self.batch_size

        # Read batch from backend
        return self._reader.read_batch(batch_size)


class _BaseReader:
    """Base class for storage-specific readers."""

    def __init__(self, data_path: str | Path, batch_size: int, shuffle: bool, **kwargs):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Worker-specific parameters for parallel processing
        self.worker_id = 0
        self.num_workers = 1

    def configure_for_worker(self, worker_id: int, num_workers: int):
        """
        Configure the reader for parallel processing.

        Args:
            worker_id: Worker ID (0-based)
            num_workers: Total number of workers
        """
        self.worker_id = worker_id
        self.num_workers = num_workers

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over transitions.

        Returns:
            Iterator yielding transitions as dictionaries
        """
        raise NotImplementedError("Subclasses must implement __iter__")

    def read_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Read a batch of transitions.

        Args:
            batch_size: Size of batch to read

        Returns:
            Dictionary with batched arrays
        """
        raise NotImplementedError("Subclasses must implement read_batch")


class HDF5Reader(_BaseReader):
    """Reader for HDF5 storage backend."""

    def __init__(self, data_path: str | Path, batch_size: int, shuffle: bool, **kwargs):
        super().__init__(data_path, batch_size, shuffle, **kwargs)

        # Import h5py here for lazy loading
        import h5py

        # Open the HDF5 file
        self.file = h5py.File(self.data_path, "r")

        # Get the total number of transitions
        self.total_transitions = self.file.attrs.get("transition_count", 0)

        # Generate indices for all transitions
        self.indices = np.arange(self.total_transitions)

        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Current position in the index array
        self.position = 0

        # Cache for group structures
        self.group_structures = {}

        # Load group structures
        for group_name in self.file:
            if group_name.startswith("group_"):
                group = self.file[group_name]
                if "keys" in group.attrs:
                    keys = json.loads(group.attrs["keys"])
                    self.group_structures[group_name] = keys

    def configure_for_worker(self, worker_id: int, num_workers: int):
        """Configure for parallel processing by splitting indices."""
        super().configure_for_worker(worker_id, num_workers)

        # Split indices among workers
        indices_per_worker = len(self.indices) // num_workers
        start_idx = worker_id * indices_per_worker
        end_idx = (
            start_idx + indices_per_worker
            if worker_id < num_workers - 1
            else len(self.indices)
        )

        # Update indices for this worker
        self.indices = self.indices[start_idx:end_idx]
        self.position = 0

    def __iter__(self):
        """Iterate over transitions in the HDF5 file."""
        self.position = 0

        while self.position < len(self.indices):
            # Get the index for the current transition
            idx = self.indices[self.position]

            # Find the group containing this transition
            transition = {}
            for group_name, keys in self.group_structures.items():
                group = self.file[group_name]

                # Check if this group contains the transition
                found = False
                for dataset_name in group:
                    if dataset_name.split("_")[0] in keys:
                        dataset = group[dataset_name]
                        start_idx = dataset.attrs.get("start_idx", 0)
                        end_idx = dataset.attrs.get("end_idx", start_idx + len(dataset))

                        if start_idx <= idx < end_idx:
                            found = True
                            # Get all keys for this transition
                            for key in keys:
                                # Find the dataset containing this key for this transition
                                relative_idx = idx - start_idx
                                for dname in group:
                                    if dname.startswith(f"{key}_"):
                                        dset = group[dname]
                                        dset_start = dset.attrs.get("start_idx", 0)
                                        dset_end = dset.attrs.get(
                                            "end_idx", dset_start + len(dset)
                                        )

                                        if dset_start <= idx < dset_end:
                                            # Get the value for this key
                                            value = dset[relative_idx]
                                            transition[key] = value
                                            break
                            break

                if found:
                    break

            # Increment position
            self.position += 1

            # Yield the transition
            if transition:
                yield transition

    def read_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Read a batch of transitions."""
        # Initialize batch
        batch = {}

        # Read transitions
        count = 0
        for transition in self:
            # Initialize batch arrays if first transition
            if not batch:
                for key, value in transition.items():
                    batch[key] = np.zeros(
                        (batch_size,) + np.shape(value), dtype=np.array(value).dtype
                    )

            # Add transition to batch
            for key, value in transition.items():
                batch[key][count] = value

            # Increment counter
            count += 1

            # Break if batch is full
            if count >= batch_size:
                break

        # Resize batch if not full
        if count < batch_size:
            for key in batch:
                batch[key] = batch[key][:count]

        return batch

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "file") and self.file:
            self.file.close()


class ParquetReader(_BaseReader):
    """Reader for Parquet storage backend."""

    def __init__(self, data_path: str | Path, batch_size: int, shuffle: bool, **kwargs):
        super().__init__(data_path, batch_size, shuffle, **kwargs)

        # Import pyarrow here for lazy loading
        import pyarrow.parquet as pq

        # Find all parquet files
        self.data_path = Path(self.data_path)
        if self.data_path.is_file():
            self.parquet_files = [self.data_path]
        else:
            self.parquet_files = list(self.data_path.glob("*.parquet"))

        # Sort files for deterministic ordering
        self.parquet_files.sort()

        # Generate file indices
        self.file_indices = []
        self.file_sizes = []
        self.total_rows = 0

        # Get metadata for each file
        for file_idx, file_path in enumerate(self.parquet_files):
            # Get number of rows in this file
            metadata = pq.read_metadata(file_path)
            num_rows = metadata.num_rows

            # Add to total
            self.total_rows += num_rows
            self.file_sizes.append(num_rows)

            # Add file index and row indices
            self.file_indices.extend(
                [(file_idx, row_idx) for row_idx in range(num_rows)]
            )

        # Shuffle indices if requested
        if self.shuffle:
            np.random.shuffle(self.file_indices)

        # Current position in the index array
        self.position = 0

        # Cache for open files
        self.open_files = {}

    def configure_for_worker(self, worker_id: int, num_workers: int):
        """Configure for parallel processing by splitting indices."""
        super().configure_for_worker(worker_id, num_workers)

        # Split indices among workers
        indices_per_worker = len(self.file_indices) // num_workers
        start_idx = worker_id * indices_per_worker
        end_idx = (
            start_idx + indices_per_worker
            if worker_id < num_workers - 1
            else len(self.file_indices)
        )

        # Update indices for this worker
        self.file_indices = self.file_indices[start_idx:end_idx]
        self.position = 0

    def __iter__(self):
        """Iterate over transitions in Parquet files."""
        self.position = 0

        while self.position < len(self.file_indices):
            # Get the file and row index
            file_idx, row_idx = self.file_indices[self.position]

            # Open the file if not already open
            if file_idx not in self.open_files:
                import pyarrow.parquet as pq

                self.open_files[file_idx] = pq.ParquetFile(self.parquet_files[file_idx])

            # Read the row
            parquet_file = self.open_files[file_idx]

            # Read a small batch containing the row
            min(100, self.file_sizes[file_idx] - row_idx)
            # Calculate row group index based on rows per row group
            rows_per_group = (
                parquet_file.metadata.num_rows / parquet_file.metadata.num_row_groups
            )
            table = parquet_file.read_row_group(
                int(row_idx // rows_per_group), columns=None
            )

            # Convert to pandas for easier row access
            df = table.to_pandas()

            # Get the row and convert to dict
            row = df.iloc[row_idx % parquet_file.metadata.row_group_size]
            transition = row.to_dict()

            # Process numpy arrays that might be stored as lists
            for key, value in transition.items():
                if isinstance(value, list):
                    try:
                        transition[key] = np.array(value)
                    except Exception:
                        pass

            # Increment position
            self.position += 1

            # Yield the transition
            yield transition

    def read_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Read a batch of transitions."""
        # Initialize batch
        batch = {}

        # Read transitions
        count = 0
        for transition in self:
            # Initialize batch arrays if first transition
            if not batch:
                for key, value in transition.items():
                    if isinstance(value, np.ndarray):
                        batch[key] = np.zeros(
                            (batch_size,) + value.shape, dtype=value.dtype
                        )
                    else:
                        batch[key] = np.zeros(batch_size, dtype=type(value))

            # Add transition to batch
            for key, value in transition.items():
                batch[key][count] = value

            # Increment counter
            count += 1

            # Break if batch is full
            if count >= batch_size:
                break

        # Resize batch if not full
        if count < batch_size:
            for key in batch:
                batch[key] = batch[key][:count]

        return batch

    def __del__(self):
        """Clean up resources."""
        self.open_files = {}


class MemoryMappedReader(_BaseReader):
    """Reader for memory-mapped storage backend."""

    def __init__(self, data_path: str | Path, batch_size: int, shuffle: bool, **kwargs):
        super().__init__(data_path, batch_size, shuffle, **kwargs)

        # Ensure path is a directory
        self.data_path = Path(data_path)
        self.mm_dir = self.data_path / "mmap_arrays"

        # Load index file
        index_path = self.data_path / "index.json"
        with open(index_path) as f:
            self.index = json.load(f)

        # Total transitions
        self.total_transitions = self.index.get("count", 0)

        # Generate indices
        self.indices = np.arange(self.total_transitions)

        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Current position
        self.position = 0

        # Load memory-mapped arrays
        self.arrays = {}
        for file_path in self.mm_dir.glob("*.npy"):
            if not file_path.stem.endswith(("_shape", "_dtype")):
                array_name = file_path.stem

                # Load shape and dtype
                shape_file = self.mm_dir / f"{array_name}_shape.json"
                dtype_file = self.mm_dir / f"{array_name}_dtype.json"

                if shape_file.exists() and dtype_file.exists():
                    with open(shape_file) as f:
                        shape = tuple(json.load(f))
                    with open(dtype_file) as f:
                        dtype_str = json.load(f)
                        dtype = np.dtype(dtype_str)

                    # Create memory-mapped array
                    self.arrays[array_name] = np.memmap(
                        file_path, dtype=dtype, mode="r", shape=shape
                    )

    def configure_for_worker(self, worker_id: int, num_workers: int):
        """Configure for parallel processing by splitting indices."""
        super().configure_for_worker(worker_id, num_workers)

        # Split indices among workers
        indices_per_worker = len(self.indices) // num_workers
        start_idx = worker_id * indices_per_worker
        end_idx = (
            start_idx + indices_per_worker
            if worker_id < num_workers - 1
            else len(self.indices)
        )

        # Update indices for this worker
        self.indices = self.indices[start_idx:end_idx]
        self.position = 0

    def __iter__(self):
        """Iterate over transitions in memory-mapped arrays."""
        self.position = 0

        while self.position < len(self.indices):
            # Get the index for the current transition
            idx = int(self.indices[self.position])

            # Find the transition info
            transition_info = None
            for info in self.index["transitions"]:
                if info["position"] == idx:
                    transition_info = info
                    break

            if transition_info is None:
                # Skip if transition info not found
                self.position += 1
                continue

            # Build the transition
            transition = {}
            for key in transition_info["keys"]:
                if key in self.arrays:
                    value = self.arrays[key][idx]

                    # Handle pickled objects
                    if self.arrays[key].dtype == np.dtype("O"):
                        import pickle

                        try:
                            value = pickle.loads(value)
                        except Exception:
                            pass

                    transition[key] = value

            # Increment position
            self.position += 1

            # Yield the transition
            yield transition

    def read_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Read a batch of transitions."""
        # Get indices for the batch
        if self.position + batch_size <= len(self.indices):
            batch_indices = self.indices[self.position : self.position + batch_size]
        else:
            batch_indices = self.indices[self.position :]

        # Get unique keys across all transitions in the batch
        keys = set()
        for idx in batch_indices:
            for info in self.index["transitions"]:
                if info["position"] == idx:
                    keys.update(info["keys"])
                    break

        # Initialize batch dictionary
        batch = {key: [] for key in keys}

        # Read each transition
        for idx in batch_indices:
            for key in keys:
                if key in self.arrays and idx < len(self.arrays[key]):
                    batch[key].append(self.arrays[key][idx])
                else:
                    # Use None for missing values
                    batch[key].append(None)

        # Convert lists to numpy arrays
        for key in batch:
            values = batch[key]
            if all(v is not None for v in values):
                try:
                    batch[key] = np.array(values)
                except Exception:
                    pass

        # Update position
        self.position += len(batch_indices)

        return batch


class FileReader(_BaseReader):
    """Reader for file-based storage backend."""

    def __init__(self, data_path: str | Path, batch_size: int, shuffle: bool, **kwargs):
        super().__init__(data_path, batch_size, shuffle, **kwargs)

        self.data_path = Path(data_path)

        # Determine if consolidated or individual mode
        if self.data_path.is_file():
            self.mode = "consolidated"

            # Check if compressed
            self.compression = None
            if self.data_path.suffix in [".gz", ".bz2", ".xz"]:
                self.compression = self.data_path.suffix[1:]

            # Load data
            with self._open_file("r") as f:
                self.data = json.load(f)
                self.transitions = self.data.get("transitions", [])
        else:
            self.mode = "individual"

            # Find all transition files
            self.transition_files = sorted(
                list(self.data_path.glob("transition_*.json")),
                key=lambda p: int(p.stem.split("_")[1]),
            )

            # No need to load all transitions in memory
            self.transitions = None

        # Generate indices
        if self.mode == "consolidated":
            self.total_transitions = len(self.transitions)
            self.indices = np.arange(self.total_transitions)
        else:
            self.total_transitions = len(self.transition_files)
            self.indices = np.arange(self.total_transitions)

        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Current position
        self.position = 0

    def _open_file(self, mode="r"):
        """
        Open the file with appropriate compression handling.

        Args:
            mode: File mode ('r', 'w', 'a')

        Returns:
            File object
        """
        if self.compression == "gz" or self.compression == "gzip":
            import gzip

            return gzip.open(self.data_path, mode + "t", encoding="utf-8")
        elif self.compression == "bz2":
            import bz2

            return bz2.open(self.data_path, mode + "t", encoding="utf-8")
        elif self.compression == "xz":
            import lzma

            return lzma.open(self.data_path, mode + "t", encoding="utf-8")
        else:
            return open(self.data_path, mode, encoding="utf-8")

    def configure_for_worker(self, worker_id: int, num_workers: int):
        """Configure for parallel processing by splitting indices."""
        super().configure_for_worker(worker_id, num_workers)

        # Split indices among workers
        indices_per_worker = len(self.indices) // num_workers
        start_idx = worker_id * indices_per_worker
        end_idx = (
            start_idx + indices_per_worker
            if worker_id < num_workers - 1
            else len(self.indices)
        )

        # Update indices for this worker
        self.indices = self.indices[start_idx:end_idx]
        self.position = 0

    def _process_transition(self, transition: dict[str, Any]) -> dict[str, Any]:
        """
        Process a transition dictionary to convert serialized data.

        Args:
            transition: Raw transition dictionary

        Returns:
            Processed transition dictionary
        """
        result = {}

        for key, value in transition.items():
            # Handle serialized objects
            if isinstance(value, dict) and "_serialized" in value:
                if value["_serialized"] == "pickle":
                    # Deserialize pickled data
                    import base64
                    import pickle

                    binary_data = base64.b64decode(value["data"])
                    result[key] = pickle.loads(binary_data)
                else:
                    # Keep as is for unknown serialization
                    result[key] = value
            elif isinstance(value, list):
                # Try to convert lists to numpy arrays
                try:
                    result[key] = np.array(value)
                except Exception:
                    result[key] = value
            else:
                result[key] = value

        return result

    def __iter__(self):
        """Iterate over transitions in files."""
        self.position = 0

        while self.position < len(self.indices):
            # Get the index for the current transition
            idx = int(self.indices[self.position])

            # Get the transition
            if self.mode == "consolidated":
                # Get from in-memory list
                transition = self.transitions[idx]
            else:
                # Load from file
                if idx < len(self.transition_files):
                    with open(self.transition_files[idx]) as f:
                        transition = json.load(f)
                else:
                    # Skip if file not found
                    self.position += 1
                    continue

            # Process the transition
            processed = self._process_transition(transition)

            # Increment position
            self.position += 1

            # Yield the transition
            yield processed

    def read_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Read a batch of transitions."""
        # Initialize batch
        batch = {}

        # Read transitions
        count = 0
        for transition in self:
            # Initialize batch arrays if first transition
            if not batch:
                for key, value in transition.items():
                    if isinstance(value, np.ndarray):
                        batch[key] = np.zeros(
                            (batch_size,) + value.shape, dtype=value.dtype
                        )
                    else:
                        batch[key] = np.zeros(batch_size, dtype=type(value))

            # Add transition to batch
            for key, value in transition.items():
                if key in batch:
                    batch[key][count] = value

            # Increment counter
            count += 1

            # Break if batch is full
            if count >= batch_size:
                break

        # Resize batch if not full
        if count < batch_size:
            for key in batch:
                batch[key] = batch[key][:count]

        return batch
