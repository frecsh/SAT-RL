"""
Concurrent operations for experience data handling.

This module provides functionality for:
1. Writer locking mechanisms for multi-actor systems
2. Sharded files for parallel writers
3. Thread-safe reader implementation
"""

import atexit
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

# Set up logging
logger = logging.getLogger(__name__)


class ConcurrentWriter:
    """
    Thread and process safe writer for experience data.

    This class provides locking mechanisms to ensure that multiple writers
    can safely write to the same experience file or set of files.
    """

    def __init__(
        self,
        base_filepath: str | Path,
        format_type: str = "jsonl",
        lock_timeout: float = 10.0,
        auto_flush: bool = True,
    ):
        """
        Initialize the concurrent writer.

        Args:
            base_filepath: Base path for the output file
            format_type: Format type ('jsonl', 'npz', 'parquet', 'hdf5')
            lock_timeout: Timeout for acquiring locks in seconds
            auto_flush: Whether to flush after every write
        """
        self.base_filepath = Path(base_filepath)
        self.format_type = format_type.lower()
        self.lock_timeout = lock_timeout
        self.auto_flush = auto_flush

        # Create directory if needed
        os.makedirs(self.base_filepath.parent, exist_ok=True)

        # Set up file extension
        if self.format_type == "jsonl":
            self.extension = ".jsonl"
        elif self.format_type == "npz":
            self.extension = ".npz"
        elif self.format_type == "parquet":
            self.extension = ".parquet"
        elif self.format_type == "hdf5":
            self.extension = ".h5"
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        # Set up lock file path
        self.lock_file = self.base_filepath.with_suffix(".lock")

        # Set up file handle
        self.file = None
        self.lock = None

        # Register cleanup handler
        atexit.register(self.close)

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False  # Don't suppress exceptions

    def open(self):
        """Open the writer and acquire the lock."""
        # Create a file lock
        self.lock = FileLock(str(self.lock_file), timeout=self.lock_timeout)

        try:
            # Only open file for formats that keep files open
            if self.format_type == "jsonl":
                # Open the file in append mode
                self.file = open(self.base_filepath.with_suffix(self.extension), "a")
            elif self.format_type in ["npz", "parquet", "hdf5"]:
                # For these formats, we don't keep the file open continuously
                # Just acquire the lock to ensure we have access
                self.lock.acquire()
            else:
                raise ValueError(f"Unsupported format type: {self.format_type}")
        except Exception as e:
            if self.lock is not None:
                if self.lock.is_locked:
                    self.lock.release()
                self.lock = None
            raise OSError(f"Could not open file {self.base_filepath}: {e}")

    def close(self):
        """Close the writer and release the lock."""
        # Close the file if it's open
        if self.file is not None:
            try:
                self.file.close()
            except Exception as e:
                logger.warning(f"Error closing file: {e}")
            finally:
                self.file = None

        # Release the lock if we have it and it's locked
        if self.lock is not None:
            try:
                if hasattr(self.lock, "is_locked") and self.lock.is_locked:
                    self.lock.release()
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")
            finally:
                self.lock = None

    def write(self, data: dict[str, Any]):
        """
        Write a single data item to the file.

        Args:
            data: Data to write
        """
        # For thread safety, use both thread lock and file lock
        with threading.RLock():
            # Ensure we have an open file
            if self.file is None and self.format_type == "jsonl":
                self.open()

            # Write based on format
            if self.format_type == "jsonl":
                self._write_jsonl(data)
            elif self.format_type == "npz":
                self._write_npz(data)
            elif self.format_type == "parquet":
                self._write_parquet(data)
            elif self.format_type == "hdf5":
                self._write_hdf5(data)

    def _write_jsonl(self, data: dict[str, Any]):
        """Write to a JSON Lines file."""
        # Use the lock to ensure atomic write
        with self.lock:
            self.file.write(json.dumps(data) + "\n")
            if self.auto_flush:
                self.file.flush()

    def _write_npz(self, data: dict[str, Any]):
        """Write to an NPZ file."""
        # For NPZ, we need to load the existing data, append, and save
        npz_path = self.base_filepath.with_suffix(self.extension)

        # Make sure we have a lock
        if self.lock is None:
            self.lock = FileLock(str(self.lock_file), timeout=self.lock_timeout)

        # Use the lock for thread safety
        with self.lock:
            # Load existing data or create new
            existing_data = {}
            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path, allow_pickle=True) as npz_data:
                        # Convert to dictionary
                        for k in npz_data.files:
                            existing_data[k] = npz_data[k]
                except Exception as e:
                    logger.warning(f"Error loading NPZ file: {e}, creating new file")

            # Update data
            for key, value in data.items():
                # Convert value to numpy array if needed
                if not isinstance(value, np.ndarray):
                    value = np.array(value)

                # If key already exists in existing_data, append
                if key in existing_data:
                    # Make sure values are properly shaped for concatenation
                    if value.ndim == 1:
                        value = value.reshape(1, -1)  # Convert 1D arrays to 2D rows

                    if existing_data[key].ndim == 1:
                        # If existing data is 1D, make it a 2D row
                        existing_data[key] = existing_data[key].reshape(1, -1)

                    # Now concatenate along the first axis (add rows)
                    existing_data[key] = np.vstack([existing_data[key], value])
                else:
                    # For new keys, ensure it's shaped correctly for future concatenation
                    if value.ndim == 1:
                        value = value.reshape(1, -1)  # Convert to 2D array (single row)
                    existing_data[key] = value

            # Save back to file
            try:
                np.savez(npz_path, **existing_data)
            except Exception as e:
                logger.error(f"Failed to save NPZ file: {e}")
                raise

    def _write_parquet(self, data: dict[str, Any]):
        """Write to a Parquet file."""
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pandas and pyarrow are required for Parquet file support"
            )

        with self.lock:
            parquet_path = self.base_filepath.with_suffix(self.extension)

            # Convert data to DataFrame
            df_row = pd.DataFrame([data])

            if os.path.exists(parquet_path):
                # Read schema from existing file
                existing_schema = pq.read_schema(parquet_path)

                # Convert DataFrame to table with matching schema
                table = pa.Table.from_pandas(df_row, schema=existing_schema)

                # Append to existing file
                with pq.ParquetWriter(
                    parquet_path, existing_schema, append=True
                ) as writer:
                    writer.write_table(table)
            else:
                # Create new file
                table = pa.Table.from_pandas(df_row)
                pq.write_table(table, parquet_path)

    def _write_hdf5(self, data: dict[str, Any]):
        """Write to an HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 file support")

        with self.lock:
            hdf5_path = self.base_filepath.with_suffix(self.extension)

            # Open the HDF5 file
            with h5py.File(hdf5_path, "a") as f:
                for key, value in data.items():
                    # Convert value to numpy array if it's not already
                    value_array = np.array(value)

                    if key in f:
                        # Resize and append to existing dataset
                        dataset = f[key]
                        current_size = dataset.shape[0]
                        new_shape = (current_size + 1,) + dataset.shape[1:]
                        dataset.resize(new_shape)
                        dataset[current_size] = value_array
                    else:
                        # Create new dataset with unlimited first dimension
                        maxshape = (None,) + value_array.shape
                        f.create_dataset(
                            key,
                            data=[value_array],
                            maxshape=maxshape,
                            compression="gzip",
                        )


class ShardedWriter:
    """
    Sharded file writer for parallel experience collection.

    This class manages multiple file shards for parallel writing from
    different actors or threads, avoiding contention.
    """

    def __init__(
        self,
        base_directory: str | Path,
        base_filename: str,
        num_shards: int = 4,
        format_type: str = "jsonl",
        max_shard_size: int = 1000000,
        lock_timeout: float = 10.0,
    ):
        """
        Initialize the sharded writer.

        Args:
            base_directory: Directory to store the shards
            base_filename: Base filename for the shards
            num_shards: Number of shards to create
            format_type: Format type ('jsonl', 'npz', 'parquet', 'hdf5')
            max_shard_size: Maximum number of records per shard
            lock_timeout: Timeout for acquiring locks in seconds
        """
        self.base_directory = Path(base_directory)
        self.base_filename = base_filename
        self.num_shards = num_shards
        self.format_type = format_type
        self.max_shard_size = max_shard_size
        self.lock_timeout = lock_timeout

        # Create directory if needed
        os.makedirs(self.base_directory, exist_ok=True)

        # Initialize writers for each shard
        self.writers = {}

        # Create or load shard metadata
        self.metadata_path = self.base_directory / f"{base_filename}_shards.json"
        self._load_or_create_metadata()

        # Lock for updating metadata
        self.metadata_lock = threading.RLock()

        # Register cleanup
        atexit.register(self.close)

    def _load_or_create_metadata(self):
        """Load or create metadata for the shards."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading shard metadata: {e}")
                self.metadata = self._create_metadata()
        else:
            self.metadata = self._create_metadata()

        # Save metadata
        self._save_metadata()

    def _create_metadata(self) -> dict[str, Any]:
        """Create new metadata for the shards."""
        return {
            "base_filename": self.base_filename,
            "num_shards": self.num_shards,
            "format_type": self.format_type,
            "created_at": time.time(),
            "updated_at": time.time(),
            "shards": {
                str(i): {
                    "path": f"{self.base_filename}_shard_{i}",
                    "records": 0,
                    "last_updated": time.time(),
                    "active": True,
                }
                for i in range(self.num_shards)
            },
        }

    def _save_metadata(self):
        """Save shard metadata to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # Update timestamp
        self.metadata["updated_at"] = time.time()

        # Save to file
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def get_shard_writer(self, shard_id: int) -> ConcurrentWriter:
        """
        Get a writer for a specific shard.

        Args:
            shard_id: ID of the shard

        Returns:
            ConcurrentWriter for the shard
        """
        if shard_id < 0 or shard_id >= self.num_shards:
            raise ValueError(
                f"Invalid shard ID: {shard_id}, must be 0-{self.num_shards-1}"
            )

        # Create the writer if it doesn't exist
        if shard_id not in self.writers:
            shard_path = (
                self.base_directory / self.metadata["shards"][str(shard_id)]["path"]
            )
            self.writers[shard_id] = ConcurrentWriter(
                shard_path, self.format_type, self.lock_timeout
            )

        return self.writers[shard_id]

    def write(self, data: dict[str, Any], shard_id: int | None = None) -> int:
        """
        Write data to a shard.

        Args:
            data: Data to write
            shard_id: Specific shard ID to write to (if None, selects least busy shard)

        Returns:
            ID of the shard that was written to
        """
        # Select a shard if not specified
        if shard_id is None:
            shard_id = self._select_shard()

        # Get the writer for the shard
        writer = self.get_shard_writer(shard_id)

        # Write the data
        writer.write(data)

        # Update metadata
        with self.metadata_lock:
            self.metadata["shards"][str(shard_id)]["records"] += 1
            self.metadata["shards"][str(shard_id)]["last_updated"] = time.time()

            # Check if we need to rotate the shard
            if self.metadata["shards"][str(shard_id)]["records"] >= self.max_shard_size:
                self._rotate_shard(shard_id)

            # Periodically save metadata
            if self.metadata["shards"][str(shard_id)]["records"] % 100 == 0:
                self._save_metadata()

        return shard_id

    def _select_shard(self) -> int:
        """
        Select the least busy shard.

        Returns:
            ID of the selected shard
        """
        with self.metadata_lock:
            # Find active shards
            active_shards = [
                int(shard_id)
                for shard_id, info in self.metadata["shards"].items()
                if info["active"]
            ]

            if not active_shards:
                # Rotate all shards if none are active
                for shard_id in range(self.num_shards):
                    self._rotate_shard(shard_id)
                active_shards = list(range(self.num_shards))

            # Select the shard with the fewest records
            return min(
                active_shards, key=lambda s: self.metadata["shards"][str(s)]["records"]
            )

    def _rotate_shard(self, shard_id: int):
        """
        Rotate a shard (close it and create a new one).

        Args:
            shard_id: ID of the shard to rotate
        """
        with self.metadata_lock:
            # Close existing writer
            if shard_id in self.writers:
                self.writers[shard_id].close()
                del self.writers[shard_id]

            # Archive the current shard
            old_path = self.metadata["shards"][str(shard_id)]["path"]
            archive_path = f"{old_path}_{int(time.time())}"

            self.metadata["shards"][str(shard_id)]["path"] = archive_path
            self.metadata["shards"][str(shard_id)]["active"] = False

            # Create a new shard
            self.metadata["shards"][str(shard_id)] = {
                "path": f"{self.base_filename}_shard_{shard_id}",
                "records": 0,
                "last_updated": time.time(),
                "active": True,
            }

            # Save metadata
            self._save_metadata()

    def close(self):
        """Close all writers."""
        # Close all writers
        for writer in self.writers.values():
            writer.close()
        self.writers = {}

        # Save metadata
        try:
            self._save_metadata()
        except FileNotFoundError:
            # Directory already cleaned up, nothing to do
            pass

    def merge_shards(self, output_path: str | Path | None = None) -> Path:
        """
        Merge all shards into a single file.

        Args:
            output_path: Path for the merged output (if None, uses base_filename)

        Returns:
            Path to the merged file
        """
        # Close all writers
        self.close()

        # Determine output path
        if output_path is None:
            output_path = self.base_directory / f"{self.base_filename}_merged"
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        os.makedirs(output_path.parent, exist_ok=True)

        # Merge based on format type
        if self.format_type == "jsonl":
            return self._merge_jsonl_shards(output_path)
        elif self.format_type == "npz":
            return self._merge_npz_shards(output_path)
        elif self.format_type == "parquet":
            return self._merge_parquet_shards(output_path)
        elif self.format_type == "hdf5":
            return self._merge_hdf5_shards(output_path)
        else:
            raise ValueError(f"Unsupported format type: {self.format_type}")

    def _merge_jsonl_shards(self, output_path: Path) -> Path:
        """
        Merge JSON Lines shards.

        Args:
            output_path: Path for the merged output

        Returns:
            Path to the merged file
        """
        output_file = output_path.with_suffix(".jsonl")

        with open(output_file, "w") as outfile:
            # Process each shard
            for shard_id, shard_info in self.metadata["shards"].items():
                shard_path = self.base_directory / shard_info["path"]
                full_path = shard_path.with_suffix(".jsonl")

                if os.path.exists(full_path):
                    with open(full_path) as infile:
                        for line in infile:
                            outfile.write(line)

        return output_file

    def _merge_npz_shards(self, output_path: Path) -> Path:
        """
        Merge NPZ shards.

        Args:
            output_path: Path for the merged output

        Returns:
            Path to the merged file
        """
        output_file = output_path.with_suffix(".npz")

        # Collect all data from shards
        combined_data = {}

        for shard_id, shard_info in self.metadata["shards"].items():
            shard_path = self.base_directory / shard_info["path"]
            full_path = shard_path.with_suffix(".npz")

            if os.path.exists(full_path):
                with np.load(full_path, allow_pickle=True) as data:
                    for key in data.files:
                        if key in combined_data:
                            combined_data[key] = np.concatenate(
                                [combined_data[key], data[key]], axis=0
                            )
                        else:
                            combined_data[key] = data[key]

        # Save combined data
        np.savez(output_file, **combined_data)

        return output_file

    def _merge_parquet_shards(self, output_path: Path) -> Path:
        """
        Merge Parquet shards.

        Args:
            output_path: Path for the merged output

        Returns:
            Path to the merged file
        """
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pandas and pyarrow are required for Parquet file support"
            )

        output_file = output_path.with_suffix(".parquet")

        # Collect all tables from shards
        tables = []

        for shard_id, shard_info in self.metadata["shards"].items():
            shard_path = self.base_directory / shard_info["path"]
            full_path = shard_path.with_suffix(".parquet")

            if os.path.exists(full_path):
                tables.append(pq.read_table(full_path))

        if tables:
            # Concatenate tables and write to output
            combined_table = pa.concat_tables(tables)
            pq.write_table(combined_table, output_file)
        else:
            # Create empty table if no shards exist
            empty_table = pa.Table.from_pandas(pd.DataFrame())
            pq.write_table(empty_table, output_file)

        return output_file

    def _merge_hdf5_shards(self, output_path: Path) -> Path:
        """
        Merge HDF5 shards.

        Args:
            output_path: Path for the merged output

        Returns:
            Path to the merged file
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 file support")

        output_file = output_path.with_suffix(".h5")

        # Create output file
        with h5py.File(output_file, "w") as out_f:
            # Process each shard
            for shard_id, shard_info in self.metadata["shards"].items():
                shard_path = self.base_directory / shard_info["path"]
                full_path = shard_path.with_suffix(".h5")

                if os.path.exists(full_path):
                    with h5py.File(full_path, "r") as in_f:
                        for key in in_f.keys():
                            if key in out_f:
                                # Dataset already exists, append data
                                in_data = in_f[key][:]
                                out_data = out_f[key]

                                current_size = out_data.shape[0]
                                new_shape = (
                                    current_size + in_data.shape[0],
                                ) + out_data.shape[1:]
                                out_data.resize(new_shape)
                                out_data[current_size:] = in_data
                            else:
                                # Create new dataset with unlimited first dimension
                                in_data = in_f[key][:]
                                maxshape = (None,) + in_data.shape[1:]
                                out_f.create_dataset(
                                    key,
                                    data=in_data,
                                    maxshape=maxshape,
                                    compression="gzip",
                                )

        return output_file


class ThreadSafeReader:
    """
    Thread-safe reader for experience data.

    This class provides mechanisms to safely read from experience files
    that might be concurrently modified by writers.
    """

    def __init__(
        self,
        filepath: str | Path,
        lock_timeout: float = 5.0,
        cache_size: int = 1000,
    ):
        """
        Initialize the thread-safe reader.

        Args:
            filepath: Path to the experience file
            lock_timeout: Timeout for acquiring locks in seconds
            cache_size: Size of the read cache
        """
        self.filepath = Path(filepath)
        self.lock_timeout = lock_timeout
        self.cache_size = cache_size

        # Lock file path
        self.lock_file = self.filepath.with_suffix(".lock")

        # Determine file format
        self.format_type = self._detect_format_type()

        # Last read position for streaming
        self.last_position = 0

        # Read cache
        self.cache = {}

        # Lock for thread safety
        self.read_lock = threading.RLock()

    def _detect_format_type(self) -> str:
        """
        Detect the format type from the file extension.

        Returns:
            Format type string
        """
        ext = self.filepath.suffix.lower()

        if ext == ".jsonl":
            return "jsonl"
        elif ext == ".npz":
            return "npz"
        elif ext == ".parquet":
            return "parquet"
        elif ext in [".h5", ".hdf5"]:
            return "hdf5"
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def read(self, index: int) -> dict[str, Any]:
        """
        Read a specific record by index.

        Args:
            index: Index of the record to read

        Returns:
            The record data
        """
        # Check cache first
        if index in self.cache:
            return self.cache[index]

        with self.read_lock:
            # Different read strategy based on format
            if self.format_type == "jsonl":
                result = self._read_jsonl(index)
            elif self.format_type == "npz":
                result = self._read_npz(index)
            elif self.format_type == "parquet":
                result = self._read_parquet(index)
            elif self.format_type == "hdf5":
                result = self._read_hdf5(index)

            # Update cache
            self._update_cache(index, result)

            return result

    def _read_jsonl(self, index: int) -> dict[str, Any]:
        """Read from a JSON Lines file."""
        # For JSONL, we need to read line by line
        # This is inefficient for random access, but we use a cache to mitigate
        with FileLock(str(self.lock_file), timeout=self.lock_timeout):
            with open(self.filepath) as f:
                for i, line in enumerate(f):
                    if i == index:
                        return json.loads(line)
            raise IndexError(f"Index {index} out of bounds")

    def _read_npz(self, index: int) -> dict[str, Any]:
        """Read from an NPZ file."""
        with FileLock(str(self.lock_file), timeout=self.lock_timeout):
            with np.load(self.filepath, allow_pickle=True) as data:
                # Check if index is in bounds for at least one array
                in_bounds = False
                for key in data.files:
                    if len(data[key]) > index:
                        in_bounds = True
                        break

                if not in_bounds:
                    raise IndexError(f"Index {index} out of bounds")

                # Construct a record from all arrays at this index
                result = {}
                for key in data.files:
                    if len(data[key]) > index:
                        result[key] = (
                            data[key][index].item()
                            if data[key][index].ndim == 0
                            else data[key][index]
                        )

                return result

    def _read_parquet(self, index: int) -> dict[str, Any]:
        """Read from a Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet file support")

        with FileLock(str(self.lock_file), timeout=self.lock_timeout):
            # Read the record at the given index
            table = pq.read_table(self.filepath, row_groups=[index // 1000])
            row_in_group = index % 1000

            try:
                # Convert to pandas for easier row access
                df_row = table.to_pandas().iloc[row_in_group]
                return df_row.to_dict()
            except IndexError:
                raise IndexError(f"Index {index} out of bounds")

    def _read_hdf5(self, index: int) -> dict[str, Any]:
        """Read from an HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 file support")

        with FileLock(str(self.lock_file), timeout=self.lock_timeout):
            with h5py.File(self.filepath, "r") as f:
                # Check if index is in bounds for at least one dataset
                in_bounds = False
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and len(f[key].shape) > 0:
                        if f[key].shape[0] > index:
                            in_bounds = True
                            break

                if not in_bounds:
                    raise IndexError(f"Index {index} out of bounds")

                # Construct a record from all datasets at this index
                result = {}
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and len(f[key].shape) > 0:
                        if f[key].shape[0] > index:
                            result[key] = (
                                f[key][index].item()
                                if f[key][index].ndim == 0
                                else f[key][index]
                            )

                return result

    def _update_cache(self, index: int, data: dict[str, Any]):
        """Update the read cache."""
        # If cache is full, remove oldest entries
        if len(self.cache) >= self.cache_size:
            # Simple approach: clear half of the cache
            keys = sorted(self.cache.keys())
            for key in keys[: len(keys) // 2]:
                del self.cache[key]

        self.cache[index] = data

    def stream_next(self, batch_size: int = 1) -> list[dict[str, Any]]:
        """
        Stream the next batch of records sequentially.

        Args:
            batch_size: Number of records to read

        Returns:
            List of records
        """
        results = []

        with self.read_lock:
            try:
                for i in range(batch_size):
                    results.append(self.read(self.last_position + i))

                # Update position for next read
                self.last_position += batch_size
            except IndexError:
                # End of file reached
                pass

        return results

    def count_records(self) -> int:
        """
        Count the total number of records.

        Returns:
            Number of records
        """
        with FileLock(str(self.lock_file), timeout=self.lock_timeout):
            if self.format_type == "jsonl":
                with open(self.filepath) as f:
                    return sum(1 for _ in f)
            elif self.format_type == "npz":
                with np.load(self.filepath, allow_pickle=True) as data:
                    # Find the longest array
                    max_len = 0
                    for key in data.files:
                        max_len = max(max_len, len(data[key]))
                    return max_len
            elif self.format_type == "parquet":
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(self.filepath)
                return parquet_file.metadata.num_rows
            elif self.format_type == "hdf5":
                import h5py

                with h5py.File(self.filepath, "r") as f:
                    # Find the longest dataset
                    max_len = 0
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset) and len(f[key].shape) > 0:
                            max_len = max(max_len, f[key].shape[0])
                    return max_len
            else:
                raise ValueError(f"Unsupported format: {self.format_type}")

    def read_all(self) -> list[dict[str, Any]]:
        """
        Read all records.

        Returns:
            List of all records
        """
        count = self.count_records()
        return [self.read(i) for i in range(count)]

    def read_batch(self, indices: list[int]) -> list[dict[str, Any]]:
        """
        Read multiple records by their indices.

        Args:
            indices: List of indices to read

        Returns:
            List of records in the same order as the indices
        """
        return [self.read(i) for i in indices]
