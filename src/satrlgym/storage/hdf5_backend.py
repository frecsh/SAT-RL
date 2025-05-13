"""
Advanced storage backend implementation using HDF5 for efficient experience storage.
Provides chunked datasets, hierarchical organization, and memory-mapped file access.
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any

import h5py
import numpy as np

# Import our type system
from ..type_system import DType, ExperienceSchema, TypeConverter, TypeSpecification

# Import the StorageBackend abstract base class
from .arrow_backend import StorageBackend

# Set up logging
logger = logging.getLogger(__name__)


class HDF5StorageBackend(StorageBackend):
    """
    HDF5 storage backend for efficient experience storage.
    Provides chunked datasets, hierarchical organization, and memory-mapped access.
    """

    def __init__(
        self,
        schema: ExperienceSchema,
        path: str,
        compression: str = "gzip",
        compression_level: int = 4,
        chunk_size: int = 1000,
        use_swmr: bool = False,
        metadata: dict[str, str] | None = None,
    ):
        """
        Initialize the HDF5 storage backend.

        Args:
            schema: The experience schema defining data types
            path: The file system path for storage
            compression: Compression algorithm ('gzip', 'lzf', 'none')
            compression_level: Compression level (1-9, higher = better compression but slower)
            chunk_size: Number of experiences per chunk
            use_swmr: Use Single Writer Multiple Reader mode for concurrent access
            metadata: Optional metadata to store with the dataset
        """
        super().__init__(schema, path)
        self.compression = compression
        self.compression_level = compression_level
        self.chunk_size = chunk_size
        self.use_swmr = use_swmr
        self.metadata = metadata or {}

        # Add default metadata
        self.metadata.update(
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "schema_version": schema.version,
                "uuid": str(uuid.uuid4()),
            }
        )

        # Initialize state
        self.total_experiences = 0
        self.file = None
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize the HDF5 file and create necessary groups."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # Check if file exists
        file_exists = os.path.exists(self.path)

        # Open file with appropriate mode
        mode = "a" if file_exists else "w"
        self.file = h5py.File(self.path, mode, libver="latest", swmr=self.use_swmr)

        # If new file, initialize structure
        if not file_exists:
            # Create metadata group
            meta_group = self.file.create_group("metadata")

            # Store metadata as attributes
            for key, value in self.metadata.items():
                meta_group.attrs[key] = value

            # Store schema as JSON
            meta_group.attrs["schema"] = json.dumps(self.schema.to_dict())

            # Create experiences group
            self.file.create_group("experiences")

            # Create index dataset (maps global index to chunk/local index)
            index_group = self.file.create_group("indices")
            index_group.create_dataset(
                "global_to_local",
                shape=(0, 2),  # [chunk_idx, local_idx]
                maxshape=(None, 2),
                dtype="int32",
                chunks=(min(1000, self.chunk_size), 2),
            )
        else:
            # Load existing metadata
            meta_group = self.file["metadata"]

            # Load existing schema
            schema_json = meta_group.attrs.get("schema")
            if schema_json:
                loaded_schema = ExperienceSchema.from_dict(json.loads(schema_json))
                # Validate schema compatibility
                if loaded_schema.version != self.schema.version:
                    logger.warning(
                        f"Schema version mismatch: file has {loaded_schema.version}, "
                        f"current is {self.schema.version}"
                    )

            # Get total experiences count
            if "indices/global_to_local" in self.file:
                self.total_experiences = len(self.file["indices/global_to_local"])

    def _get_field_datasets(self, chunk_idx: int) -> dict[str, h5py.Dataset]:
        """
        Get all field datasets for a specific chunk.

        Args:
            chunk_idx: Chunk index

        Returns:
            Dict mapping field names to datasets
        """
        exp_group = self.file["experiences"]
        chunk_group_name = f"chunk_{chunk_idx:06d}"

        # Create chunk group if it doesn't exist
        if chunk_group_name not in exp_group:
            chunk_group = exp_group.create_group(chunk_group_name)
        else:
            chunk_group = exp_group[chunk_group_name]

        datasets = {}

        # Create or get datasets for each field
        for field_name, field_spec in self.schema.field_specs.items():
            if field_name not in chunk_group:
                # Create dataset with appropriate type and shape
                dataset = self._create_field_dataset(
                    chunk_group, field_name, field_spec
                )
            else:
                dataset = chunk_group[field_name]

            datasets[field_name] = dataset

        return datasets

    def _create_field_dataset(
        self, group: h5py.Group, field_name: str, field_spec: TypeSpecification
    ) -> h5py.Dataset:
        """
        Create a dataset for a field in the HDF5 file.

        Args:
            group: HDF5 group to create the dataset in
            field_name: Field name
            field_spec: Field type specification

        Returns:
            h5py.Dataset: The created dataset
        """
        # Handle nested structures (create group)
        if field_spec.nested_spec:
            nested_group = group.create_group(field_name)

            # Create datasets for each nested field
            for nested_name, nested_spec in field_spec.nested_spec.items():
                self._create_field_dataset(nested_group, nested_name, nested_spec)

            return nested_group

        # Handle tuple types (create group with numbered items)
        elif field_spec.is_tuple:
            tuple_group = group.create_group(field_name)

            # Create datasets for each tuple item
            for i, item_spec in enumerate(field_spec.tuple_specs):
                self._create_field_dataset(tuple_group, f"item_{i}", item_spec)

            return tuple_group

        # Handle array types
        elif field_spec.shape is not None:
            # Determine dataset shape and dtype
            dtype = self._dtype_to_hdf5_dtype(field_spec.dtype)

            # Replace -1 in shape with None for extendable dimensions
            shape = list(field_spec.shape)

            for i, dim in enumerate(shape):
                if dim == -1:
                    shape[i] = 0  # Start with size 0, will be extended

            # Add leading dimension for multiple experiences
            full_shape = tuple([0] + shape)

            # Calculate maxshape
            maxshape = tuple([None] + [None if s == 0 else s for s in shape])

            # Calculate chunks
            # For variable-sized dimensions, use reasonable defaults
            chunks = list(full_shape)
            chunks[0] = min(
                self.chunk_size, 100
            )  # Chunk size for experiences dimension

            for i, dim in enumerate(chunks[1:], 1):
                if dim == 0:  # Variable dimension
                    chunks[i] = 10  # Default chunk size for variable dimensions

            # Create compression args
            compression_args = {}
            if self.compression != "none":
                compression_args["compression"] = self.compression
                if self.compression == "gzip":
                    compression_args["compression_opts"] = self.compression_level

            # Create dataset
            return group.create_dataset(
                field_name,
                shape=full_shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=tuple(chunks),
                **compression_args,
            )

        # Handle scalar types
        else:
            dtype = self._dtype_to_hdf5_dtype(field_spec.dtype)

            # Create dataset with variable length strings if needed
            if field_spec.dtype == DType.STRING:
                dtype = h5py.special_dtype(vlen=str)

            # Create compression args
            compression_args = {}
            if self.compression != "none":
                compression_args["compression"] = self.compression
                if self.compression == "gzip":
                    compression_args["compression_opts"] = self.compression_level

            # Create dataset for scalars (leading dimension for multiple experiences)
            return group.create_dataset(
                field_name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(min(self.chunk_size, 1000),),
                **compression_args,
            )

    def _dtype_to_hdf5_dtype(self, dtype: DType) -> np.dtype:
        """
        Convert our DType enum to HDF5 compatible numpy dtype.

        Args:
            dtype: The data type from DType enum

        Returns:
            np.dtype: The numpy dtype
        """
        type_map = {
            DType.FLOAT32: np.float32,
            DType.FLOAT64: np.float64,
            DType.INT8: np.int8,
            DType.INT16: np.int16,
            DType.INT32: np.int32,
            DType.INT64: np.int64,
            DType.UINT8: np.uint8,
            DType.UINT16: np.uint16,
            DType.UINT32: np.uint32,
            DType.UINT64: np.uint64,
            DType.BOOL: np.bool_,
            DType.STRING: np.dtype(
                "O"
            ),  # Will be converted to h5py.special_dtype later
        }

        if dtype not in type_map:
            logger.warning(f"Unknown dtype: {dtype}, defaulting to float32")
            return np.float32

        return type_map[dtype]

    def write_batch(self, experiences: list[dict[str, Any]]) -> bool:
        """
        Write a batch of experiences to storage.

        Args:
            experiences: List of experience dictionaries

        Returns:
            bool: True if write was successful
        """
        if not experiences:
            return True

        if self.file is None:
            logger.error("Storage not initialized")
            return False

        try:
            # Validate experiences against schema
            for exp in experiences:
                valid, error = self.schema.validate_sample(exp)
                if not valid:
                    logger.error(f"Invalid experience: {error}")
                    return False

            # Determine current chunk index and local start index
            chunk_idx = self.total_experiences // self.chunk_size
            local_start_idx = self.total_experiences % self.chunk_size
            experiences_in_chunk = min(
                len(experiences), self.chunk_size - local_start_idx
            )

            # Write experiences to current chunk
            if experiences_in_chunk > 0:
                self._write_to_chunk(
                    chunk_idx, local_start_idx, experiences[:experiences_in_chunk]
                )

            # Write remaining experiences to new chunks if needed
            remaining_experiences = experiences[experiences_in_chunk:]
            next_chunk_idx = chunk_idx + 1
            while remaining_experiences:
                # Take up to chunk_size experiences
                chunk_experiences = remaining_experiences[: self.chunk_size]
                remaining_experiences = remaining_experiences[self.chunk_size :]

                # Write to new chunk
                self._write_to_chunk(next_chunk_idx, 0, chunk_experiences)
                next_chunk_idx += 1

            # Update total experiences count
            self.total_experiences += len(experiences)

            # Update the global index
            self._update_global_index(experiences)

            # Flush file to disk
            self.file.flush()

            return True

        except Exception as e:
            logger.error(f"Error writing to HDF5 file: {e}")
            return False

    def _write_to_chunk(
        self, chunk_idx: int, local_start_idx: int, experiences: list[dict[str, Any]]
    ) -> None:
        """
        Write experiences to a specific chunk.

        Args:
            chunk_idx: Chunk index
            local_start_idx: Start index within the chunk
            experiences: List of experiences to write
        """
        # Get datasets for this chunk
        datasets = self._get_field_datasets(chunk_idx)

        # Write each experience field to corresponding dataset
        for field_name, field_spec in self.schema.field_specs.items():
            # Extract field values from all experiences
            field_values = [exp.get(field_name) for exp in experiences]

            # Write values to dataset
            self._write_field_values(
                datasets[field_name], field_values, field_spec, local_start_idx
            )

    def _write_field_values(
        self,
        dataset_or_group,
        values: list[Any],
        spec: TypeSpecification,
        start_idx: int,
    ) -> None:
        """
        Write field values to a dataset or group.

        Args:
            dataset_or_group: HDF5 dataset or group
            values: List of values to write
            spec: Type specification
            start_idx: Start index
        """
        # Handle nested structures
        if spec.nested_spec:
            for nested_name, nested_spec in spec.nested_spec.items():
                # Extract values for this nested field
                nested_values = [
                    v[nested_name] if v is not None else None for v in values
                ]

                # Write to nested dataset or group
                self._write_field_values(
                    dataset_or_group[nested_name], nested_values, nested_spec, start_idx
                )
            return

        # Handle tuple types
        elif spec.is_tuple:
            for i, item_spec in enumerate(spec.tuple_specs):
                # Extract values for this tuple item
                item_values = [v[i] if v is not None else None for v in values]

                # Write to tuple item dataset or group
                self._write_field_values(
                    dataset_or_group[f"item_{i}"], item_values, item_spec, start_idx
                )
            return

        # Handle array or scalar types (actual datasets)
        # Convert values to numpy arrays
        np_values = []
        for val in values:
            if val is None:
                # Use a default value appropriate for the dtype
                if spec.dtype in [DType.FLOAT32, DType.FLOAT64]:
                    np_val = np.nan
                elif spec.dtype == DType.STRING:
                    np_val = ""
                else:
                    np_val = 0
            else:
                # Convert to numpy
                np_val = TypeConverter.to_numpy(val, spec)
            np_values.append(np_val)

        # Convert list of arrays to single array
        if spec.shape is not None:
            # For array types, stack along a new first dimension
            np_array = np.stack(np_values)
        else:
            # For scalar types, create a 1D array
            np_array = np.array(np_values)

        # Resize dataset if needed
        end_idx = start_idx + len(values)
        current_size = dataset_or_group.shape[0]
        if end_idx > current_size:
            new_shape = list(dataset_or_group.shape)
            new_shape[0] = end_idx
            dataset_or_group.resize(tuple(new_shape))

        # Write to dataset
        dataset_or_group[start_idx:end_idx] = np_array

    def _update_global_index(self, experiences: list[dict[str, Any]]) -> None:
        """
        Update the global index mapping for new experiences.

        Args:
            experiences: List of new experiences
        """
        # Get index dataset
        index_dataset = self.file["indices/global_to_local"]

        # Calculate new indices
        new_indices = []
        for i in range(len(experiences)):
            global_idx = self.total_experiences - len(experiences) + i
            chunk_idx = global_idx // self.chunk_size
            local_idx = global_idx % self.chunk_size
            new_indices.append([chunk_idx, local_idx])

        # Create numpy array
        new_indices_array = np.array(new_indices, dtype="int32")

        # Resize index dataset
        old_size = index_dataset.shape[0]
        new_size = old_size + len(new_indices)
        index_dataset.resize((new_size, 2))

        # Write new indices
        index_dataset[old_size:new_size] = new_indices_array

    def read_batch(
        self, batch_size: int = 32, indices: list[int] | None = None
    ) -> list[dict[str, Any]]:
        """
        Read a batch of experiences from storage.

        Args:
            batch_size: Number of experiences to read
            indices: Optional list of specific indices to read

        Returns:
            List of experience dictionaries
        """
        if self.file is None:
            logger.error("Storage not initialized")
            return []

        if self.total_experiences == 0:
            logger.warning("No experiences in storage")
            return []

        try:
            # If indices are provided, use them
            if indices is not None:
                # Lookup indices in the global index
                chunk_local_indices = self._get_chunk_local_indices(indices)

                # Group by chunk
                chunk_groups = defaultdict(list)
                for global_idx, (chunk_idx, local_idx) in zip(
                    indices, chunk_local_indices
                ):
                    chunk_groups[chunk_idx].append((global_idx, local_idx))

                # Read from each chunk
                result = []
                for chunk_idx, idx_pairs in chunk_groups.items():
                    _, local_indices = zip(*idx_pairs)
                    experiences = self._read_from_chunk(chunk_idx, list(local_indices))

                    # Remap to global indices
                    for (global_idx, local_idx), exp in zip(idx_pairs, experiences):
                        result.append(exp)

                return result

            # Otherwise, randomly select indices
            else:
                if batch_size > self.total_experiences:
                    batch_size = self.total_experiences

                # Generate random indices
                random_indices = np.random.choice(
                    self.total_experiences, size=batch_size, replace=False
                )

                return self.read_batch(indices=random_indices.tolist())

        except Exception as e:
            logger.error(f"Error reading from HDF5 file: {e}")
            return []

    def _get_chunk_local_indices(
        self, global_indices: list[int]
    ) -> list[tuple[int, int]]:
        """
        Convert global indices to (chunk_idx, local_idx) pairs.

        Args:
            global_indices: List of global indices

        Returns:
            List of (chunk_idx, local_idx) pairs
        """
        # Get index dataset
        index_dataset = self.file["indices/global_to_local"]

        # Read indices
        result = []
        for global_idx in global_indices:
            if 0 <= global_idx < self.total_experiences:
                chunk_idx, local_idx = index_dataset[global_idx]
                result.append((int(chunk_idx), int(local_idx)))
            else:
                logger.warning(
                    f"Index {global_idx} out of range [0, {self.total_experiences})"
                )
                # Use default values
                result.append((0, 0))

        return result

    def _read_from_chunk(
        self, chunk_idx: int, local_indices: list[int]
    ) -> list[dict[str, Any]]:
        """
        Read experiences from a specific chunk.

        Args:
            chunk_idx: Chunk index
            local_indices: List of local indices

        Returns:
            List of experiences
        """
        if self.file is None:
            return []

        # Check if chunk exists
        exp_group = self.file["experiences"]
        chunk_group_name = f"chunk_{chunk_idx:06d}"

        if chunk_group_name not in exp_group:
            logger.warning(f"Chunk {chunk_idx} does not exist")
            return []

        chunk_group = exp_group[chunk_group_name]

        # Read each field for the requested indices
        result = [{} for _ in range(len(local_indices))]

        # Read each field
        for field_name, field_spec in self.schema.field_specs.items():
            if field_name not in chunk_group:
                logger.warning(f"Field {field_name} not found in chunk {chunk_idx}")
                continue

            # Read field values
            field_values = self._read_field_values(
                chunk_group[field_name], field_spec, local_indices
            )

            # Add to result
            for i, val in enumerate(field_values):
                result[i][field_name] = val

        return result

    def _read_field_values(
        self, dataset_or_group, spec: TypeSpecification, indices: list[int]
    ) -> list[Any]:
        """
        Read field values from a dataset or group.

        Args:
            dataset_or_group: HDF5 dataset or group
            spec: Type specification
            indices: List of indices to read

        Returns:
            List of values
        """
        # Handle nested structures
        if spec.nested_spec:
            # Initialize result
            result = [{} for _ in range(len(indices))]

            # Read each nested field
            for nested_name, nested_spec in spec.nested_spec.items():
                nested_values = self._read_field_values(
                    dataset_or_group[nested_name], nested_spec, indices
                )

                # Add to result
                for i, val in enumerate(nested_values):
                    result[i][nested_name] = val

            return result

        # Handle tuple types
        elif spec.is_tuple:
            # Initialize result
            result = [tuple() for _ in range(len(indices))]

            # Read each tuple item
            for i, item_spec in enumerate(spec.tuple_specs):
                item_values = self._read_field_values(
                    dataset_or_group[f"item_{i}"], item_spec, indices
                )

                # Add to result
                for j, val in enumerate(item_values):
                    result[j] = result[j] + (val,)

            return result

        # Handle array or scalar types (actual datasets)
        # Read from dataset
        values = [dataset_or_group[idx] for idx in indices]

        return values

    def get_size(self) -> int:
        """
        Get the number of experiences in storage.

        Returns:
            int: Number of experiences
        """
        return self.total_experiences

    def close(self) -> None:
        """Close the storage backend and release resources."""
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None


# Additional helper functions


def create_hdf5_storage(
    schema: ExperienceSchema, path: str, **kwargs
) -> HDF5StorageBackend:
    """
    Create an HDF5 storage backend.

    Args:
        schema: Experience schema
        path: Storage path
        **kwargs: Additional arguments for HDF5StorageBackend

    Returns:
        HDF5StorageBackend: The storage backend
    """
    return HDF5StorageBackend(schema, path, **kwargs)


def load_hdf5_storage(path: str) -> HDF5StorageBackend:
    """
    Load an existing HDF5 storage backend.

    Args:
        path: Storage path

    Returns:
        HDF5StorageBackend: The storage backend
    """
    # Open file to read metadata
    with h5py.File(path, "r") as f:
        # Load schema
        schema_json = f["metadata"].attrs.get("schema")
        if not schema_json:
            raise ValueError(f"No schema found in HDF5 file: {path}")

        # Create schema
        schema = ExperienceSchema.from_dict(json.loads(schema_json))

    # Create storage backend
    return HDF5StorageBackend(schema, path)
