"""
Advanced storage backend implementation using Apache Arrow/Parquet for efficient experience storage.
Provides zero-copy reading, column-based compression, and predicate pushdown capabilities.
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Import our type system
from ..type_system import DType, ExperienceSchema, TypeConverter, TypeSpecification

# Set up logging
logger = logging.getLogger(__name__)


class StorageBackend:
    """Abstract base class for experience storage backends."""

    def __init__(self, schema: ExperienceSchema, path: str):
        """
        Initialize the storage backend.

        Args:
            schema: The experience schema defining data types
            path: The file system path for storage
        """
        self.schema = schema
        self.path = path

    def write_batch(self, experiences: list[dict[str, Any]]) -> bool:
        """
        Write a batch of experiences to storage.

        Args:
            experiences: List of experience dictionaries

        Returns:
            bool: True if write was successful
        """
        raise NotImplementedError("Subclasses must implement write_batch")

    def read_batch(
        self, batch_size: int, indices: list[int] | None = None
    ) -> list[dict[str, Any]]:
        """
        Read a batch of experiences from storage.

        Args:
            batch_size: Number of experiences to read
            indices: Optional list of specific indices to read

        Returns:
            List of experience dictionaries
        """
        raise NotImplementedError("Subclasses must implement read_batch")

    def get_size(self) -> int:
        """
        Get the number of experiences in storage.

        Returns:
            int: Number of experiences
        """
        raise NotImplementedError("Subclasses must implement get_size")

    def close(self) -> None:
        """Close the storage backend and release resources."""


class ArrowStorageBackend(StorageBackend):
    """
    Apache Arrow/Parquet storage backend for efficient experience storage.
    Provides zero-copy reading, column-based compression, and predicate pushdown.
    """

    def __init__(
        self,
        schema: ExperienceSchema,
        path: str,
        compression: str = "zstd",
        use_memory_map: bool = True,
        max_file_size_mb: int = 1024,
        metadata: dict[str, str] | None = None,
    ):
        """
        Initialize the Arrow storage backend.

        Args:
            schema: The experience schema defining data types
            path: The file system path for storage
            compression: Compression algorithm ('zstd', 'gzip', 'lz4', 'snappy', 'none')
            use_memory_map: Whether to use memory-mapped files for reading
            max_file_size_mb: Maximum file size in MB before creating a new file
            metadata: Optional metadata to store with the dataset
        """
        super().__init__(schema, path)
        self.compression = compression
        self.use_memory_map = use_memory_map
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
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
        self.current_file_idx = 0
        self.current_file_size = 0
        self.total_experiences = 0
        self._initialize_storage()

        # Build arrow schema
        self.arrow_schema = self._build_arrow_schema()

        # Track file indices for each chunk of data
        self.file_indices = []
        self.index_map = {}  # Maps global indices to (file_idx, local_idx)

    def _initialize_storage(self) -> None:
        """Initialize the storage directory structure."""
        os.makedirs(self.path, exist_ok=True)

        # Create metadata file
        metadata_path = os.path.join(self.path, "metadata.json")
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w") as f:
                json.dump(
                    {"metadata": self.metadata, "schema": self.schema.to_dict()},
                    f,
                    indent=2,
                )

        # Create index directory
        index_dir = os.path.join(self.path, "indices")
        os.makedirs(index_dir, exist_ok=True)

        # Load existing index if it exists
        index_path = os.path.join(index_dir, "global_index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index_data = json.load(f)
                self.file_indices = index_data["file_indices"]
                self.total_experiences = index_data["total_experiences"]
                self.current_file_idx = index_data.get("current_file_idx", 0)

                # Rebuild index map
                for file_idx, count in enumerate(self.file_indices):
                    start_idx = sum(self.file_indices[:file_idx])
                    for local_idx in range(count):
                        global_idx = start_idx + local_idx
                        self.index_map[global_idx] = (file_idx, local_idx)

    def _build_arrow_schema(self) -> pa.Schema:
        """
        Convert our ExperienceSchema to PyArrow Schema.

        Returns:
            pa.Schema: The PyArrow schema
        """
        fields = []

        for field_name, field_spec in self.schema.field_specs.items():
            arrow_field = self._spec_to_arrow_field(field_name, field_spec)
            fields.append(arrow_field)

        return pa.schema(fields)

    def _spec_to_arrow_field(self, name: str, spec: TypeSpecification) -> pa.Field:
        """
        Convert a TypeSpecification to a PyArrow Field.

        Args:
            name: Field name
            spec: Type specification

        Returns:
            pa.Field: The PyArrow field
        """
        # Handle nested dictionary structures
        if spec.nested_spec:
            nested_fields = [
                self._spec_to_arrow_field(nested_name, nested_spec)
                for nested_name, nested_spec in spec.nested_spec.items()
            ]
            return pa.field(name, pa.struct(nested_fields))

        # Handle tuple types
        elif spec.is_tuple:
            tuple_types = [
                self._spec_to_arrow_type(item_spec) for item_spec in spec.tuple_specs
            ]
            return pa.field(
                name,
                pa.list_(
                    pa.struct(
                        [pa.field(f"item_{i}", t) for i, t in enumerate(tuple_types)]
                    )
                ),
            )

        # Handle array types
        elif spec.shape is not None:
            base_type = self._dtype_to_arrow_type(spec.dtype)

            # Convert shape to Arrow type
            if len(spec.shape) == 1:
                # 1D array
                return pa.field(name, pa.list_(base_type))
            else:
                # Multi-dimensional array
                # Use nested lists for multi-dimensional arrays
                array_type = base_type
                for dim in reversed(spec.shape):
                    array_type = pa.list_(array_type)
                return pa.field(name, array_type)

        # Handle scalar types
        else:
            return pa.field(name, self._dtype_to_arrow_type(spec.dtype))

    def _spec_to_arrow_type(self, spec: TypeSpecification) -> pa.DataType:
        """
        Convert a TypeSpecification to a PyArrow DataType.

        Args:
            spec: Type specification

        Returns:
            pa.DataType: The PyArrow data type
        """
        # Handle nested dictionary structures
        if spec.nested_spec:
            nested_fields = [
                self._spec_to_arrow_field(nested_name, nested_spec)
                for nested_name, nested_spec in spec.nested_spec.items()
            ]
            return pa.struct(nested_fields)

        # Handle tuple types
        elif spec.is_tuple:
            tuple_types = [
                self._spec_to_arrow_type(item_spec) for item_spec in spec.tuple_specs
            ]
            return pa.list_(
                pa.struct([pa.field(f"item_{i}", t) for i, t in enumerate(tuple_types)])
            )

        # Handle array types
        elif spec.shape is not None:
            base_type = self._dtype_to_arrow_type(spec.dtype)

            # Convert shape to Arrow type
            if len(spec.shape) == 1:
                # 1D array
                return pa.list_(base_type)
            else:
                # Multi-dimensional array
                # Use nested lists for multi-dimensional arrays
                array_type = base_type
                for dim in reversed(spec.shape):
                    array_type = pa.list_(array_type)
                return array_type

        # Handle scalar types
        else:
            return self._dtype_to_arrow_type(spec.dtype)

    def _dtype_to_arrow_type(self, dtype: DType) -> pa.DataType:
        """
        Convert our DType enum to PyArrow DataType.

        Args:
            dtype: The data type from DType enum

        Returns:
            pa.DataType: The PyArrow data type
        """
        type_map = {
            DType.FLOAT32: pa.float32(),
            DType.FLOAT64: pa.float64(),
            DType.INT8: pa.int8(),
            DType.INT16: pa.int16(),
            DType.INT32: pa.int32(),
            DType.INT64: pa.int64(),
            DType.UINT8: pa.uint8(),
            DType.UINT16: pa.uint16(),
            DType.UINT32: pa.uint32(),
            DType.UINT64: pa.uint64(),
            DType.BOOL: pa.bool_(),
            DType.STRING: pa.string(),
        }

        if dtype not in type_map:
            logger.warning(f"Unknown dtype: {dtype}, defaulting to float32")
            return pa.float32()

        return type_map[dtype]

    def _get_current_file_path(self) -> str:
        """
        Get the path for the current data file.

        Returns:
            str: File path
        """
        return os.path.join(
            self.path, f"experiences_{self.current_file_idx:06d}.parquet"
        )

    def _create_arrow_table(self, experiences: list[dict[str, Any]]) -> pa.Table:
        """
        Convert a list of experiences to an Arrow Table.

        Args:
            experiences: List of experience dictionaries

        Returns:
            pa.Table: Arrow Table containing the experiences
        """
        # Group by field
        field_arrays = defaultdict(list)

        for exp in experiences:
            for field_name, field_value in exp.items():
                if field_name in self.schema.field_specs:
                    field_arrays[field_name].append(field_value)

        # Convert to Arrow arrays
        arrow_arrays = {}
        for field_name, values in field_arrays.items():
            field_spec = self.schema.field_specs[field_name]
            arrow_arrays[field_name] = self._values_to_arrow_array(values, field_spec)

        return pa.Table.from_pydict(arrow_arrays, schema=self.arrow_schema)

    def _values_to_arrow_array(
        self, values: list[Any], spec: TypeSpecification
    ) -> pa.Array:
        """
        Convert a list of values to an Arrow Array based on type specification.

        Args:
            values: List of values
            spec: Type specification

        Returns:
            pa.Array: Arrow Array
        """
        # Convert to numpy first for numerical arrays
        if spec.shape is not None and spec.dtype != DType.STRING:
            # Handle arrays with known shape
            numpy_values = []
            for val in values:
                numpy_val = TypeConverter.to_numpy(val, spec)
                numpy_values.append(numpy_val)

            # Create Arrow array from numpy
            return pa.array(numpy_values)

        # Handle nested structures
        elif spec.nested_spec:
            nested_arrays = {}
            for nested_name, nested_spec in spec.nested_spec.items():
                nested_values = [v[nested_name] for v in values]
                nested_arrays[nested_name] = self._values_to_arrow_array(
                    nested_values, nested_spec
                )

            return pa.StructArray.from_arrays(
                list(nested_arrays.values()), list(nested_arrays.keys())
            )

        # Handle tuple types
        elif spec.is_tuple:
            tuple_arrays = []
            for i, item_spec in enumerate(spec.tuple_specs):
                item_values = [v[i] for v in values]
                tuple_arrays.append(self._values_to_arrow_array(item_values, item_spec))

            struct_array = pa.StructArray.from_arrays(
                tuple_arrays, [f"item_{i}" for i in range(len(spec.tuple_specs))]
            )
            return pa.ListArray.from_arrays(
                pa.array(range(len(values) + 1)), struct_array
            )

        # Handle scalar values
        else:
            arrow_type = self._dtype_to_arrow_type(spec.dtype)
            return pa.array(values, type=arrow_type)

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

        # Validate experiences against schema
        for exp in experiences:
            valid, error = self.schema.validate_sample(exp)
            if not valid:
                logger.error(f"Invalid experience: {error}")
                return False

        # Convert experiences to Arrow Table
        table = self._create_arrow_table(experiences)

        # Check if we need to create a new file
        current_file_path = self._get_current_file_path()
        if os.path.exists(current_file_path):
            file_size = os.path.getsize(current_file_path)
            if file_size + table.nbytes > self.max_file_size_bytes:
                # Increment file index
                self.current_file_idx += 1
                current_file_path = self._get_current_file_path()
                self.current_file_size = 0

        # Write to Parquet file
        try:
            # Configure compression
            compression_options = {
                "zstd": pq.write_table,
                "gzip": pq.write_table,
                "lz4": pq.write_table,
                "snappy": pq.write_table,
                "none": lambda t, path: pq.write_table(t, path, compression="none"),
            }

            write_func = compression_options.get(
                self.compression.lower(),
                lambda t, path: pq.write_table(t, path, compression="zstd"),
            )

            # Check if file exists
            if os.path.exists(current_file_path):
                # Read existing file and append
                existing_table = pq.read_table(current_file_path)
                table = pa.concat_tables([existing_table, table])

            # Write table to file
            write_func(table, current_file_path)

            # Update state
            batch_size = len(experiences)
            self.total_experiences += batch_size

            # Update index
            if len(self.file_indices) <= self.current_file_idx:
                self.file_indices.append(0)

            # Track local and global indices
            start_idx = sum(self.file_indices[: self.current_file_idx])
            for local_idx in range(batch_size):
                global_idx = (
                    start_idx + self.file_indices[self.current_file_idx] + local_idx
                )
                self.index_map[global_idx] = (
                    self.current_file_idx,
                    self.file_indices[self.current_file_idx] + local_idx,
                )

            # Update file index count
            self.file_indices[self.current_file_idx] += batch_size

            # Update current file size estimation
            self.current_file_size = os.path.getsize(current_file_path)

            # Update global index file
            self._update_global_index()

            return True

        except Exception as e:
            logger.error(f"Error writing to Parquet file: {e}")
            return False

    def _update_global_index(self) -> None:
        """Update the global index file with current state."""
        index_path = os.path.join(self.path, "indices", "global_index.json")
        with open(index_path, "w") as f:
            json.dump(
                {
                    "file_indices": self.file_indices,
                    "total_experiences": self.total_experiences,
                    "current_file_idx": self.current_file_idx,
                },
                f,
            )

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
        if self.total_experiences == 0:
            logger.warning("No experiences in storage")
            return []

        # If indices are provided, use them
        if indices is not None:
            # Group indices by file
            file_groups = defaultdict(list)
            for global_idx in indices:
                if global_idx not in self.index_map:
                    logger.warning(f"Index {global_idx} not found in storage")
                    continue

                file_idx, local_idx = self.index_map[global_idx]
                file_groups[file_idx].append(local_idx)

            # Read experiences from each file
            experiences = []
            for file_idx, local_indices in file_groups.items():
                file_experiences = self._read_from_file(file_idx, local_indices)
                experiences.extend(file_experiences)

            return experiences

        # Otherwise, randomly select indices
        else:
            if batch_size > self.total_experiences:
                batch_size = self.total_experiences

            # Generate random indices
            random_indices = np.random.choice(
                self.total_experiences, size=batch_size, replace=False
            )

            return self.read_batch(indices=random_indices.tolist())

    def _read_from_file(
        self, file_idx: int, local_indices: list[int]
    ) -> list[dict[str, Any]]:
        """
        Read experiences from a specific file.

        Args:
            file_idx: File index
            local_indices: List of local indices to read

        Returns:
            List of experience dictionaries
        """
        file_path = os.path.join(self.path, f"experiences_{file_idx:06d}.parquet")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        try:
            # Read the file with memory mapping if enabled
            table = pq.read_table(file_path, memory_map=self.use_memory_map)

            # Use predicate pushdown if possible
            if len(local_indices) < table.num_rows / 2:
                # Convert indices to Arrow array
                indices_array = pa.array(local_indices, type=pa.int64())

                # Create row indices column
                row_indices = pa.array(range(table.num_rows), type=pa.int64())

                # Filter using Arrow compute
                mask = pc.is_in(row_indices, indices_array)
                table = table.filter(mask)

            # Convert to Python dictionaries
            experiences = self._arrow_table_to_experiences(table)

            # If we used predicate pushdown, we got all requested rows
            # Otherwise, we need to filter manually
            if len(local_indices) >= table.num_rows / 2:
                # Filter to only the requested indices
                experiences = [
                    experiences[idx] for idx in local_indices if idx < len(experiences)
                ]

            return experiences

        except Exception as e:
            logger.error(f"Error reading from Parquet file: {e}")
            return []

    def _arrow_table_to_experiences(self, table: pa.Table) -> list[dict[str, Any]]:
        """
        Convert an Arrow Table to a list of experience dictionaries.

        Args:
            table: Arrow Table

        Returns:
            List of experience dictionaries
        """
        # Convert to pandas (efficient for conversion to Python objects)
        # This leverages zero-copy when possible
        df = table.to_pandas()

        # Convert DataFrame to list of dictionaries
        experiences = df.to_dict("records")

        return experiences

    def get_size(self) -> int:
        """
        Get the number of experiences in storage.

        Returns:
            int: Number of experiences
        """
        return self.total_experiences

    def close(self) -> None:
        """Close the storage backend and release resources."""
        # Update global index
        self._update_global_index()

    def get_iterator(
        self, batch_size: int = 32, shuffle: bool = False
    ) -> Iterator[list[dict[str, Any]]]:
        """
        Get an iterator over all experiences.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data

        Returns:
            Iterator yielding batches of experiences
        """
        # Calculate total number of batches
        total_batches = (self.total_experiences + batch_size - 1) // batch_size

        # Generate indices for all experiences
        indices = list(range(self.total_experiences))

        # Shuffle if requested
        if shuffle:
            np.random.shuffle(indices)

        # Yield batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.total_experiences)
            batch_indices = indices[start_idx:end_idx]

            yield self.read_batch(indices=batch_indices)

    def filter(self, predicate_fn) -> list[int]:
        """
        Filter experiences using a predicate function.

        Args:
            predicate_fn: Function that takes an experience and returns a boolean

        Returns:
            List of indices that match the predicate
        """
        matching_indices = []

        # Iterate through all experiences in batches
        for batch in self.get_iterator(batch_size=100):
            for exp_idx, exp in enumerate(batch):
                if predicate_fn(exp):
                    # Calculate global index
                    global_idx = exp_idx + len(matching_indices)
                    matching_indices.append(global_idx)

        return matching_indices


# Additional helper functions


def create_arrow_storage(
    schema: ExperienceSchema, path: str, **kwargs
) -> ArrowStorageBackend:
    """
    Create an Arrow storage backend.

    Args:
        schema: Experience schema
        path: Storage path
        **kwargs: Additional arguments for ArrowStorageBackend

    Returns:
        ArrowStorageBackend: The storage backend
    """
    return ArrowStorageBackend(schema, path, **kwargs)


def load_arrow_storage(path: str) -> ArrowStorageBackend:
    """
    Load an existing Arrow storage backend.

    Args:
        path: Storage path

    Returns:
        ArrowStorageBackend: The storage backend
    """
    # Load metadata and schema
    metadata_path = os.path.join(path, "metadata.json")
    with open(metadata_path) as f:
        data = json.load(f)
        metadata = data["metadata"]
        schema_dict = data["schema"]

    # Create schema
    schema = ExperienceSchema.from_dict(schema_dict)

    # Create storage backend
    return ArrowStorageBackend(schema, path, metadata=metadata)


class ArrowDatasetIterator:
    """Iterator for Arrow dataset that can be used with PyTorch DataLoader."""

    def __init__(
        self,
        storage: ArrowStorageBackend,
        batch_size: int = 32,
        shuffle: bool = True,
        transform=None,
    ):
        """
        Initialize the iterator.

        Args:
            storage: Arrow storage backend
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            transform: Optional transform function
        """
        self.storage = storage
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.iterator = None

    def __iter__(self):
        """Get iterator."""
        self.iterator = self.storage.get_iterator(
            batch_size=self.batch_size, shuffle=self.shuffle
        )
        return self

    def __next__(self):
        """Get next batch."""
        try:
            batch = next(self.iterator)
            if self.transform:
                batch = self.transform(batch)
            return batch
        except StopIteration:
            raise StopIteration
