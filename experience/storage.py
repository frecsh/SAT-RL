import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import pickle
from filelock import FileLock
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import threading
from utils.compression import CompressionConfig, CompressionFormat, CompressionLevel, FieldCompressionManager, CompressionService


class ExperienceStorage:
    """Base class for experience storage backends with compression support."""
    
    def __init__(self, 
                 path: str, 
                 compression_manager: Optional[FieldCompressionManager] = None):
        """
        Initialize the storage handler.
        
        Args:
            path: Path to the storage file
            compression_manager: Manager for field-specific compression settings
        """
        self.path = path
        self.compression_manager = compression_manager or FieldCompressionManager(
            default_config=CompressionConfig(format=CompressionFormat.NONE)
        )
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
    def _compress_field(self, field_name: str, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Compress a field's data using its specific compression settings."""
        config = self.compression_manager.get_field_config(field_name)
        return CompressionService.compress(data, config)
    
    def _decompress_field(self, field_name: str, data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Decompress a field's data using the metadata."""
        return CompressionService.decompress(data, metadata)


class ParquetExperienceStorage(ExperienceStorage):
    """Storage backend using Apache Arrow/Parquet with compression support."""
    
    def __init__(self, 
                 path: str, 
                 compression_manager: Optional[FieldCompressionManager] = None):
        """Initialize Parquet storage."""
        super().__init__(path, compression_manager)
        self.file_lock = FileLock(f"{path}.lock")
        
    def write_batch(self, data: Dict[str, np.ndarray], metadata: Optional[Dict[str, Any]] = None):
        """
        Write a batch of experience data with field-specific compression.
        
        Args:
            data: Dictionary of field name to numpy arrays
            metadata: Optional metadata to store with the batch
        """
        with self._lock:
            # Process and potentially compress each field
            compressed_data = {}
            compression_metadata = {}
            
            for field_name, field_data in data.items():
                # Convert numpy arrays to bytes for compression
                serialized = pickle.dumps(field_data)
                
                if self.compression_manager.should_compress_field(field_name):
                    compressed, field_meta = self._compress_field(field_name, serialized)
                    compressed_data[field_name] = compressed
                    compression_metadata[field_name] = field_meta
                else:
                    compressed_data[field_name] = serialized
                    compression_metadata[field_name] = {"compression": "none"}
            
            # Create Arrow table with compressed data
            fields = []
            arrays = []
            
            for field_name, compressed in compressed_data.items():
                fields.append(pa.field(field_name, pa.binary()))
                arrays.append(pa.array([compressed]))
            
            # Add compression metadata as a separate column
            fields.append(pa.field('__compression_meta__', pa.binary()))
            arrays.append(pa.array([pickle.dumps(compression_metadata)]))
            
            # Add general metadata if provided
            if metadata:
                fields.append(pa.field('__metadata__', pa.binary()))
                arrays.append(pa.array([pickle.dumps(metadata)]))
            
            table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
            
            # Write to Parquet with file lock to ensure thread safety
            with self.file_lock:
                if os.path.exists(self.path):
                    # Append to existing file
                    pq.write_to_dataset(
                        table, 
                        root_path=os.path.dirname(self.path),
                        partition_cols=[],
                        basename_template=os.path.basename(self.path).split('.')[0] + '_{i}.parquet',
                        existing_data_behavior='overwrite_or_ignore'
                    )
                else:
                    # Create new file
                    pq.write_table(table, self.path)
    
    def read_batch(self, fields: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Read and decompress a batch of experience data.
        
        Args:
            fields: Optional list of field names to read (None for all)
            
        Returns:
            Dictionary of field name to numpy arrays
        """
        with self._lock:
            if not os.path.exists(self.path):
                return {}
            
            # Read with predicate pushdown if specific fields requested
            if fields:
                columns = fields + ['__compression_meta__']
                table = pq.read_table(self.path, columns=columns)
            else:
                table = pq.read_table(self.path)
            
            # Extract compression metadata
            compression_meta_col = table.column('__compression_meta__')
            compression_meta_bytes = compression_meta_col[0].as_py()
            compression_metadata = pickle.loads(compression_meta_bytes)
            
            # Process and decompress each field
            result = {}
            for field_name in table.column_names:
                if field_name in ('__compression_meta__', '__metadata__'):
                    continue
                
                field_data = table.column(field_name)[0].as_py()
                field_meta = compression_metadata.get(field_name, {"compression": "none"})
                
                # Decompress if needed
                if field_meta.get("compression") != "none":
                    decompressed = self._decompress_field(field_name, field_data, field_meta)
                    result[field_name] = pickle.loads(decompressed)
                else:
                    result[field_name] = pickle.loads(field_data)
            
            return result


class HDF5ExperienceStorage(ExperienceStorage):
    """Storage backend using HDF5 with compression support."""
    
    def __init__(self, 
                 path: str, 
                 compression_manager: Optional[FieldCompressionManager] = None,
                 chunk_size: int = 1000):
        """
        Initialize HDF5 storage.
        
        Args:
            path: Path to the HDF5 file
            compression_manager: Manager for field-specific compression settings
            chunk_size: Size of chunks for storage
        """
        super().__init__(path, compression_manager)
        self.chunk_size = chunk_size
        self.file_lock = FileLock(f"{path}.lock")
        
    def write_batch(self, data: Dict[str, np.ndarray], metadata: Optional[Dict[str, Any]] = None):
        """
        Write a batch of experience data with compression.
        
        Args:
            data: Dictionary of field name to numpy arrays
            metadata: Optional metadata to store with the batch
        """
        with self._lock:
            # Process and potentially compress each field
            compressed_data = {}
            compression_metadata = {}
            
            for field_name, field_data in data.items():
                # Convert numpy arrays to bytes for compression
                serialized = pickle.dumps(field_data)
                
                if self.compression_manager.should_compress_field(field_name):
                    compressed, field_meta = self._compress_field(field_name, serialized)
                    compressed_data[field_name] = compressed
                    compression_metadata[field_name] = field_meta
                else:
                    compressed_data[field_name] = serialized
                    compression_metadata[field_name] = {"compression": "none"}
            
            # Write to HDF5 with file lock
            with self.file_lock:
                mode = 'a' if os.path.exists(self.path) else 'w'
                with h5py.File(self.path, mode) as f:
                    # Create or update compression metadata - store as uint8 array to handle NULL bytes
                    if '__compression_meta__' in f:
                        del f['__compression_meta__']
                    meta_bytes = pickle.dumps(compression_metadata)
                    f.create_dataset('__compression_meta__', data=np.frombuffer(meta_bytes, dtype=np.uint8))
                    
                    # Add general metadata if provided - also as uint8 array
                    if metadata:
                        if '__metadata__' in f:
                            del f['__metadata__']
                        meta_bytes = pickle.dumps(metadata)
                        f.create_dataset('__metadata__', data=np.frombuffer(meta_bytes, dtype=np.uint8))
                    
                    # Store each compressed field
                    for field_name, compressed in compressed_data.items():
                        if field_name in f:
                            del f[field_name]
                        # Convert binary data to uint8 array to handle NULL bytes
                        byte_array = np.frombuffer(compressed, dtype=np.uint8)
                        f.create_dataset(
                            field_name, 
                            data=byte_array,
                            chunks=(min(self.chunk_size, len(byte_array)),) if len(byte_array) > 0 else None
                        )
    
    def read_batch(self, fields: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Read and decompress a batch of experience data.
        
        Args:
            fields: Optional list of field names to read (None for all)
            
        Returns:
            Dictionary of field name to numpy arrays
        """
        with self._lock:
            if not os.path.exists(self.path):
                return {}
            
            result = {}
            with h5py.File(self.path, 'r') as f:
                # Extract compression metadata - convert uint8 array back to bytes
                compression_meta_bytes = f['__compression_meta__'][()].tobytes()
                compression_metadata = pickle.loads(compression_meta_bytes)
                
                # Read requested fields
                field_names = fields or [k for k in f.keys() 
                                       if k not in ('__compression_meta__', '__metadata__')]
                
                for field_name in field_names:
                    if field_name in ('__compression_meta__', '__metadata__') or field_name not in f:
                        continue
                    
                    # Read byte array and convert back to bytes
                    field_bytes = f[field_name][()].tobytes()
                    field_meta = compression_metadata.get(field_name, {"compression": "none"})
                    
                    # Decompress if needed
                    if field_meta.get("compression") != "none":
                        decompressed = self._decompress_field(field_name, field_bytes, field_meta)
                        result[field_name] = pickle.loads(decompressed)
                    else:
                        result[field_name] = pickle.loads(field_bytes)
            
            return result


class MemoryMappedExperienceStorage(ExperienceStorage):
    """Storage backend using memory-mapped files with compression support."""
    
    def __init__(self, 
                 path: str, 
                 compression_manager: Optional[FieldCompressionManager] = None):
        """Initialize memory-mapped storage."""
        super().__init__(path, compression_manager)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.index_path = f"{self.path}.index"
        self.meta_path = f"{self.path}.meta"
        self.file_lock = FileLock(f"{self.path}.lock")
        
    def write_batch(self, data: Dict[str, np.ndarray], metadata: Optional[Dict[str, Any]] = None):
        """Write a batch of experience data with compression."""
        with self._lock:
            # Process and potentially compress each field
            compressed_fields = {}
            compression_metadata = {}
            
            for field_name, field_data in data.items():
                # Convert numpy arrays to bytes for compression
                serialized = pickle.dumps(field_data)
                
                if self.compression_manager.should_compress_field(field_name):
                    compressed, field_meta = self._compress_field(field_name, serialized)
                    compressed_fields[field_name] = compressed
                    compression_metadata[field_name] = field_meta
                else:
                    compressed_fields[field_name] = serialized
                    compression_metadata[field_name] = {"compression": "none"}
            
            # Create index of field sizes and offsets
            index = {}
            with self.file_lock:
                with open(self.path, 'ab') as f:
                    for field_name, compressed in compressed_fields.items():
                        offset = f.tell()
                        size = len(compressed)
                        f.write(compressed)
                        index[field_name] = {'offset': offset, 'size': size}
                
                # Update index file
                existing_index = {}
                if os.path.exists(self.index_path):
                    with open(self.index_path, 'r') as f:
                        existing_index = json.load(f)
                
                existing_index.update(index)
                with open(self.index_path, 'w') as f:
                    json.dump(existing_index, f)
                
                # Update metadata file
                existing_meta = {'compression': {}}
                if os.path.exists(self.meta_path):
                    with open(self.meta_path, 'r') as f:
                        existing_meta = json.load(f)
                
                # Update compression metadata
                existing_meta['compression'].update(compression_metadata)
                
                # Add general metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        if key != 'compression':  # Don't overwrite compression metadata
                            existing_meta[key] = value
                
                with open(self.meta_path, 'w') as f:
                    json.dump(existing_meta, f)
    
    def read_batch(self, fields: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Read and decompress a batch of experience data."""
        with self._lock:
            if not os.path.exists(self.path) or not os.path.exists(self.index_path):
                return {}
            
            # Read index
            with open(self.index_path, 'r') as f:
                index = json.load(f)
            
            # Read compression metadata
            with open(self.meta_path, 'r') as f:
                metadata = json.load(f)
                compression_metadata = metadata.get('compression', {})
            
            # Determine fields to read
            field_names = fields or list(index.keys())
            
            # Read and decompress fields
            result = {}
            with open(self.path, 'rb') as f:
                for field_name in field_names:
                    if field_name not in index:
                        continue
                    
                    field_info = index[field_name]
                    offset = field_info['offset']
                    size = field_info['size']
                    
                    f.seek(offset)
                    field_data = f.read(size)
                    
                    field_meta = compression_metadata.get(field_name, {"compression": "none"})
                    
                    # Decompress if needed
                    if field_meta.get("compression") != "none":
                        decompressed = self._decompress_field(field_name, field_data, field_meta)
                        result[field_name] = pickle.loads(decompressed)
                    else:
                        result[field_name] = pickle.loads(field_data)
            
            return result


def create_storage(storage_type: str, path: str, 
                   compression_config: Optional[Dict[str, Any]] = None) -> ExperienceStorage:
    """
    Factory function to create appropriate storage backend.
    
    Args:
        storage_type: Type of storage ('parquet', 'hdf5', or 'mmap')
        path: Path to the storage file
        compression_config: Optional compression configuration
        
    Returns:
        Configured storage backend
    """
    # Set up compression manager
    compression_manager = FieldCompressionManager()
    
    if compression_config:
        # Set default compression
        if "default" in compression_config:
            default_cfg = compression_config["default"]
            format_name = default_cfg.get("format", "none")
            level = default_cfg.get("level", 5)
            compression_manager.default_config = CompressionConfig(
                format=CompressionFormat(format_name),
                level=level
            )
        
        # Set field-specific compression
        for field_name, field_cfg in compression_config.get("fields", {}).items():
            format_name = field_cfg.get("format", "none")
            level = field_cfg.get("level", 5)
            compression_manager.set_field_config(
                field_name,
                CompressionConfig(format=CompressionFormat(format_name), level=level)
            )
    
    # Create appropriate storage backend
    if storage_type == "parquet":
        return ParquetExperienceStorage(path, compression_manager)
    elif storage_type == "hdf5":
        chunk_size = compression_config.get("chunk_size", 1000) if compression_config else 1000
        return HDF5ExperienceStorage(path, compression_manager, chunk_size=chunk_size)
    elif storage_type == "mmap":
        return MemoryMappedExperienceStorage(path, compression_manager)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")