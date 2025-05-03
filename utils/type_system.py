"""
Type system utilities for enhanced experience format.
Provides explicit dtype specification, support for complex observation spaces,
validation utilities, and cross-language type conversion handling.
"""

from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Type
import numpy as np
import logging
import json
import inspect
from collections import OrderedDict

# Set up logging
logger = logging.getLogger(__name__)

class DType(Enum):
    """Enum representing supported data types with explicit size specifications."""
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOL = "bool"
    STRING = "string"


class TypeSpecification:
    """
    Class to define and enforce type specifications for experience data.
    Supports primitive types, arrays, and complex nested structures.
    """
    
    def __init__(self, dtype: DType, shape: Optional[Tuple[int, ...]] = None, 
                 nested_spec: Optional[Dict[str, 'TypeSpecification']] = None,
                 is_tuple: bool = False, tuple_specs: Optional[List['TypeSpecification']] = None):
        """
        Initialize a type specification.
        
        Args:
            dtype: The data type from DType enum
            shape: Optional shape for array types (None for scalars)
            nested_spec: Specification for nested dictionary structures
            is_tuple: Whether this spec represents a tuple type
            tuple_specs: List of specs for each tuple element
        """
        self.dtype = dtype
        self.shape = shape
        self.nested_spec = nested_spec
        self.is_tuple = is_tuple
        self.tuple_specs = tuple_specs
        
        # Validate the specification parameters
        if is_tuple and tuple_specs is None:
            raise ValueError("tuple_specs must be provided when is_tuple is True")
        if nested_spec is not None and is_tuple:
            raise ValueError("Cannot specify both nested_spec and is_tuple")
    
    def __str__(self):
        if self.nested_spec:
            nested_str = {k: str(v) for k, v in self.nested_spec.items()}
            return f"Dict({nested_str})"
        elif self.is_tuple:
            tuple_str = ', '.join(str(s) for s in self.tuple_specs)
            return f"Tuple({tuple_str})"
        else:
            shape_str = f"{self.shape}" if self.shape else "scalar"
            return f"{self.dtype.value}[{shape_str}]"
    
    def to_dict(self) -> Dict:
        """Convert the specification to a dictionary for serialization."""
        result = {"dtype": self.dtype.value}
        if self.shape:
            result["shape"] = self.shape
        if self.nested_spec:
            result["nested_spec"] = {k: v.to_dict() for k, v in self.nested_spec.items()}
        if self.is_tuple:
            result["is_tuple"] = True
            result["tuple_specs"] = [s.to_dict() for s in self.tuple_specs]
        return result
    
    @classmethod
    def from_dict(cls, spec_dict: Dict) -> 'TypeSpecification':
        """Create a TypeSpecification from a dictionary."""
        nested_spec = None
        tuple_specs = None
        
        if "nested_spec" in spec_dict:
            nested_spec = {k: cls.from_dict(v) for k, v in spec_dict["nested_spec"].items()}
        
        if "tuple_specs" in spec_dict:
            tuple_specs = [cls.from_dict(s) for s in spec_dict["tuple_specs"]]
        
        return cls(
            dtype=DType(spec_dict["dtype"]),
            shape=spec_dict.get("shape"),
            nested_spec=nested_spec,
            is_tuple=spec_dict.get("is_tuple", False),
            tuple_specs=tuple_specs
        )
    
    @classmethod
    def from_numpy_dtype(cls, np_dtype) -> DType:
        """Convert numpy dtype to our DType enum."""
        dtype_str = str(np_dtype)
        mapping = {
            "float32": DType.FLOAT32,
            "float64": DType.FLOAT64,
            "int8": DType.INT8,
            "int16": DType.INT16,
            "int32": DType.INT32,
            "int64": DType.INT64, 
            "uint8": DType.UINT8,
            "uint16": DType.UINT16,
            "uint32": DType.UINT32,
            "uint64": DType.UINT64,
            "bool": DType.BOOL,
            "object": DType.STRING  # Default for Python objects/strings
        }
        
        for k, v in mapping.items():
            if k in dtype_str:
                return v
        
        logger.warning(f"Unknown numpy dtype: {dtype_str}, defaulting to FLOAT32")
        return DType.FLOAT32
    
    @classmethod
    def infer_from_value(cls, value: Any) -> 'TypeSpecification':
        """Infer type specification from a value."""
        if isinstance(value, dict):
            nested_spec = {k: cls.infer_from_value(v) for k, v in value.items()}
            return cls(dtype=DType.STRING, nested_spec=nested_spec)
        
        elif isinstance(value, tuple):
            tuple_specs = [cls.infer_from_value(item) for item in value]
            return cls(dtype=DType.STRING, is_tuple=True, tuple_specs=tuple_specs)
        
        elif isinstance(value, np.ndarray):
            return cls(
                dtype=cls.from_numpy_dtype(value.dtype),
                shape=value.shape
            )
        
        elif isinstance(value, (int, np.integer)):
            return cls(dtype=DType.INT64)
        
        elif isinstance(value, (float, np.floating)):
            return cls(dtype=DType.FLOAT64)
        
        elif isinstance(value, bool):
            return cls(dtype=DType.BOOL)
        
        elif isinstance(value, str):
            return cls(dtype=DType.STRING)
        
        else:
            logger.warning(f"Unknown type: {type(value)}, defaulting to STRING")
            return cls(dtype=DType.STRING)


class TypeValidator:
    """
    Validator for checking compatibility of data with type specifications.
    Provides detailed error messages for validation failures.
    """
    
    @staticmethod
    def validate(value: Any, spec: TypeSpecification) -> Tuple[bool, str]:
        """
        Validate that a value conforms to a type specification.
        
        Args:
            value: The value to validate
            spec: The type specification
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle nested dictionary structures
        if spec.nested_spec:
            if not isinstance(value, dict):
                return False, f"Expected dict, got {type(value)}"
            
            # Check all required keys are present
            for key in spec.nested_spec:
                if key not in value:
                    return False, f"Missing required key: {key}"
            
            # Validate each field
            for key, field_spec in spec.nested_spec.items():
                valid, error = TypeValidator.validate(value[key], field_spec)
                if not valid:
                    return False, f"Invalid field '{key}': {error}"
            
            return True, ""
            
        # Handle tuple types
        elif spec.is_tuple:
            if not isinstance(value, tuple):
                return False, f"Expected tuple, got {type(value)}"
            
            if len(value) != len(spec.tuple_specs):
                return False, f"Expected tuple of length {len(spec.tuple_specs)}, got {len(value)}"
            
            # Validate each tuple element
            for i, (item, item_spec) in enumerate(zip(value, spec.tuple_specs)):
                valid, error = TypeValidator.validate(item, item_spec)
                if not valid:
                    return False, f"Invalid tuple element at index {i}: {error}"
            
            return True, ""
            
        # Handle array types
        elif spec.shape is not None:
            if not isinstance(value, np.ndarray):
                return False, f"Expected numpy array, got {type(value)}"
            
            # Check shape compatibility
            if len(value.shape) != len(spec.shape):
                return False, f"Expected {len(spec.shape)} dimensions, got {len(value.shape)}"
            
            for i, (dim, spec_dim) in enumerate(zip(value.shape, spec.shape)):
                # -1 in spec means any size is acceptable for that dimension
                if spec_dim != -1 and dim != spec_dim:
                    return False, f"Shape mismatch at dimension {i}: expected {spec_dim}, got {dim}"
            
            # Check dtype compatibility (approximate check)
            value_dtype = TypeSpecification.from_numpy_dtype(value.dtype)
            if value_dtype != spec.dtype:
                return False, f"Dtype mismatch: expected {spec.dtype.value}, got {value_dtype.value}"
                
            return True, ""
            
        # Handle scalar types
        else:
            # Map Python types to our DType enum for scalar comparison
            type_map = {
                int: [DType.INT8, DType.INT16, DType.INT32, DType.INT64],
                float: [DType.FLOAT32, DType.FLOAT64],
                bool: [DType.BOOL],
                str: [DType.STRING]
            }
            
            for py_type, dtypes in type_map.items():
                if isinstance(value, py_type) and spec.dtype in dtypes:
                    return True, ""
            
            return False, f"Type mismatch: expected {spec.dtype.value}, got {type(value).__name__}"


class TypeConverter:
    """
    Utility for converting data between different framework formats.
    Supports conversions between numpy, PyTorch, TensorFlow, and native Python types.
    """
    
    @staticmethod
    def to_numpy(value: Any, spec: TypeSpecification) -> Union[np.ndarray, Dict, Tuple]:
        """Convert a value to numpy format based on its specification."""
        if spec.nested_spec:
            return {k: TypeConverter.to_numpy(value[k], field_spec) 
                    for k, field_spec in spec.nested_spec.items()}
        
        elif spec.is_tuple:
            return tuple(TypeConverter.to_numpy(item, item_spec)
                         for item, item_spec in zip(value, spec.tuple_specs))
        
        # Already numpy array
        elif isinstance(value, np.ndarray):
            target_dtype = spec.dtype.value
            if value.dtype != target_dtype:
                return value.astype(target_dtype)
            return value
            
        # Convert scalar to numpy
        else:
            target_dtype = spec.dtype.value
            return np.array(value, dtype=target_dtype)
    
    @staticmethod
    def to_torch(value: Any, spec: TypeSpecification) -> Any:
        """Convert a value to PyTorch format based on its specification."""
        # Import here to avoid hard dependency
        try:
            import torch
        except ImportError:
            logger.error("PyTorch is not installed. Cannot convert to torch tensors.")
            raise
        
        if spec.nested_spec:
            return {k: TypeConverter.to_torch(value[k], field_spec)
                    for k, field_spec in spec.nested_spec.items()}
        
        elif spec.is_tuple:
            return tuple(TypeConverter.to_torch(item, item_spec)
                         for item, item_spec in zip(value, spec.tuple_specs))
        
        # Convert numpy array to torch tensor
        elif isinstance(value, np.ndarray):
            # Map numpy dtypes to torch dtypes
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
                "uint8": torch.uint8,
                "bool": torch.bool
            }
            torch_dtype = dtype_map.get(spec.dtype.value, torch.float32)
            return torch.tensor(value, dtype=torch_dtype)
        
        # Convert scalar to torch tensor
        else:
            torch_dtype = {
                DType.FLOAT32: torch.float32,
                DType.FLOAT64: torch.float64,
                DType.INT8: torch.int8, 
                DType.INT16: torch.int16,
                DType.INT32: torch.int32,
                DType.INT64: torch.int64,
                DType.UINT8: torch.uint8,
                DType.BOOL: torch.bool,
                DType.STRING: None  # Strings cannot be directly converted to tensors
            }.get(spec.dtype)
            
            if spec.dtype == DType.STRING:
                return value  # Keep strings as Python strings
            return torch.tensor(value, dtype=torch_dtype)
    
    @staticmethod
    def to_tensorflow(value: Any, spec: TypeSpecification) -> Any:
        """Convert a value to TensorFlow format based on its specification."""
        # Import here to avoid hard dependency
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow is not installed. Cannot convert to tf tensors.")
            raise
        
        if spec.nested_spec:
            return {k: TypeConverter.to_tensorflow(value[k], field_spec)
                    for k, field_spec in spec.nested_spec.items()}
        
        elif spec.is_tuple:
            return tuple(TypeConverter.to_tensorflow(item, item_spec)
                         for item, item_spec in zip(value, spec.tuple_specs))
        
        # Convert numpy array to tf tensor
        elif isinstance(value, np.ndarray):
            # Map numpy dtypes to tf dtypes
            dtype_map = {
                "float32": tf.float32,
                "float64": tf.float64,
                "int8": tf.int8,
                "int16": tf.int16,
                "int32": tf.int32,
                "int64": tf.int64,
                "uint8": tf.uint8,
                "bool": tf.bool,
                "string": tf.string
            }
            tf_dtype = dtype_map.get(spec.dtype.value, tf.float32)
            return tf.convert_to_tensor(value, dtype=tf_dtype)
        
        # Convert scalar to tf tensor
        else:
            tf_dtype = {
                DType.FLOAT32: tf.float32,
                DType.FLOAT64: tf.float64,
                DType.INT8: tf.int8,
                DType.INT16: tf.int16,
                DType.INT32: tf.int32,
                DType.INT64: tf.int64,
                DType.UINT8: tf.uint8,
                DType.BOOL: tf.bool,
                DType.STRING: tf.string
            }.get(spec.dtype)
            
            return tf.convert_to_tensor(value, dtype=tf_dtype)


class ExperienceSchema:
    """
    Schema definition for the entire experience format.
    Manages the collection of type specifications for all fields.
    """
    
    def __init__(self):
        """Initialize an empty schema."""
        self.field_specs: Dict[str, TypeSpecification] = {}
        self.version = "1.0"
    
    def add_field(self, name: str, spec: TypeSpecification):
        """Add a field specification to the schema."""
        self.field_specs[name] = spec
    
    def validate_sample(self, sample: Dict) -> Tuple[bool, str]:
        """Validate a single experience sample against the schema."""
        # Check all required fields are present
        for field in self.field_specs:
            if field not in sample:
                return False, f"Missing required field: {field}"
        
        # Validate each field
        for field, value in sample.items():
            if field in self.field_specs:
                valid, error = TypeValidator.validate(value, self.field_specs[field])
                if not valid:
                    return False, f"Invalid field '{field}': {error}"
        
        return True, ""
    
    def to_dict(self) -> Dict:
        """Convert the schema to a dictionary for serialization."""
        return {
            "version": self.version,
            "fields": {k: v.to_dict() for k, v in self.field_specs.items()}
        }
    
    def to_json(self) -> str:
        """Convert the schema to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, schema_dict: Dict) -> 'ExperienceSchema':
        """Create a schema from a dictionary."""
        schema = cls()
        schema.version = schema_dict.get("version", "1.0")
        
        for field, spec_dict in schema_dict.get("fields", {}).items():
            schema.field_specs[field] = TypeSpecification.from_dict(spec_dict)
        
        return schema
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExperienceSchema':
        """Create a schema from a JSON string."""
        schema_dict = json.loads(json_str)
        return cls.from_dict(schema_dict)
    
    @classmethod
    def infer_from_sample(cls, sample: Dict) -> 'ExperienceSchema':
        """Infer a schema from a sample experience."""
        schema = cls()
        
        for field, value in sample.items():
            schema.field_specs[field] = TypeSpecification.infer_from_value(value)
        
        return schema


def create_default_sat_schema() -> ExperienceSchema:
    """Create a default schema for SAT solver experiences."""
    schema = ExperienceSchema()
    
    # Define observations (state)
    obs_spec = TypeSpecification(
        dtype=DType.FLOAT32,
        nested_spec={
            "variables": TypeSpecification(dtype=DType.INT8, shape=(-1,)),  # -1 means variable length
            "clauses": TypeSpecification(dtype=DType.INT8, shape=(-1, -1)),  # Variable number of clauses
            "assignment": TypeSpecification(dtype=DType.INT8, shape=(-1,)),  # Current assignment
            "clause_sat": TypeSpecification(dtype=DType.BOOL, shape=(-1,)),  # Clause satisfaction status
        }
    )
    schema.add_field("observation", obs_spec)
    
    # Define action
    schema.add_field("action", TypeSpecification(dtype=DType.INT32))
    
    # Define reward
    schema.add_field("reward", TypeSpecification(dtype=DType.FLOAT32))
    
    # Define next observation
    schema.add_field("next_observation", obs_spec)
    
    # Define terminal flag
    schema.add_field("done", TypeSpecification(dtype=DType.BOOL))
    
    # Define metadata
    meta_spec = TypeSpecification(
        dtype=DType.STRING,
        nested_spec={
            "episode_id": TypeSpecification(dtype=DType.INT64),
            "step": TypeSpecification(dtype=DType.INT32),
            "timestamp": TypeSpecification(dtype=DType.FLOAT64),
            "clause_count": TypeSpecification(dtype=DType.INT32),
            "variable_count": TypeSpecification(dtype=DType.INT32),
            "ratio": TypeSpecification(dtype=DType.FLOAT32)
        }
    )
    schema.add_field("metadata", meta_spec)
    
    return schema


# Type conversion reference documentation
TYPE_CONVERSION_DOCS = """
# Type Conversion Reference

This document describes how types are converted between different frameworks in the SAT+RL system.

## Basic Type Mappings

| SAT+RL Type | Python | NumPy | PyTorch | TensorFlow | Arrow | HDF5 |
|-------------|--------|-------|---------|------------|-------|------|
| FLOAT32     | float  | float32 | torch.float32 | tf.float32 | Arrow.Float32 | np.float32 |
| FLOAT64     | float  | float64 | torch.float64 | tf.float64 | Arrow.Float64 | np.float64 |
| INT8        | int    | int8    | torch.int8    | tf.int8    | Arrow.Int8    | np.int8    |
| INT16       | int    | int16   | torch.int16   | tf.int16   | Arrow.Int16   | np.int16   |
| INT32       | int    | int32   | torch.int32   | tf.int32   | Arrow.Int32   | np.int32   |
| INT64       | int    | int64   | torch.int64   | tf.int64   | Arrow.Int64   | np.int64   |
| UINT8       | int    | uint8   | torch.uint8   | tf.uint8   | Arrow.UInt8   | np.uint8   |
| UINT16      | int    | uint16  | -             | -          | Arrow.UInt16  | np.uint16  |
| UINT32      | int    | uint32  | -             | -          | Arrow.UInt32  | np.uint32  |
| UINT64      | int    | uint64  | -             | -          | Arrow.UInt64  | np.uint64  |
| BOOL        | bool   | bool    | torch.bool    | tf.bool    | Arrow.Bool    | np.bool    |
| STRING      | str    | object  | -             | tf.string  | Arrow.String  | h5py.special_dtype(vlen=str) |

## Complex Types

### Dict Observations
Dictionary observations are preserved as Python dictionaries with individual fields converted according to their specifications.

### Tuple Observations
Tuple observations are preserved as Python tuples with individual elements converted according to their specifications.

### Nested Structures
Nested structures (dictionaries containing dictionaries or tuples) maintain their structure with recursive conversion of their elements.

## Framework-Specific Notes

### PyTorch
- String values remain as Python strings
- Conversion to PyTorch tensors uses registered conversion functions
- Tensors are moved to the appropriate device (CPU/GPU) based on the global configuration

### TensorFlow
- TensorFlow eager tensors are used by default
- For tf.data pipelines, conversion happens during dataset creation
- Graph mode is supported through specialized constructors

### NumPy
- NumPy arrays are the default storage format
- Zero-copy views are used when possible to avoid memory duplication

### Arrow/Parquet
- Arrow tables use the schema definition for field types
- Dictionary encoding is used for repeated strings
- LZ4 compression is applied by default for binary data
"""

def get_type_conversion_docs():
    """Return the type conversion documentation."""
    return TYPE_CONVERSION_DOCS