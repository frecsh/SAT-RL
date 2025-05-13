"""
Type system for experience data.

This module provides type specifications and conversion utilities
for experience data storage and retrieval.
"""

from enum import Enum
from typing import Any


class DataType(Enum):
    """Enum defining data types for experience storage."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    STRING = "string"
    BYTES = "bytes"


class TypeSpecification:
    """
    Specification for data types in experience storage.

    This class provides a unified interface for defining data types
    that can be mapped to specific backend implementations.
    """

    def __init__(self, data_type: DataType, shape: list[int] | None = None):
        """
        Initialize a type specification.

        Args:
            data_type: The DataType enum value for this specification
            shape: Optional shape information for tensors/arrays
        """
        self.data_type = data_type
        self.shape = shape or []

    def __repr__(self) -> str:
        shape_str = f"[{', '.join(map(str, self.shape))}]" if self.shape else ""
        return f"TypeSpecification({self.data_type.value}{shape_str})"

    def to_dict(self) -> dict[str, Any]:
        """Convert the type specification to a dictionary."""
        return {"data_type": self.data_type.value, "shape": self.shape}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypeSpecification":
        """Create a type specification from a dictionary."""
        return cls(data_type=DataType(data["data_type"]), shape=data.get("shape", []))


class TypeConverter:
    """
    Conversion utilities between different type systems.

    This class handles conversion between our internal type system
    and various backend-specific type systems (numpy, torch, TF, etc.)
    """

    @staticmethod
    def to_numpy_dtype(spec: TypeSpecification) -> str:
        """Convert internal type to numpy dtype."""
        mapping = {
            DataType.FLOAT32: "float32",
            DataType.FLOAT64: "float64",
            DataType.INT32: "int32",
            DataType.INT64: "int64",
            DataType.BOOL: "bool",
            DataType.STRING: "object",
            DataType.BYTES: "object",
        }
        return mapping[spec.data_type]

    @staticmethod
    def from_numpy_dtype(dtype_str: str) -> TypeSpecification:
        """Convert numpy dtype to internal type specification."""
        mapping = {
            "float32": DataType.FLOAT32,
            "float64": DataType.FLOAT64,
            "int32": DataType.INT32,
            "int64": DataType.INT64,
            "bool": DataType.BOOL,
            "object": DataType.STRING,  # Default assumption
        }
        return TypeSpecification(mapping.get(dtype_str, DataType.STRING))

    @staticmethod
    def to_torch_dtype(spec: TypeSpecification) -> str:
        """Convert internal type to PyTorch dtype."""
        import torch

        mapping = {
            DataType.FLOAT32: torch.float32,
            DataType.FLOAT64: torch.float64,
            DataType.INT32: torch.int32,
            DataType.INT64: torch.int64,
            DataType.BOOL: torch.bool,
            DataType.STRING: torch.string,  # May need adjustment based on PyTorch version
        }
        return mapping[spec.data_type]

    @staticmethod
    def to_tensorflow_dtype(spec: TypeSpecification) -> str:
        """Convert internal type to TensorFlow dtype."""
        import tensorflow as tf

        mapping = {
            DataType.FLOAT32: tf.float32,
            DataType.FLOAT64: tf.float64,
            DataType.INT32: tf.int32,
            DataType.INT64: tf.int64,
            DataType.BOOL: tf.bool,
            DataType.STRING: tf.string,
        }
        return mapping[spec.data_type]
