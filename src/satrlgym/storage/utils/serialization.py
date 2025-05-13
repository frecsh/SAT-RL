"""
Serialization utilities for storage.

This module provides utilities for serializing and deserializing data types
that aren't natively supported by JSON, such as NumPy arrays and data types.
"""

import base64
import json
import pickle
from typing import Any

import numpy as np


def serialize_multidim_array(arr: np.ndarray) -> bytes:
    """
    Serialize a multi-dimensional numpy array to binary format.

    Args:
        arr: NumPy array (multi-dimensional) to serialize

    Returns:
        Binary representation of the array
    """
    # Use pickle for serializing multi-dimensional arrays
    return pickle.dumps(arr)


def deserialize_multidim_array(data: bytes) -> np.ndarray:
    """
    Deserialize a binary representation back to a numpy array.

    Args:
        data: Binary data to deserialize

    Returns:
        NumPy array reconstructed from binary data
    """
    # Use pickle for deserializing multi-dimensional arrays
    return pickle.loads(data)


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder with support for NumPy arrays and data types.

    This encoder converts NumPy arrays, scalars, and data types to their
    Python equivalents for JSON serialization.
    """

    def default(self, obj: Any) -> Any:
        """
        Convert NumPy types to JSON-serializable types.

        Args:
            obj: Object to encode

        Returns:
            JSON-serializable equivalent

        Raises:
            TypeError: For types that cannot be serialized
        """
        if isinstance(obj, np.ndarray):
            # For arrays, convert to lists and preserve dtype info
            return {
                "_type": "numpy.ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        elif isinstance(obj, np.number):
            # For scalar values, convert to Python scalar
            return obj.item()
        elif isinstance(obj, np.bool_):
            # Special case for boolean
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating, np.complexfloating)):
            # For scalar types
            return obj.item()
        elif isinstance(obj, np.dtype):
            # For dtype objects
            return str(obj)
        elif isinstance(obj, (complex, np.complex64, np.complex128)):
            # For complex numbers
            return {"_type": "complex", "real": obj.real, "imag": obj.imag}
        elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
            # For objects with tolist method
            return obj.tolist()
        elif isinstance(obj, bytes):
            # For binary data, use base64 encoding
            return {"_type": "bytes", "data": base64.b64encode(obj).decode("ascii")}
        elif hasattr(obj, "__dict__"):
            # For custom objects, try to convert __dict__
            try:
                # Attempt pickle serialization as a fallback
                serialized = base64.b64encode(pickle.dumps(obj)).decode("ascii")
                return {"_type": "pickled", "data": serialized}
            except Exception:
                return str(obj)

        # Let the parent class handle it or raise TypeError
        return super().default(obj)


def encode_numpy_objects(obj: Any) -> Any:
    """
    Encode NumPy objects recursively for JSON serialization.

    Args:
        obj: Object to encode

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return {
            "_type": "numpy.ndarray",
            "data": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating, np.complexfloating)):
        return obj.item()
    elif isinstance(obj, np.dtype):
        return str(obj)
    elif isinstance(obj, (complex, np.complex64, np.complex128)):
        return {"_type": "complex", "real": obj.real, "imag": obj.imag}
    elif isinstance(obj, dict):
        return {key: encode_numpy_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [encode_numpy_objects(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(encode_numpy_objects(item) for item in obj)
    elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return {"_type": "bytes", "data": base64.b64encode(obj).decode("ascii")}
    elif hasattr(obj, "__dict__"):
        try:
            serialized = base64.b64encode(pickle.dumps(obj)).decode("ascii")
            return {"_type": "pickled", "data": serialized}
        except Exception:
            return str(obj)
    else:
        return obj


def decode_numpy_objects(obj: Any) -> Any:
    """
    Decode NumPy objects from JSON data.

    Args:
        obj: JSON-decoded object

    Returns:
        Decoded object with NumPy types restored
    """
    if isinstance(obj, dict):
        if "_type" in obj:
            if obj["_type"] == "numpy.ndarray":
                # Reconstruct NumPy array
                arr = np.array(obj["data"])
                if "dtype" in obj:
                    # Convert to proper dtype if specified
                    arr = arr.astype(np.dtype(obj["dtype"]))
                if "shape" in obj and obj["shape"] != arr.shape:
                    # Reshape if necessary
                    arr = arr.reshape(obj["shape"])
                return arr
            elif obj["_type"] == "complex":
                # Reconstruct complex number
                return complex(obj["real"], obj["imag"])
            elif obj["_type"] == "bytes":
                # Decode base64 to bytes
                return base64.b64decode(obj["data"])
            elif obj["_type"] == "pickled":
                # Unpickle object
                try:
                    return pickle.loads(base64.b64decode(obj["data"]))
                except Exception:
                    return obj
        else:
            # Process each key-value pair recursively
            return {key: decode_numpy_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Process each item in the list recursively
        return [decode_numpy_objects(item) for item in obj]
    else:
        # Return other types as is
        return obj
