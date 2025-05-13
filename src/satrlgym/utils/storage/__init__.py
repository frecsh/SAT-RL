"""
DEPRECATED: This module has been moved to satrlgym.storage.backends

This compatibility layer will be removed in a future version.
"""

import warnings

from satrlgym.storage.backends.file import FileExperienceStorage
from satrlgym.storage.backends.hdf5 import HDF5ExperienceStorage
from satrlgym.storage.backends.memory import MemoryMappedExperienceStorage
from satrlgym.storage.backends.parquet import ParquetExperienceStorage
from satrlgym.storage.io.reader import ExperienceReader
from satrlgym.storage.io.writer import ExperienceWriter
from satrlgym.storage.utils.serialization import NumpyEncoder

warnings.warn(
    "The satrlgym.utils.storage module is deprecated. "
    "Please use satrlgym.storage.backends instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import from new location

# Re-export
__all__ = [
    "FileExperienceStorage",
    "HDF5ExperienceStorage",
    "MemoryMappedExperienceStorage",
    "ParquetExperienceStorage",
    "ExperienceReader",
    "ExperienceWriter",
    "NumpyEncoder",
]
