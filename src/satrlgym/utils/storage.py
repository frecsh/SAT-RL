"""
DEPRECATED: This module is no longer maintained.

Please update your imports to use the new structure:
- Base storage classes: from satrlgym.storage import ExperienceStorage, StorageBase
- Backends: from satrlgym.storage.backends.file import FileExperienceStorage, etc.
- IO: from satrlgym.storage.io.reader import ExperienceReader, etc.
- Registry: from satrlgym.storage.registry import register_storage, get_storage_class
- Utils: from satrlgym.storage.utils.serialization import NumpyEncoder
"""

# Raise a more helpful error message when this module is imported
raise ImportError(
    "The 'satrlgym.utils.storage' module has been deprecated. "
    "Please update your imports to use the direct modules instead. "
    "See the storage_cleanup_summary.md document for details on the new import structure."
)
