"""
Storage registry module.

This module provides functionality to register and retrieve storage backends.
"""


# Registry to store storage backends
_STORAGE_REGISTRY: dict[str, type] = {}


def register_storage(name: str, storage_class: type) -> None:
    """
    Register a storage backend.

    Args:
        name: Name of the storage backend
        storage_class: Class implementing the storage backend
    """
    if name in _STORAGE_REGISTRY:
        raise ValueError(f"Storage backend '{name}' already registered")

    _STORAGE_REGISTRY[name] = storage_class


def get_storage_class(name: str) -> type:
    """
    Get a registered storage class by name.

    Args:
        name: Name of the storage backend

    Returns:
        The storage class

    Raises:
        ValueError: If the storage backend is not registered
    """
    if name not in _STORAGE_REGISTRY:
        raise ValueError(f"Storage backend '{name}' not registered")

    return _STORAGE_REGISTRY[name]


def get_storage_backend(name: str) -> type:
    """
    Get a registered storage class by name.

    Args:
        name: Name of the storage backend

    Returns:
        The storage class

    Raises:
        ValueError: If the storage backend is not registered
    """
    if name not in _STORAGE_REGISTRY:
        raise ValueError(
            f"Storage backend '{name}' not registered. Available backends: {list_storage_backends()}"
        )

    return _STORAGE_REGISTRY[name]


def list_storage_backends() -> list:
    """
    List all registered storage backends.

    Returns:
        List of registered backend names
    """
    return list(_STORAGE_REGISTRY.keys())


# Add alias for backward compatibility
list_available_backends = list_storage_backends
