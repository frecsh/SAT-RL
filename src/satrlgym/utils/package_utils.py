"""
Package utilities for getting version information and other package details.
This module provides modern alternatives to the deprecated pkg_resources API.
"""

import importlib.metadata


def get_version(package_name: str) -> str:
    """
    Get the version of an installed package using importlib.metadata.

    Args:
        package_name: Name of the package

    Returns:
        Version string of the package or "unknown" if not found
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_distribution_info(package_name: str) -> dict:
    """
    Get information about a distribution using importlib.metadata.

    Args:
        package_name: Name of the package

    Returns:
        Dictionary with distribution information
    """
    try:
        dist = importlib.metadata.distribution(package_name)
        metadata = {
            "name": dist.metadata.get("Name", package_name),
            "version": dist.version,
            "summary": dist.metadata.get("Summary", ""),
            "author": dist.metadata.get("Author", ""),
            "author_email": dist.metadata.get("Author-email", ""),
            "license": dist.metadata.get("License", ""),
            "requires": list(dist.requires or []),
        }
        return metadata
    except importlib.metadata.PackageNotFoundError:
        return {"name": package_name, "version": "unknown"}


def package_exists(package_name: str) -> bool:
    """
    Check if a package exists using importlib.metadata.

    Args:
        package_name: Name of the package

    Returns:
        True if package exists, False otherwise
    """
    try:
        importlib.metadata.distribution(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def get_all_packages() -> list:
    """
    Get a list of all installed packages using importlib.metadata.

    Returns:
        List of dictionaries with package name and version
    """
    packages = []
    for dist in importlib.metadata.distributions():
        packages.append({"name": dist.metadata.get("Name"), "version": dist.version})
    return packages


def get_entry_points(group: str = None) -> dict:
    """
    Get entry points using importlib.metadata.

    Args:
        group: Optional group filter

    Returns:
        Dictionary mapping entry point names to their loaded values
    """
    results = {}

    # In Python 3.10+, entry_points() takes a group parameter directly
    entries = (
        importlib.metadata.entry_points(group=group)
        if group
        else importlib.metadata.entry_points()
    )

    # In 3.10+, entry_points() returns a SelectableGroups object that we need to iterate differently
    if group:
        for entry in entries:
            results[entry.name] = entry.load
    else:
        for group_name, entries_list in entries.items():
            for entry in entries_list:
                if group_name not in results:
                    results[group_name] = {}
                results[group_name][entry.name] = entry.load

    # For Python 3.8 and 3.9

    return results
