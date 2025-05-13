"""
Configuration management system for SAT solver experiments.
Uses OmegaConf for flexible configuration handling.
"""

import logging
import os
from typing import Any

try:
    import omegaconf
    from omegaconf import DictConfig, OmegaConf

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    DictConfig = dict  # Type alias for when OmegaConf is not available

# Set up logging
logger = logging.getLogger(__name__)


class SolverConfig:
    """
    Configuration manager for SAT solvers.
    Handles loading, merging, and accessing configuration parameters.
    """

    DEFAULT_CONFIG = {
        "solver": {
            "name": "walksat",
            "timeout": 30.0,
            "max_iterations": 10000,
        },
        "problem": {
            "num_vars": 20,
            "num_clauses": 85,
            "seed": 42,
        },
        "logging": {
            "level": "INFO",
            "file": None,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "experiment": {
            "name": "default",
            "runs": 5,
            "save_results": True,
            "results_dir": "./results",
        },
    }

    def __init__(self, config_path: str | None = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        # Load the default configuration
        self.config = self._load_default_config()

        # Load the configuration file if provided
        if config_path:
            self._load_config_file(config_path)

    def _load_default_config(self) -> DictConfig | dict[str, Any]:
        """
        Load the default configuration.

        Returns:
            Default configuration
        """
        if OMEGACONF_AVAILABLE:
            return OmegaConf.create(self.DEFAULT_CONFIG)
        else:
            logger.warning("OmegaConf not available, using dict-based config")
            return self.DEFAULT_CONFIG.copy()

    def _load_config_file(self, config_path: str) -> None:
        """
        Load configuration from a file.

        Args:
            config_path: Path to configuration file
        """
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return

        try:
            if OMEGACONF_AVAILABLE:
                # Load the configuration using OmegaConf
                file_config = OmegaConf.load(config_path)
                # Merge with default configuration
                self.config = OmegaConf.merge(self.config, file_config)
            else:
                # Fallback to basic file loading
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    import yaml

                    with open(config_path) as f:
                        file_config = yaml.safe_load(f)
                elif config_path.endswith(".json"):
                    import json

                    with open(config_path) as f:
                        file_config = json.load(f)
                else:
                    logger.error(
                        f"Unsupported configuration file format: {config_path}"
                    )
                    return

                # Merge with default configuration
                self._merge_dicts(self.config, file_config)
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")

    def _merge_dicts(self, target: dict, source: dict) -> None:
        """
        Recursively merge two dictionaries.

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_dicts(target[key], value)
            else:
                target[key] = value

    def update(self, config_dict: dict[str, Any]) -> None:
        """
        Update the configuration with the given dictionary.

        Args:
            config_dict: Dictionary to update the configuration with
        """
        if OMEGACONF_AVAILABLE:
            self.config = OmegaConf.merge(self.config, config_dict)
        else:
            self._merge_dicts(self.config, config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        Supports dot notation for nested keys (e.g., "solver.timeout").

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if OMEGACONF_AVAILABLE:
            try:
                return OmegaConf.select(self.config, key)
            except (omegaconf.errors.OmegaConfBaseException, KeyError):
                return default
        else:
            # Manually traverse the key path
            keys = key.split(".")
            value = self.config
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        Supports dot notation for nested keys (e.g., "solver.timeout").

        Args:
            key: Configuration key
            value: Value to set
        """
        if OMEGACONF_AVAILABLE:
            OmegaConf.update(self.config, key, value)
        else:
            # Manually traverse the key path
            keys = key.split(".")
            target = self.config
            for i, k in enumerate(keys[:-1]):
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        if OMEGACONF_AVAILABLE:
            return OmegaConf.to_container(self.config, resolve=True)
        else:
            return self.config

    def save(self, file_path: str) -> None:
        """
        Save the configuration to a file.

        Args:
            file_path: Path to save the configuration to
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            if OMEGACONF_AVAILABLE:
                OmegaConf.save(self.config, file_path)
            else:
                # Determine the file format based on extension
                if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                    import yaml

                    with open(file_path, "w") as f:
                        yaml.dump(self.config, f, default_flow_style=False)
                elif file_path.endswith(".json"):
                    import json

                    with open(file_path, "w") as f:
                        json.dump(self.config, f, indent=2)
                else:
                    logger.error(
                        f"Unsupported file format for configuration: {file_path}"
                    )
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key

        Returns:
            Configuration value
        """
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the configuration.

        Args:
            key: Configuration key

        Returns:
            True if the key exists, False otherwise
        """
        return self.get(key) is not None


# Create a global configuration instance
config = SolverConfig()


def load_config(config_path: str | None = None) -> SolverConfig:
    """
    Load configuration from a file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration instance
    """
    global config
    if config_path:
        config = SolverConfig(config_path)
    return config


def get_config() -> SolverConfig:
    """
    Get the global configuration instance.

    Returns:
        Configuration instance
    """
    return config
