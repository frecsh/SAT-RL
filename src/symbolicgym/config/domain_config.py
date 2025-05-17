"""Domain configuration utilities for SymbolicGym."""

import os

import yaml


def load_domain_config(domain, config_dir=None):
    """Load YAML config for a given domain."""
    if config_dir is None:
        config_dir = os.path.join(
            os.path.dirname(__file__), "../../..", "config/curricula"
        )
    config_path = os.path.join(config_dir, f"{domain}_curriculum.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {config_path}")
    return config
