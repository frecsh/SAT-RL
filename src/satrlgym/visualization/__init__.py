"""
Visualization components for SatRLGym.

This package provides visualization tools for CNF formulas, agent behaviors,
and performance metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Import directly from visualization.py module to avoid circular imports
from satrlgym.visualization.visualization import (
    DatasetVisualization,
    compute_episode_statistics,
    create_tensorboard_log,
    create_transition_dataframe,
    generate_analysis_notebook,
    generate_summary_report,
    log_episode_metrics,
    log_to_wandb,
    plot_episode_rewards,
    plot_observation_statistics,
    plot_transition_history,
    visualize_episode_data,
)

# Export all the required functions
__all__ = [
    "DatasetVisualization",
    "plot_transition_history",
    "plot_episode_rewards",
    "plot_observation_statistics",
    "visualize_episode_data",
    "compute_episode_statistics",
    "create_transition_dataframe",
    "generate_summary_report",
    "create_tensorboard_log",
    "log_episode_metrics",
    "log_to_wandb",
    "generate_analysis_notebook",
]
