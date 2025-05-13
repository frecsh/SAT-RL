"""
Environment module for SAT reinforcement learning.

This module provides environments for reinforcement learning on SAT problems.
"""

from src.satrlgym.envs.core import SatGymEnv
from src.satrlgym.envs.rewards import (
    dense_reward,
    get_reward_function,
    learning_reward,
    sparse_reward,
)

# Register the environment with Gym
try:
    import gymnasium as gym
    from gymnasium.envs.registration import register

    register(
        id="SatRLGym-v0",
        entry_point="src.satrlgym.envs.core:SatGymEnv",
        kwargs={
            "formula": {
                "clauses": [[1, -2, 3], [-1, 2], [2, -3]],
                "num_vars": 3,
                "name": "default_instance",
            }
        },
    )
except ImportError:
    pass  # Gymnasium not available
