"""
SATRLGym: A reinforcement learning environment for SAT solving.
"""

from satrlgym import environments, experience, utils

__version__ = "0.1.0"

# Import and register all environments
from gymnasium.envs.registration import register

register(
    id="SatRLGym-v0",
    entry_point="satrlgym.environments.base_env:SATEnv",
    max_episode_steps=1000,
)

# Import main package modules
