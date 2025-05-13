"""
Base environment classes for SAT reinforcement learning.
"""

import gymnasium as gym
import numpy as np


class SATEnv(gym.Env):
    """
    Base class for SAT environments.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "clauses": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
                "variables": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            }
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        observation = {
            "clauses": np.zeros((1,), dtype=np.float32),
            "variables": np.zeros((1,), dtype=np.float32),
        }
        info = {}
        return observation, info

    def step(self, action):
        """Take a step in the environment"""
        observation = {
            "clauses": np.zeros((1,), dtype=np.float32),
            "variables": np.zeros((1,), dtype=np.float32),
        }
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
