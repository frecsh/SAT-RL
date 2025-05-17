"""Abstract base environment for cross-domain symbolic RL.
Supports graph and vector representations, and pluggable backend logic.
"""

import gymnasium as gym


class AbstractSymbolicEnv(gym.Env):
    """Abstract environment for SAT, SymPy, Z3, etc.
    Provides unified observation builder and backend delegation.
    """

    def __init__(self, backend=None, observation_mode="dict", **kwargs):
        self.backend = backend  # e.g., SATFeedback, SymPyFeedback, etc.
        self.observation_mode = observation_mode  # "dict", "flat", "graph"
        self._init_backend(**kwargs)

    def _init_backend(self, **kwargs):
        if self.backend is not None:
            self.state = self.backend.init_state(**kwargs)
        else:
            self.state = None

    def reset(self, **kwargs):
        self._init_backend(**kwargs)
        obs = self.build_observation(self.state)
        return obs, {}

    def step(self, action):
        self.state = self.backend.apply_action(self.state, action)
        obs = self.build_observation(self.state)
        reward = self.backend.get_reward(self.state)
        done = self.backend.is_done(self.state)
        info = self.backend.get_info(self.state)
        return obs, reward, done, False, info

    def build_observation(self, state):
        if self.observation_mode == "dict":
            return self.backend.get_observation_dict(state)
        elif self.observation_mode == "flat":
            return self.backend.get_observation_flat(state)
        elif self.observation_mode == "graph":
            return self.backend.get_observation_graph(state)
        else:
            raise ValueError(f"Unknown observation_mode: {self.observation_mode}")
