"""Abstract base class for all symbolic RL environments in SymbolicGym.
Defines the interface for state, action, and feedback integration.
"""

from abc import ABC, abstractmethod


class AbstractSymbolicEnv(ABC):
    """Abstract base class for symbolic RL environments."""

    @abstractmethod
    def reset(self, seed=None):
        """Reset the environment to an initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Apply an action and return (observation, reward, done, info)."""
        pass

    @abstractmethod
    def get_feedback(self):
        """Return the current symbolic feedback vector."""
        pass
