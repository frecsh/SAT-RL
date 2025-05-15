from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract agent interface for cross-domain RL."""

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def observe(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
