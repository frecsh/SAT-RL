"""
SymbolicFeedback interface for pluggable domain feedback in SymbolicGym.
Defines the contract for state initialization, action application, and feedback extraction.
"""

from abc import ABC, abstractmethod


class SymbolicFeedback(ABC):
    """Interface for symbolic feedback backends."""

    @abstractmethod
    def init_state(self, *args, **kwargs):
        """Initialize and return the initial symbolic state."""
        pass

    @abstractmethod
    def apply_action(self, state, action):
        """Apply an action to the state and return the new state."""
        pass

    @abstractmethod
    def get_feedback(self, state):
        """Extract a feedback vector from the current state."""
        pass
