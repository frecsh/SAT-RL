"""
Unified symbolic action interface for all domains.
"""
from abc import ABC, abstractmethod


class SymbolicActionSpace(ABC):
    """Unified symbolic action interface for all domains."""

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def contains(self, action):
        pass

    @abstractmethod
    def to_human(self, action):
        pass


class SymbolicAction(ABC):
    """Abstract base class for symbolic actions."""

    @abstractmethod
    def as_tuple(self):
        """Return a tuple representation of the action."""
        pass


class SATBranchAction(SymbolicAction):
    """SAT branching decision: pick variable and polarity."""

    def __init__(self, variable, polarity):
        self.variable = variable  # int
        self.polarity = polarity  # bool (True=positive, False=negative)

    def as_tuple(self):
        return (self.variable, self.polarity)


class SymPyTransformAction(SymbolicAction):
    """SymPy transformation: apply a named transformation."""

    def __init__(self, transform_name):
        self.transform_name = transform_name

    def as_tuple(self):
        return (self.transform_name,)
