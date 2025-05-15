"""
Abstract metric collector interface for SymbolicGym.
"""
from abc import ABC, abstractmethod


class MetricCollector(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass

    # For backward compatibility
    def collect(self, *args, **kwargs):
        self.update(*args, **kwargs)
