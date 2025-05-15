"""
Abstract curriculum interface for SymbolicGym.
"""
from abc import ABC, abstractmethod


class CurriculumBase(ABC):
    """Domain-agnostic curriculum interface for cross-domain learning."""

    @abstractmethod
    def sample_task(self):
        pass

    @abstractmethod
    def update(self, result):
        pass

    @abstractmethod
    def get_difficulty(self):
        pass
