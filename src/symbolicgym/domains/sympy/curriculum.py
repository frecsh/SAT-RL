"""
SymPy expression complexity curriculum for SymbolicGym.
"""
import random

from symbolicgym.curriculum.curriculum_base import CurriculumBase


class SymPyComplexityCurriculum(CurriculumBase):
    """Curriculum for SymPy: increases expression complexity."""

    def __init__(self, min_terms=2, max_terms=10, step=1):
        self.terms = min_terms
        self.max_terms = max_terms
        self.step = step

    def sample_task(self):
        # Return a random expression with current number of terms
        return {"num_terms": self.terms}

    def update(self, result):
        if result.get("solved", False) and self.terms < self.max_terms:
            self.terms += self.step

    def get_difficulty(self):
        return self.terms
