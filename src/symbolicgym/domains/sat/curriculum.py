"""
SAT auto-curriculum for SymbolicGym.
"""
import random

from symbolicgym.curriculum.curriculum_base import CurriculumBase


class SATAutoCurriculum(CurriculumBase):
    """Auto-curriculum for SAT: gradually increases clause/variable ratio (3.0 → 4.2)."""

    def __init__(self, min_ratio=3.0, max_ratio=4.2, step=0.05):
        self.ratio = min_ratio
        self.max_ratio = max_ratio
        self.step = step

    def sample_task(self):
        # Return a random SAT instance with current c/v ratio
        return {"c_v_ratio": self.ratio}

    def update(self, result):
        # Increase ratio if agent solves current level
        if result.get("solved", False) and self.ratio < self.max_ratio:
            self.ratio += self.step

    def get_difficulty(self):
        return self.ratio
