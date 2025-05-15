"""
Multi-dimensional feedback vectors for symbolic reasoning domains.

This module provides utilities for constructing and manipulating rich latent-space
feedback vectors that capture multiple aspects of symbolic reasoning progress.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FeedbackDimension:
    """Represents a single dimension in the feedback vector."""

    name: str
    description: str
    range: tuple[float, float]
    is_normalized: bool = True


class SymbolicFeedbackVector:
    """Rich latent-space feedback vector for symbolic reasoning domains."""

    def __init__(self, dimensions: List[FeedbackDimension]):
        """Initialize feedback vector with specified dimensions.

        Args:
            dimensions: List of FeedbackDimension objects defining the vector space
        """
        self.dimensions = dimensions
        self._values = np.zeros(len(dimensions))
        self._dim_map = {dim.name: i for i, dim in enumerate(dimensions)}

    def set_value(self, dim_name: str, value: float):
        """Set the value for a specific feedback dimension."""
        if dim_name not in self._dim_map:
            raise KeyError(f"Unknown feedback dimension: {dim_name}")

        dim_idx = self._dim_map[dim_name]
        dim = self.dimensions[dim_idx]

        # Normalize value if needed
        if dim.is_normalized:
            min_val, max_val = dim.range
            value = (value - min_val) / (max_val - min_val)

        self._values[dim_idx] = value

    def get_value(self, dim_name: str) -> float:
        """Get the value of a specific feedback dimension."""
        if dim_name not in self._dim_map:
            raise KeyError(f"Unknown feedback dimension: {dim_name}")
        return self._values[self._dim_map[dim_name]]

    def as_array(self) -> np.ndarray:
        """Return the feedback vector as a numpy array."""
        return self._values.copy()

    def as_dict(self) -> Dict[str, float]:
        """Return the feedback vector as a dictionary mapping dimension names to values."""
        return {dim.name: self._values[i] for i, dim in enumerate(self.dimensions)}

    @classmethod
    def create_sat_feedback(cls) -> "SymbolicFeedbackVector":
        """Create a feedback vector for SAT domain with standard dimensions."""
        dimensions = [
            FeedbackDimension(
                "clause_satisfaction", "Percentage of satisfied clauses", (0.0, 1.0)
            ),
            FeedbackDimension(
                "variable_decisiveness",
                "How decisive variable assignments are",
                (0.0, 1.0),
            ),
            FeedbackDimension(
                "search_diversity", "Diversity of search space exploration", (0.0, 1.0)
            ),
            FeedbackDimension(
                "constraint_tension",
                "Tension between competing constraints",
                (-1.0, 1.0),
            ),
            FeedbackDimension(
                "proof_progress", "Progress towards UNSAT proof", (0.0, 1.0)
            ),
        ]
        return cls(dimensions)

    @classmethod
    def create_sympy_feedback(cls) -> "SymbolicFeedbackVector":
        """Create a feedback vector for SymPy domain with standard dimensions."""
        dimensions = [
            FeedbackDimension(
                "expression_complexity",
                "Complexity of current expression",
                (0.0, float("inf")),
                False,
            ),
            FeedbackDimension(
                "solution_progress", "Progress towards target form", (0.0, 1.0)
            ),
            FeedbackDimension(
                "term_reduction", "Success in term reduction", (0.0, 1.0)
            ),
            FeedbackDimension(
                "pattern_matching", "Success in pattern matching", (0.0, 1.0)
            ),
            FeedbackDimension(
                "algebraic_insight", "Measure of algebraic insight", (0.0, 1.0)
            ),
        ]
        return cls(dimensions)

    @classmethod
    def create_z3_feedback(cls) -> "SymbolicFeedbackVector":
        """Create a feedback vector for Z3 domain with standard dimensions."""
        dimensions = [
            FeedbackDimension(
                "constraint_satisfaction",
                "Degree of constraint satisfaction",
                (0.0, 1.0),
            ),
            FeedbackDimension(
                "theory_combination", "Success in theory combination", (0.0, 1.0)
            ),
            FeedbackDimension(
                "decision_level", "Current decision level", (0.0, float("inf")), False
            ),
            FeedbackDimension(
                "conflict_density", "Density of conflicts encountered", (0.0, 1.0)
            ),
            FeedbackDimension("lemma_quality", "Quality of learned lemmas", (0.0, 1.0)),
        ]
        return cls(dimensions)
