"""
Representation module for SAT-RL.

This module contains classes and utilities for representing SAT problems
in formats suitable for reinforcement learning.
"""

from satrlgym.representation.basic_representation import (
    ClauseSatisfactionEncoder,
    HistoryTracker,
    ModularObservationPreprocessor,
    ObservationEncoder,
    VariableAssignmentEncoder,
)
from satrlgym.representation.gnn_representation import (
    GNNObservationEncoder,
    GraphBuilder,
    MessagePassingLayer,
    SATGraphNN,
)

__all__ = [
    # Basic representation components
    "ModularObservationPreprocessor",
    "VariableAssignmentEncoder",
    "ClauseSatisfactionEncoder",
    "HistoryTracker",
    "ObservationEncoder",
    # GNN representation components
    "GraphBuilder",
    "SATGraphNN",
    "MessagePassingLayer",
    "GNNObservationEncoder",
]
