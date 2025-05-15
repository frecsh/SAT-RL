"""
SymPy transformation action space: list and apply named transformations.
"""
import numpy as np
import sympy as sp
from sympy import expand, factor, simplify

from symbolicgym.core.action_spaces import SymPyTransformAction


def simplify(expr):
    return sp.simplify(expr)


def expand(expr):
    return sp.expand(expr)


def factor(expr):
    return sp.factor(expr)


def available_transformations():
    """Return a list of available transformation names."""
    return ["simplify", "expand", "factor"]


def apply_transformation(expr, action: SymPyTransformAction):
    """Apply the named transformation to the expression."""
    if action.transform_name == "simplify":
        return sp.simplify(expr)
    elif action.transform_name == "expand":
        return sp.expand(expr)
    elif action.transform_name == "factor":
        return sp.factor(expr)
    else:
        raise ValueError(f"Unknown transformation: {action.transform_name}")


action_registry = {
    "simplify": simplify,
    "expand": expand,
    "factor": factor,
}


class SymPyTransformationActionSpace:
    """
    Action space for SymPy transformations: e.g., simplify, expand, factor.
    action = int index of transformation
    """

    TRANSFORMATIONS = [simplify, expand, factor]
    NAMES = ["simplify", "expand", "factor"]

    def __init__(self):
        self.n = len(self.TRANSFORMATIONS)

    def sample(self):
        return np.random.randint(self.n)

    def contains(self, action):
        return 0 <= action < self.n

    def to_human(self, action):
        return self.NAMES[action]

    def apply(self, expr, action):
        return self.TRANSFORMATIONS[action](expr)
