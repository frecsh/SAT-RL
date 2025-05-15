"""
SymPy baselines for SymbolicGym.
"""
from sympy import Eq, expand, factor, simplify


def baseline_simplify(expr):
    return simplify(expr)


def baseline_expand(expr):
    return expand(expr)


def baseline_factor(expr):
    return factor(expr)


def baseline_prove(equation):
    # Returns True if equation is an identity
    return Eq(equation.lhs, equation.rhs).simplify() == True
