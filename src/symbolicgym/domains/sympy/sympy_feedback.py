"""SymPyFeedback: SymbolicFeedback implementation for SymPy domain in SymbolicGym."""

import sympy as sp

from symbolicgym.core.feedback_vectors import SymbolicFeedbackVector
from symbolicgym.core.symbolic_feedback import SymbolicFeedback


class SymPyFeedback(SymbolicFeedback):
    """Symbolic feedback backend for symbolic algebra tasks (SymPy)."""

    def init_state(self, expr="x**2 + 2*x + 1", target="(x + 1)**2"):
        sp.symbols("x")
        return {"expr": sp.sympify(expr), "target": sp.sympify(target)}

    def apply_action(self, state, action):
        # Action: a function that takes and returns a sympy expression
        state = dict(state)
        state["expr"] = action(state["expr"])
        return state

    def get_feedback(self, state):
        expr = state["expr"]
        target = state["target"]
        # Expression complexity: number of nodes in the expression tree
        complexity = sp.count_ops(expr)
        # Solution progress: 1 if expr == target, else 0
        progress = 1.0 if sp.simplify(expr - target) == 0 else 0.0
        vec = SymbolicFeedbackVector.create_sympy_feedback()
        vec.set_value("expression_complexity", float(complexity))
        vec.set_value("solution_progress", progress)
        # Set other dimensions to 0 for now
        for dim in ["term_reduction", "pattern_matching", "algebraic_insight"]:
            vec.set_value(dim, 0.0)
        return vec
