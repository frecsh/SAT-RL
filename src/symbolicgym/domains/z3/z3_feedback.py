"""
Z3Feedback: SymbolicFeedback stub for SMT (Z3) domain in SymbolicGym.
"""
from symbolicgym.core.feedback_vectors import SymbolicFeedbackVector
from symbolicgym.core.symbolic_feedback import SymbolicFeedback


class Z3Feedback(SymbolicFeedback):
    """Symbolic feedback backend for SMT (Z3) tasks."""

    def init_state(self, constraints=None, assignments=None):
        if constraints is None:
            constraints = ["x > 0", "x < 5"]
        if assignments is None:
            assignments = {}
        return {"constraints": constraints, "assignments": assignments}

    def apply_action(self, state, action):
        # Action: (variable, value)
        var, val = action
        state = dict(state)
        state["assignments"] = dict(state["assignments"])
        state["assignments"][var] = val
        return state

    def get_feedback(self, state):
        # For demonstration, treat constraints as Python expressions
        assignments = state["assignments"]
        constraints = state["constraints"]
        satisfied = 0
        for c in constraints:
            try:
                if eval(c, {}, assignments):
                    satisfied += 1
            except Exception:
                pass
        constraint_satisfaction = satisfied / len(constraints) if constraints else 1.0
        # Dummy theory combination: 1.0 if more than one variable assigned
        theory_combination = 1.0 if len(assignments) > 1 else 0.0
        vec = SymbolicFeedbackVector.create_z3_feedback()
        vec.set_value("constraint_satisfaction", constraint_satisfaction)
        vec.set_value("theory_combination", theory_combination)
        # Set other dimensions to 0 for now
        for dim in ["decision_level", "conflict_density", "lemma_quality"]:
            vec.set_value(dim, 0.0)
        return vec
