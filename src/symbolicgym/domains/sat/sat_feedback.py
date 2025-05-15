"""
SATFeedback: SymbolicFeedback implementation for SAT domain in SymbolicGym.
"""
from symbolicgym.core.feedback_vectors import SymbolicFeedbackVector
from symbolicgym.core.symbolic_feedback import SymbolicFeedback


class SATFeedback(SymbolicFeedback):
    """Symbolic feedback backend for SAT problems."""

    def init_state(self, num_vars=3, clauses=None):
        if clauses is None:
            clauses = [[1, -2], [-1, 3]]
        return {"assignments": {}, "clauses": clauses, "num_vars": num_vars}

    def apply_action(self, state, action):
        # Action: (variable, value)
        var, val = action
        state = dict(state)  # shallow copy
        state["assignments"] = dict(state["assignments"])
        state["assignments"][var] = val
        return state

    def get_feedback(self, state):
        assignments = state["assignments"]
        clauses = state["clauses"]
        num_vars = state["num_vars"]
        # Clause satisfaction: fraction of satisfied clauses
        satisfied = 0
        for clause in clauses:
            if any(
                (lit > 0 and assignments.get(abs(lit), None) is True)
                or (lit < 0 and assignments.get(abs(lit), None) is False)
                for lit in clause
            ):
                satisfied += 1
        clause_satisfaction = satisfied / len(clauses) if clauses else 1.0
        # Variable decisiveness: fraction of assigned variables
        variable_decisiveness = len(assignments) / num_vars if num_vars else 1.0
        vec = SymbolicFeedbackVector.create_sat_feedback()
        vec.set_value("clause_satisfaction", clause_satisfaction)
        vec.set_value("variable_decisiveness", variable_decisiveness)
        # Set other dimensions to 0 for now
        for dim in ["search_diversity", "constraint_tension", "proof_progress"]:
            vec.set_value(dim, 0.0)
        return vec
