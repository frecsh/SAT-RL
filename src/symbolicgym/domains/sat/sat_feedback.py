"""
SATFeedback: SymbolicFeedback implementation for SAT domain in SymbolicGym.
"""
import numpy as np

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

    def clause_is_satisfied(self, clause, assignments):
        return any(
            (lit > 0 and assignments.get(abs(lit), None) is True)
            or (lit < 0 and assignments.get(abs(lit), None) is False)
            for lit in clause
        )

    def get_observation_dict(self, state):
        # Return a dict observation (default for SAT)
        return {
            "clauses": np.array(
                [
                    1.0
                    if self.clause_is_satisfied(clause, state["assignments"])
                    else 0.0
                    for clause in state["clauses"]
                ],
                dtype=np.float32,
            ),
            "variables": np.array(
                [
                    1.0 if state["assignments"].get(i + 1, 0) == 1 else -1.0
                    for i in range(state["num_vars"])
                ]
            ),
            "variable_assignment": {
                str(i + 1): int(state["assignments"].get(i + 1, 0))
                for i in range(state["num_vars"])
            },
        }

    def get_observation_flat(self, state):
        # Return a flat vector observation
        obs_dict = self.get_observation_dict(state)
        flat = []
        for v in obs_dict.values():
            if isinstance(v, dict):
                flat.extend(list(v.values()))
            elif isinstance(v, np.ndarray):
                flat.extend(v.flatten())
            else:
                flat.append(v)
        return np.array(flat, dtype=np.float32)

    def get_observation_graph(self, state):
        # Placeholder: return a tuple (nodes, edges) for GNNs
        # nodes: variable assignments, edges: clause-variable relations
        nodes = np.array(
            [state["assignments"].get(i + 1, 0) for i in range(state["num_vars"])]
        ).reshape(-1, 1)
        edges = []
        for ci, clause in enumerate(state["clauses"]):
            for lit in clause:
                vi = abs(lit) - 1
                edges.append((vi, ci))
        return {"nodes": nodes, "edges": np.array(edges)}
