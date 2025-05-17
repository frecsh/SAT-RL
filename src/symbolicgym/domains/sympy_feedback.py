"""SymPy feedback backend for SymbolicGym, supporting shared observation API."""

import numpy as np


class SymPyFeedback:
    def init_state(self, expr=None):
        # expr: symbolic expression (placeholder)
        return {"expr": expr, "steps": 0}

    def apply_action(self, state, action):
        # Apply a symbolic transformation (placeholder)
        state = dict(state)
        state["steps"] += 1
        return state

    def get_reward(self, state):
        # Placeholder: reward for simplification
        return 1.0 if state["steps"] > 0 else 0.0

    def is_done(self, state):
        # Placeholder: done if steps > 0
        return state["steps"] > 0

    def get_info(self, state):
        return {"steps": state["steps"]}

    def get_observation_dict(self, state):
        return {"expr": state["expr"], "steps": state["steps"]}

    def get_observation_flat(self, state):
        # Placeholder: flatten steps only
        return np.array([state["steps"]], dtype=np.float32)

    def get_observation_graph(self, state):
        # Placeholder: return expr as node
        return {"nodes": np.array([state["expr"]]), "edges": np.zeros((0, 2))}
