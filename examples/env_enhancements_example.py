"""
Example usage of graph-based state representations, action spaces, and reward system in SymbolicGym.
"""
import sympy as sp

from symbolicgym.core.action_spaces import SATBranchAction, SymPyTransformAction
from symbolicgym.core.rewards import LinearAnnealingRewardShaping
from symbolicgym.domains.sat.actions import pick_branching_action
from symbolicgym.domains.sympy.actions import (
    apply_transformation,
    available_transformations,
)
from symbolicgym.representation.graph_representation import incidence_graph
from symbolicgym.representation.matrix_representation import incidence_coo_matrix
from symbolicgym.representation.networkx_representation import sat_to_networkx

# --- Graph-based state representations ---
num_vars = 3
clauses = [[1, -2], [-1, 3]]
print("Incidence graph nodes:", incidence_graph(num_vars, clauses).nodes)
print("COO matrix:\n", incidence_coo_matrix(num_vars, clauses).toarray())
print("NetworkX graph edges:", sat_to_networkx(num_vars, clauses).edges(data=True))

# --- Action spaces ---
assignments = {1: True}
action = pick_branching_action(assignments, num_vars)
print("SAT branching action:", action.as_tuple())

expr = sp.sympify("(x + 1)**2")
for tname in available_transformations():
    action = SymPyTransformAction(tname)
    result = apply_transformation(expr, action)
    print(f"SymPy action {tname}: {result}")

# --- Reward system ---
shaping = LinearAnnealingRewardShaping(
    dense_weight=1.0, sparse_weight=2.0, anneal_steps=5
)
info = {"dense": 0.5, "sparse": 1.0}
for i in range(7):
    reward = shaping.compute_reward(None, None, None, info)
    print(f"Step {i}: reward={reward:.3f}, weights={shaping.get_shaping_weights()}")
