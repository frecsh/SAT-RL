"""
SAT branching decision action head: pick variable and polarity.
"""
import numpy as np

from symbolicgym.core.action_spaces import SATBranchAction


def pick_branching_action(assignments, num_vars):
    """Pick the next unassigned variable and a polarity (True/False)."""
    for v in range(1, num_vars + 1):
        if v not in assignments:
            return SATBranchAction(v, True)  # Always pick True for demo
    return None  # All assigned


class SATBranchingActionSpace:
    """
    Action space for SAT branching: pick variable and polarity.
    action = (var_idx, polarity) where polarity in {0,1}
    """

    def __init__(self, num_vars):
        self.num_vars = num_vars

    def sample(self):
        return (np.random.randint(self.num_vars), np.random.randint(2))

    def contains(self, action):
        var, pol = action
        return 0 <= var < self.num_vars and pol in (0, 1)

    def to_human(self, action):
        var, pol = action
        return f"Flip variable v{var+1} to {'True' if pol else 'False'}"
