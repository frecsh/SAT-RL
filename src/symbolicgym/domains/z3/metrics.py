"""Z3 domain metrics for SymbolicGym."""

import time


class Z3Metrics:
    """Stateful Z3 metric collector for episodic RL and interpretability."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.constraints_before = 0
        self.constraints_after = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def set_constraints_before(self, n):
        self.constraints_before = n

    def set_constraints_after(self, n):
        self.constraints_after = n

    def compute(self):
        solver_time = (
            (self.end_time - self.start_time)
            if (self.start_time and self.end_time)
            else None
        )
        reduction_rate = (
            (self.constraints_before - self.constraints_after) / self.constraints_before
            if self.constraints_before
            else 0
        )
        return {"solver_time": solver_time, "constraint_reduction_rate": reduction_rate}


def z3_constraint_reduction_rate(initial_constraints, final_constraints):
    return 1.0 - float(len(final_constraints)) / max(1, len(initial_constraints))


def z3_solver_time(start_time, end_time):
    return end_time - start_time
