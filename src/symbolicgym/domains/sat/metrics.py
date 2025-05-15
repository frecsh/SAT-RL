"""
SAT domain metrics for SymbolicGym.
"""
import time


class SATMetrics:
    """Stateful SAT metric collector for episodic RL and interpretability."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.conflicts = 0
        self.flips = 0
        self.solved = False

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.solved = True

    def add_conflict(self, n=1):
        self.conflicts += n

    def add_flip(self, n=1):
        self.flips += n

    def compute(self):
        time_to_solution = (
            (self.end_time - self.start_time)
            if (self.start_time and self.end_time)
            else None
        )
        flips_per_sat = self.flips if self.solved else None
        return {
            "time_to_solution": time_to_solution,
            "conflicts": self.conflicts,
            "flips_per_sat": flips_per_sat,
        }


def sat_time_to_solution(start_time, end_time):
    return end_time - start_time


def sat_conflicts(conflict_count):
    return conflict_count


def sat_flips_per_sat(num_flips, solved):
    return num_flips if solved else None
