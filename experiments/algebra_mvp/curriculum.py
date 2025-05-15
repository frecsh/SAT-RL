# Curriculum generator for Symbolic Algebra MVP RL Experiment

import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from symbolicgym.utils.verifier import is_valid_problem

EASY_PROBLEMS = ["x + 2 = 4", "2*x + 3 = 7", "x + 2 = x + 5"]

MEDIUM_PROBLEMS = ["2*(x + 1) = 8", "3*x - 4 = 2*x + 5", "x/2 + 7 = 10"]
HARD_PROBLEMS = ["x**2 + 2*x + 1 = 0", "x**2 - 4*x + 4 = 0", "x**2 + x = 6"]


class AdaptiveCurriculum:
    def __init__(self, easy, medium, hard, window=10, up_thresh=0.8, down_thresh=0.3):
        self.easy = [p for p in easy if is_valid_problem(p)]
        self.medium = [p for p in medium if is_valid_problem(p)]
        self.hard = [p for p in hard if is_valid_problem(p)]
        self.window = window
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.history = []
        self.level = 0  # 0: easy, 1: medium, 2: hard

    def record(self, solved):
        self.history.append(solved)
        if len(self.history) > self.window:
            self.history.pop(0)
        rate = sum(self.history) / len(self.history)
        if self.level == 0 and rate > self.up_thresh:
            self.level = 1
        elif self.level == 1:
            if rate > self.up_thresh:
                self.level = 2
            elif rate < self.down_thresh:
                self.level = 0
        elif self.level == 2 and rate < self.down_thresh:
            self.level = 1

    def sample(self):
        if self.level == 0:
            return random.choice(self.easy)
        elif self.level == 1:
            return random.choice(self.medium)
        else:
            return random.choice(self.hard)


# Singleton curriculum instance
_curric = AdaptiveCurriculum(EASY_PROBLEMS, MEDIUM_PROBLEMS, HARD_PROBLEMS)


def get_problem():
    return _curric.sample()


def record_result(solved):
    _curric.record(solved)
