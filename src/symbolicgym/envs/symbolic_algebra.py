import re

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sympy import sympify

from src.symbolicgym.utils.feedback import get_symbolic_feedback


def tokenize_expression(expr_str):
    # Simple tokenizer: split on operators, parens, numbers, variables
    tokens = re.findall(r"[A-Za-z]+|\d+|[=+\-*/^()]", expr_str)
    return tokens


class SymbolicAlgebraEnv(gym.Env):
    """Symbolic Algebra Environment for RL (MVP)"""

    def __init__(self, max_steps=20, initial_expression="x + 2 = 4"):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.initial_expression = initial_expression
        self.action_space = spaces.Discrete(10)  # Placeholder: number of rewrite rules
        # Set max_length for Text and Sequence spaces
        max_expr_len = 64
        max_tokens = 16
        self.observation_space = spaces.Dict(
            {
                "expression": spaces.Text(max_length=max_expr_len),
                "tokens": spaces.Sequence(
                    spaces.Text(max_length=16), max_length=max_tokens
                ),
                "feedback": spaces.Box(low=-100, high=100, shape=(4,), dtype=float),
            }
        )
        self.state = None

    def reset(self, *, seed=None, options=None, expression=None):
        self.current_step = 0
        expr = expression or self.initial_expression
        feedback = get_symbolic_feedback(expr)
        tokens = tokenize_expression(str(expr))
        self.state = {
            "expression": str(expr),
            "tokens": tokens,
            "feedback": np.array(list(feedback.values()), dtype=float),
        }
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        # Placeholder: apply rewrite rule (simulate progress)
        # For demo, after 3 steps, mark as solved
        if self.current_step >= 3:
            self.state["feedback"][0] = 1  # is_solved
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
        info = {}
        return self.state, reward, done, False, info

    def render(self, mode="human"):
        print(f"Current expression: {self.state['expression']}")


# Registration (in __init__.py or here for MVP)
gym.envs.registration.register(
    id="SymbolicAlgebra-v0",
    entry_point="symbolicgym.envs.symbolic_algebra:SymbolicAlgebraEnv",
)
