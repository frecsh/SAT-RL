import gymnasium as gym
import numpy as np
import sympy as sp


class SymPyGymEnv(gym.Env):
    """A simple symbolic math environment for SymbolicGym."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        expr_str="x + x + 1",
        reward_mode="sparse",
        render_mode=None,
        max_steps=20,
        seed=None,
    ):
        self.expr = sp.sympify(expr_str)
        self.original_expr = self.expr
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(2)  # 0: expand, 1: factor (for demo)
        self.observation_space = gym.spaces.Box(-1e6, 1e6, shape=(1,), dtype=np.float32)
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.expr = self.original_expr
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        if action == 0:
            self.expr = sp.expand(self.expr)
        elif action == 1:
            self.expr = sp.factor(self.expr)
        done = self.current_step >= self.max_steps
        reward = (
            1.0 if sp.simplify(self.expr) == sp.simplify(self.original_expr) else 0.0
        )
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([float(self.expr.count_ops())], dtype=np.float32)

    def render(self, mode="human"):
        print(f"Current expression: {self.expr}")
