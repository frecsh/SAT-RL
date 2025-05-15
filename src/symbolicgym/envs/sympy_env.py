import gymnasium as gym
import numpy as np
import sympy as sp


class SymPyGymEnv(gym.Env):
    """Feature-complete SymPy transformation environment for SymbolicGym."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        expr=None,
        reward_mode="sparse",
        render_mode=None,
        max_steps=100,
        seed=None,
    ):
        self.expr = expr or sp.sympify("x + 1")
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(3)  # e.g., expand, factor, simplify
        self.observation_space = gym.spaces.Dict(
            {
                "expr": gym.spaces.Box(
                    -1e6, 1e6, shape=(1,), dtype=np.float32
                ),  # Placeholder
                "step": gym.spaces.Discrete(max_steps + 1),
            }
        )
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.current_expr = self.expr
        self._prev_score = self._score(self.current_expr)
        obs, info = self._get_obs_info()
        return obs, info

    def step(self, action):
        self.current_step += 1
        # Map action to transformation
        if action == 0:
            self.current_expr = sp.expand(self.current_expr)
        elif action == 1:
            self.current_expr = sp.factor(self.current_expr)
        elif action == 2:
            self.current_expr = sp.simplify(self.current_expr)
        score = self._score(self.current_expr)
        obs, info = self._get_obs_info()
        prev_score = getattr(self, "_prev_score", 0.0)
        if self.reward_mode == "sparse":
            reward = 1.0 if self._is_solved(self.current_expr) else 0.0
        elif self.reward_mode == "dense":
            reward = score
        elif self.reward_mode == "learning":
            reward = (score - prev_score) + (
                1.0 if self._is_solved(self.current_expr) else 0.0
            )
        else:
            reward = 0.0
        self._prev_score = score
        terminated = self._is_solved(self.current_expr)
        truncated = self.current_step >= self.max_steps and not terminated
        return obs, reward, terminated, truncated, info

    def _score(self, expr):
        # Example: negative of expression complexity (smaller is better)
        return -sp.count_ops(expr)

    def _is_solved(self, expr):
        # Example: solved if expression is fully simplified (no ops)
        return sp.count_ops(expr) == 1

    def _get_obs_info(self):
        obs = {
            "expr": np.array(
                [float(sp.count_ops(self.current_expr))], dtype=np.float32
            ),
            "step": self.current_step,
        }
        info = {
            "expr_str": str(self.current_expr),
            "step": self.current_step,
            "score": self._score(self.current_expr),
        }
        return obs, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print("Expr:", self.current_expr)
