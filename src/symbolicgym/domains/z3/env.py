import gymnasium as gym
import numpy as np
import z3


class Z3GymEnv(gym.Env):
    """A simple Z3 constraint satisfaction environment for SymbolicGym."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        constraints=None,
        reward_mode="sparse",
        render_mode=None,
        max_steps=20,
        seed=None,
    ):
        # Example: constraints = [z3.Int('x') + 1 == 2]
        self.constraints = constraints or [z3.Int("x") + 1 == 2]
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.solver = z3.Solver()
        for c in self.constraints:
            self.solver.add(c)
        self.action_space = gym.spaces.Discrete(2)  # 0: add x==1, 1: add x==2 (demo)
        self.observation_space = gym.spaces.Box(-1e6, 1e6, shape=(1,), dtype=np.float32)
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.solver = z3.Solver()
        for c in self.constraints:
            self.solver.add(c)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        x = z3.Int("x")
        if action == 0:
            self.solver.add(x == 1)
        elif action == 1:
            self.solver.add(x == 2)
        done = self.current_step >= self.max_steps or self.solver.check() == z3.sat
        reward = 1.0 if self.solver.check() == z3.sat else 0.0
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # For demo: number of constraints
        return np.array([len(self.solver.assertions())], dtype=np.float32)

    def render(self, mode="human"):
        print(f"Current constraints: {self.solver.assertions()}")
