import gymnasium as gym
import numpy as np

try:
    import z3
except ImportError:
    z3 = None


class Z3GymEnv(gym.Env):
    """Feature-complete Z3 constraint satisfaction environment for SymbolicGym."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        constraints=None,
        reward_mode="sparse",
        render_mode=None,
        max_steps=100,
        seed=None,
    ):
        if z3 is None:
            raise ImportError("z3-solver is required for Z3GymEnv")
        self.constraints = constraints or [z3.Bool("x"), z3.Bool("y")]
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(len(self.constraints))
        self.observation_space = gym.spaces.Dict(
            {
                "assignments": gym.spaces.Box(
                    0, 1, shape=(len(self.constraints),), dtype=np.int32
                ),
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
        self.assignments = [0 for _ in self.constraints]
        self._prev_score = self._score(self.assignments)
        obs, info = self._get_obs_info()
        return obs, info

    def step(self, action):
        self.current_step += 1
        # Flip assignment for the selected variable
        self.assignments[action] = 1 - self.assignments[action]
        score = self._score(self.assignments)
        obs, info = self._get_obs_info()
        prev_score = getattr(self, "_prev_score", 0.0)
        if self.reward_mode == "sparse":
            reward = 1.0 if self._is_solved(self.assignments) else 0.0
        elif self.reward_mode == "dense":
            reward = score
        elif self.reward_mode == "learning":
            reward = (score - prev_score) + (
                1.0 if self._is_solved(self.assignments) else 0.0
            )
        else:
            reward = 0.0
        self._prev_score = score
        terminated = self._is_solved(self.assignments)
        truncated = self.current_step >= self.max_steps and not terminated
        return obs, reward, terminated, truncated, info

    def _score(self, assignments):
        # Example: number of satisfied constraints
        s = z3.Solver()
        for i, val in enumerate(assignments):
            s.add(self.constraints[i] if val else z3.Not(self.constraints[i]))
        if s.check() == z3.sat:
            return 1.0
        return 0.0

    def _is_solved(self, assignments):
        return self._score(assignments) == 1.0

    def _get_obs_info(self):
        obs = {
            "assignments": np.array(self.assignments, dtype=np.int32),
            "step": self.current_step,
        }
        info = {
            "assignments": self.assignments[:],
            "step": self.current_step,
            "score": self._score(self.assignments),
        }
        return obs, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print("Assignments:", self.assignments)
        print("Constraints:", self.constraints)
