import gymnasium as gym
import numpy as np


class SatGymEnv(gym.Env):
    """Feature-complete SAT environment for SymbolicGym."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        formula=None,
        reward_mode="sparse",
        render_mode=None,
        max_steps=100,
        seed=None,
    ):
        self.formula = formula or {"num_vars": 3, "clauses": [[1, 2], [-1, 2], [1, -2]]}
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.num_vars = self.formula["num_vars"]
        self.clauses = self.formula["clauses"]
        self.num_clauses = len(self.clauses)
        self.action_space = gym.spaces.Discrete(self.num_vars)
        self.observation_space = gym.spaces.Dict(
            {
                "clauses": gym.spaces.Box(
                    -self.num_vars,
                    self.num_vars,
                    shape=(self.num_clauses,),
                    dtype=np.float32,
                ),
                "variables": gym.spaces.Box(
                    -1, 1, shape=(self.num_vars,), dtype=np.float32
                ),
                "variable_assignment": gym.spaces.Dict(
                    {str(i + 1): gym.spaces.Discrete(2) for i in range(self.num_vars)}
                ),
                "clause_satisfaction": gym.spaces.Box(
                    0, 1, shape=(self.num_clauses,), dtype=np.float32
                ),
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
        # Random initial assignment
        self.variable_assignment = {
            i + 1: self.np_random.integers(0, 2) for i in range(self.num_vars)
        }
        self._prev_satisfaction_ratio = 0.0
        obs, info = self._get_obs_info()
        return obs, info

    def step(self, action):
        self.current_step += 1
        var = action + 1
        self.variable_assignment[var] = 1 - self.variable_assignment[var]
        obs, info = self._get_obs_info()
        prev_ratio = getattr(self, "_prev_satisfaction_ratio", 0.0)
        satisfaction_ratio = info["satisfaction_ratio"]
        solved = info["solved"]

        if self.reward_mode == "sparse":
            reward = 1.0 if solved else 0.0
        elif self.reward_mode == "dense":
            reward = satisfaction_ratio
        elif self.reward_mode == "learning":
            reward = (satisfaction_ratio - prev_ratio) + (1.0 if solved else 0.0)
        else:
            reward = 0.0

        self._prev_satisfaction_ratio = satisfaction_ratio
        terminated = solved
        truncated = self.current_step >= self.max_steps and not terminated
        return obs, reward, terminated, truncated, info

    def _get_obs_info(self):
        # Clause satisfaction
        def clause_is_satisfied(clause, assignment):
            return any(
                (assignment[abs(lit)] == 1 if lit > 0 else assignment[abs(lit)] == 0)
                for lit in clause
            )

        satisfied_clauses = [
            clause_is_satisfied(clause, self.variable_assignment)
            for clause in self.clauses
        ]
        satisfaction_ratio = (
            float(sum(satisfied_clauses)) / self.num_clauses
            if self.num_clauses
            else 0.0
        )
        solved = all(satisfied_clauses)
        clause_satisfaction = np.array(
            [1.0 if sat else 0.0 for sat in satisfied_clauses], dtype=np.float32
        )
        obs = {
            "clauses": clause_satisfaction.copy(),
            "variables": np.array(
                [
                    1.0 if self.variable_assignment[i + 1] == 1 else -1.0
                    for i in range(self.num_vars)
                ],
                dtype=np.float32,
            ),
            "variable_assignment": {
                str(k): int(v) for k, v in self.variable_assignment.items()
            },
            "clause_satisfaction": clause_satisfaction.copy(),
        }
        info = {
            "num_satisfied_clauses": sum(satisfied_clauses),
            "total_clauses": self.num_clauses,
            "satisfaction_ratio": satisfaction_ratio,
            "solved": solved,
            "satisfied_clauses": satisfied_clauses,
            "step": self.current_step,
            "assignment": self.variable_assignment.copy(),
        }
        return obs, info

    def compute_satisfaction(self, variable_assignment):
        def clause_is_satisfied(clause, assignment):
            return any(
                (assignment[abs(lit)] == 1 if lit > 0 else assignment[abs(lit)] == 0)
                for lit in clause
            )

        satisfied_clauses = [
            clause_is_satisfied(clause, variable_assignment) for clause in self.clauses
        ]
        satisfaction_ratio = (
            float(sum(satisfied_clauses)) / self.num_clauses
            if self.num_clauses
            else 0.0
        )
        return satisfied_clauses, satisfaction_ratio

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print("Assignment:", self.variable_assignment)
        print("Clauses:", self.clauses)
