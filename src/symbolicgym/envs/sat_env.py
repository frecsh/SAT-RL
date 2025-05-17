import gymnasium as gym
import numpy as np

from symbolicgym.domains.sat_feedback import SATFeedback
from symbolicgym.envs.base_env import BaseEnv


class SatEnv(BaseEnv):
    """Feature-complete SAT environment for SymbolicGym.

    Feedback metric descriptions:
        - clause_satisfaction: Fraction of SAT clauses currently satisfied by the agent’s assignment.
        - variable_decisiveness: Fraction of variables that have been assigned a value (measures how “decided” the agent is).
        - search_diversity: The standard deviation of the variable assignments. High diversity means the agent is exploring a variety of assignments, while low diversity means assignments are similar (less exploration).
        - constraint_tension: The average absolute sum of literals in each clause. This can reflect how “tight” or “conflicted” the current formula is under the agent’s assignment.
        - proof_progress: A composite metric: clause_satisfaction * variable_decisiveness. It measures how much progress the agent has made toward a full, satisfying assignment.

    These metrics help analyze not just whether agents are solving the SAT, but how they are exploring, coordinating, and progressing.

    Example usage for feedback-in-reward:
    env = SatEnv(feedback_in_reward=True, feedback_reward_scale=0.05,
                    feedback_metrics_to_use=['clause_satisfaction','proof_progress'],
                    feedback_metric_weights=[0.7, 0.3])
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        formula=None,
        reward_mode="sparse",
        render_mode=None,
        max_steps=100,
        seed=None,
        n_agents=1,
        multi_agent_mode=False,
        agent_ids=None,
        observation_mode="dict",
        backend=None,
        feedback_in_reward=False,
        feedback_reward_scale=0.01,
        feedback_metrics_to_use=None,
        feedback_metric_weights=None,
        debug=False,
    ):
        super().__init__(config=None, debug=debug)
        self.formula = formula or {"num_vars": 3, "clauses": [[1, 2], [-1, 2], [1, -2]]}
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.num_vars = self.formula["num_vars"]
        self.clauses = self.formula["clauses"]
        self.num_clauses = len(self.clauses)
        self.n_agents = n_agents
        self.multi_agent_mode = multi_agent_mode
        self.agent_ids = agent_ids or [f"agent_{i}" for i in range(n_agents)]
        self.observation_mode = observation_mode
        self.backend = backend
        self.feedback_in_reward = feedback_in_reward
        self.feedback_reward_scale = feedback_reward_scale
        self.feedback_metrics_to_use = feedback_metrics_to_use or [
            "clause_satisfaction",
            "variable_decisiveness",
            "search_diversity",
            "constraint_tension",
            "proof_progress",
        ]
        self.feedback_metric_weights = feedback_metric_weights or [1.0] * len(
            self.feedback_metrics_to_use
        )
        if self.multi_agent_mode:
            self.action_space = gym.spaces.Dict(
                {aid: gym.spaces.Discrete(self.num_vars) for aid in self.agent_ids}
            )
            self.observation_space = gym.spaces.Dict(
                {
                    aid: gym.spaces.Dict(
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
                                {
                                    str(i + 1): gym.spaces.Discrete(2)
                                    for i in range(self.num_vars)
                                }
                            ),
                            "clause_satisfaction": gym.spaces.Box(
                                0, 1, shape=(self.num_clauses,), dtype=np.float32
                            ),
                            "feedback_vector": gym.spaces.Box(
                                -np.inf, np.inf, shape=(8,), dtype=np.float32
                            ),
                        }
                    )
                    for aid in self.agent_ids
                }
            )
        else:
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
                        {
                            str(i + 1): gym.spaces.Discrete(2)
                            for i in range(self.num_vars)
                        }
                    ),
                    "clause_satisfaction": gym.spaces.Box(
                        0, 1, shape=(self.num_clauses,), dtype=np.float32
                    ),
                    "feedback_vector": gym.spaces.Box(
                        -np.inf, np.inf, shape=(8,), dtype=np.float32
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
        if self.multi_agent_mode:
            self.variable_assignment = {
                aid: {
                    i + 1: self.np_random.integers(0, 2) for i in range(self.num_vars)
                }
                for aid in self.agent_ids
            }
            self._prev_satisfaction_ratio = {aid: 0.0 for aid in self.agent_ids}
            obs = {}
            info = {}
            for aid in self.agent_ids:
                obs[aid], info[aid] = self._get_obs_info(self.variable_assignment[aid])
                feedback = SATFeedback().get_feedback(
                    self.variable_assignment[aid], self.clauses, self.num_vars
                )
                obs[aid]["feedback_vector"] = np.array(
                    list(feedback.values()), dtype=np.float32
                )
            if self.debug:
                self.log_debug_info()
            return obs, info
        else:
            self.variable_assignment = {
                i + 1: self.np_random.integers(0, 2) for i in range(self.num_vars)
            }
            self._prev_satisfaction_ratio = 0.0
            obs, info = self._get_obs_info(self.variable_assignment)
            feedback = SATFeedback().get_feedback(
                self.variable_assignment, self.clauses, self.num_vars
            )
            obs["feedback_vector"] = np.array(list(feedback.values()), dtype=np.float32)
            if self.debug:
                self.log_debug_info()
            return obs, info

    def step(self, action, messages=None):
        comm_penalty = 0.0
        feedback_vectors = {}
        if self.multi_agent_mode and messages is not None:
            for aid, msg in messages.items():
                comm_penalty += len(str(msg)) * 0.001
        obs, reward, terminated, truncated, info = None, None, None, None, None
        if self.multi_agent_mode:
            obs, reward, terminated, truncated, info = {}, {}, {}, {}, {}
            for aid in self.agent_ids:
                act = action[aid] if isinstance(action, dict) else action
                var = act + 1
                self.variable_assignment[aid][var] = (
                    1 - self.variable_assignment[aid][var]
                )
                obs[aid], info[aid], feedback_vectors[aid] = self._get_obs_info(
                    self.variable_assignment[aid], return_feedback=True
                )
                feedback = SATFeedback().get_feedback(
                    self.variable_assignment[aid], self.clauses, self.num_vars
                )
                obs[aid]["feedback_vector"] = np.array(
                    list(feedback.values()), dtype=np.float32
                )
                prev_ratio = self._prev_satisfaction_ratio.get(aid, 0.0)
                satisfaction_ratio = info[aid]["satisfaction_ratio"]
                solved = info[aid]["solved"]
                if self.reward_mode == "sparse":
                    reward[aid] = 1.0 if solved else 0.0
                elif self.reward_mode == "dense":
                    reward[aid] = satisfaction_ratio
                elif self.reward_mode == "learning":
                    reward[aid] = (satisfaction_ratio - prev_ratio) + (
                        1.0 if solved else 0.0
                    )
                else:
                    reward[aid] = 0.0
                if self.feedback_in_reward:
                    feedback_vec = np.array(
                        [feedback[m] for m in self.feedback_metrics_to_use],
                        dtype=np.float32,
                    )
                    reward[aid] += (
                        float(np.dot(feedback_vec, self.feedback_metric_weights))
                        * self.feedback_reward_scale
                    )
                reward[aid] -= comm_penalty
                self._prev_satisfaction_ratio[aid] = satisfaction_ratio
                terminated[aid] = solved
                truncated[aid] = self.current_step >= self.max_steps and not solved
            self.current_step += 1
            if self.debug:
                self.log_debug_info()
            return obs, reward, terminated, truncated, info, feedback_vectors
        else:
            self.current_step += 1
            var = action + 1
            self.variable_assignment[var] = 1 - self.variable_assignment[var]
            obs, info, feedback_vector = self._get_obs_info(
                self.variable_assignment, return_feedback=True
            )
            feedback = SATFeedback().get_feedback(
                self.variable_assignment, self.clauses, self.num_vars
            )
            obs["feedback_vector"] = np.array(list(feedback.values()), dtype=np.float32)
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
            if self.feedback_in_reward:
                feedback_vec = np.array(
                    [feedback[m] for m in self.feedback_metrics_to_use],
                    dtype=np.float32,
                )
                reward += (
                    float(np.dot(feedback_vec, self.feedback_metric_weights))
                    * self.feedback_reward_scale
                )
            reward -= comm_penalty
            self._prev_satisfaction_ratio = satisfaction_ratio
            terminated = solved
            truncated = self.current_step >= self.max_steps and not terminated
            if self.debug:
                self.log_debug_info()
            return obs, reward, terminated, truncated, info

    def _get_obs_info(self, variable_assignment=None, return_feedback=False):
        # If using backend, delegate observation building
        if self.backend is not None:
            state = {
                "assignments": variable_assignment,
                "clauses": self.clauses,
                "num_vars": self.num_vars,
            }
            if self.observation_mode == "dict":
                obs = self.backend.get_observation_dict(state)
            elif self.observation_mode == "flat":
                obs = self.backend.get_observation_flat(state)
            elif self.observation_mode == "graph":
                obs = self.backend.get_observation_graph(state)
            else:
                raise ValueError(f"Unknown observation_mode: {self.observation_mode}")
            info = {"assignment": variable_assignment}
            feedback_vector = None
            if return_feedback:
                feedback_vector = None  # Could call backend.get_feedback(state)
                return obs, info, feedback_vector
            return obs, info

        assignment = (
            variable_assignment
            if variable_assignment is not None
            else self.variable_assignment
        )

        def clause_is_satisfied(clause, assignment):
            return any(
                (assignment[abs(lit)] == 1 if lit > 0 else assignment[abs(lit)] == 0)
                for lit in clause
            )

        satisfied_clauses = [
            clause_is_satisfied(clause, assignment) for clause in self.clauses
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
                [1.0 if assignment[i + 1] == 1 else -1.0 for i in range(self.num_vars)],
                dtype=np.float32,
            ),
            "variable_assignment": {str(k): int(v) for k, v in assignment.items()},
            "clause_satisfaction": clause_satisfaction.copy(),
        }
        info = {
            "num_satisfied_clauses": sum(satisfied_clauses),
            "total_clauses": self.num_clauses,
            "satisfaction_ratio": satisfaction_ratio,
            "solved": solved,
            "satisfied_clauses": satisfied_clauses,
            "step": self.current_step,
            "assignment": assignment.copy(),
        }
        # Richer feedback: add diversity, tension, progress
        feedback_vector = {
            "clause_satisfaction": satisfaction_ratio,
            "variable_decisiveness": len(assignment) / self.num_vars
            if self.num_vars
            else 1.0,
            "search_diversity": np.std(list(assignment.values()))
            if assignment
            else 0.0,
            "constraint_tension": float(
                sum([abs(sum(clause)) for clause in self.clauses])
            )
            / (self.num_clauses or 1),
            "proof_progress": satisfaction_ratio
            * len(assignment)
            / (self.num_vars or 1),
        }
        if return_feedback:
            return obs, info, feedback_vector
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

    def log_debug_info(self):
        # Advanced logging: clause satisfaction, feedback vectors, agent actions, etc.
        # Example:
        print(
            f"[DEBUG] Clause satisfaction: {getattr(self, 'clause_satisfaction', None)}"
        )
        print(f"[DEBUG] Feedback vector: {getattr(self, 'feedback_vector', None)}")
        print(f"[DEBUG] Last agent actions: {getattr(self, 'last_actions', None)}")
        # ...add more as needed...
