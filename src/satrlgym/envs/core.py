"""
Core environment implementation for SAT reinforcement learning.

This module provides the main environment class for SAT problems.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Discrete

from src.satrlgym.envs.rewards import get_reward_function
from src.satrlgym.utils.cnf import compute_satisfied_clauses


class SatGymEnv(gym.Env):
    """
    Gym environment for SAT problems.

    This environment represents SAT problems as reinforcement learning tasks
    where the agent assigns values to variables.

    Attributes:
        formula: The SAT formula
        reward_mode: Type of reward function to use
        max_steps: Maximum number of steps per episode
        num_vars: Number of variables in the formula
        num_clauses: Number of clauses in the formula
        action_space: Space of valid actions
        observation_space: Space of valid observations
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}

    def __init__(
        self,
        formula: dict[str, Any],
        reward_mode: str = "sparse",
        max_steps: int | None = None,
        render_mode: str | None = None,
        **kwargs,
    ):
        """
        Initialize the SAT environment.

        Args:
            formula: SAT formula dict with 'clauses' and 'num_vars'
            reward_mode: Type of reward function ('sparse', 'dense', 'learning')
            max_steps: Maximum number of steps per episode
            render_mode: How to render the environment
            **kwargs: Additional arguments
        """
        super().__init__()

        # Store formula
        self.clauses = formula["clauses"]
        self.num_vars = formula["num_vars"]
        self.num_clauses = len(self.clauses)

        # Environment parameters
        self.reward_mode = reward_mode
        self.max_steps = max_steps or (2 * self.num_vars)
        self.render_mode = render_mode

        # Set up spaces
        self.action_space = Discrete(self.num_vars)

        # Simple observation: One float for each variable and clause
        self.observation_space = DictSpace(
            {
                "variables": Box(
                    low=-1.0, high=1.0, shape=(self.num_vars,), dtype=np.float32
                ),
                "clauses": Box(
                    low=0.0, high=1.0, shape=(self.num_clauses,), dtype=np.float32
                ),
                # Use Dict space to represent variable assignments; keys are integers,
                # values are booleans
                "variable_assignment": DictSpace(
                    {str(i): Discrete(2) for i in range(1, self.num_vars + 1)}
                ),
                "clause_satisfaction": Box(
                    low=0, high=1, shape=(self.num_clauses,), dtype=bool
                ),
            }
        )

        # Get reward function
        self.reward_fn = get_reward_function(reward_mode)

        # Initialize state
        self.variable_assignment = {}
        self.steps_taken = 0
        self.rng = None
        self.satisfied_clauses = []

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            A tuple of (observation, info)
        """
        # Reset random number generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        elif self.rng is None:
            self.rng = np.random.RandomState()

        # Reset state
        self.variable_assignment = {}
        self.steps_taken = 0

        # Create an array representation for the observation
        variables = np.zeros(self.num_vars, dtype=np.float32)
        clauses = np.zeros(self.num_clauses, dtype=np.float32)

        # Calculate satisfied clauses
        self.satisfied_clauses = compute_satisfied_clauses(
            self.clauses, self.variable_assignment, partial=True
        )

        # Prepare observation
        obs = {
            "variables": variables,
            "clauses": clauses,
            # Include the variable assignment for compatibility with tests
            "variable_assignment": self.variable_assignment,
            # Add clause_satisfaction for tests
            "clause_satisfaction": np.zeros(self.num_clauses, dtype=bool),
        }

        # Prepare info
        info = {
            "num_satisfied_clauses": self.satisfied_clauses,
            "total_clauses": self.num_clauses,
            "steps_taken": self.steps_taken,
            "satisfaction_ratio": (
                self.satisfied_clauses / self.num_clauses
                if self.num_clauses > 0
                else 0.0
            ),
        }

        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Take an action in the environment.

        Args:
            action: Index of variable to flip (0-indexed)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Increment step counter
        self.steps_taken += 1

        # Get the 1-indexed variable from 0-indexed action
        var_idx = action + 1

        # Track previous satisfaction for reward calculation
        prev_satisfied = self.satisfied_clauses

        # Toggle the variable value - use string keys for consistency with observation space
        var_key = str(var_idx)
        if var_idx in self.variable_assignment:
            self.variable_assignment[var_idx] = not self.variable_assignment[var_idx]
        elif var_key in self.variable_assignment:
            self.variable_assignment[var_key] = not self.variable_assignment[var_key]
        else:
            # Use string key for consistency with observation space
            self.variable_assignment[var_key] = True

        # Recalculate satisfied clauses
        self.satisfied_clauses = compute_satisfied_clauses(
            self.clauses, self.variable_assignment, partial=True
        )

        # Check if the formula is satisfied
        is_solved = self.satisfied_clauses == self.num_clauses

        # Check termination conditions
        terminated = is_solved
        truncated = self.steps_taken >= self.max_steps

        # Calculate reward
        reward = self.reward_fn(
            solved=is_solved,
            terminated=(terminated or truncated),
            num_satisfied_before=prev_satisfied,
            num_satisfied_after=self.satisfied_clauses,
            total_clauses=self.num_clauses,
            steps=self.steps_taken,
        )

        # Create an array representation for the observation
        variables = np.zeros(self.num_vars, dtype=np.float32)
        for var, val in self.variable_assignment.items():
            # Handle string keys (from DictSpace) or int keys
            var_idx = int(var) if isinstance(var, str) else var
            if 1 <= var_idx <= self.num_vars:  # Ensure var is in bounds
                variables[var_idx - 1] = 1.0 if val else -1.0

        # Convert the satisfied clauses count to an array of clause satisfaction
        clauses = np.zeros(self.num_clauses, dtype=np.float32)

        # Compute which clauses are satisfied
        satisfied_indices = []
        for i, clause in enumerate(self.clauses):
            clause_satisfied = False
            for literal in clause:
                var_idx = abs(literal)
                expected_value = literal > 0  # True if positive, False if negative

                if (
                    var_idx in self.variable_assignment
                    and self.variable_assignment[var_idx] == expected_value
                ):
                    clause_satisfied = True
                    break

            if clause_satisfied:
                clauses[i] = 1.0
                satisfied_indices.append(i)

        # Create array of clause satisfaction states
        clause_satisfaction = np.zeros(self.num_clauses, dtype=bool)
        for i in satisfied_indices:
            clause_satisfaction[i] = True

        # Prepare observation
        obs = {
            "variables": variables,
            "clauses": clauses,
            # Include the variable assignment for compatibility with tests
            "variable_assignment": self.variable_assignment,
            # Add clause_satisfaction for tests
            "clause_satisfaction": clause_satisfaction,
        }

        # Prepare info
        info = {
            "num_satisfied_clauses": self.satisfied_clauses,
            "total_clauses": self.num_clauses,
            "satisfaction_ratio": (
                self.satisfied_clauses / self.num_clauses
                if self.num_clauses > 0
                else 0.0
            ),
            "solved": is_solved,
            "steps_taken": self.steps_taken,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns:
            Rendering of the environment depending on render_mode
        """
        if self.render_mode is None:
            return

        if self.render_mode == "ansi":
            output = []
            output.append(f"Step: {self.steps_taken}")
            output.append(f"Variable assignments: {self.variable_assignment}")
            output.append(
                f"Satisfied clauses: {len(self.satisfied_clauses)}/{self.num_clauses}"
            )
            return "\n".join(output)

        return None
