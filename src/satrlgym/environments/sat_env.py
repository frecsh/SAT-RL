"""
Concrete implementation of the SAT environment interface.
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np

from .base_env import SATEnv

# Set up logging
logger = logging.getLogger(__name__)


class SimpleSATEnv(SATEnv):
    """
    A simple implementation of the SAT environment.
    """

    def __init__(
        self,
        clauses: list[list[int]] | None = None,
        num_vars: int = 0,
        max_steps: int = 1000,
        reward_config: dict[str, Any] | None = None,
        observation_type: str = "simple",
    ):
        """
        Initialize the simple SAT environment.

        Args:
            clauses: List of clauses, where each clause is a list of literals
            num_vars: Number of variables in the problem
            max_steps: Maximum number of steps per episode
            reward_config: Configuration for reward shaping
            observation_type: Type of observation to use ("simple", "binary", "graph")
        """
        super().__init__(clauses, num_vars, max_steps, reward_config)

        # Store observation type
        self.observation_type = observation_type

        # Update observation space based on observation type
        if observation_type == "binary":
            # Binary representation: assignment as 0/1 array
            self.observation_space = gym.spaces.Dict(
                {
                    "assignment": gym.spaces.Box(
                        low=0, high=1, shape=(self.num_vars * 2,), dtype=np.int8
                    ),
                    "clause_status": gym.spaces.Box(
                        low=0, high=1, shape=(len(self.clauses),), dtype=np.int8
                    ),
                    "steps": gym.spaces.Box(
                        low=0, high=self.max_steps, shape=(1,), dtype=np.int32
                    ),
                }
            )
        elif observation_type == "graph":
            # Graph representation: adjacency matrix + node features
            # This is a placeholder - in practice, you would use a proper graph
            # representation library
            max_clause_size = (
                max([len(clause) for clause in self.clauses]) if self.clauses else 3
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "var_features": gym.spaces.Box(
                        low=-1, high=1, shape=(self.num_vars, 2), dtype=np.int8
                    ),
                    "clause_features": gym.spaces.Box(
                        low=0, high=1, shape=(len(self.clauses), 1), dtype=np.int8
                    ),
                    "adjacency": gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(
                            self.num_vars + len(self.clauses),
                            self.num_vars + len(self.clauses),
                        ),
                        dtype=np.int8,
                    ),
                    "steps": gym.spaces.Box(
                        low=0, high=self.max_steps, shape=(1,), dtype=np.int32
                    ),
                }
            )

    def _get_observation(self) -> dict[str, np.ndarray]:
        """
        Get the current observation based on the observation type.

        Returns:
            Observation dictionary
        """
        if self.observation_type == "binary":
            # Binary representation: one-hot encode each variable assignment
            assignment = np.zeros((self.num_vars * 2,), dtype=np.int8)
            for i in range(1, self.num_vars + 1):
                if self.assignment[i] > 0:
                    assignment[i - 1] = 1  # Variable i is True
                elif self.assignment[i] < 0:
                    assignment[i - 1 + self.num_vars] = 1  # Variable i is False

            # Compute clause status
            clause_status = np.zeros((len(self.clauses),), dtype=np.int8)
            for i, clause in enumerate(self.clauses):
                for lit in clause:
                    var_idx = abs(lit)
                    if (lit > 0 and self.assignment[var_idx] > 0) or (
                        lit < 0 and self.assignment[var_idx] < 0
                    ):
                        clause_status[i] = 1
                        break

            return {
                "assignment": assignment,
                "clause_status": clause_status,
                "steps": np.array([self.steps_taken], dtype=np.int32),
            }

        elif self.observation_type == "graph":
            # Graph representation: adjacency matrix + node features
            # Variables are the first self.num_vars nodes, clauses are the rest

            # Variable features: [assigned, is_positive]
            var_features = np.zeros((self.num_vars, 2), dtype=np.int8)
            for i in range(1, self.num_vars + 1):
                if self.assignment[i] != 0:
                    var_features[i - 1, 0] = 1
                    var_features[i - 1, 1] = 1 if self.assignment[i] > 0 else 0

            # Clause features: [is_satisfied]
            clause_features = np.zeros((len(self.clauses), 1), dtype=np.int8)
            for i, clause in enumerate(self.clauses):
                for lit in clause:
                    var_idx = abs(lit)
                    if (lit > 0 and self.assignment[var_idx] > 0) or (
                        lit < 0 and self.assignment[var_idx] < 0
                    ):
                        clause_features[i, 0] = 1
                        break

            # Adjacency matrix between variables and clauses
            total_nodes = self.num_vars + len(self.clauses)
            adjacency = np.zeros((total_nodes, total_nodes), dtype=np.int8)

            # Connect variables to clauses they appear in
            for i, clause in enumerate(self.clauses):
                clause_node = self.num_vars + i
                for lit in clause:
                    var_node = abs(lit) - 1
                    adjacency[var_node, clause_node] = 1
                    adjacency[clause_node, var_node] = 1

            return {
                "var_features": var_features,
                "clause_features": clause_features,
                "adjacency": adjacency,
                "steps": np.array([self.steps_taken], dtype=np.int32),
            }

        else:  # Default simple observation
            # Simple representation: assignment and clause status
            clause_status = np.zeros((len(self.clauses),), dtype=np.int8)
            for i, clause in enumerate(self.clauses):
                for lit in clause:
                    var_idx = abs(lit)
                    if (lit > 0 and self.assignment[var_idx] > 0) or (
                        lit < 0 and self.assignment[var_idx] < 0
                    ):
                        clause_status[i] = 1
                        break

            return {
                "assignment": self.assignment.copy(),
                "clause_status": clause_status,
                "steps": np.array([self.steps_taken], dtype=np.int32),
            }

    def _calculate_reward(self, prev_satisfied: int, current_satisfied: int) -> float:
        """
        Calculate the reward based on the change in satisfied clauses.

        Args:
            prev_satisfied: Number of satisfied clauses before action
            current_satisfied: Number of satisfied clauses after action

        Returns:
            Reward value
        """
        # Base reward from clause satisfaction change
        reward = (current_satisfied - prev_satisfied) * self.reward_config.get(
            "clause_reward", 0.1
        )

        # Step penalty to encourage efficiency
        reward += self.reward_config.get("step_penalty", -0.01)

        return reward

    def _count_satisfied_clauses(self, assignment: np.ndarray | None = None) -> int:
        """
        Count the number of satisfied clauses for the given assignment.

        Args:
            assignment: Assignment to check, or current assignment if None

        Returns:
            Number of satisfied clauses
        """
        if assignment is None:
            assignment = self.assignment

        count = 0
        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                var_idx = abs(lit)
                if (lit > 0 and assignment[var_idx] > 0) or (
                    lit < 0 and assignment[var_idx] < 0
                ):
                    satisfied = True
                    break
            if satisfied:
                count += 1

        return count

    def _is_satisfied(self) -> bool:
        """
        Check if all clauses are satisfied with the current assignment.

        Returns:
            True if all clauses are satisfied, False otherwise
        """
        return self._count_satisfied_clauses() == len(self.clauses)

    def render(self, mode: str = "human") -> np.ndarray | None:
        """
        Render the environment.

        Args:
            mode: Rendering mode ('human', 'ansi', 'rgb_array')

        Returns:
            Optional rendering output depending on the mode
        """
        # Use the parent class implementation
        return super().render(mode)

    @classmethod
    def from_dimacs(cls, file_path: str, **kwargs) -> "SimpleSATEnv":
        """
        Create a SimpleSATEnv from a DIMACS CNF file.

        Args:
            file_path: Path to the DIMACS CNF file
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            SimpleSATEnv instance
        """
        clauses = []
        num_vars = 0

        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("c"):
                        # Skip comments and empty lines
                        continue
                    elif line.startswith("p cnf"):
                        # Parse header
                        parts = line.split()
                        if len(parts) >= 4:
                            num_vars = int(parts[2])
                    else:
                        # Parse clause
                        literals = [int(x) for x in line.split() if x != "0"]
                        if literals:  # Skip empty clauses
                            clauses.append(literals)
        except Exception as e:
            logger.error(f"Error reading DIMACS file: {e}")
            raise

        return cls(clauses=clauses, num_vars=num_vars, **kwargs)
