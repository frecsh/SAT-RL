"""
Basic Representation Framework for SAT-RL.

This module contains classes for representing SAT problems in formats suitable for
reinforcement learning, with different encoding strategies for variables and clauses.
"""

from collections import deque

import numpy as np
import torch


class ObservationEncoder:
    """Base class for observation encoders."""

    def encode(self, sat_state: dict) -> np.ndarray | torch.Tensor | dict:
        """
        Encode a SAT state into a representation suitable for ML models.

        Args:
            sat_state: Dictionary containing the SAT problem state

        Returns:
            An encoded representation of the state
        """
        raise NotImplementedError("Subclasses must implement encode method")

    def get_observation_space(self) -> dict:
        """
        Get the observation space specification.

        Returns:
            A dictionary describing the shape and type of the observation space
        """
        raise NotImplementedError(
            "Subclasses must implement get_observation_space method"
        )


class VariableAssignmentEncoder(ObservationEncoder):
    """Encodes variable assignments in different formats."""

    def __init__(self, num_variables: int, encoding_type: str = "one-hot"):
        """
        Initialize the variable assignment encoder.

        Args:
            num_variables: Number of variables in the SAT problem
            encoding_type: Type of encoding to use ("one-hot", "binary", or "continuous")
        """
        self.num_variables = num_variables
        self.encoding_type = encoding_type

        if encoding_type not in ["one-hot", "binary", "continuous"]:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def encode(self, var_assignments: dict[int, bool]) -> np.ndarray | torch.Tensor:
        """
        Encode variable assignments.

        Args:
            var_assignments: Dictionary mapping variable indices to boolean assignments

        Returns:
            Encoded representation of variable assignments
        """
        if self.encoding_type == "one-hot":
            # One-hot encoding: [unassigned, False, True] for each variable
            encoding = np.zeros((self.num_variables, 3), dtype=np.float32)
            for var_idx in range(self.num_variables):
                if var_idx + 1 in var_assignments:
                    # Variables are 1-indexed in SAT problems
                    value = var_assignments[var_idx + 1]
                    encoding[var_idx, 2 if value else 1] = 1.0
                else:
                    # Unassigned
                    encoding[var_idx, 0] = 1.0
            return encoding

        elif self.encoding_type == "binary":
            # Binary encoding: 1 for True, 0 for False, -1 for unassigned
            encoding = np.full(self.num_variables, -1, dtype=np.float32)
            for var_idx in range(self.num_variables):
                if var_idx + 1 in var_assignments:
                    encoding[var_idx] = float(var_assignments[var_idx + 1])
            return encoding

        elif self.encoding_type == "continuous":
            # Continuous encoding: 1.0 for True, 0.0 for False, 0.5 for unassigned
            encoding = np.full(self.num_variables, 0.5, dtype=np.float32)
            for var_idx in range(self.num_variables):
                if var_idx + 1 in var_assignments:
                    encoding[var_idx] = 1.0 if var_assignments[var_idx + 1] else 0.0
            return encoding

    def get_observation_space(self) -> dict:
        """Get the observation space specification."""
        if self.encoding_type == "one-hot":
            return {"shape": (self.num_variables, 3), "dtype": np.float32}
        else:  # binary or continuous
            return {"shape": (self.num_variables,), "dtype": np.float32}


class ClauseSatisfactionEncoder(ObservationEncoder):
    """Encodes clause satisfaction status in different formats."""

    def __init__(self, num_clauses: int, encoding_type: str = "binary"):
        """
        Initialize the clause satisfaction encoder.

        Args:
            num_clauses: Number of clauses in the SAT problem
            encoding_type: Type of encoding to use ("binary" or "percentage")
        """
        self.num_clauses = num_clauses
        self.encoding_type = encoding_type

        if encoding_type not in ["binary", "percentage"]:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def encode(self, clause_satisfaction: list[bool] | dict[int, bool]) -> np.ndarray:
        """
        Encode clause satisfaction status.

        Args:
            clause_satisfaction: List or dictionary indicating satisfaction state of clauses

        Returns:
            Encoded representation of clause satisfaction
        """
        if self.encoding_type == "binary":
            # Binary encoding: 1 for satisfied, 0 for unsatisfied
            if isinstance(clause_satisfaction, dict):
                encoding = np.zeros(self.num_clauses, dtype=np.float32)
                for clause_idx, satisfied in clause_satisfaction.items():
                    encoding[clause_idx] = float(satisfied)
            else:  # List
                encoding = np.array(
                    [float(sat) for sat in clause_satisfaction], dtype=np.float32
                )
            return encoding

        elif self.encoding_type == "percentage":
            # Percentage-based encoding: 1.0 for fully satisfied, 0.0 for unsatisfied
            # For partially satisfied clauses (some literals true), value between 0-1
            if isinstance(clause_satisfaction, dict):
                encoding = np.zeros(self.num_clauses, dtype=np.float32)
                for clause_idx, value in clause_satisfaction.items():
                    # If value is already a float (percentage), use it directly
                    # Otherwise convert boolean to 0.0/1.0
                    encoding[clause_idx] = (
                        value if isinstance(value, float) else float(value)
                    )
            else:  # List
                encoding = np.array(clause_satisfaction, dtype=np.float32)
            return encoding

    def get_observation_space(self) -> dict:
        """Get the observation space specification."""
        return {"shape": (self.num_clauses,), "dtype": np.float32}


class HistoryTracker:
    """Tracks the history of previous assignments."""

    def __init__(self, history_length: int, num_variables: int):
        """
        Initialize the history tracker.

        Args:
            history_length: Number of past assignments to track
            num_variables: Number of variables in the SAT problem
        """
        self.history_length = history_length
        self.num_variables = num_variables
        # Store (variable_idx, assignment) tuples, where variable_idx is 0-indexed
        self.history = deque(maxlen=history_length)

    def add_assignment(self, variable_idx: int, assignment: bool):
        """
        Add a new assignment to the history.

        Args:
            variable_idx: Index of the variable (1-indexed)
            assignment: Boolean value assigned to the variable
        """
        self.history.append((variable_idx - 1, assignment))  # Convert to 0-indexed

    def get_history_encoding(self) -> np.ndarray:
        """
        Get encoded history of assignments.

        Returns:
            Array with shape (history_length, 2) where each entry contains
            [variable_idx, assignment_value]
        """
        encoding = np.zeros((self.history_length, 2), dtype=np.float32)

        for i, (var_idx, assignment) in enumerate(self.history):
            if i >= self.history_length:
                break
            encoding[i, 0] = var_idx / self.num_variables  # Normalize variable index
            encoding[i, 1] = float(assignment)

        return encoding

    def clear(self):
        """Clear the history."""
        self.history.clear()


class ModularObservationPreprocessor:
    """
    Modular preprocessor that combines multiple encoders for SAT problem representations.
    """

    def __init__(
        self,
        num_variables: int,
        num_clauses: int,
        variable_encoding: str = "one-hot",
        clause_encoding: str = "binary",
        history_length: int = 0,
    ):
        """
        Initialize the modular observation preprocessor.

        Args:
            num_variables: Number of variables in the SAT problem
            num_clauses: Number of clauses in the SAT problem
            variable_encoding: Encoding type for variables
            clause_encoding: Encoding type for clauses
            history_length: Number of past assignments to track (0 to disable)
        """
        self.num_variables = num_variables
        self.num_clauses = num_clauses
        self.variable_encoder = VariableAssignmentEncoder(
            num_variables, variable_encoding
        )
        self.clause_encoder = ClauseSatisfactionEncoder(num_clauses, clause_encoding)
        self.history_length = history_length

        if history_length > 0:
            self.history_tracker = HistoryTracker(history_length, num_variables)
        else:
            self.history_tracker = None

    def encode(self, sat_state: dict) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Encode a SAT state into a structured observation (alias for process_observation).

        Args:
            sat_state: Dictionary with SAT problem state information

        Returns:
            Dictionary with encoded observations
        """
        return self.process_observation(sat_state)

    def process_observation(
        self, sat_state: dict
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Process a SAT state into a structured observation.

        Args:
            sat_state: Dictionary with keys:
                - 'variable_assignments': Dict mapping variable indices to boolean values
                - 'clause_satisfaction': List or Dict of clause satisfaction states
                - 'new_assignment': Optional tuple (variable_idx, assignment) for history tracking

        Returns:
            Dictionary with encoded observations
        """
        observation = {}

        # Encode variable assignments (handle both keys for compatibility)
        if "variable_assignments" in sat_state:
            observation["variables"] = self.variable_encoder.encode(
                sat_state["variable_assignments"]
            )
        elif "variable_assignment" in sat_state:  # Add support for new key naming
            observation["variables"] = self.variable_encoder.encode(
                sat_state["variable_assignment"]
            )

        # Encode clause satisfaction
        if "clause_satisfaction" in sat_state:
            observation["clauses"] = self.clause_encoder.encode(
                sat_state["clause_satisfaction"]
            )

        # Track and encode history if enabled
        if self.history_tracker is not None:
            if "new_assignment" in sat_state:
                var_idx, assignment = sat_state["new_assignment"]
                self.history_tracker.add_assignment(var_idx, assignment)

            observation["history"] = self.history_tracker.get_history_encoding()

        return observation

    def reset(self):
        """Reset the preprocessor state (e.g., history)."""
        if self.history_tracker:
            self.history_tracker.clear()

    def get_observation_space(self) -> dict:
        """
        Get the complete observation space specification.

        Returns:
            Dictionary describing the structure and shape of the observation space
        """
        obs_space = {
            "variables": self.variable_encoder.get_observation_space(),
            "clauses": self.clause_encoder.get_observation_space(),
        }

        if self.history_tracker:
            obs_space["history"] = {
                "shape": (self.history_length, 2),
                "dtype": np.float32,
            }

        return obs_space
