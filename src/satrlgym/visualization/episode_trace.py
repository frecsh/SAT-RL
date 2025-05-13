"""
Episode trace visualization.

This module provides a tool for recording and visualizing the trace of
an agent's episode, including states, actions, and rewards.
"""

import json
from typing import Any

import numpy as np


class EpisodeTraceViewer:
    """
    A tool for recording and visualizing episode traces.

    Attributes:
        steps: List of recorded steps
    """

    def __init__(self):
        """
        Initialize an empty episode trace.
        """
        self.steps = []
        self.metadata = {}

    def record_step(
        self, action: int, reward: float, state: dict[str, Any], info: dict[str, Any]
    ):
        """
        Record a step in the episode trace.

        Args:
            action: The action taken
            reward: The reward received
            state: The state observation
            info: Additional information
        """
        # Deep copy to avoid modifying the original objects
        state_copy = self._safe_copy(state)
        info_copy = self._safe_copy(info)

        step_data = {
            "action": action,
            "reward": reward,
            "state": state_copy,
            "info": info_copy,
        }

        self.steps.append(step_data)

    def set_metadata(self, metadata: dict[str, Any]):
        """
        Set metadata for the episode.

        Args:
            metadata: Dictionary of metadata
        """
        self.metadata = self._safe_copy(metadata)

    def _safe_copy(self, obj):
        """
        Create a safe copy of an object, handling numpy arrays.

        Args:
            obj: Object to copy

        Returns:
            A copy of the object
        """
        if isinstance(obj, dict):
            return {k: self._safe_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_copy(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "copy"):
            return obj.copy()
        else:
            return obj

    def generate_summary(self) -> str:
        """
        Generate a text summary of the episode.

        Returns:
            A string with the summary
        """
        if not self.steps:
            return "No steps recorded."

        # Calculate total reward
        total_reward = sum(step["reward"] for step in self.steps)

        # Get final state
        final_state = self.steps[-1]["state"]
        final_info = self.steps[-1]["info"]

        # Format the summary
        lines = []
        lines.append(f"Episode Summary:")
        lines.append(f"Total steps: {len(self.steps)}")
        lines.append(f"Total reward: {total_reward:.4f}")

        if "num_satisfied_clauses" in final_info and "total_clauses" in final_info:
            satisfaction = (
                f"{final_info['num_satisfied_clauses']}/{final_info['total_clauses']}"
            )
            lines.append(f"Final clause satisfaction: {satisfaction}")

        if "solved" in final_info:
            lines.append(f"Problem solved: {final_info['solved']}")

        lines.append("\nFinal state:")
        try:
            var_assignment = final_state.get("variable_assignment", {})
            if var_assignment:
                vars_true = [v for v, val in var_assignment.items() if val]
                vars_false = [v for v, val in var_assignment.items() if not val]
                lines.append(f"  Variables set to True: {vars_true}")
                lines.append(f"  Variables set to False: {vars_false}")
        except BaseException:
            lines.append("  [Error formatting variable assignments]")

        return "\n".join(lines)

    def save(self, file_path: str):
        """
        Save the episode trace to a file.

        Args:
            file_path: Path where to save the trace
        """
        data = {"metadata": self.metadata, "steps": self.steps}

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "EpisodeTraceViewer":
        """
        Load an episode trace from a file.

        Args:
            file_path: Path to the trace file

        Returns:
            An EpisodeTraceViewer instance with the loaded data
        """
        with open(file_path) as f:
            data = json.load(f)

        viewer = cls()
        viewer.metadata = data.get("metadata", {})
        viewer.steps = data.get("steps", [])

        return viewer
