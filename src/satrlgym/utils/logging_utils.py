"""
Structured logging utilities for the SAT+RL project.

This module provides a StructuredLogger class that can log events in various formats
and a NumpyJSONEncoder for serializing numpy arrays to JSON.
"""

import csv
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle NumPy arrays and scalars."""

    def default(self, obj):
        """Convert numpy objects to standard Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class StructuredLogger:
    """
    A logger for structured data in various formats.

    This logger can output data in JSON Lines or CSV format.
    It maintains separate files for different event types.
    """

    FORMAT_JSON = "json"
    FORMAT_CSV = "csv"

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        format_type: str = "json",
        visualize_ready: bool = False,
    ):
        """
        Initialize the structured logger.

        Args:
            output_dir: Directory to save log files in
            experiment_name: Name of the experiment (used in filenames)
            format_type: Format to save logs in ("json" or "csv")
            visualize_ready: Whether to format logs for easy visualization
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.format_type = format_type
        self.visualize_ready = visualize_ready

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize file handles and headers
        self.files = {}
        self.headers = {}
        self.write_counts = {}

        # Metadata for visualization
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "log_files": {},
        }

    def _get_file(self, event_type: str) -> tuple:
        """
        Get the file handle for a given event type.

        Args:
            event_type: Type of event (used in filename)

        Returns:
            Tuple of (file_handle, is_new)
        """
        if event_type not in self.files:
            # Determine file extension
            ext = ".jsonl" if self.format_type == self.FORMAT_JSON else ".csv"
            filename = f"{self.experiment_name}_{event_type}{ext}"
            filepath = os.path.join(self.output_dir, filename)

            # Track this file in metadata
            self.metadata["log_files"][event_type] = filepath

            # Open file and initialize
            file = open(
                filepath,
                "w",
                newline="" if self.format_type == self.FORMAT_CSV else None,
            )
            self.files[event_type] = file
            self.write_counts[event_type] = 0
            return file, True

        return self.files[event_type], False

    def _write_event(self, event_type: str, data: dict[str, Any]):
        """
        Write an event to the appropriate log file.

        Args:
            event_type: Type of event
            data: Data to log
        """
        file, is_new = self._get_file(event_type)

        if self.format_type == self.FORMAT_JSON:
            # Write as JSON
            file.write(json.dumps(data, cls=NumpyJSONEncoder) + "\n")
            file.flush()
        else:
            # Write as CSV
            writer = csv.DictWriter(file, fieldnames=list(data.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(data)
            file.flush()

        self.write_counts[event_type] += 1

    def log_reward(
        self, episode: int, step: int, reward: float, info: dict | None = None
    ):
        """
        Log a reward event.

        Args:
            episode: Episode number
            step: Step within the episode
            reward: Reward value
            info: Additional information
        """
        data = {
            "episode": episode,
            "step": step,
            "reward": reward,
            "timestamp": time.time(),
            "info": info or {},
        }
        self._write_event("reward", data)

    def log_clause_stats(
        self, episode: int, step: int, satisfied_count: int, total_count: int
    ):
        """
        Log clause satisfaction statistics.

        Args:
            episode: Episode number
            step: Step within the episode
            satisfied_count: Number of satisfied clauses
            total_count: Total number of clauses
        """
        data = {
            "episode": episode,
            "step": step,
            "satisfied_count": satisfied_count,
            "total_count": total_count,
            "satisfaction_ratio": (
                satisfied_count / total_count if total_count > 0 else 0
            ),
            "timestamp": time.time(),
        }
        self._write_event("clause_stats", data)

    def log_agent_decision(
        self,
        episode: int,
        step: int,
        agent_id: str,
        action: Any,
        observation: dict,
        action_probs: dict | None = None,
    ):
        """
        Log an agent's decision.

        Args:
            episode: Episode number
            step: Step within the episode
            agent_id: Identifier for the agent
            action: Action taken
            observation: Observation that led to the action
            action_probs: Probabilities for each action (if available)
        """
        # Convert action_probs to strings as keys for JSON compatibility
        action_probs_serializable = {str(k): v for k, v in (action_probs or {}).items()}

        data = {
            "episode": episode,
            "step": step,
            "agent_id": agent_id,
            "action": action,
            "action_probs": action_probs_serializable,
            "observation": observation,
            "timestamp": time.time(),
        }
        self._write_event("agent_decision", data)

    def log_exception(
        self,
        episode: int,
        step: int,
        exception_type: str,
        exception_message: str,
        stack_trace: str,
    ):
        """
        Log an exception.

        Args:
            episode: Episode number
            step: Step within the episode
            exception_type: Type of the exception
            exception_message: Exception message
            stack_trace: Stack trace
        """
        data = {
            "episode": episode,
            "step": step,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "stack_trace": stack_trace,
            "timestamp": time.time(),
        }
        self._write_event("exception", data)

    def close(self):
        """Close all open file handles."""
        for file in self.files.values():
            file.close()
        self.files = {}

    def finalize(self) -> str:
        """
        Finalize logging and write metadata file if visualization is enabled.

        Returns:
            Path to metadata file if visualization is enabled, empty string otherwise
        """
        # Close all files
        self.close()

        if self.visualize_ready:
            # Update metadata with end time
            self.metadata["end_time"] = datetime.now().isoformat()

            # Record write counts
            self.metadata["record_counts"] = self.write_counts

            # Write metadata file
            metadata_path = os.path.join(
                self.output_dir, f"{self.experiment_name}_viz_metadata.json"
            )
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

            return metadata_path

        return ""


def create_logger(
    experiment_name: str,
    output_dir: str = "logs",
    format_type: str = "json",
    visualize_ready: bool = False,
) -> StructuredLogger:
    """
    Create a structured logger with default settings.

    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to save logs in
        format_type: Format to save logs in ("json" or "csv")
        visualize_ready: Whether to format logs for easy visualization

    Returns:
        StructuredLogger instance
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    return StructuredLogger(
        output_dir=output_dir,
        experiment_name=experiment_name,
        format_type=format_type,
        visualize_ready=visualize_ready,
    )


class LoggingManager:
    """
    Manager for configuring Python's built-in logging system alongside structured logging.

    This class provides a way to configure both structured logging for data
    and standard logging for messages and debugging.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ):
        """
        Initialize the logging manager.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save logs in
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Configure Python's logging system
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(min(console_level, file_level))

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Create file handler
        log_file = os.path.join(output_dir, f"{experiment_name}_log.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Create structured logger
        self.structured_logger = create_logger(
            experiment_name=experiment_name,
            output_dir=output_dir,
            format_type="json",
            visualize_ready=True,
        )

    def get_logger(self) -> logging.Logger:
        """Get the Python logger."""
        return self.logger

    def get_structured_logger(self) -> StructuredLogger:
        """Get the structured data logger."""
        return self.structured_logger

    def close(self):
        """Close all loggers."""
        self.structured_logger.close()

        # Remove handlers from Python logger
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
