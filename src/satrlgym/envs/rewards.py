"""
Reward functions for SAT environments.

This module provides various reward functions for reinforcement learning on SAT problems.
"""

from collections.abc import Callable


def sparse_reward(
    solved: bool,
    terminated: bool,
    num_satisfied_before: int,
    num_satisfied_after: int,
    total_clauses: int,
    steps: int,
    **kwargs,
) -> float:
    """
    A sparse reward function: 1 for solving the problem, 0 otherwise.

    Args:
        solved: Whether the problem is solved
        terminated: Whether the episode has terminated
        num_satisfied_before: Number of satisfied clauses before action
        num_satisfied_after: Number of satisfied clauses after action
        total_clauses: Total number of clauses
        steps: Number of steps taken so far
        **kwargs: Additional arguments

    Returns:
        The reward value (1.0 for solving, 0.0 otherwise)
    """
    if solved:
        return 1.0
    return 0.0


def dense_reward(
    solved: bool,
    terminated: bool,
    num_satisfied_before: int,
    num_satisfied_after: int,
    total_clauses: int,
    steps: int,
    **kwargs,
) -> float:
    """
    A dense reward function based on clause satisfaction progress.

    Args:
        solved: Whether the problem is solved
        terminated: Whether the episode has terminated
        num_satisfied_before: Number of satisfied clauses before action
        num_satisfied_after: Number of satisfied clauses after action
        total_clauses: Total number of clauses
        steps: Number of steps taken so far
        **kwargs: Additional arguments

    Returns:
        The reward value (higher for more clauses satisfied)
    """
    # Reward for solving
    if solved:
        return 2.0

    # Reward for clause satisfaction progress
    delta = num_satisfied_after - num_satisfied_before

    if delta > 0:
        # Positive reward for improvement
        return 0.1 * delta
    elif delta < 0:
        # Negative reward for regression
        return 0.2 * delta
    else:
        # Small negative reward for no change (encourage exploration)
        return -0.01


def learning_reward(
    solved: bool,
    terminated: bool,
    num_satisfied_before: int,
    num_satisfied_after: int,
    total_clauses: int,
    steps: int,
    explored_ratio: float = 0.0,
    **kwargs,
) -> float:
    """
    A learning-oriented reward with exploration bonuses.

    Args:
        solved: Whether the problem is solved
        terminated: Whether the episode has terminated
        num_satisfied_before: Number of satisfied clauses before action
        num_satisfied_after: Number of satisfied clauses after action
        total_clauses: Total number of clauses
        steps: Number of steps taken so far
        explored_ratio: Ratio of state space explored (0-1)
        **kwargs: Additional arguments

    Returns:
        The reward value with exploration bonuses
    """
    # Base reward similar to dense reward
    base_reward = dense_reward(
        solved,
        terminated,
        num_satisfied_before,
        num_satisfied_after,
        total_clauses,
        steps,
    )

    # Add exploration bonus
    exploration_bonus = 0.05 * (1.0 - explored_ratio)

    return base_reward + exploration_bonus


def get_reward_function(reward_type: str) -> Callable:
    """
    Get a reward function by name.

    Args:
        reward_type: Name of the reward function ('sparse', 'dense', or 'learning')

    Returns:
        The corresponding reward function

    Raises:
        ValueError: If the reward type is not recognized
    """
    reward_functions = {
        "sparse": sparse_reward,
        "dense": dense_reward,
        "learning": learning_reward,
    }

    if reward_type not in reward_functions:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available types: {list(reward_functions.keys())}"
        )

    return reward_functions[reward_type]
