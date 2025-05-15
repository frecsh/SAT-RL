"""
Reward calculation functions for SymbolicGym SAT environments.
"""


def sparse_reward(
    new_satisfied_clauses, num_clauses, prev_ratio, solved, step, max_steps
):
    """Sparse reward: 1.0 if solved, else 0.0."""
    return 1.0 if solved else 0.0


def dense_reward(
    new_satisfied_clauses, num_clauses, prev_ratio, solved, step, max_steps
):
    """Dense reward: proportional to satisfaction ratio, clamped to [0.0, 1.0]."""
    if num_clauses == 0:
        return 0.0
    ratio = float(new_satisfied_clauses) / num_clauses
    return max(0.0, min(1.0, ratio))


def learning_reward(
    new_satisfied_clauses, num_clauses, prev_ratio, solved, step, max_steps
):
    """Learning reward: difference in satisfaction ratio, bonus for solving, clamped to [-1.0, 1.0]."""
    if num_clauses == 0:
        return 0.0
    ratio = float(new_satisfied_clauses) / num_clauses
    reward = ratio - prev_ratio
    if solved:
        reward += 1.0
    return max(-1.0, min(1.0, reward))
