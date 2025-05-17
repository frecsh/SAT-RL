"""Feedback vector dimension interpreter for SymbolicGym."""


def interpret_feedback_vector(feedback_vector, domain):
    """Interpret feedback vector dimensions for a given domain.

    Args:
        feedback_vector: np.ndarray or list
        domain: str
    Returns:
        interpretation: dict
    """
    if domain == "sat":
        return {f"clause_{i}": v for i, v in enumerate(feedback_vector)}
    elif domain == "sympy":
        return {f"feature_{i}": v for i, v in enumerate(feedback_vector)}
    elif domain == "z3":
        return {f"constraint_{i}": v for i, v in enumerate(feedback_vector)}
    else:
        return {f"dim_{i}": v for i, v in enumerate(feedback_vector)}
