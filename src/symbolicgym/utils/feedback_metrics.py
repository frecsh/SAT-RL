"""
Utility for mapping feedback vector indices to metric names for SatEnv/SATFeedback.
"""

FEEDBACK_VECTOR_METRICS = [
    "clause_satisfaction",  # 0
    "variable_decisiveness",  # 1
    "search_diversity",  # 2
    "constraint_tension",  # 3
    "proof_progress",  # 4
    "clause_centrality",  # 5
    "assignment_entropy",  # 6
    "clause_length_var",  # 7
]

FEEDBACK_VECTOR_INDEX_TO_NAME = {
    i: name for i, name in enumerate(FEEDBACK_VECTOR_METRICS)
}
FEEDBACK_VECTOR_NAME_TO_INDEX = {
    name: i for i, name in enumerate(FEEDBACK_VECTOR_METRICS)
}


def feedback_index_to_metric(idx):
    """Return the metric name for a given feedback vector index."""
    if 0 <= idx < len(FEEDBACK_VECTOR_METRICS):
        return FEEDBACK_VECTOR_METRICS[idx]
    return f"unknown_{idx}"


# Example usage:
#   from symbolicgym.utils.feedback_metrics import FEEDBACK_VECTOR_METRICS, FEEDBACK_VECTOR_INDEX_TO_NAME, FEEDBACK_VECTOR_NAME_TO_INDEX
#   metric_name = FEEDBACK_VECTOR_INDEX_TO_NAME[2]  # 'search_diversity'
#   idx = FEEDBACK_VECTOR_NAME_TO_INDEX['proof_progress']
