"""
Interpretability metrics for SymbolicGym.
"""

import numpy as np
from scipy.stats import entropy


def mean_clause_attention_entropy(attention_matrix):
    """Compute mean entropy of clause attention weights (rows: clauses, cols: variables)."""
    # Normalize rows to sum to 1
    probs = np.array(attention_matrix)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    entropies = entropy(probs, axis=1)
    return float(np.mean(entropies))


def percentage_flips_matching_vsids(agent_flips, oracle_flips):
    """Compute percentage of agent variable flips matching MiniSAT VSIDS picks."""
    if not agent_flips or not oracle_flips:
        return 0.0
    matches = sum(1 for a, o in zip(agent_flips, oracle_flips) if a == o)
    return 100.0 * matches / min(len(agent_flips), len(oracle_flips))


def feedback_vector_alignment(feedback_vectors, human_metrics):
    """Compute alignment (Pearson correlation) between feedback vector dims and human metrics.
    feedback_vectors: np.ndarray shape (n, d)
    human_metrics: np.ndarray shape (n, m)
    Returns: mean absolute correlation across all pairs
    """
    feedback_vectors = np.array(feedback_vectors)
    human_metrics = np.array(human_metrics)
    if feedback_vectors.shape[0] != human_metrics.shape[0]:
        raise ValueError("Mismatched number of samples")
    corrs = []
    for i in range(feedback_vectors.shape[1]):
        for j in range(human_metrics.shape[1]):
            c = np.corrcoef(feedback_vectors[:, i], human_metrics[:, j])[0, 1]
            corrs.append(abs(c))
    return float(np.mean(corrs)) if corrs else 0.0
