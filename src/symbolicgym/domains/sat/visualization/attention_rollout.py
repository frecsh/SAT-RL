"""Attention rollout for GNN (Graph-LIME style) for SymbolicGym SAT domain."""

import numpy as np


def attention_rollout(attn_weights):
    """Compute attention rollout for a stack of attention weights.

    Args:
        attn_weights: list of np.ndarray (each shape: N x N)

    Returns:
        rollout: np.ndarray (N x N)
    """
    result = np.eye(attn_weights[0].shape[0])
    for attn in attn_weights:
        result = attn @ result
    return result
