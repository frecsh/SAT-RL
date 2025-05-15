"""
Representational Similarity Analysis (RSA) for SymbolicGym.
"""
import numpy as np


def compute_rsa(matrix_a, matrix_b):
    """
    Compute RSA (correlation of similarity matrices) between two sets of representations.
    Args:
        matrix_a: np.ndarray (N x D)
        matrix_b: np.ndarray (N x D)
    Returns:
        rsa_score: float
    """

    def sim(mat):
        return np.corrcoef(mat)

    sim_a = sim(matrix_a)
    sim_b = sim(matrix_b)
    # Flatten upper triangle
    iu = np.triu_indices_from(sim_a, k=1)
    return np.corrcoef(sim_a[iu], sim_b[iu])[0, 1]
