"""
Sparse COO matrix representation for SAT incidence graphs.
"""
import numpy as np
from scipy.sparse import coo_matrix


def incidence_coo_matrix(num_vars, clauses):
    """Return a scipy.sparse.coo_matrix (shape: n_clauses x n_vars)."""
    row, col, data = [], [], []
    for i, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            row.append(i)
            col.append(v)
            data.append(1 if lit > 0 else -1)
    n_clauses = len(clauses)
    mat = coo_matrix((data, (row, col)), shape=(n_clauses, num_vars))
    return mat


def sat_incidence_coo(clauses, num_vars):
    """
    Return a sparse COO matrix (shape: [num_vars, num_clauses]) for SAT incidence graph.
    Args:
        clauses: List of lists, each inner list is a clause (ints, positive/negative for polarity)
        num_vars: Number of variables (int)
    Returns:
        coo_matrix: scipy.sparse.coo_matrix, 1 if var appears in clause, 0 otherwise
    """
    row, col, data = [], [], []
    for j, clause in enumerate(clauses):
        for lit in clause:
            row.append(abs(lit) - 1)
            col.append(j)
            data.append(1)
    return coo_matrix((data, (row, col)), shape=(num_vars, len(clauses)))
