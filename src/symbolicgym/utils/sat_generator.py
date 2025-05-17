"""
Random SAT instance generator for benchmarking and experiment suites.
"""
import random

import numpy as np


def generate_random_ksat(n_vars, n_clauses, k=3, seed=None):
    """Generate a random k-SAT formula as a list of clauses."""
    rng = np.random.default_rng(seed)
    clauses = []
    for _ in range(n_clauses):
        vars_ = rng.choice(np.arange(1, n_vars + 1), size=k, replace=False)
        signs = rng.choice([-1, 1], size=k)
        clause = list(vars_ * signs)
        clauses.append(clause)
    return {"num_vars": n_vars, "clauses": clauses}


def batch_generate_ksat(n_vars, n_clauses, k=3, n_instances=10, seed=None):
    """Batch-generate random k-SAT formulas."""
    seeds = [seed + i if seed is not None else None for i in range(n_instances)]
    return [generate_random_ksat(n_vars, n_clauses, k, s) for s in seeds]


def save_cnf(formula, path):
    """Save a formula dict to DIMACS CNF file."""
    with open(path, "w") as f:
        f.write(f"p cnf {formula['num_vars']} {len(formula['clauses'])}\n")
        for clause in formula["clauses"]:
            f.write(" ".join(str(lit) for lit in clause) + " 0\n")


def load_cnf(path):
    """Load a DIMACS CNF file into a formula dict."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("c")]
    header = [l for l in lines if l.startswith("p")][0]
    n_vars, n_clauses = map(int, header.split()[2:4])
    clause_lines = [l for l in lines if not l.startswith("p")]
    clauses = [[int(x) for x in line.split() if int(x) != 0] for line in clause_lines]
    return {"num_vars": n_vars, "clauses": clauses}
