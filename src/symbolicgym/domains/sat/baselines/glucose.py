"""Glucose 4.2 integration for SymbolicGym SAT domain.
Requires glucose binary in PATH.
"""

from pysat.solvers import Glucose42


def run_glucose(cnf_clauses, num_vars, timeout=30):
    """Run Glucose (via PySAT) on a CNF problem.

    Args:
        cnf_clauses: list of lists (ints)
        num_vars: int
        timeout: seconds (not used, PySAT Glucose does not support timeouts directly)

    Returns:
        result: dict with 'satisfiable', 'solution', 'stats'
    """
    # Glucose42 supports the latest features; you can also use Glucose3 or Glucose4
    with Glucose42(bootstrap_with=cnf_clauses) as solver:
        sat = solver.solve()
        solution = solver.get_model() if sat else []
        stats = solver.accum_stats()
    return {"satisfiable": sat, "solution": solution, "stats": stats}
