"""MiniSAT 2.2 integration for SymbolicGym SAT domain.
Requires MiniSAT binary in PATH.
"""

from pysat.solvers import Minisat22


def run_minisat(cnf_clauses, num_vars, timeout=30):
    """Run MiniSAT (via PySAT) on a CNF problem.

    Args:
        cnf_clauses: list of lists (ints)
        num_vars: int
        timeout: seconds (ignored, as PySAT does not support timeouts directly)

    Returns:
        result: dict with 'satisfiable', 'solution', 'stats'
    """
    stats = {}
    with Minisat22(bootstrap_with=cnf_clauses, use_timer=True) as solver:
        sat = solver.solve()
        if sat:
            solution = solver.get_model()
        else:
            solution = []
        stats = solver.accum_stats()
        stats["time"] = solver.time()
    return {"satisfiable": sat, "solution": solution, "stats": stats}
