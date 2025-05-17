"""Z3 baselines for SymbolicGym."""

from z3 import Solver, sat


def baseline_z3_solve(constraints):
    s = Solver()
    for c in constraints:
        s.add(c)
    result = s.check()
    if result == sat:
        m = s.model()
        return {"satisfiable": True, "model": m}
    else:
        return {"satisfiable": False, "model": None}
