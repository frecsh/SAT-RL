"""Loader for industrial SATCOMP traces in DIMACS CNF format for SymbolicGym."""


def load_dimacs_cnf(filepath):
    """Load a SAT problem from a DIMACS CNF file.
    Returns: {'num_vars': int, 'clauses': List[List[int]]}.
    """
    clauses = []
    num_vars = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                num_vars = int(parts[2])
                continue
            # Clause line
            clause = [int(x) for x in line.split() if x != "0"]
            if clause:
                clauses.append(clause)
    if num_vars is None:
        raise ValueError("DIMACS file missing problem line (p cnf ...)")
    return {"num_vars": num_vars, "clauses": clauses}
