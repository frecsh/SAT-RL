"""CNF file loading utilities for SymbolicGym."""

from symbolicgym.domains.sat.loaders.satcomp_loader import load_dimacs_cnf


def load_cnf_file(filepath):
    """Load a SAT problem from a DIMACS CNF file. Returns {'num_vars': int, 'clauses': List[List[int]]}."""
    return load_dimacs_cnf(filepath)
