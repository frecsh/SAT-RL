"""Example usage of SymbolicGym CNF loader utility."""

from symbolicgym.utils.cnf import load_cnf_file

# Example: create a small DIMACS file and load it
cnf_content = """
c Example CNF
p cnf 4 3
1 -2 0
2 3 -1 0
4 0
"""
with open("example.cnf", "w") as f:
    f.write(cnf_content)

problem = load_cnf_file("example.cnf")
print("Number of variables:", problem["num_vars"])
print("Clauses:", problem["clauses"])

import os

os.remove("example.cnf")
