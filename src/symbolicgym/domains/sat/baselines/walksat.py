"""
WalkSAT integration for SymbolicGym SAT domain.
Requires walksat binary in PATH.
"""
import os
import subprocess
import tempfile


def run_walksat(cnf_clauses, num_vars, timeout=30, max_flips=10000000):
    """
    Run WalkSAT on a CNF problem.
    Args:
        cnf_clauses: list of lists (ints)
        num_vars: int
        timeout: seconds
        max_flips: int
    Returns:
        result: dict with 'satisfiable', 'solution', 'stats'
    """
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(f"p cnf {num_vars} {len(cnf_clauses)}\n")
        for clause in cnf_clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")
        f.flush()
        cnf_path = f.name
    try:
        proc = subprocess.run(
            ["walksat", "-maxflips", str(max_flips), cnf_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout
        sat = "solution" in output.lower() or "satisfiable" in output.lower()
        solution = []
        for line in output.splitlines():
            if line.startswith("v "):
                solution += [int(x) for x in line[2:].split() if x != "0"]
        stats = output
    finally:
        os.remove(cnf_path)
    return {"satisfiable": sat, "solution": solution, "stats": stats}
