# Mini SAT Solver (DPLL-based)
# We'll implement a basic SAT solver using recursion, backtracking, and unit propagation

from typing import List, Dict, Optional

# A clause is a list of integers. Positive = var is True, Negative = var is False
Clause = List[int]
Formula = List[Clause]
Assignment = Dict[int, bool]

def is_satisfied(clause: Clause, assignment: Assignment) -> bool:
    return any((lit > 0 and assignment.get(abs(lit), None) == True) or
               (lit < 0 and assignment.get(abs(lit), None) == False) for lit in clause)

def is_conflict(clause: Clause, assignment: Assignment) -> bool:
    return all((lit > 0 and assignment.get(abs(lit), None) == False) or
               (lit < 0 and assignment.get(abs(lit), None) == True) for lit in clause)

def unit_clause(formula: Formula, assignment: Assignment) -> Optional[int]:
    for clause in formula:
        unassigned = [lit for lit in clause if abs(lit) not in assignment]
        if len(unassigned) == 1:
            return unassigned[0]
    return None

def simplify(formula: Formula, assignment: Assignment) -> Formula:
    new_formula = []
    for clause in formula:
        if is_satisfied(clause, assignment):
            continue
        new_clause = [lit for lit in clause if abs(lit) not in assignment]
        new_formula.append(new_clause)
    return new_formula

def dpll(formula: Formula, assignment: Assignment = {}) -> Optional[Assignment]:
    # Unit propagation
    while True:
        unit = unit_clause(formula, assignment)
        if unit is None:
            break
        assignment[abs(unit)] = unit > 0
        formula = simplify(formula, assignment)

    # Check if satisfied
    if all(is_satisfied(clause, assignment) for clause in formula):
        return assignment

    # Check if conflict
    if any(is_conflict(clause, assignment) for clause in formula):
        return None

    # Choose unassigned variable
    unassigned_vars = {abs(lit) for clause in formula for lit in clause} - assignment.keys()
    if not unassigned_vars:
        return None

    var = unassigned_vars.pop()
    for val in [True, False]:
        new_assignment = assignment.copy()
        new_assignment[var] = val
        result = dpll(formula, new_assignment)
        if result is not None:
            return result

    return None

# Example usage:
if __name__ == "__main__":
    # (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
    formula = [
        [1, 2],
        [-1, 3],
        [-2, -3]
    ]

    solution = dpll(formula)
    if solution:
        print("SATISFIABLE:", solution)
    else:
        print("UNSATISFIABLE")
