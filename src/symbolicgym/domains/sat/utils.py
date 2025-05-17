"""SAT-specific utility functions for SymbolicGym SAT domain."""
# TODO: Move SAT utilities here from legacy codebase.


def evaluate_clause(clause, assignments):
    """Return True if the clause is satisfied by the current assignments."""
    return any(
        (lit > 0 and assignments.get(abs(lit), None) is True)
        or (lit < 0 and assignments.get(abs(lit), None) is False)
        for lit in clause
    )


def is_satisfied(clauses, assignments):
    """Return True if all clauses are satisfied."""
    return all(evaluate_clause(clause, assignments) for clause in clauses)


def random_assignment(num_vars):
    """Generate a random assignment for num_vars variables."""
    import random

    return {i + 1: random.choice([True, False]) for i in range(num_vars)}
