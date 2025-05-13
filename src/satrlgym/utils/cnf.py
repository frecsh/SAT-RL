"""
CNF file handling utilities.

This module provides functions for loading, parsing, and manipulating
CNF formulas in DIMACS format, as well as checking solutions and computing satisfied clauses.
"""

import os
from typing import Any, TextIO


def load_cnf_file(file_path: str) -> tuple[list[list[int]], dict[str, Any]]:
    """
    Load a CNF formula from a DIMACS file.

    Args:
        file_path: Path to the CNF file in DIMACS format

    Returns:
        Tuple of (formula, metadata)
        - formula: List of clauses (each clause is a list of literals)
        - metadata: Dictionary with metadata (variables, clauses, etc.)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CNF file not found: {file_path}")

    with open(file_path) as f:
        dimacs_content = f.read()

    return parse_dimacs(dimacs_content)


def parse_dimacs(source: str | TextIO) -> tuple[list[list[int]], dict[str, Any]]:
    """
    Parse CNF formula from DIMACS format.

    Args:
        source: DIMACS content as a string or file-like object

    Returns:
        Tuple of (formula, metadata)
        - formula: List of clauses (each clause is a list of literals)
        - metadata: Dictionary with metadata (variables, clauses, etc.)

    Raises:
        ValueError: If the format is invalid
    """
    # Convert string to lines if needed
    if isinstance(source, str):
        lines = source.strip().split("\n")
    else:
        lines = source.readlines()

    formula = []
    metadata = {"comments": [], "num_variables": 0, "num_clauses": 0, "source_info": ""}

    found_problem_line = False
    current_clause = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Handle comments
        if line.startswith("c"):
            metadata["comments"].append(line[1:].strip())
            continue

        # Handle problem line
        if line.startswith("p"):
            if found_problem_line:
                raise ValueError("Multiple problem lines in CNF file")

            parts = line.split()
            if len(parts) < 4 or parts[1] != "cnf":
                raise ValueError(f"Invalid problem line: {line}")

            try:
                metadata["num_variables"] = int(parts[2])
                metadata["num_clauses"] = int(parts[3])
            except ValueError:
                raise ValueError(f"Invalid numbers in problem line: {line}")

            found_problem_line = True
            continue

        # Handle clause data
        values = [int(x) for x in line.split()]

        # Check for continuation or end of clause
        for value in values:
            if value == 0:
                if current_clause:
                    formula.append(current_clause)
                    current_clause = []
            else:
                current_clause.append(value)

    # Add the last clause if needed
    if current_clause:
        formula.append(current_clause)

    # Validate
    if not found_problem_line:
        raise ValueError("No problem line found in CNF file")

    if len(formula) != metadata["num_clauses"]:
        raise ValueError(
            f"Expected {metadata['num_clauses']} clauses, but found {len(formula)}"
        )

    return formula, metadata


def formula_to_dimacs(
    formula: list[list[int]],
    num_variables: int | None = None,
    comments: list[str] | None = None,
) -> str:
    """
    Convert a formula to DIMACS format.

    Args:
        formula: List of clauses (each clause is a list of literals)
        num_variables: Number of variables (computed if not provided)
        comments: List of comment lines to include

    Returns:
        DIMACS format string representation
    """
    if comments is None:
        comments = []

    # Calculate the number of variables if not provided
    if num_variables is None:
        all_vars = set()
        for clause in formula:
            all_vars.update(abs(lit) for lit in clause)
        num_variables = max(all_vars) if all_vars else 0

    # Build the DIMACS string
    lines = []

    # Add comments
    for comment in comments:
        lines.append(f"c {comment}")

    # Add problem line
    lines.append(f"p cnf {num_variables} {len(formula)}")

    # Add clauses
    for clause in formula:
        lines.append(" ".join(map(str, clause)) + " 0")

    return "\n".join(lines)


def save_cnf_file(
    file_path: str,
    formula: list[list[int]],
    num_variables: int | None = None,
    comments: list[str] | None = None,
) -> None:
    """
    Save a CNF formula to a DIMACS file.

    Args:
        file_path: Path to save the CNF file
        formula: List of clauses (each clause is a list of literals)
        num_variables: Number of variables (computed if not provided)
        comments: List of comment lines to include
    """
    dimacs_str = formula_to_dimacs(formula, num_variables, comments)

    with open(file_path, "w") as f:
        f.write(dimacs_str)


def check_solution(
    formula: list[list[int]], assignment: dict[int, bool], partial: bool = False
) -> bool | None:
    """
    Check if a variable assignment satisfies a CNF formula.

    Args:
        formula: List of clauses, each clause being a list of literals
        assignment: Dictionary mapping variable indices to Boolean values
        partial: Whether to allow partial assignments

    Returns:
        True if the assignment satisfies the formula,
        False if it does not satisfy the formula,
        None if the assignment is partial and partial=True
    """
    result = True

    for clause in formula:
        clause_satisfied = False
        clause_unknown = False

        for literal in clause:
            var_idx = abs(literal)
            expected_value = literal > 0  # True if positive, False if negative

            if var_idx not in assignment:
                if partial:
                    clause_unknown = True
                    break
                continue

            if assignment[var_idx] == expected_value:
                clause_satisfied = True
                break

        if not clause_satisfied and not clause_unknown:
            result = False
            break

    if partial and result and clause_unknown:
        return None  # Partial assignment, no conclusion yet

    return result


def compute_satisfied_clauses(
    formula: list[list[int]], assignment: dict[int, bool], partial: bool = False
) -> int:
    """
    Count the number of clauses satisfied by an assignment.

    Args:
        formula: List of clauses, each clause being a list of literals
        assignment: Dictionary mapping variable indices to Boolean values
        partial: Whether to allow partial assignments

    Returns:
        Number of satisfied clauses
    """
    satisfied_count = 0

    for clause in formula:
        clause_satisfied = False

        for literal in clause:
            var_idx = abs(literal)
            expected_value = literal > 0  # True if positive, False if negative

            if var_idx in assignment and assignment[var_idx] == expected_value:
                clause_satisfied = True
                break

        if clause_satisfied:
            satisfied_count += 1

    return satisfied_count
