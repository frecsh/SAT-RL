#!/usr/bin/env python3
"""
Functions for generating and manipulating SAT problems.
"""

import random
import numpy as np
from typing import List, Tuple, Set, Optional

def generate_sat_problem(n_vars: int, n_clauses: int, clause_length: int = 3, seed: Optional[int] = None) -> List[List[int]]:
    """
    Generate a random k-SAT problem.
    
    Args:
        n_vars: Number of variables
        n_clauses: Number of clauses
        clause_length: Number of literals per clause (default: 3 for 3-SAT)
        seed: Random seed for reproducibility
        
    Returns:
        List of clauses, where each clause is a list of literals
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    clauses = []
    
    for _ in range(n_clauses):
        # Select random variables for this clause
        vars_in_clause = random.sample(range(1, n_vars + 1), clause_length)
        
        # Randomly negate some variables
        clause = [var if random.random() < 0.5 else -var for var in vars_in_clause]
        
        clauses.append(clause)
    
    return clauses

def load_sat_from_file(filepath: str) -> Tuple[List[List[int]], int]:
    """
    Load a SAT problem from a DIMACS CNF file.
    
    Args:
        filepath: Path to the CNF file
        
    Returns:
        Tuple of (clauses, n_vars)
    """
    clauses = []
    n_vars = 0
    
    with open(filepath, 'r', errors='ignore') as f:
        current_clause = []
        
        for line in f:
            line = line.strip()
            
            # Skip comments
            if line.startswith('c'):
                continue
            
            # Process problem line (p cnf <vars> <clauses>)
            if line.startswith('p cnf'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        n_vars = int(parts[2])
                    except ValueError:
                        # Try to extract from filename if parsing fails
                        filename = filepath.split('/')[-1]
                        if 'uf' in filename:
                            try:
                                n_vars = int(filename.split('-')[0].replace('uf', ''))
                            except ValueError:
                                n_vars = 100  # Default if all else fails
                continue
            
            # Process literals
            try:
                for token in line.split():
                    if token in ['%', '0%']:  # Handle special characters
                        continue
                    
                    lit = int(token)
                    
                    if lit == 0:  # End of clause
                        if current_clause:
                            clauses.append(current_clause)
                            current_clause = []
                    else:
                        current_clause.append(lit)
            except ValueError:
                # Skip lines with parsing errors
                continue
        
        # Don't forget the last clause if file doesn't end with 0
        if current_clause:
            clauses.append(current_clause)
    
    # If we couldn't determine n_vars from the file, estimate from the clauses
    if n_vars == 0 and clauses:
        n_vars = max([abs(lit) for clause in clauses for lit in clause])
    
    return clauses, n_vars

def is_satisfied(clauses: List[List[int]], assignment: List[int]) -> bool:
    """
    Check if a given assignment satisfies all clauses.
    
    Args:
        clauses: List of clauses, where each clause is a list of literals
        assignment: List of variable assignments (positive = True, negative = False)
        
    Returns:
        True if all clauses are satisfied, False otherwise
    """
    # Convert assignment list to a set for O(1) lookups
    true_lits = set(assignment)
    
    for clause in clauses:
        # A clause is satisfied if any of its literals is satisfied
        satisfied = False
        
        for lit in clause:
            if lit in true_lits:
                satisfied = True
                break
        
        if not satisfied:
            return False
    
    return True

def count_satisfied_clauses(clauses: List[List[int]], assignment: List[int]) -> int:
    """
    Count how many clauses are satisfied by a given assignment.
    
    Args:
        clauses: List of clauses, where each clause is a list of literals
        assignment: List of variable assignments (positive = True, negative = False)
        
    Returns:
        Number of satisfied clauses
    """
    # Convert assignment list to a set for O(1) lookups
    true_lits = set(assignment)
    
    satisfied_count = 0
    
    for clause in clauses:
        # A clause is satisfied if any of its literals is satisfied
        for lit in clause:
            if lit in true_lits:
                satisfied_count += 1
                break
    
    return satisfied_count

def flip_variable(solution: List[int], var_idx: int) -> List[int]:
    """
    Flip the value of a variable in the solution.
    
    Args:
        solution: List of variable assignments
        var_idx: Index of the variable to flip
        
    Returns:
        New solution with the variable flipped
    """
    new_solution = solution.copy()
    new_solution[var_idx] = -new_solution[var_idx]
    return new_solution

def random_assignment(n_vars: int, seed: Optional[int] = None) -> List[int]:
    """
    Generate a random assignment for a SAT problem.
    
    Args:
        n_vars: Number of variables
        seed: Random seed for reproducibility
        
    Returns:
        List of literals representing the assignment (positive = True, negative = False)
    """
    if seed is not None:
        random.seed(seed)
    
    return [i if random.random() < 0.5 else -i for i in range(1, n_vars + 1)]

def random_walksat(clauses: List[List[int]], n_vars: int, 
                   max_flips: int = 100000, 
                   random_probability: float = 0.5,
                   seed: Optional[int] = None) -> Tuple[Optional[List[int]], bool]:
    """
    Simple WalkSAT implementation to solve SAT problems.
    
    Args:
        clauses: List of clauses, where each clause is a list of literals
        n_vars: Number of variables
        max_flips: Maximum number of variable flips
        random_probability: Probability of making a random choice
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (assignment, solved) where assignment is the variable assignment
        and solved indicates whether all clauses are satisfied
    """
    if seed is not None:
        random.seed(seed)
    
    # Start with a random assignment
    assignment = set([i if random.random() < 0.5 else -i for i in range(1, n_vars + 1)])
    
    for _ in range(max_flips):
        # Find unsatisfied clauses
        unsatisfied = []
        
        for i, clause in enumerate(clauses):
            if not any(lit in assignment for lit in clause):
                unsatisfied.append(i)
        
        # If all clauses are satisfied, we're done
        if not unsatisfied:
            return list(assignment), True
        
        # Choose a random unsatisfied clause
        clause_idx = random.choice(unsatisfied)
        clause = clauses[clause_idx]
        
        # Choose which variable to flip
        if random.random() < random_probability:
            # Random flip
            var = abs(random.choice(clause))
        else:
            # Greedy flip - choose the variable that maximizes satisfied clauses
            best_var = None
            best_score = -1
            
            for lit in clause:
                var = abs(lit)
                
                # Flip the variable
                if var in assignment:
                    assignment.remove(var)
                    assignment.add(-var)
                else:
                    assignment.remove(-var)
                    assignment.add(var)
                
                # Count satisfied clauses
                score = sum(1 for c in clauses if any(l in assignment for l in c))
                
                # Restore the variable
                if var in assignment:
                    assignment.remove(var)
                    assignment.add(-var)
                else:
                    assignment.remove(-var)
                    assignment.add(var)
                
                if score > best_score:
                    best_score = score
                    best_var = var
            
            var = best_var
        
        # Flip the chosen variable
        if var in assignment:
            assignment.remove(var)
            assignment.add(-var)
        else:
            assignment.remove(-var)
            assignment.add(var)
    
    # If we get here, we've exceeded max_flips
    return list(assignment), False

# Define some test problems
SMALL_PROBLEM = {
    "name": "small_problem",
    "num_vars": 10,
    "clauses": generate_sat_problem(10, 30)
}

MEDIUM_PROBLEM = {
    "name": "medium_problem",
    "num_vars": 20,
    "clauses": generate_sat_problem(20, 85)
}

LARGE_PROBLEM = {
    "name": "large_problem",
    "num_vars": 50,
    "clauses": generate_sat_problem(50, 210)
}

HARD_PROBLEM = {
    "name": "hard_problem",
    "num_vars": 30,
    "clauses": generate_sat_problem(30, 130)
}

EASY_PROBLEM = {
    "name": "easy_problem",
    "num_vars": 15,
    "clauses": generate_sat_problem(15, 40)
}

PROBLEM_COLLECTION = [
    SMALL_PROBLEM,
    MEDIUM_PROBLEM,
    LARGE_PROBLEM,
    HARD_PROBLEM,
    EASY_PROBLEM
]

if __name__ == "__main__":
    # Example usage
    problem = generate_sat_problem(20, 85)
    print(f"Generated random 3-SAT problem with 20 variables and 85 clauses")
    
    # Try to solve it with WalkSAT
    assignment, solved = random_walksat(problem, 20)
    
    if solved:
        print(f"Found solution: {assignment}")
    else:
        print(f"Could not find solution after max flips")