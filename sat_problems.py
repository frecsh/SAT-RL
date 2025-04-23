import random
import numpy as np

# Define the generate_random_sat function first
def generate_random_sat(num_vars, num_clauses, clause_length=3):
    """Generate a random SAT problem"""
    clauses = []
    for _ in range(num_clauses):
        # Generate a clause with unique variables
        variables = random.sample(range(1, num_vars + 1), clause_length)
        clause = []
        for var in variables:
            sign = random.choice([1, -1])
            clause.append(var * sign)
        clauses.append(clause)
    
    return {
        "name": f"random_{num_vars}v_{num_clauses}c",
        "clauses": clauses,
        "num_vars": num_vars
    }

# Collection of SAT problems for testing

# Original small problem (3 variables, 3 clauses)
SMALL_PROBLEM = {
    "name": "small_basic",
    "clauses": [[1, -2, 3], [-1, 2], [2, -3]],
    "num_vars": 3
}

# Medium sized problem (10 variables, 16 clauses) 
MEDIUM_PROBLEM = {
    "name": "medium_standard",
    "clauses": [
        [1, -3, 5], [-1, 2, 7], [2, -4, -8], [3, 6, -9],
        [-2, 5, 10], [-5, -7, 8], [4, 9, -10], [-3, -6, 8],
        [1, -7, -10], [3, -5, -8], [2, 4, 9], [-1, -4, 6],
        [5, 8, -9], [-2, -6, 10], [1, 3, -8], [-4, 7, 9]
    ],
    "num_vars": 10
}

# Hard problem (15 variables, 40 clauses)
HARD_PROBLEM = {
    "name": "hard_complex",
    "clauses": [
        [1, -2, 3], [-1, 4, 5], [2, -6, 7], [-3, -4, 8], 
        [5, -7, -9], [6, 10, 11], [-8, -10, 12], [9, -11, -13],
        [1, -5, -10], [-2, 6, -12], [3, 7, 13], [4, -8, 14],
        [-6, -9, 15], [8, 10, -14], [-7, 11, -15], [1, 2, 9],
        [-3, -5, -11], [4, 6, 12], [-1, 8, 13], [2, -10, 14],
        [-4, 7, 15], [3, 9, -12], [5, -8, -13], [-2, -7, -14],
        [1, 10, -15], [4, -6, 11], [-5, 7, 12], [2, -9, -13],
        [3, 6, 14], [-1, -8, 15], [5, -10, -11], [-3, 8, -12],
        [-4, 9, 13], [6, -7, -14], [1, -5, 15], [-2, 4, 11],
        [3, -6, -13], [7, 8, 10], [-9, 12, 14], [5, -11, -15]
    ],
    "num_vars": 15
}

# A problem known to be UNSAT (unsatisfiable)
UNSAT_PROBLEM = {
    "name": "unsat_example",
    "clauses": [
        [1], [2], [3], [-1, -2, -3]  # Forces 1=True, 2=True, 3=True, but also one must be False
    ],
    "num_vars": 3
}

# Phase transition problem (clause-to-var ratio ~4.3, known to be hard)
PHASE_PROBLEM = generate_random_sat(20, 85)  # 20 variables, 85 clauses
PHASE_PROBLEM["name"] = "phase_transition"

def is_satisfiable(problem, max_checks=10000):
    """Check if a SAT problem is satisfiable (brute force for small problems)"""
    num_vars = problem["num_vars"]
    clauses = problem["clauses"]
    
    if num_vars > 10:
        # For larger problems, try random assignments
        for _ in range(max_checks):
            assignment = {var: random.choice([True, False]) for var in range(1, num_vars+1)}
            if check_assignment(assignment, clauses):
                return True, assignment
        return None, None  # Unknown if satisfiable
        
    # For small problems, check all possible assignments (2^n)
    for i in range(2**num_vars):
        # Convert i to binary representation as assignment
        assignment = {}
        for var in range(1, num_vars + 1):
            assignment[var] = ((i >> (var-1)) & 1) == 1  # Convert to boolean
        
        if check_assignment(assignment, clauses):
            return True, assignment
            
    return False, None  # Definitely unsatisfiable

def check_assignment(assignment, clauses):
    """Check if an assignment satisfies all clauses"""
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var = abs(lit)
            if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False
    return True  # All clauses are satisfied

# Generate a collection of random problems of different sizes
PROBLEM_COLLECTION = [
    SMALL_PROBLEM,
    MEDIUM_PROBLEM,
    HARD_PROBLEM,
    UNSAT_PROBLEM,
    generate_random_sat(5, 10),
    generate_random_sat(10, 20),
    generate_random_sat(15, 35),
    PHASE_PROBLEM
]