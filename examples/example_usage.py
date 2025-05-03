#!/usr/bin/env python
"""
Example usage of the unified SAT solver architecture.
This demonstrates how to use the new solver interfaces, registry, and configuration system.
"""

import os
import sys
import random
import time
import argparse
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our solver package
from solvers import (
    SolverRegistry, 
    SolverBase, 
    SolverResult, 
    SolverStatus,
    get_config, 
    load_config
)

def generate_random_problem(num_vars: int, num_clauses: int, clause_size: int = 3) -> List[List[int]]:
    """
    Generate a random SAT problem instance.
    
    Args:
        num_vars: Number of variables
        num_clauses: Number of clauses
        clause_size: Number of literals per clause
        
    Returns:
        List of clauses, where each clause is a list of literals
    """
    clauses = []
    for _ in range(num_clauses):
        # Generate a random clause
        variables = random.sample(range(1, num_vars + 1), clause_size)
        clause = [var * random.choice([-1, 1]) for var in variables]
        clauses.append(clause)
    
    return clauses

def verify_solution(clauses: List[List[int]], solution: List[int]) -> bool:
    """
    Verify that a solution satisfies all clauses.
    
    Args:
        clauses: List of clauses
        solution: List of literals representing the solution
        
    Returns:
        True if all clauses are satisfied, False otherwise
    """
    # Convert solution to a dict for easier lookup
    assignment = {abs(lit): lit > 0 for lit in solution}
    
    # Check each clause
    for i, clause in enumerate(clauses):
        clause_satisfied = False
        for lit in clause:
            var = abs(lit)
            if var in assignment and ((lit > 0) == assignment[var]):
                clause_satisfied = True
                break
        
        if not clause_satisfied:
            return False
    
    return True

def main():
    """
    Main function to run example.
    """
    parser = argparse.ArgumentParser(description="SAT Solver Example")
    parser.add_argument("--solver", type=str, default=None, help="Solver to use")
    parser.add_argument("--vars", type=int, default=20, help="Number of variables")
    parser.add_argument("--clauses", type=int, default=85, help="Number of clauses")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--list-solvers", action="store_true", help="List available solvers")
    args = parser.parse_args()
    
    # List available solvers if requested
    if args.list_solvers:
        solvers = SolverRegistry.list_solvers()
        print("Available solvers:")
        for solver in solvers:
            print(f"  - {solver}")
        return
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
        
    # Override configuration with command-line arguments
    if args.vars:
        config["problem.num_vars"] = args.vars
    if args.clauses:
        config["problem.num_clauses"] = args.clauses
    if args.timeout:
        config["solver.timeout"] = args.timeout
    if args.solver:
        config["solver.name"] = args.solver
    
    # Get the solver name from config
    solver_name = config.get("solver.name")
    
    # Generate a random problem
    num_vars = config.get("problem.num_vars")
    num_clauses = config.get("problem.num_clauses")
    
    logger.info(f"Generating random problem with {num_vars} variables and {num_clauses} clauses")
    clauses = generate_random_problem(num_vars, num_clauses)
    
    # Display problem information
    print(f"Problem: {num_vars} variables, {num_clauses} clauses")
    print(f"First 3 clauses: {clauses[:3]}")
    
    try:
        # Create the solver
        solver = SolverRegistry.create(solver_name, num_vars=num_vars)
        print(f"Using solver: {solver_name or 'default'}")
        
        # Add clauses to the solver
        solver.add_clauses(clauses)
        
        # Configure the solver with any specific parameters
        solver_config = {k.split(".", 2)[2]: v 
                       for k, v in config.to_dict().items() 
                       if k.startswith(f"solver.{solver_name}.")}
        if solver_config:
            solver.configure(solver_config)
        
        # Solve the problem
        print("Solving...")
        start_time = time.time()
        result = solver.solve(timeout=config.get("solver.timeout"))
        
        # Display the result
        print(result)
        
        # Verify the solution if one was found
        if result.is_sat and result.solution:
            is_valid = verify_solution(clauses, result.solution)
            print(f"Solution verification: {'Valid' if is_valid else 'Invalid'}")
            
            if is_valid:
                print(f"Solution (first 10): {result.solution[:10]}...")
        elif result.status == SolverStatus.UNSATISFIABLE:
            print("Problem is UNSATISFIABLE")
        else:
            print(f"No complete solution found. Best partial solution satisfies "
                  f"{result.satisfied_clauses}/{result.total_clauses} clauses "
                  f"({result.satisfaction_ratio:.2%})")
        
        # Display solver statistics
        stats = solver.get_statistics()
        print("\nSolver Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())