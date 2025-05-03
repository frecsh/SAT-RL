#!/usr/bin/env python3
"""
SAT Oracle Solver - Interface for traditional SAT solving methods
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Import available solvers
from sat_problems import generate_sat_problem, random_walksat, is_satisfied, count_satisfied_clauses
try:
    from pysat.solvers import Glucose3, Lingeling, Minisat22
    PYSAT_AVAILABLE = True
except ImportError:
    print("Warning: PySAT not available. Some solving methods will be disabled.")
    PYSAT_AVAILABLE = False

# Try to import the multi-agent oracle implementation
try:
    from multi_q_sat_oracle import MultiQLearningSATOracle
    MULTI_Q_AVAILABLE = True
except ImportError:
    MULTI_Q_AVAILABLE = False

# Try to import the core oracle
try:
    from sat_oracle import SATOracle
    SAT_ORACLE_AVAILABLE = True
except ImportError:
    SAT_ORACLE_AVAILABLE = False


class SATOracleSolver:
    """
    Oracle solver for SAT problems.
    Provides a unified interface to various SAT solving methods.
    """
    
    def __init__(self, n_vars: int, n_clauses: int = None, method: str = "dpll", logs_dir: str = None):
        """
        Initialize the oracle solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            n_clauses: Number of clauses (if None, will use 4.2 * n_vars)
            method: Solving method to use (dpll, cdcl, walksat, or multi-agent)
            logs_dir: Directory to save logs
        """
        self.n_vars = n_vars
        self.n_clauses = n_clauses if n_clauses else int(4.2 * n_vars)
        self.method = method.lower()
        self.logs_dir = logs_dir
        
        # Generate a SAT problem if clauses aren't provided
        self.clauses = generate_sat_problem(n_vars, self.n_clauses)
        
        # Track history for visualization
        self.history = {
            "satisfaction_ratio": [],
            "time": [],
            "solution_changes": []
        }
        
        # Track best solution
        self.best_solution = None
        self.best_satisfied = 0
        self.best_ratio = 0.0
        
        # Validate solver method
        valid_methods = ["dpll", "cdcl", "walksat", "oracle", "multiagent", "multi-agent", 
                         "glucose", "lingeling", "minisat"]
        if self.method not in valid_methods:
            print(f"Warning: Unknown method '{method}'. Falling back to DPLL.")
            self.method = "dpll"
            
        # Check if required dependencies are available
        if (self.method in ["dpll", "cdcl", "glucose", "lingeling", "minisat"] and 
            not PYSAT_AVAILABLE):
            print(f"Warning: Method '{method}' requires PySAT. Falling back to WalkSAT.")
            self.method = "walksat"
        
        if self.method in ["multiagent", "multi-agent"] and not MULTI_Q_AVAILABLE:
            print(f"Warning: Method '{method}' requires multi_q_sat_oracle.py. Falling back to WalkSAT.")
            self.method = "walksat"
    
    def _update_best_solution(self, solution):
        """Update the best solution found so far"""
        satisfied = count_satisfied_clauses(self.clauses, solution)
        ratio = satisfied / len(self.clauses)
        
        if satisfied > self.best_satisfied:
            self.best_solution = solution.copy()
            self.best_satisfied = satisfied
            self.best_ratio = ratio
            
        # Record in history
        self.history["satisfaction_ratio"].append(ratio)
        self.history["time"].append(time.time())
        self.history["solution_changes"].append(solution.copy())
    
    def solve(self) -> Tuple[List[int], int, int]:
        """
        Solve the SAT problem using the selected method.
        
        Returns:
            Tuple of (solution, satisfied_clauses, total_clauses)
        """
        start_time = time.time()
        solution = None
        
        # Use the appropriate solver based on method
        if self.method == "walksat":
            print("Using WalkSAT solver...")
            solution, solved = random_walksat(
                self.clauses, self.n_vars, max_flips=100000, 
                max_tries=30, random_probability=0.3
            )
        
        elif self.method in ["multiagent", "multi-agent"]:
            print("Using Multi-agent Oracle solver...")
            if MULTI_Q_AVAILABLE:
                # Create and run multi-agent solver
                solver = MultiQLearningSATOracle(
                    self.n_vars, self.clauses, 
                    n_agents=5, epsilon=0.1, oracle_weight=0.4
                )
                solution, stats = solver.solve(max_episodes=300, early_stopping=True)
                self.history["satisfaction_ratio"] = stats.get('satisfaction_history', [])
        
        elif self.method == "oracle":
            print("Using core SAT Oracle...")
            if SAT_ORACLE_AVAILABLE:
                # Use the base oracle implementation 
                # SATOracle expects a dictionary with 'clauses' and 'num_vars' keys
                problem_data = {"clauses": self.clauses, "num_vars": self.n_vars}
                oracle = SATOracle(problem_data)
                
                # The method is actually called _find_optimal_solution instead of compute_optimal_solution
                oracle._find_optimal_solution()  # Use the correct method name
                solution = oracle.optimal_solution
                
                # Convert from bit array to literal list
                if solution is not None:
                    solution = [i+1 if bit else -(i+1) for i, bit in enumerate(solution)]
        
        elif self.method in ["dpll", "cdcl", "glucose", "lingeling", "minisat"]:
            print(f"Using PySAT solver: {self.method}...")
            if PYSAT_AVAILABLE:
                # Select appropriate solver
                if self.method == "glucose" or self.method == "cdcl":
                    solver = Glucose3()
                elif self.method == "lingeling":
                    solver = Lingeling()
                elif self.method == "minisat":
                    solver = Minisat22()
                else:
                    # Default to Glucose3
                    solver = Glucose3()
                
                # Add clauses to solver
                for clause in self.clauses:
                    solver.add_clause(clause)
                
                # Solve
                if solver.solve():
                    solution = solver.get_model()
                
                solver.delete()
        
        # If no solution found, return the best partial solution
        if solution is None:
            solution = [np.random.choice([-i, i]) for i in range(1, self.n_vars + 1)]
        
        # Count satisfied clauses
        satisfied = count_satisfied_clauses(self.clauses, solution)
        
        # Update best solution one last time
        self._update_best_solution(solution)
        
        # Log solve time
        solve_time = time.time() - start_time
        print(f"Solved in {solve_time:.2f} seconds. "
              f"Satisfied {satisfied}/{len(self.clauses)} clauses ({satisfied/len(self.clauses):.2%})")
        
        return solution, satisfied, len(self.clauses)
    
    def get_history(self) -> Dict[str, Any]:
        """Get the history of the solving process"""
        return self.history


# Demo code
if __name__ == "__main__":
    # Demo the SAT Oracle Solver
    n_vars = 20
    n_clauses = 85
    
    print(f"Testing SATOracleSolver on {n_vars}-variable, {n_clauses}-clause problem")
    
    # Try different methods
    for method in ["walksat", "dpll", "oracle", "multiagent"]:
        try:
            print(f"\nTrying {method} method:")
            solver = SATOracleSolver(n_vars, n_clauses, method=method)
            solution, satisfied, total = solver.solve()
            print(f"Satisfied: {satisfied}/{total} clauses ({satisfied/total:.2%})")
        except Exception as e:
            print(f"Error with {method} method: {e}")