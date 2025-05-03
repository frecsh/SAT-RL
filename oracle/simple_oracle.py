"""
Simple oracle implementation for SAT solving based on DPLL algorithm.
"""

import numpy as np
import time
import random
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

from .base_oracle import OracleBase

# Set up logging
logger = logging.getLogger(__name__)

class SimpleDPLLOracle(OracleBase):
    """
    A simple oracle implementation based on the DPLL algorithm.
    Provides guidance based on unit propagation and symbolic reasoning.
    """
    
    def __init__(
        self,
        clauses: List[List[int]],
        num_vars: int,
        oracle_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the DPLL-based oracle.
        
        Args:
            clauses: List of clauses, where each clause is a list of literals
            num_vars: Number of variables in the problem
            oracle_config: Configuration parameters for the oracle
        """
        super().__init__(clauses, num_vars, oracle_config)
        
        # Additional initialization for DPLL oracle
        self.max_depth = oracle_config.get('max_depth', 10)
        self.timeout = oracle_config.get('timeout', 1.0)  # seconds
        self.rng = random.Random(oracle_config.get('seed', None))
        
        # Variable activity heuristics
        self.variable_activity = np.ones(num_vars + 1)  # 1-indexed
    
    def query(self, 
             state: Dict[str, np.ndarray], 
             available_actions: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Query the oracle for guidance based on the current state.
        Uses unit propagation and basic heuristics to suggest actions.
        
        Args:
            state: Current state observation
            available_actions: Optional list of available actions
            
        Returns:
            Dictionary containing oracle guidance
        """
        start_time = time.time()
        self.stats['queries'] += 1
        
        # Extract current assignment
        if 'assignment' in state:
            assignment = state['assignment'].copy()
        else:
            logger.warning("No assignment found in state, using empty assignment")
            assignment = np.zeros(self.num_vars + 1, dtype=np.int8)
        
        # Find unassigned variables
        unassigned = [i for i in range(1, self.num_vars + 1) if assignment[i] == 0]
        
        if not unassigned:
            # All variables assigned, nothing to recommend
            result = {
                "recommended_action": None,
                "confidence": 0.0,
                "explanation": "All variables already assigned"
            }
        else:
            # Try unit propagation first
            unit_prop_result = self._unit_propagation(assignment)
            
            if unit_prop_result['propagations']:
                # Unit propagation found implications
                var = unit_prop_result['propagations'][0][0]
                value = unit_prop_result['propagations'][0][1]
                
                # Convert to action index
                action = (var - 1) + (1 if value > 0 else 0) * self.num_vars
                
                result = {
                    "recommended_action": action,
                    "confidence": 0.9,  # High confidence for unit propagation
                    "explanation": f"Unit propagation: var {var} must be {value > 0}"
                }
            else:
                # Use VSIDS-like heuristic to pick highest activity variable
                var_activities = [(i, self.variable_activity[i]) for i in unassigned]
                var_activities.sort(key=lambda x: x[1], reverse=True)
                
                best_var = var_activities[0][0]
                
                # Try both polarities and see which one satisfies more clauses
                pos_assignment = assignment.copy()
                pos_assignment[best_var] = 1
                pos_satisfied = self._count_satisfied_clauses(pos_assignment)
                
                neg_assignment = assignment.copy()
                neg_assignment[best_var] = -1
                neg_satisfied = self._count_satisfied_clauses(neg_assignment)
                
                if pos_satisfied >= neg_satisfied:
                    action = (best_var - 1) + 1 * self.num_vars  # Positive value
                    value = 1
                else:
                    action = (best_var - 1) + 0 * self.num_vars  # Negative value
                    value = -1
                
                result = {
                    "recommended_action": action,
                    "confidence": 0.7,  # Medium confidence for heuristic choice
                    "explanation": f"VSIDS heuristic: var {best_var} to {value > 0}"
                }
        
        # Add action values if available_actions were provided
        if available_actions:
            action_values = np.zeros(len(available_actions))
            
            # Assign values based on unit propagation and heuristics
            for i, action in enumerate(available_actions):
                var_id = (action % self.num_vars) + 1
                value = 1 if action // self.num_vars else -1
                
                # Check if this matches any unit propagation
                matches_unit_prop = any(var_id == var and value == val 
                                        for var, val in unit_prop_result['propagations'])
                
                # Higher value for unit propagations
                if matches_unit_prop:
                    action_values[i] = 0.9
                else:
                    # Value based on variable activity
                    action_values[i] = min(0.7, self.variable_activity[var_id] / 
                                        max(1, np.max(self.variable_activity)))
            
            result["action_values"] = action_values
        
        # Update query time statistics
        query_time = time.time() - start_time
        self.stats['avg_query_time'] = ((self.stats['avg_query_time'] * (self.stats['queries'] - 1) + 
                                      query_time) / self.stats['queries'])
        
        return result
    
    def _unit_propagation(self, assignment: np.ndarray) -> Dict[str, Any]:
        """
        Perform unit propagation on the current assignment.
        
        Args:
            assignment: Current variable assignments
            
        Returns:
            Dictionary containing propagation results
                - propagations: List of (var, value) pairs to be propagated
                - unit_clauses: List of clauses that became unit clauses
        """
        propagations = []
        unit_clauses = []
        
        # Analyze each clause
        for clause_idx, clause in enumerate(self.clauses):
            # Check if this clause is already satisfied
            satisfied = False
            unassigned_lits = []
            
            for lit in clause:
                var_idx = abs(lit)
                if (lit > 0 and assignment[var_idx] > 0) or (lit < 0 and assignment[var_idx] < 0):
                    satisfied = True
                    break
                elif assignment[var_idx] == 0:
                    unassigned_lits.append(lit)
            
            if satisfied:
                continue
                
            if not unassigned_lits:
                # Conflict: no literals can satisfy this clause
                continue
                
            if len(unassigned_lits) == 1:
                # Unit clause found
                lit = unassigned_lits[0]
                var_idx = abs(lit)
                value = 1 if lit > 0 else -1
                propagations.append((var_idx, value))
                unit_clauses.append(clause_idx)
                
        return {
            "propagations": propagations,
            "unit_clauses": unit_clauses
        }
    
    def _count_satisfied_clauses(self, assignment: np.ndarray) -> int:
        """
        Count the number of satisfied clauses for the given assignment.
        
        Args:
            assignment: Variable assignments
            
        Returns:
            Number of satisfied clauses
        """
        count = 0
        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                var_idx = abs(lit)
                if (lit > 0 and assignment[var_idx] > 0) or (lit < 0 and assignment[var_idx] < 0):
                    satisfied = True
                    break
            if satisfied:
                count += 1
                
        return count
    
    def evaluate_solution(self, assignment: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a complete solution.
        
        Args:
            assignment: Variable assignments
            
        Returns:
            Dictionary containing evaluation results
        """
        satisfied_clauses = self._count_satisfied_clauses(assignment)
        total_clauses = len(self.clauses)
        
        return {
            "is_satisfied": satisfied_clauses == total_clauses,
            "satisfied_clauses": satisfied_clauses,
            "total_clauses": total_clauses,
            "satisfaction_ratio": satisfied_clauses / total_clauses if total_clauses else 1.0
        }
    
    def provide_demonstration(self, partial_assignment: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate a demonstration solution using DPLL algorithm.
        
        Args:
            partial_assignment: Optional partial assignment to start from
            
        Returns:
            Dictionary containing the demonstration
        """
        # Start with provided partial assignment or create new one
        if partial_assignment is not None:
            assignment = partial_assignment.copy()
        else:
            assignment = np.zeros(self.num_vars + 1, dtype=np.int8)
            
        # Use DPLL algorithm to solve
        start_time = time.time()
        steps = []
        solved, solution, step_history = self._dpll(assignment, steps)
        solve_time = time.time() - start_time
        
        if solved:
            return {
                "assignment": solution,
                "is_satisfied": True,
                "steps": step_history,
                "solve_time": solve_time
            }
        else:
            return {
                "assignment": solution,
                "is_satisfied": False,
                "steps": step_history,
                "solve_time": solve_time,
                "explanation": "DPLL could not find a solution in the allowed time"
            }
    
    def _dpll(
        self, 
        assignment: np.ndarray, 
        steps: List[Dict[str, Any]],
        depth: int = 0
    ) -> Tuple[bool, np.ndarray, List[Dict[str, Any]]]:
        """
        DPLL recursive algorithm for SAT solving.
        
        Args:
            assignment: Current partial assignment
            steps: List to track step history
            depth: Current recursion depth
            
        Returns:
            Tuple of (solved, solution, steps)
        """
        if depth > self.max_depth or time.time() - self._start_time > self.timeout:
            return False, assignment, steps
        
        # Unit propagation
        prop_result = self._unit_propagation(assignment)
        
        # Apply unit propagations
        new_assignment = assignment.copy()
        for var, value in prop_result['propagations']:
            new_assignment[var] = value
            steps.append({
                "action": "propagate",
                "variable": var,
                "value": value,
                "depth": depth
            })
            
            # Update variable activity
            self.variable_activity[var] += 1.0
        
        # Check if solution is complete
        if np.count_nonzero(new_assignment[1:]) == self.num_vars:
            result = self.evaluate_solution(new_assignment)
            if result["is_satisfied"]:
                return True, new_assignment, steps
            else:
                # Unsatisfiable assignment after all variables assigned
                return False, new_assignment, steps
        
        # Get unassigned variables
        unassigned = [i for i in range(1, self.num_vars + 1) if new_assignment[i] == 0]
        
        if not unassigned:
            # No unassigned variables but not all satisfied - unsatisfiable
            return False, new_assignment, steps
        
        # Choose a variable using variable activity
        var_activities = [(i, self.variable_activity[i]) for i in unassigned]
        var_activities.sort(key=lambda x: x[1], reverse=True)
        chosen_var = var_activities[0][0]
        
        # Try positive value first
        pos_assignment = new_assignment.copy()
        pos_assignment[chosen_var] = 1
        steps.append({
            "action": "branch",
            "variable": chosen_var,
            "value": 1,
            "depth": depth
        })
        
        solved, solution, steps = self._dpll(pos_assignment, steps, depth + 1)
        if solved:
            return True, solution, steps
        
        # Try negative value
        neg_assignment = new_assignment.copy()
        neg_assignment[chosen_var] = -1
        steps.append({
            "action": "branch",
            "variable": chosen_var,
            "value": -1,
            "depth": depth
        })
        
        return self._dpll(neg_assignment, steps, depth + 1)
    
    def reset(self) -> None:
        """
        Reset the oracle's state for a new problem.
        """
        self.variable_activity = np.ones(self.num_vars + 1)
        self._start_time = time.time()