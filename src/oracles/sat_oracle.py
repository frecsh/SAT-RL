import numpy as np
import time
try:
    from pysat.solvers import Glucose3
    PYSAT_AVAILABLE = True
except ImportError:
    print("PySAT not available. Install with: pip install python-sat")
    PYSAT_AVAILABLE = False

class SATOracle:
    """
    A traditional SAT solver that provides feedback on agent trajectories,
    highlighting problematic clauses that agents consistently fail to satisfy.
    """
    def __init__(self, problem):
        self.problem = problem
        self.clauses = problem["clauses"]
        self.num_vars = problem["num_vars"]
        self.clause_difficulty = np.zeros(len(self.clauses))  # Track difficulty of each clause
        self.clause_satisfaction_count = np.zeros(len(self.clauses))  # Track satisfaction frequency
        self.total_evaluations = 0
        self.pysat_available = PYSAT_AVAILABLE
        self.optimal_solution = None
        self._find_optimal_solution()  # Find a solution with traditional solver
        
    def _find_optimal_solution(self):
        """Find a solution using a traditional SAT solver"""
        if not self.pysat_available:
            return None
            
        try:
            start_time = time.time()
            solver = Glucose3()
            for clause in self.clauses:
                solver.add_clause(clause)
                
            if solver.solve():
                model = solver.get_model()
                # Convert model to our format
                solution = np.zeros(self.num_vars, dtype=np.int32)
                for lit in model:
                    var = abs(lit)
                    if var <= self.num_vars:  # Only consider variables in our problem
                        solution[var-1] = 1 if lit > 0 else 0
                
                self.optimal_solution = solution
                print(f"Oracle found solution in {time.time() - start_time:.4f}s")
                return solution
            else:
                print("Problem is UNSAT according to traditional solver")
                return None
        except Exception as e:
            print(f"Error in traditional solver: {e}")
            return None
    
    def evaluate_trajectory(self, trajectory):
        """
        Evaluate a trajectory of state-action pairs and identify problematic clauses
        
        Args:
            trajectory: List of (state, action, reward) tuples
            
        Returns:
            Difficulty scores for each clause
        """
        self.total_evaluations += 1
        
        # Analyze the final action in the trajectory
        if not trajectory:
            return self.clause_difficulty
            
        final_state, final_action, final_reward = trajectory[-1]
        
        # Check which clauses are satisfied by the final action
        assignment = {i + 1: bool(bit) for i, bit in enumerate(final_action)}
        satisfied_clauses = self._get_satisfied_clauses(assignment)
        
        # Update satisfaction counts
        for i in range(len(self.clauses)):
            if i in satisfied_clauses:
                self.clause_satisfaction_count[i] += 1
            else:
                # Increase difficulty score for unsatisfied clauses
                self.clause_difficulty[i] += 1
        
        # Normalize difficulty scores
        if self.total_evaluations > 0:
            satisfaction_rate = self.clause_satisfaction_count / self.total_evaluations
            self.clause_difficulty = 1 - satisfaction_rate  # Higher value = more difficult
        
        return self.clause_difficulty
    
    def _get_satisfied_clauses(self, assignment):
        """
        Get indices of clauses satisfied by the given assignment
        
        Args:
            assignment: Dictionary mapping variables to boolean values
            
        Returns:
            Set of clause indices that are satisfied
        """
        satisfied = set()
        for i, clause in enumerate(self.clauses):
            for lit in clause:
                var = abs(lit)
                if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                    satisfied.add(i)
                    break
                    
        return satisfied
    
    def critique(self, trajectory):
        """
        Provide detailed feedback on a trajectory
        
        Args:
            trajectory: List of (state, action, reward) tuples
            
        Returns:
            dict: Feedback including unsatisfied clauses and suggestions
        """
        if not trajectory:
            return {"error": "Empty trajectory"}
            
        final_state, final_action, final_reward = trajectory[-1]
        assignment = {i + 1: bool(bit) for i, bit in enumerate(final_action)}
        
        satisfied_clauses = self._get_satisfied_clauses(assignment)
        unsatisfied_indices = [i for i in range(len(self.clauses)) if i not in satisfied_clauses]
        
        # Get the most problematic clauses
        difficulty_ranking = sorted(
            [(i, self.clause_difficulty[i]) for i in range(len(self.clauses))],
            key=lambda x: x[1], reverse=True
        )
        
        # Generate suggestions based on optimal solution if available
        suggestions = {}
        if self.optimal_solution is not None and unsatisfied_indices:
            opt_assignment = {i + 1: bool(bit) for i, bit in enumerate(self.optimal_solution)}
            
            for clause_idx in unsatisfied_indices:
                clause = self.clauses[clause_idx]
                # Find variables in this clause that differ from optimal solution
                for lit in clause:
                    var = abs(lit)
                    if assignment.get(var, False) != opt_assignment.get(var, False):
                        suggestions[var] = opt_assignment.get(var)
        
        return {
            "satisfied_clauses": list(satisfied_clauses),
            "unsatisfied_clauses": unsatisfied_indices,
            "difficulty_ranking": difficulty_ranking[:3],  # Top 3 most difficult clauses
            "suggestions": suggestions
        }

def test_oracle():
    """Test the SAT Oracle functionality"""
    from src.sat_problems import MEDIUM_PROBLEM
    
    oracle = SATOracle(MEDIUM_PROBLEM)
    
    # Test with a random trajectory
    n_vars = MEDIUM_PROBLEM["num_vars"]
    random_action = np.random.randint(0, 2, n_vars)
    trajectory = [(np.zeros(n_vars), random_action, 0.5)]
    
    difficulty = oracle.evaluate_trajectory(trajectory)
    feedback = oracle.critique(trajectory)
    
    print("\nClause difficulty scores:")
    for i, score in enumerate(difficulty):
        print(f"Clause {i}: {score:.2f}")
    
    print("\nOracle critique:")
    print(f"Satisfied clauses: {feedback['satisfied_clauses']}")
    print(f"Unsatisfied clauses: {feedback['unsatisfied_clauses']}")
    print(f"Most difficult clauses: {feedback['difficulty_ranking']}")
    print(f"Suggestions: {feedback['suggestions']}")
    
    if oracle.optimal_solution is not None:
        print(f"\nOptimal solution: {oracle.optimal_solution}")

if __name__ == "__main__":
    test_oracle()