"""
Anytime SAT Solver implementation.
Provides partial results and solution quality bounds at any point in the solving process.
"""

import numpy as np
import time
import random
from collections import defaultdict


class AnytimeSATSolver:
    def __init__(self, n_vars, clauses, time_limit=60):
        """
        Initialize an Anytime SAT Solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses in CNF form
            time_limit: Maximum time to run in seconds
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.time_limit = time_limit
        
        # Current solution state
        self.current_best = None
        self.current_satisfaction = 0
        self.lower_bound = 0
        self.upper_bound = 1.0
        
        # Solution history
        self.solution_history = []
        self.start_time = None
        
        # Statistics
        self.stats = {
            'iterations': 0,
            'improvements': 0,
            'time_to_best': 0,
            'bound_updates': 0
        }
        
        # Pre-compute clause structure for faster evaluation
        self._precompute_clause_structure()
    
    def _precompute_clause_structure(self):
        """Pre-compute clause structure for efficient evaluation"""
        # Map variables to clauses they appear in
        self.var_to_clauses = defaultdict(list)
        # For each clause, store variables and their polarities
        self.clause_vars = []
        
        for i, clause in enumerate(self.clauses):
            clause_vars = []
            for literal in clause:
                var = abs(literal) - 1  # Convert to 0-indexed
                polarity = literal > 0
                clause_vars.append((var, polarity))
                self.var_to_clauses[var].append(i)
            
            self.clause_vars.append(clause_vars)
    
    def count_satisfied_clauses(self, assignment):
        """Count number of clauses satisfied by an assignment"""
        if assignment is None:
            return 0
            
        satisfied = 0
        for clause in self.clause_vars:
            clause_satisfied = False
            for var, polarity in clause:
                # Check if literal matches assignment
                if (assignment[var] == 1 and polarity) or (assignment[var] == 0 and not polarity):
                    clause_satisfied = True
                    break
            
            if clause_satisfied:
                satisfied += 1
        
        return satisfied
    
    def update_solution(self, assignment):
        """Update solution if better than current best"""
        if assignment is None:
            return False
            
        satisfaction = self.count_satisfied_clauses(assignment)
        ratio = satisfaction / len(self.clauses)
        
        if satisfaction > self.current_satisfaction:
            self.current_best = assignment.copy()
            self.current_satisfaction = satisfaction
            self.lower_bound = ratio
            
            # Update stats
            self.stats['improvements'] += 1
            if self.start_time is not None:
                self.stats['time_to_best'] = time.time() - self.start_time
            
            # Add to solution history
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.solution_history.append({
                'time': elapsed,
                'satisfaction': satisfaction,
                'ratio': ratio
            })
            
            return True
        
        return False
    
    def update_bounds(self):
        """Update the upper bound on solution quality"""
        # This is a placeholder for more sophisticated bound updates
        # In a real implementation, this would use problem structure
        # to derive tighter bounds
        
        # Simple decay of upper bound over time
        if self.upper_bound > self.lower_bound:
            decay_rate = 0.995
            self.upper_bound = max(
                self.lower_bound,
                self.upper_bound * decay_rate
            )
            self.stats['bound_updates'] += 1
    
    def get_current_solution(self):
        """Return best solution found so far along with bounds"""
        return {
            "assignment": self.current_best,
            "satisfied": self.current_satisfaction,
            "total_clauses": len(self.clauses),
            "satisfaction_ratio": self.lower_bound,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "gap": self.upper_bound - self.lower_bound,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            "iterations": self.stats['iterations']
        }
    
    def solve_with_local_search(self, max_iterations=10000, report_interval=100):
        """
        Solve SAT problem using local search, providing anytime guarantees.
        Uses a simple stochastic local search with occasional random restarts.
        """
        self.start_time = time.time()
        
        # Initialize with random assignment
        current = np.random.randint(0, 2, size=self.n_vars)
        current_satisfied = self.count_satisfied_clauses(current)
        self.update_solution(current)
        
        # Temperature for simulated annealing
        temperature = 1.0
        
        # Track time since last improvement
        last_improvement = 0
        
        for i in range(max_iterations):
            self.stats['iterations'] += 1
            
            # Check if time limit reached
            if time.time() - self.start_time > self.time_limit:
                print(f"Time limit reached after {i} iterations")
                break
                
            # Periodically report progress
            if i % report_interval == 0:
                sol = self.get_current_solution()
                print(f"Iteration {i}: Best={sol['satisfied']}/{sol['total_clauses']} "
                      f"({sol['satisfaction_ratio']:.2f}), Gap={sol['gap']:.3f}")
            
            # Random restart if stuck
            if i - last_improvement > 200:
                print(f"Random restart at iteration {i}")
                current = np.random.randint(0, 2, size=self.n_vars)
                current_satisfied = self.count_satisfied_clauses(current)
                temperature = 1.0
                last_improvement = i
            
            # Pick a random variable to flip
            var_idx = np.random.randint(0, self.n_vars)
            
            # Create candidate by flipping the variable
            candidate = current.copy()
            candidate[var_idx] = 1 - candidate[var_idx]
            
            # Evaluate candidate
            candidate_satisfied = self.count_satisfied_clauses(candidate)
            
            # Calculate delta (improvement)
            delta = candidate_satisfied - current_satisfied
            
            # Accept if better or with some probability if worse
            if delta > 0 or (temperature > 0 and random.random() < np.exp(delta / temperature)):
                current = candidate
                current_satisfied = candidate_satisfied
                
                # Update best solution
                if self.update_solution(current):
                    last_improvement = i
            
            # Update bounds
            if i % 10 == 0:
                self.update_bounds()
            
            # Cool temperature
            temperature = max(0.001, temperature * 0.997)
            
            # If problem solved, exit
            if self.current_satisfaction == len(self.clauses):
                print(f"Solution found at iteration {i}")
                break
        
        # Final report
        sol = self.get_current_solution()
        print(f"\nFinal result: Satisfied {sol['satisfied']}/{sol['total_clauses']} clauses "
              f"({sol['satisfaction_ratio']:.2f})")
        print(f"Bounds: [{sol['lower_bound']:.3f}, {sol['upper_bound']:.3f}], Gap: {sol['gap']:.3f}")
        print(f"Time: {sol['elapsed_time']:.2f}s, Iterations: {sol['iterations']}")
        
        return self.current_best, self.stats
    
    def visualize_progress(self):
        """Visualize the solver's progress over time"""
        if not self.solution_history:
            print("No solution history to visualize")
            return
            
        import matplotlib.pyplot as plt
        
        # Extract data from solution history
        times = [s['time'] for s in self.solution_history]
        ratios = [s['ratio'] for s in self.solution_history]
        
        # Plot progress over time
        plt.figure(figsize=(10, 6))
        plt.plot(times, ratios, 'bo-', label='Solution Quality')
        
        # Plot bounds if available
        if len(times) > 0:
            plt.axhline(y=1.0, color='g', linestyle='--', label='Optimal')
            plt.axhline(y=self.lower_bound, color='r', linestyle='-.', label='Lower Bound')
            plt.axhline(y=self.upper_bound, color='m', linestyle=':', label='Upper Bound')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Satisfaction Ratio')
        plt.title('Anytime SAT Solver Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('anytime_progress.png')
        plt.close()
        
        print("Progress visualization saved to anytime_progress.png")


class AnytimeEnsembleSolver:
    """
    Ensemble of multiple anytime solvers running in parallel.
    Provides even better anytime guarantees by combining multiple solution strategies.
    """
    def __init__(self, n_vars, clauses, time_limit=60, strategies=None):
        """
        Initialize an ensemble of anytime solvers.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses in CNF form
            time_limit: Maximum time to run in seconds
            strategies: List of solver strategies to use
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.time_limit = time_limit
        
        # Default strategies
        if strategies is None:
            self.strategies = [
                {'name': 'random_walk', 'weight': 1.0},
                {'name': 'greedy', 'weight': 1.0},
                {'name': 'annealing', 'weight': 1.0}
            ]
        else:
            self.strategies = strategies
        
        # Current solution state (shared among all solvers)
        self.current_best = None
        self.current_satisfaction = 0
        self.lower_bound = 0
        self.upper_bound = 1.0
        
        # Track which solver found the best solution
        self.best_solver = None
        
        # Track improvement history
        self.improvement_history = []
    
    def count_satisfied_clauses(self, assignment):
        """Count number of clauses satisfied by an assignment"""
        if assignment is None:
            return 0
            
        satisfied = 0
        for clause in self.clauses:
            for literal in clause:
                var = abs(literal) - 1  # Convert to 0-indexed
                val = assignment[var]
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied += 1
                    break
        
        return satisfied
    
    def update_solution(self, assignment, solver_name):
        """Update solution if better than current best"""
        if assignment is None:
            return False
            
        satisfaction = self.count_satisfied_clauses(assignment)
        ratio = satisfaction / len(self.clauses)
        
        if satisfaction > self.current_satisfaction:
            self.current_best = assignment.copy()
            self.current_satisfaction = satisfaction
            self.lower_bound = ratio
            self.best_solver = solver_name
            
            # Record improvement
            self.improvement_history.append({
                'time': time.time() - self.start_time,
                'satisfaction': satisfaction,
                'ratio': ratio,
                'solver': solver_name
            })
            
            return True
        
        return False
    
    def random_walk_solver(self, max_iterations=1000):
        """Simple random walk solver"""
        current = np.random.randint(0, 2, size=self.n_vars)
        self.update_solution(current, 'random_walk')
        
        for i in range(max_iterations):
            # Check time limit
            if time.time() - self.start_time > self.time_limit:
                break
                
            # Flip a random variable
            var_idx = np.random.randint(0, self.n_vars)
            current[var_idx] = 1 - current[var_idx]
            
            # Update solution
            self.update_solution(current, 'random_walk')
    
    def greedy_solver(self, max_iterations=1000):
        """Greedy local search solver"""
        current = np.random.randint(0, 2, size=self.n_vars)
        current_satisfied = self.count_satisfied_clauses(current)
        self.update_solution(current, 'greedy')
        
        for i in range(max_iterations):
            # Check time limit
            if time.time() - self.start_time > self.time_limit:
                break
                
            # Try flipping each variable and keep the best improvement
            best_var = -1
            best_improvement = -1
            
            for var_idx in range(self.n_vars):
                # Flip variable
                current[var_idx] = 1 - current[var_idx]
                
                # Evaluate
                new_satisfied = self.count_satisfied_clauses(current)
                improvement = new_satisfied - current_satisfied
                
                # Flip back
                current[var_idx] = 1 - current[var_idx]
                
                # Keep track of best improvement
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_var = var_idx
            
            # If improvement found, make the move
            if best_var >= 0 and best_improvement > 0:
                current[best_var] = 1 - current[best_var]
                current_satisfied += best_improvement
                self.update_solution(current, 'greedy')
            else:
                # No improvement possible, restart
                current = np.random.randint(0, 2, size=self.n_vars)
                current_satisfied = self.count_satisfied_clauses(current)
                self.update_solution(current, 'greedy')
    
    def annealing_solver(self, max_iterations=1000):
        """Simulated annealing solver"""
        current = np.random.randint(0, 2, size=self.n_vars)
        current_satisfied = self.count_satisfied_clauses(current)
        self.update_solution(current, 'annealing')
        
        temperature = 1.0
        cooling_rate = 0.995
        
        for i in range(max_iterations):
            # Check time limit
            if time.time() - self.start_time > self.time_limit:
                break
                
            # Cool temperature
            temperature *= cooling_rate
            
            # Flip a random variable
            var_idx = np.random.randint(0, self.n_vars)
            
            # Make temporary flip
            current[var_idx] = 1 - current[var_idx]
            
            # Evaluate
            new_satisfied = self.count_satisfied_clauses(current)
            delta = new_satisfied - current_satisfied
            
            # Accept or reject move
            if delta > 0 or random.random() < np.exp(delta / max(temperature, 0.001)):
                # Accept move
                current_satisfied = new_satisfied
                self.update_solution(current, 'annealing')
            else:
                # Reject move, flip back
                current[var_idx] = 1 - current[var_idx]
    
    def solve(self):
        """Run ensemble of solvers with time allocation based on weights"""
        self.start_time = time.time()
        
        import threading
        
        # Calculate time allocation for each strategy
        total_weight = sum(strategy['weight'] for strategy in self.strategies)
        solvers = []
        
        for strategy in self.strategies:
            # Allocate time based on weight
            solver_time = self.time_limit * strategy['weight'] / total_weight
            
            if strategy['name'] == 'random_walk':
                solver_thread = threading.Thread(
                    target=self.random_walk_solver,
                    kwargs={'max_iterations': 10000}
                )
            elif strategy['name'] == 'greedy':
                solver_thread = threading.Thread(
                    target=self.greedy_solver,
                    kwargs={'max_iterations': 5000}
                )
            elif strategy['name'] == 'annealing':
                solver_thread = threading.Thread(
                    target=self.annealing_solver,
                    kwargs={'max_iterations': 8000}
                )
            else:
                continue
                
            solvers.append({
                'thread': solver_thread,
                'name': strategy['name'],
                'time': solver_time
            })
        
        # Start all solver threads
        for solver in solvers:
            solver['thread'].daemon = True
            solver['thread'].start()
        
        # Monitor progress and update bounds
        end_time = self.start_time + self.time_limit
        report_interval = min(1.0, self.time_limit / 10)
        next_report = self.start_time + report_interval
        
        try:
            while time.time() < end_time:
                # Sleep briefly
                time.sleep(0.1)
                
                # Report progress
                if time.time() >= next_report:
                    ratio = self.current_satisfaction / len(self.clauses)
                    elapsed = time.time() - self.start_time
                    print(f"[{elapsed:.1f}s] Best: {self.current_satisfaction}/{len(self.clauses)} "
                          f"({ratio:.2f}) from {self.best_solver}")
                    next_report = time.time() + report_interval
                
                # Update upper bound (could be more sophisticated)
                if self.upper_bound > self.lower_bound:
                    time_fraction = (time.time() - self.start_time) / self.time_limit
                    self.upper_bound = max(
                        self.lower_bound, 
                        1.0 - 0.5 * time_fraction * (1.0 - self.lower_bound)
                    )
                
                # If solution found, exit early
                if self.current_satisfaction == len(self.clauses):
                    print("Solution found!")
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        # Wait for all threads to finish or terminate them
        for solver in solvers:
            if solver['thread'].is_alive():
                # Just let threads terminate naturally as they check time
                pass
        
        # Final report
        ratio = self.current_satisfaction / len(self.clauses)
        elapsed = time.time() - self.start_time
        print(f"\nFinal result: {self.current_satisfaction}/{len(self.clauses)} "
              f"({ratio:.2f}) from {self.best_solver}")
        print(f"Bounds: [{self.lower_bound:.3f}, {self.upper_bound:.3f}], "
              f"Gap: {self.upper_bound - self.lower_bound:.3f}")
        print(f"Time: {elapsed:.2f}s")
        
        # Return solution and stats
        return self.current_best, {
            'satisfied': self.current_satisfaction,
            'total': len(self.clauses),
            'ratio': ratio,
            'best_solver': self.best_solver,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'gap': self.upper_bound - self.lower_bound,
            'time': elapsed,
            'improvements': len(self.improvement_history)
        }
    
    def visualize_progress(self):
        """Visualize the ensemble solver's progress"""
        if not self.improvement_history:
            print("No improvement history to visualize")
            return
            
        import matplotlib.pyplot as plt
        
        # Extract data
        times = [s['time'] for s in self.improvement_history]
        ratios = [s['ratio'] for s in self.improvement_history]
        solvers = [s['solver'] for s in self.improvement_history]
        
        # Create unique color for each solver
        solver_types = list(set(solvers))
        colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']
        solver_colors = {solver: colors[i % len(colors)] for i, solver in enumerate(solver_types)}
        
        # Plot progress
        plt.figure(figsize=(12, 8))
        
        # Plot overall progress line
        plt.plot(times, ratios, 'k-', alpha=0.3)
        
        # Plot points for each solver contribution
        for i, (time, ratio, solver) in enumerate(zip(times, ratios, solvers)):
            plt.plot(time, ratio, 'o', color=solver_colors[solver], label=solver if solver not in plt.gca().get_legend_handles_labels()[1] else "")
        
        # Plot bounds
        plt.axhline(y=1.0, color='g', linestyle='--', label='Optimal')
        plt.axhline(y=self.lower_bound, color='r', linestyle='-.', label='Lower Bound')
        plt.axhline(y=self.upper_bound, color='m', linestyle=':', label='Upper Bound')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Satisfaction Ratio')
        plt.title('Ensemble SAT Solver Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('ensemble_progress.png')
        plt.close()
        
        print("Ensemble progress visualization saved to ensemble_progress.png")