"""
Anytime SAT Solver implementation.
Provides partial results and solution quality bounds at any point in the solving process.
"""

import numpy as np
import time
import random
from collections import defaultdict
import multiprocessing
import threading
from collections import deque
import queue
import matplotlib.pyplot as plt


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
    Anytime Ensemble Solver for SAT Problems.
    This module combines multiple solving strategies and runs them in parallel,
    taking the best solution found within a given time limit.
    """

    def __init__(self, n_vars, clauses, time_limit=30, strategies=None):
        """
        Initialize an anytime ensemble SAT solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses in CNF form
            time_limit: Time limit in seconds
            strategies: List of strategies to use, with weights
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.time_limit = time_limit
        
        # Default strategies if none provided
        self.strategies = strategies or [
            {'name': 'random_walk', 'weight': 1.0},
            {'name': 'greedy', 'weight': 2.0},
            {'name': 'annealing', 'weight': 3.0},
            {'name': 'walksat', 'weight': 4.0}
        ]
        
        # Best solution found by any strategy
        self.best_solution = None
        self.best_satisfied = 0
        self.best_strategy = None
        
        # Thread-safe queue for results
        self.result_queue = queue.Queue()
        
        # Progress tracking
        self.progress_history = {}
        self.progress_tracker = defaultdict(list)
    
    def _count_satisfied_clauses(self, assignment):
        """Count how many clauses are satisfied by the given assignment"""
        satisfied = 0
        
        for clause in self.clauses:
            # Check if any literal in the clause is satisfied
            for literal in clause:
                var_idx = abs(literal) - 1  # Convert to 0-indexed
                val = assignment[var_idx]
                
                # Check if the literal is satisfied
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied += 1
                    break
        
        return satisfied
    
    def _update_best_solution(self, solution, satisfied, strategy_name):
        """Update best solution if the new one is better"""
        with threading.Lock():
            if satisfied > self.best_satisfied:
                self.best_satisfied = satisfied
                self.best_solution = solution.copy()
                self.best_strategy = strategy_name
                self.result_queue.put((solution.copy(), satisfied, strategy_name))
    
    def _random_walk_solver(self, max_flips=10000):
        """Random walk solver strategy"""
        strategy_name = "random_walk"
        # Start with random assignment
        solution = np.random.randint(0, 2, size=self.n_vars)
        satisfied = self._count_satisfied_clauses(solution)
        self._update_best_solution(solution, satisfied, strategy_name)
        
        start_time = time.time()
        last_improvement = start_time
        
        flip = 0
        while time.time() - start_time < self.time_limit and satisfied < len(self.clauses):
            # Randomly flip a variable
            var_to_flip = random.randint(0, self.n_vars - 1)
            solution[var_to_flip] = 1 - solution[var_to_flip]  # Flip 0->1 or 1->0
            
            # Evaluate new solution
            new_satisfied = self._count_satisfied_clauses(solution)
            
            # Accept if better, otherwise revert with some probability
            if new_satisfied >= satisfied:
                satisfied = new_satisfied
                self._update_best_solution(solution, satisfied, strategy_name)
                last_improvement = time.time()
                
                # Log progress
                self.progress_tracker[strategy_name].append((time.time() - start_time, satisfied))
            else:
                # Revert with 80% probability
                if random.random() < 0.8:
                    solution[var_to_flip] = 1 - solution[var_to_flip]  # Flip back
                else:
                    satisfied = new_satisfied
            
            flip += 1
            
            # Break if no improvement for a while
            if time.time() - last_improvement > self.time_limit * 0.2:
                # Restart with new random assignment
                solution = np.random.randint(0, 2, size=self.n_vars)
                satisfied = self._count_satisfied_clauses(solution)
                last_improvement = time.time()
                self.progress_tracker[strategy_name].append((time.time() - start_time, satisfied))
        
        return solution, satisfied
    
    def _greedy_solver(self):
        """Greedy SAT solver strategy"""
        strategy_name = "greedy"
        solution = np.random.randint(0, 2, size=self.n_vars)
        satisfied = self._count_satisfied_clauses(solution)
        self._update_best_solution(solution, satisfied, strategy_name)
        
        start_time = time.time()
        last_improvement = start_time
        
        while time.time() - start_time < self.time_limit and satisfied < len(self.clauses):
            # Try flipping each variable and keep the best improvement
            best_var = -1
            best_improvement = -1
            
            for var in range(self.n_vars):
                # Flip this variable
                solution[var] = 1 - solution[var]
                new_satisfied = self._count_satisfied_clauses(solution)
                improvement = new_satisfied - satisfied
                
                # Revert flip
                solution[var] = 1 - solution[var]
                
                # Track best improvement
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_var = var
            
            # If no improvement possible, do a random flip
            if best_improvement <= 0:
                if time.time() - last_improvement > self.time_limit * 0.1:
                    # Restart if stuck
                    solution = np.random.randint(0, 2, size=self.n_vars)
                    satisfied = self._count_satisfied_clauses(solution)
                    last_improvement = time.time()
                else:
                    # Random flip
                    var_to_flip = random.randint(0, self.n_vars - 1)
                    solution[var_to_flip] = 1 - solution[var_to_flip]
                    satisfied = self._count_satisfied_clauses(solution)
            else:
                # Apply best flip
                solution[best_var] = 1 - solution[best_var]
                satisfied += best_improvement
                self._update_best_solution(solution, satisfied, strategy_name)
                last_improvement = time.time()
            
            # Log progress
            self.progress_tracker[strategy_name].append((time.time() - start_time, satisfied))
        
        return solution, satisfied
    
    def _simulated_annealing_solver(self):
        """Simulated annealing solver strategy"""
        strategy_name = "annealing"
        solution = np.random.randint(0, 2, size=self.n_vars)
        satisfied = self._count_satisfied_clauses(solution)
        self._update_best_solution(solution, satisfied, strategy_name)
        
        start_time = time.time()
        
        # Annealing parameters
        temperature = 1.0
        cooling_rate = 0.995
        
        while time.time() - start_time < self.time_limit and satisfied < len(self.clauses):
            # Choose a random variable to flip
            var_to_flip = random.randint(0, self.n_vars - 1)
            solution[var_to_flip] = 1 - solution[var_to_flip]
            
            # Evaluate new solution
            new_satisfied = self._count_satisfied_clauses(solution)
            delta = new_satisfied - satisfied
            
            # Accept if better or with probability based on temperature
            accept = False
            if delta > 0:
                accept = True
            elif temperature > 0.01:  # Avoid division by zero
                # Probability of accepting worse solution decreases with temperature
                accept_probability = np.exp(delta / temperature)
                if random.random() < accept_probability:
                    accept = True
            
            if accept:
                satisfied = new_satisfied
                self._update_best_solution(solution, satisfied, strategy_name)
            else:
                # Revert flip
                solution[var_to_flip] = 1 - solution[var_to_flip]
            
            # Cool down
            temperature *= cooling_rate
            
            # Reheat if temperature gets too low
            if temperature < 0.01:
                temperature = 0.5
                # Partially randomize solution
                for i in range(self.n_vars // 10):
                    idx = random.randint(0, self.n_vars - 1)
                    solution[idx] = 1 - solution[idx]
                satisfied = self._count_satisfied_clauses(solution)
            
            # Log progress
            self.progress_tracker[strategy_name].append((time.time() - start_time, satisfied))
        
        return solution, satisfied
    
    def _walksat_solver(self, p_random=0.2):
        """WalkSAT solver strategy"""
        strategy_name = "walksat"
        solution = np.random.randint(0, 2, size=self.n_vars)
        satisfied = self._count_satisfied_clauses(solution)
        self._update_best_solution(solution, satisfied, strategy_name)
        
        start_time = time.time()
        last_improvement = start_time
        
        while time.time() - start_time < self.time_limit and satisfied < len(self.clauses):
            # Find unsatisfied clauses
            unsatisfied_clauses = []
            for i, clause in enumerate(self.clauses):
                is_satisfied = False
                for literal in clause:
                    var_idx = abs(literal) - 1
                    if (literal > 0 and solution[var_idx] == 1) or (literal < 0 and solution[var_idx] == 0):
                        is_satisfied = True
                        break
                if not is_satisfied:
                    unsatisfied_clauses.append(i)
            
            # If all clauses satisfied, we're done
            if not unsatisfied_clauses:
                break
            
            # Pick a random unsatisfied clause
            clause_idx = random.choice(unsatisfied_clauses)
            clause = self.clauses[clause_idx]
            
            # With probability p, flip a random variable in the clause
            if random.random() < p_random:
                # Random flip within the unsatisfied clause
                literal = random.choice(clause)
                var_idx = abs(literal) - 1
                solution[var_idx] = 1 - solution[var_idx]
            else:
                # Greedy flip - try each variable in the clause and pick best
                best_var = -1
                best_new_satisfied = -1
                
                for literal in clause:
                    var_idx = abs(literal) - 1
                    # Try flipping
                    solution[var_idx] = 1 - solution[var_idx]
                    new_satisfied = self._count_satisfied_clauses(solution)
                    
                    if new_satisfied > best_new_satisfied:
                        best_var = var_idx
                        best_new_satisfied = new_satisfied
                    
                    # Revert flip
                    solution[var_idx] = 1 - solution[var_idx]
                
                # Apply best flip
                solution[best_var] = 1 - solution[best_var]
                satisfied = best_new_satisfied
            
            # Evaluate new solution
            new_satisfied = self._count_satisfied_clauses(solution)
            
            if new_satisfied > satisfied:
                satisfied = new_satisfied
                self._update_best_solution(solution, satisfied, strategy_name)
                last_improvement = time.time()
            else:
                satisfied = new_satisfied
            
            # Log progress
            self.progress_tracker[strategy_name].append((time.time() - start_time, satisfied))
            
            # Check if stuck and possibly restart
            if time.time() - last_improvement > self.time_limit * 0.15:
                solution = np.random.randint(0, 2, size=self.n_vars)
                satisfied = self._count_satisfied_clauses(solution)
                last_improvement = time.time()
        
        return solution, satisfied
    
    def _strategy_thread(self, strategy_name, *args):
        """Thread function to run a solving strategy"""
        try:
            if strategy_name == 'random_walk':
                return self._random_walk_solver(*args)
            elif strategy_name == 'greedy':
                return self._greedy_solver(*args)
            elif strategy_name == 'annealing':
                return self._simulated_annealing_solver(*args)
            elif strategy_name == 'walksat':
                return self._walksat_solver(*args)
            else:
                print(f"Unknown strategy: {strategy_name}")
                return np.zeros(self.n_vars), 0
        except Exception as e:
            print(f"Error in {strategy_name} thread: {e}")
            return np.zeros(self.n_vars), 0
    
    def solve(self):
        """Solve SAT problem using ensemble of strategies"""
        print(f"Running ensemble solver with time limit {self.time_limit} seconds")
        start_time = time.time()
        
        # Create and start threads for each strategy
        threads = []
        for strategy in self.strategies:
            strategy_name = strategy['name']
            print(f"Starting {strategy_name} solver")
            
            if strategy_name == 'random_walk':
                thread = threading.Thread(target=lambda: self._strategy_thread(strategy_name))
            elif strategy_name == 'greedy':
                thread = threading.Thread(target=lambda: self._strategy_thread(strategy_name))
            elif strategy_name == 'annealing':
                thread = threading.Thread(target=lambda: self._strategy_thread(strategy_name))
            elif strategy_name == 'walksat':
                thread = threading.Thread(target=lambda: self._strategy_thread(strategy_name, 0.2))
            else:
                print(f"Skipping unknown strategy: {strategy_name}")
                continue
            
            thread.daemon = True  # Allow program to exit even if thread is running
            thread.start()
            threads.append(thread)
        
        # Wait until time limit expires
        remaining_time = max(0, self.time_limit - (time.time() - start_time))
        status_interval = min(5, remaining_time / 5)
        last_status = time.time()
        
        while time.time() - start_time < self.time_limit and self.best_satisfied < len(self.clauses):
            # Check for new results
            while not self.result_queue.empty():
                solution, satisfied, strategy = self.result_queue.get()
                if satisfied > self.best_satisfied:
                    print(f"New best solution from {strategy}: {satisfied}/{len(self.clauses)}")
            
            # Print status updates
            if time.time() - last_status > status_interval:
                print(f"Running... Best so far: {self.best_satisfied}/{len(self.clauses)} from {self.best_strategy}")
                last_status = time.time()
            
            time.sleep(0.1)  # Sleep briefly to reduce CPU usage
        
        # Time's up or solution found, prepare stats
        self.progress_history = dict(self.progress_tracker)
        
        total_time = time.time() - start_time
        print(f"\nEnsemble solver completed in {total_time:.2f} seconds")
        print(f"Best solution satisfies {self.best_satisfied}/{len(self.clauses)} clauses using {self.best_strategy}")
        
        # Create visualization
        self.visualize_progress()
        
        # Return best solution and stats
        stats = {
            'satisfied': self.best_satisfied,
            'total': len(self.clauses),
            'best_strategy': self.best_strategy,
            'time': total_time
        }
        
        return self.best_solution, stats
    
    def visualize_progress(self):
        """Visualize progress of different strategies over time"""
        if not self.progress_history:
            print("No progress data available to visualize")
            return
        
        plt.figure(figsize=(12, 6))
        
        for strategy, progress in self.progress_history.items():
            if not progress:
                continue
                
            # Extract times and satisfaction values
            times = [p[0] for p in progress]
            satisfied = [p[1] for p in progress]
            
            plt.plot(times, satisfied, label=strategy)
        
        # Add horizontal line for total clauses
        plt.axhline(y=len(self.clauses), color='r', linestyle='--', label='Goal')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Satisfied Clauses')
        plt.title('SAT Solving Progress by Strategy')
        plt.legend()
        plt.grid(True)
        plt.savefig('ensemble_progress.png')
        plt.close()
        print("Progress visualization saved to ensemble_progress.png")