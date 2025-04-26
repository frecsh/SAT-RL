"""
Curriculum Learning approach for SAT solving.
Gradually increases problem difficulty to overcome the phase transition barrier.
"""

import numpy as np
import random
from deep_q_sat_agent import DeepQLearningAgent
from improved_sat_gan import ImprovedSATGAN
import copy


class CurriculumSATLearner:
    def __init__(self, n_vars, initial_ratio=3.0, target_ratio=4.2, step=0.2):
        """
        Initialize a Curriculum Learning approach for SAT solving.
        
        Args:
            n_vars: Number of variables in the SAT problem
            initial_ratio: Initial clause-to-variable ratio (start with easier problems)
            target_ratio: Target clause-to-variable ratio (typically around phase transition)
            step: Step size for increasing difficulty
        """
        self.n_vars = n_vars
        self.initial_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.step = step
        
        # Track best models at each difficulty level
        self.models = {}
        
        # Track best solution found
        self.best_solution = None
        self.best_satisfied = 0
    
    def generate_sat_problem(self, ratio, unique_clauses=True):
        """Generate a random SAT problem with the given clause-to-variable ratio"""
        n_clauses = int(ratio * self.n_vars)
        clauses = []
        
        # Set to track unique clauses if requested
        clause_set = set()
        
        while len(clauses) < n_clauses:
            # Generate a random clause with 3 literals (3-SAT)
            vars_in_clause = random.sample(range(1, self.n_vars + 1), 3)
            
            # Randomly negate some variables
            clause = [var if random.random() > 0.5 else -var for var in vars_in_clause]
            
            # Convert clause to tuple for set operations
            clause_tuple = tuple(sorted(clause, key=abs))
            
            # Add clause if unique or if we're not enforcing uniqueness
            if not unique_clauses or clause_tuple not in clause_set:
                clauses.append(clause)
                if unique_clauses:
                    clause_set.add(clause_tuple)
        
        return clauses
    
    def count_satisfied_clauses(self, assignment, clauses):
        """Count number of clauses satisfied by an assignment"""
        satisfied = 0
        for clause in clauses:
            for literal in clause:
                var = abs(literal) - 1  # Convert to 0-indexed
                val = assignment[var]
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied += 1
                    break
        return satisfied
    
    def evaluate_solution(self, solution, clauses):
        """Evaluate a solution for a specific set of clauses"""
        if solution is None:
            return 0
            
        satisfied = self.count_satisfied_clauses(solution, clauses)
        
        # Update best solution if current is better for the target problem
        if self.target_clauses is not None and satisfied > self.best_satisfied:
            target_satisfied = self.count_satisfied_clauses(solution, self.target_clauses)
            if target_satisfied > self.best_satisfied:
                self.best_satisfied = target_satisfied
                self.best_solution = solution.copy()
            
        return satisfied
    
    def solve_with_agent(self, clauses, agent_type='dqn', previous_agent=None, max_episodes=500):
        """Solve a SAT problem with the specified agent type"""
        if agent_type == 'dqn':
            # Use Deep Q-Network
            agent = DeepQLearningAgent(self.n_vars, clauses)
            if previous_agent is not None and hasattr(previous_agent, 'model'):
                # Transfer weights from previous model if compatible
                try:
                    agent.model.set_weights(previous_agent.model.get_weights())
                    print("Transferred weights from previous DQN model")
                except:
                    print("Could not transfer weights - model architectures may differ")
            
            solution, stats = agent.solve(max_episodes=max_episodes)
            return solution, agent, stats
            
        elif agent_type == 'gan':
            # Use Improved SATGAN
            agent = ImprovedSATGAN(self.n_vars, clauses)
            if previous_agent is not None and isinstance(previous_agent, ImprovedSATGAN):
                # Initialize with previous solutions if available
                if hasattr(previous_agent, 'best_solution') and previous_agent.best_solution is not None:
                    initial_solutions = [previous_agent.best_solution]
                    agent.train_with_experience_replay(initial_solutions=initial_solutions)
                else:
                    agent.train_with_experience_replay()
            else:
                agent.train_with_experience_replay()
                
            solution = agent.solve(max_generations=50)
            return solution, agent, {'satisfied': self.count_satisfied_clauses(solution, clauses)}
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def solve_with_curriculum(self, agent_type='dqn', max_attempts=5):
        """
        Solve SAT problem using curriculum learning.
        Gradually increase problem difficulty from initial_ratio to target_ratio.
        """
        # Generate target problem (the one we ultimately want to solve)
        self.target_clauses = self.generate_sat_problem(self.target_ratio)
        
        # Start with easy problems
        current_ratio = self.initial_ratio
        current_step = self.step
        previous_agent = None
        current_agent_type = agent_type
        level = 0
        
        print(f"Starting curriculum learning from ratio {current_ratio} to {self.target_ratio}")
        
        while current_ratio <= self.target_ratio + 0.001:  # Add small epsilon for floating point comparison
            level += 1
            print(f"\n--- Curriculum Level {level}: Ratio = {current_ratio:.2f} ---")
            
            # Generate problem at current difficulty
            current_clauses = self.generate_sat_problem(current_ratio)
            
            # Try to solve with current agent type
            attempts = 0
            solved = False
            
            while attempts < max_attempts and not solved:
                attempts += 1
                
                # Solve current level
                print(f"Attempt {attempts}/{max_attempts} with {current_agent_type}")
                solution, agent, stats = self.solve_with_agent(
                    current_clauses, 
                    agent_type=current_agent_type,
                    previous_agent=previous_agent
                )
                
                # Check if solved
                satisfied = self.count_satisfied_clauses(solution, current_clauses)
                solved = satisfied == len(current_clauses)
                
                if solved:
                    print(f"Level {level} (ratio {current_ratio:.2f}) solved!")
                    
                    # Store successful model
                    self.models[current_ratio] = agent
                    
                    # Update previous agent
                    previous_agent = agent
                    
                    # Evaluate solution on target problem
                    self.evaluate_solution(solution, self.target_clauses)
                    
                else:
                    print(f"Attempt {attempts} failed: {satisfied}/{len(current_clauses)} clauses satisfied")
                    
                    # If multiple failures with the current agent type, try another
                    if attempts >= 2 and current_agent_type == 'dqn':
                        print("Switching to GAN agent")
                        current_agent_type = 'gan'
                    elif attempts >= 2 and current_agent_type == 'gan':
                        print("Switching to DQN agent")
                        current_agent_type = 'dqn'
            
            if solved:
                # Advance curriculum
                if current_ratio < self.target_ratio:
                    current_ratio += current_step
                    current_ratio = min(current_ratio, self.target_ratio)
                else:
                    # We've reached our target ratio and solved it
                    break
            else:
                # Reduce step size if stuck
                current_step = current_step / 2
                print(f"Reducing step size to {current_step:.3f}")
                
                # If step size becomes too small, try a different approach
                if current_step < 0.05:
                    print("Step size too small, skipping to target ratio")
                    current_ratio = self.target_ratio
        
        # Final attempt directly on target problem if not already solved
        if self.best_satisfied < len(self.target_clauses):
            print("\n--- Final attempt on target problem ---")
            
            # Try both agent types on the target problem
            for agent_type in ['dqn', 'gan']:
                print(f"Attempting target problem with {agent_type}")
                solution, agent, stats = self.solve_with_agent(
                    self.target_clauses, 
                    agent_type=agent_type,
                    previous_agent=self.models.get(max(self.models.keys())) if self.models else None,
                    max_episodes=1000
                )
                
                # Evaluate solution
                satisfied = self.evaluate_solution(solution, self.target_clauses)
                print(f"{agent_type} achieved {satisfied}/{len(self.target_clauses)} satisfied clauses")
        
        # Return best solution found
        return self.best_solution, self.best_satisfied
    
    def visualize_curriculum_progress(self):
        """Visualize the progress of curriculum learning"""
        import matplotlib.pyplot as plt
        
        if not self.models:
            print("No models to visualize")
            return
        
        ratios = sorted(self.models.keys())
        
        # Plot the success rate at each difficulty level
        plt.figure(figsize=(10, 6))
        plt.plot(ratios, [1.0] * len(ratios), 'o-', label='Success Rate')
        plt.axvline(x=4.2, color='r', linestyle='--', label='Phase Transition')
        plt.xlabel('Clause-to-Variable Ratio')
        plt.ylabel('Success Rate')
        plt.title('Curriculum Learning Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('curriculum_progress.png')
        plt.close()
        
        print(f"Progress visualization saved to curriculum_progress.png")