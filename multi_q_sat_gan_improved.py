#!/usr/bin/env python3
"""
Multi-agent Q-learning implementation for solving SAT problems.
This version integrates the improved Progressive GAN generator to replace random
exploration with learned patterns from previous solutions.
"""

import numpy as np
import random
import time
import torch
import os
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict

from sat_problems import count_satisfied_clauses, is_satisfied
from multi_q_sat import MultiQLearningSAT
from progressive_sat_gan import ProgressiveSATGAN

class MultiQLearningSATProgressiveGAN(MultiQLearningSAT):
    """
    Multi-agent Q-learning implementation with improved Progressive GAN guidance.
    Uses a progressive GAN to generate promising initial states and exploration paths.
    """
    
    def __init__(self, n_vars: int, clauses: List[List[int]], 
                 n_agents: int = 5, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01,
                 gan_exploration_ratio: float = 0.7,
                 gan_training_interval: int = 50,
                 gan_min_solutions: int = 20,
                 progressive_stages: int = 3,
                 models_dir: str = "models"):
        """
        Initialize the GAN-guided multi-agent Q-learning SAT solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses, where each clause is a list of literals
            n_agents: Number of agents (default: 5)
            learning_rate: Learning rate for Q-learning (default: 0.1)
            discount_factor: Discount factor for future rewards (default: 0.95)
            epsilon: Initial exploration rate (default: 0.1)
            epsilon_decay: Rate at which epsilon decays (default: 0.995)
            min_epsilon: Minimum exploration rate (default: 0.01)
            gan_exploration_ratio: Ratio of GAN-generated vs random exploration (default: 0.7)
            gan_training_interval: Episodes between GAN training (default: 50)
            gan_min_solutions: Minimum solutions needed for GAN training (default: 20)
            progressive_stages: Number of GAN training stages (default: 3)
            models_dir: Directory to save models (default: "models")
        """
        # Initialize base class
        super().__init__(n_vars, clauses, n_agents, learning_rate, 
                       discount_factor, epsilon, epsilon_decay, min_epsilon)
        
        # GAN-specific parameters
        self.gan_exploration_ratio = gan_exploration_ratio
        self.gan_training_interval = gan_training_interval
        self.gan_min_solutions = gan_min_solutions
        self.progressive_stages = progressive_stages
        self.models_dir = models_dir
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Solution buffer for GAN training
        self.solution_buffer = []
        self.solution_set = set()  # For quick duplicate checking
        
        # GAN model
        self.gan = ProgressiveSATGAN(
            n_vars=n_vars,
            clauses=clauses,
            latent_dim=64,
            clause_weight=1.0,
            diversity_weight=0.2
        )
        
        # Load existing model if available
        model_path = os.path.join(models_dir, f"sat_gan_{n_vars}_{len(clauses)}.pth")
        if os.path.exists(model_path):
            try:
                self.gan = ProgressiveSATGAN.load(model_path, clauses)
                self.gan_available = True
                print(f"Loaded existing GAN model from {model_path}")
            except Exception as e:
                print(f"Error loading GAN model: {str(e)}")
                self.gan_available = False
        else:
            self.gan_available = False
            
        # Stats tracking
        self.gan_stats = {
            'uses': 0,
            'successes': 0,
            'training_count': 0,
            'solution_buffer_size': 0
        }
    
    def _hash_assignment(self, assignment: List[int]) -> str:
        """
        Convert an assignment to a hashable representation for duplicate detection.
        
        Args:
            assignment: Variable assignment
            
        Returns:
            Hashable representation of the assignment
        """
        return ','.join(str(lit) for lit in sorted(assignment, key=abs))
    
    def _add_to_solution_buffer(self, solution: List[int], quality: float) -> None:
        """
        Add a solution to the buffer for GAN training, avoiding duplicates.
        
        Args:
            solution: Variable assignment
            quality: Solution quality (0.0-1.0)
        """
        # Only add reasonably good solutions
        if quality < 0.7:
            return
            
        # Check for duplicates
        solution_hash = self._hash_assignment(solution)
        if solution_hash in self.solution_set:
            return
            
        # Add to buffer and set
        self.solution_buffer.append((solution.copy(), quality))
        self.solution_set.add(solution_hash)
        
        # Update stats
        self.gan_stats['solution_buffer_size'] = len(self.solution_buffer)
    
    def _train_gan(self) -> None:
        """Train the GAN on collected solutions."""
        if len(self.solution_buffer) < self.gan_min_solutions:
            return
            
        print(f"\nTraining GAN on {len(self.solution_buffer)} solutions...")
        
        # Extract solutions and weights
        solutions = [s for s, _ in self.solution_buffer]
        
        # Ensure solutions have the right format and size
        normalized_solutions = []
        for sol in solutions:
            # Convert from variable representation to binary representation
            # Ensure all solutions have exactly n_vars elements
            if len(sol) != self.n_vars:
                # Truncate or pad with random values if necessary 
                if len(sol) > self.n_vars:
                    normalized_sol = sol[:self.n_vars]
                else:
                    normalized_sol = sol + [random.choice([1, -1]) for _ in range(self.n_vars - len(sol))]
            else:
                normalized_sol = sol
                
            # Ensure values are either 1 or -1
            normalized_sol = [1 if lit > 0 else -1 for lit in normalized_sol]
            normalized_solutions.append(normalized_sol)
        
        # Train the GAN
        try:
            start_time = time.time()
            
            # Progressive training with stages
            self.gan.train(
                solutions=normalized_solutions,
                epochs_per_stage=30,
                batch_size=min(32, len(normalized_solutions)),
                min_solutions_per_stage=min(20, len(normalized_solutions) // 2)
            )
            
            elapsed = time.time() - start_time
            print(f"GAN training completed in {elapsed:.1f} seconds")
            
            # Update stats
            self.gan_stats['training_count'] += 1
            self.gan_available = True
            
            # Save the model
            model_path = os.path.join(self.models_dir, f"sat_gan_{self.n_vars}_{len(self.clauses)}.pth")
            self.gan.save(model_path)
            print(f"Saved GAN model to {model_path}")
            
        except Exception as e:
            print(f"Error during GAN training: {str(e)}")
            self.gan_available = False
    
    def _get_initial_state(self) -> List[int]:
        """
        Get an initial state for an episode, using the GAN when available.
        
        Returns:
            Initial variable assignment
        """
        # Use GAN with probability gan_exploration_ratio if available
        if self.gan_available and random.random() < self.gan_exploration_ratio:
            try:
                # Generate a batch of assignments and pick the best one
                assignments = self.gan.generate(num_samples=5, temperature=1.0)
                
                # Find the best assignment by clause satisfaction
                best_satisfied = 0
                best_assignment = None
                
                for assignment in assignments:
                    satisfied = count_satisfied_clauses(self.clauses, assignment)
                    if satisfied > best_satisfied:
                        best_satisfied = satisfied
                        best_assignment = assignment
                
                if best_assignment:
                    # Track GAN usage statistics
                    self.gan_stats['uses'] += 1
                    
                    # Check if this satisfies the problem
                    if is_satisfied(self.clauses, best_assignment):
                        self.gan_stats['successes'] += 1
                        
                    return best_assignment
                    
            except Exception as e:
                print(f"Error using GAN: {str(e)}")
                # Fall back to random assignment
        
        # Default to random assignment
        return [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
    
    def solve(self, max_episodes: int = 1000, early_stopping: bool = True, 
             timeout: Optional[int] = None) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """
        Solve the SAT problem using multi-agent Q-learning with GAN guidance.
        
        Args:
            max_episodes: Maximum number of episodes
            early_stopping: Whether to stop early when a solution is found
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (solution, stats) where solution is the variable assignment
            and stats contains statistics about the solving process
        """
        start_time = time.time()
        episodes_completed = 0
        solved = False
        timed_out = False
        
        # Initialize with a GAN-generated or random assignment
        current_state = self._get_initial_state()
        
        for episode in range(max_episodes):
            episodes_completed = episode + 1
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                timed_out = True
                break
            
            # Each episode starts with a GAN-generated state with probability (1-epsilon),
            # otherwise continues from the current state
            if random.random() < self.epsilon:
                current_state = self._get_initial_state()
            
            # Each agent takes an action in turn
            for agent_idx in range(self.n_agents):
                # Select action
                action = self._select_action(agent_idx, current_state)
                
                # Take action
                next_state = self._take_action(current_state, action)
                
                # Calculate reward
                reward = self._get_reward(current_state, next_state)
                
                # Update Q-value
                self._update_q_value(agent_idx, current_state, action, reward, next_state)
                
                # Track best solution
                self._track_best_solution(next_state)
                
                # Add solution to buffer if it's good
                solution_quality = count_satisfied_clauses(self.clauses, next_state) / len(self.clauses)
                self._add_to_solution_buffer(next_state, solution_quality)
                
                # Update current state
                current_state = next_state
                
                # Check if problem is solved
                if is_satisfied(self.clauses, current_state):
                    solved = True
                    # Add the solution to buffer with maximum quality
                    self._add_to_solution_buffer(current_state, 1.0)
                    if early_stopping:
                        break
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Train the GAN periodically
            if episode > 0 and episode % self.gan_training_interval == 0:
                self._train_gan()
            
            # Early stopping if solved
            if solved and early_stopping:
                break
        
        # If we haven't trained the GAN yet and have enough solutions, train it now
        if not self.gan_available and len(self.solution_buffer) >= self.gan_min_solutions:
            self._train_gan()
        
        # Prepare solution
        solution = self.best_assignment if self.best_assignment else current_state
        
        # Calculate statistics
        end_time = time.time()
        solve_time = end_time - start_time
        
        # GAN stats
        if self.gan_stats['uses'] > 0:
            gan_success_rate = self.gan_stats['successes'] / self.gan_stats['uses']
        else:
            gan_success_rate = 0.0
        
        # Add GAN statistics to the return stats
        stats = {
            'solved': solved,
            'episodes': episodes_completed,
            'time': solve_time,
            'best_satisfied': self.best_satisfied,
            'best_satisfaction_ratio': self.best_satisfaction_ratio,
            'final_epsilon': self.epsilon,
            'timed_out': timed_out,
            'gan_uses': self.gan_stats['uses'],
            'gan_successes': self.gan_stats['successes'],
            'gan_success_rate': gan_success_rate,
            'gan_training_count': self.gan_stats['training_count'],
            'solution_buffer_size': self.gan_stats['solution_buffer_size']
        }
        
        return solution, stats


# Example usage
if __name__ == "__main__":
    from sat_problems import generate_sat_problem
    
    # Generate a small SAT problem
    n_vars = 20
    n_clauses = 85
    clauses = generate_sat_problem(n_vars, n_clauses)
    
    print(f"Solving SAT problem with {n_vars} variables and {n_clauses} clauses")
    
    # Create solver with progressive GAN guidance
    solver = MultiQLearningSATProgressiveGAN(
        n_vars, 
        clauses,
        gan_exploration_ratio=0.7, 
        gan_training_interval=50
    )
    
    # Solve
    print("Starting to solve...")
    solution, stats = solver.solve(max_episodes=500, early_stopping=True)
    
    # Print results
    print(f"\nResults:")
    print(f"Solved: {stats['solved']}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Time: {stats['time']:.2f} seconds")
    print(f"Satisfaction: {stats['best_satisfied']}/{len(clauses)} clauses ({stats['best_satisfaction_ratio']:.2%})")
    print(f"GAN uses: {stats['gan_uses']}")
    print(f"GAN success rate: {stats['gan_success_rate']:.2%}")
    print(f"GAN training count: {stats['gan_training_count']}")
    
    if stats['solved']:
        print(f"Solution found: {solution}")