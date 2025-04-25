#!/usr/bin/env python3
"""
GAN-powered multi-agent Q-learning implementation for solving SAT problems.
This version uses a trained GAN model to generate promising variable assignments
that the RL agents can refine, replacing random exploration with learned patterns.
"""

import numpy as np
import random
import time
import torch
from typing import List, Tuple, Dict, Any, Optional
from sat_problems import count_satisfied_clauses, is_satisfied
from multi_q_sat import MultiQLearningSAT
from sat_gan import SATGAN

class MultiQLearningSATGAN(MultiQLearningSAT):
    """
    GAN-powered multi-agent Q-learning implementation for SAT problems.
    Uses learned patterns from a GAN instead of random exploration.
    """
    
    def __init__(self, n_vars: int, clauses: List[List[int]], 
                 gan_model: SATGAN = None,
                 gan_model_path: Optional[str] = None,
                 n_agents: int = 5, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95, 
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, 
                 min_epsilon: float = 0.01,
                 gan_exploration_ratio: float = 0.8,
                 gan_temperature: float = 1.2):
        """
        Initialize the GAN-powered multi-agent Q-learning SAT solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses, where each clause is a list of literals
            gan_model: Pre-trained SATGAN model
            gan_model_path: Path to a saved SATGAN model (used if gan_model is None)
            n_agents: Number of agents (default: 5)
            learning_rate: Learning rate for Q-learning (default: 0.1)
            discount_factor: Discount factor for future rewards (default: 0.95)
            epsilon: Initial exploration rate (default: 0.1)
            epsilon_decay: Rate at which epsilon decays (default: 0.995)
            min_epsilon: Minimum exploration rate (default: 0.01)
            gan_exploration_ratio: Ratio of exploration steps that use GAN (default: 0.8)
            gan_temperature: Temperature for GAN sampling (default: 1.2)
        """
        super().__init__(n_vars, clauses, n_agents, learning_rate, 
                        discount_factor, epsilon, epsilon_decay, min_epsilon)
        
        # GAN exploration parameters
        self.gan_exploration_ratio = gan_exploration_ratio
        self.gan_temperature = gan_temperature
        self.gan = gan_model
        
        # Load GAN model if not provided but path is
        if self.gan is None and gan_model_path is not None:
            self.gan = SATGAN.load(gan_model_path, clauses)
        
        # Track solution history for periodic GAN retraining
        self.solution_history = []
        self.new_solutions_since_retrain = 0
        self.gan_generated_solutions = 0
        self.gan_generations_used = 0
    
    def _get_gan_assignment(self) -> List[int]:
        """
        Generate a variable assignment using the GAN model.
        
        Returns:
            Variable assignment as a list of literals
        """
        if self.gan is None:
            # Fall back to random assignment if GAN is not available
            return [i if random.random() < 0.5 else -i for i in range(1, self.n_vars + 1)]
        
        # Generate a single assignment with some temperature for diversity
        assignments = self.gan.generate(num_samples=1, temperature=self.gan_temperature)
        self.gan_generated_solutions += 1
        
        return assignments[0]
    
    def _record_solution(self, assignment: List[int], quality: float):
        """
        Record a promising solution for potential GAN retraining.
        
        Args:
            assignment: Variable assignment
            quality: Solution quality (e.g., satisfaction ratio)
        """
        # Only record if it's a good solution
        if quality > 0.8:  # Record solutions that satisfy at least 80% of clauses
            self.solution_history.append(assignment.copy())
            self.new_solutions_since_retrain += 1
    
    def _retrain_gan_if_needed(self, force: bool = False):
        """
        Retrain the GAN model if enough new solutions have been found.
        
        Args:
            force: Whether to force retraining regardless of solution count
        """
        # Skip if no GAN model or not enough new solutions
        if self.gan is None or (self.new_solutions_since_retrain < 50 and not force):
            return
        
        # Limit training data size
        if len(self.solution_history) > 1000:
            # Keep more recent solutions with higher probability
            self.solution_history = random.sample(self.solution_history, 500)
        
        # Only retrain if we have enough solutions
        if len(self.solution_history) >= 50:
            print(f"Retraining GAN with {len(self.solution_history)} solutions...")
            
            # Train for fewer epochs during retraining
            self.gan.train(
                solutions=self.solution_history,
                batch_size=min(32, len(self.solution_history) // 2),
                epochs=20,
                eval_interval=5
            )
            
            self.new_solutions_since_retrain = 0
            print("GAN retraining complete")
    
    def solve(self, max_episodes: int = 1000, early_stopping: bool = True, 
              timeout: Optional[int] = None, gan_retrain_interval: int = 100) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """
        Solve the SAT problem using GAN-powered multi-agent Q-learning.
        
        Args:
            max_episodes: Maximum number of episodes
            early_stopping: Whether to stop early when a solution is found
            timeout: Timeout in seconds
            gan_retrain_interval: Episode interval for GAN retraining
            
        Returns:
            Tuple of (solution, stats) where solution is the variable assignment
            and stats contains statistics about the solving process
        """
        start_time = time.time()
        episodes_completed = 0
        solved = False
        timed_out = False
        
        # Initialize with a GAN-generated assignment if possible, otherwise random
        if self.gan is not None:
            current_state = self._get_gan_assignment()
            self.gan_generations_used += 1
        else:
            current_state = [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
        
        for episode in range(max_episodes):
            episodes_completed = episode + 1
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                timed_out = True
                break
            
            # For exploration, use GAN-generated assignments with probability gan_exploration_ratio
            # otherwise use random assignment (to maintain some diversity)
            if random.random() < self.epsilon:
                if self.gan is not None and random.random() < self.gan_exploration_ratio:
                    current_state = self._get_gan_assignment()
                    self.gan_generations_used += 1
                else:
                    current_state = [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
            
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
                
                # Record promising solutions for GAN retraining
                satisfaction_ratio = count_satisfied_clauses(self.clauses, next_state) / len(self.clauses)
                self._record_solution(next_state, satisfaction_ratio)
                
                # Update current state
                current_state = next_state
                
                # Check if problem is solved
                if is_satisfied(self.clauses, current_state):
                    solved = True
                    
                    # Always record full solutions
                    self._record_solution(current_state, 1.0)
                    
                    if early_stopping:
                        break
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Periodically retrain the GAN
            if episode % gan_retrain_interval == 0 and episode > 0:
                self._retrain_gan_if_needed()
            
            # Early stopping if solved
            if solved and early_stopping:
                break
        
        # Final GAN retraining if we've found a solution
        if solved and self.gan is not None:
            self._retrain_gan_if_needed(force=True)
        
        # Prepare solution
        solution = self.best_assignment if self.best_assignment else current_state
        
        # Calculate statistics
        end_time = time.time()
        solve_time = end_time - start_time
        
        stats = {
            'solved': solved,
            'episodes': episodes_completed,
            'time': solve_time,
            'best_satisfied': self.best_satisfied,
            'best_satisfaction_ratio': self.best_satisfaction_ratio,
            'final_epsilon': self.epsilon,
            'timed_out': timed_out,
            'gan_generated_solutions': self.gan_generated_solutions,
            'gan_generations_used': self.gan_generations_used,
            'solutions_collected': len(self.solution_history)
        }
        
        # Save the GAN model if it was used and improved
        if self.gan is not None and solved:
            try:
                self.gan.save("models/sat_gan_improved.pth")
                print("Saved improved GAN model to models/sat_gan_improved.pth")
            except Exception as e:
                print(f"Failed to save improved GAN model: {str(e)}")
        
        return solution, stats

# Advanced version with iterative GAN training during solution process
class AdaptiveMultiQLearningSATGAN(MultiQLearningSATGAN):
    """
    Advanced version of GAN-powered RL with adaptive exploration strategy.
    This version dynamically adjusts the GAN exploration ratio based on
    solution quality and periodically reinitializes weaker agents.
    """
    
    def __init__(self, n_vars: int, clauses: List[List[int]], 
                 gan_model: SATGAN = None,
                 gan_model_path: Optional[str] = None,
                 n_agents: int = 5, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95, 
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, 
                 min_epsilon: float = 0.01,
                 gan_exploration_ratio: float = 0.8,
                 gan_temperature: float = 1.2,
                 adaptation_interval: int = 20):
        """
        Initialize the adaptive GAN-powered multi-agent Q-learning SAT solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses, where each clause is a list of literals
            gan_model: Pre-trained SATGAN model
            gan_model_path: Path to a saved SATGAN model (used if gan_model is None)
            n_agents: Number of agents (default: 5)
            learning_rate: Learning rate for Q-learning (default: 0.1)
            discount_factor: Discount factor for future rewards (default: 0.95)
            epsilon: Initial exploration rate (default: 0.1)
            epsilon_decay: Rate at which epsilon decays (default: 0.995)
            min_epsilon: Minimum exploration rate (default: 0.01)
            gan_exploration_ratio: Ratio of exploration steps that use GAN (default: 0.8)
            gan_temperature: Temperature for GAN sampling (default: 1.2)
            adaptation_interval: Interval for adapting exploration strategy (default: 20)
        """
        super().__init__(n_vars, clauses, gan_model, gan_model_path, n_agents,
                        learning_rate, discount_factor, epsilon, epsilon_decay,
                        min_epsilon, gan_exploration_ratio, gan_temperature)
        
        self.adaptation_interval = adaptation_interval
        
        # Track agent performance for adaptive strategy
        self.agent_performance = [0.0] * n_agents
        self.exploration_success_rate = 0.0
        self.random_success_rate = 0.0
        self.exploration_attempts = 1  # Avoid division by zero
        self.random_attempts = 1
    
    def _adaptive_exploration(self) -> List[int]:
        """
        Use an adaptive exploration strategy that dynamically balances
        GAN-generated assignments and random assignments based on success rates.
        
        Returns:
            A new state assignment for exploration
        """
        # Dynamically adjust exploration ratio based on success rates
        if self.exploration_success_rate > self.random_success_rate:
            # GAN exploration is more successful, increase its probability
            exploration_prob = min(0.95, self.gan_exploration_ratio * 1.05)
        else:
            # Random exploration is more successful, increase its probability
            exploration_prob = max(0.2, self.gan_exploration_ratio * 0.95)
        
        # Update the exploration ratio
        self.gan_exploration_ratio = exploration_prob
        
        # Use the updated exploration ratio to decide
        if self.gan is not None and random.random() < self.gan_exploration_ratio:
            assignment = self._get_gan_assignment()
            self.gan_generations_used += 1
            self.exploration_attempts += 1
            return assignment
        else:
            self.random_attempts += 1
            return [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
    
    def _reinitialize_weak_agents(self):
        """
        Reinitialize Q-tables for the worst performing agents to
        escape local optima and encourage exploration of new strategies.
        """
        # Skip if too early in the process
        if sum(self.agent_performance) == 0:
            return
        
        # Find the worst performing agent
        worst_idx = np.argmin(self.agent_performance)
        worst_performance = self.agent_performance[worst_idx]
        
        # Only reinitialize if significantly worse than average
        avg_performance = sum(self.agent_performance) / len(self.agent_performance)
        if worst_performance < avg_performance * 0.7:
            print(f"Reinitializing agent {worst_idx} (performance: {worst_performance:.4f})")
            
            # Reinitialize its Q-table
            self.q_tables[worst_idx] = {}
            
            # Reset its performance metric
            self.agent_performance[worst_idx] = avg_performance
    
    def _update_exploration_success(self, state: List[int], next_state: List[int], is_gan_exploration: bool):
        """
        Update success metrics for adaptive exploration strategy.
        
        Args:
            state: Current state before exploration
            next_state: State after exploration
            is_gan_exploration: Whether GAN was used for this exploration
        """
        # Calculate satisfaction improvement
        current_satisfied = count_satisfied_clauses(self.clauses, state)
        next_satisfied = count_satisfied_clauses(self.clauses, next_state)
        improvement = next_satisfied - current_satisfied
        
        # Consider it a success if there's any improvement
        success = improvement > 0
        
        # Update appropriate success rate
        if is_gan_exploration:
            self.exploration_success_rate = (self.exploration_success_rate * 
                                           (self.exploration_attempts - 1) + 
                                           (1 if success else 0)) / self.exploration_attempts
        else:
            self.random_success_rate = (self.random_success_rate * 
                                       (self.random_attempts - 1) + 
                                       (1 if success else 0)) / self.random_attempts
    
    def _update_agent_performance(self, agent_idx: int, reward: float):
        """
        Update the performance metric for an agent.
        
        Args:
            agent_idx: Index of the agent
            reward: Reward received by the agent
        """
        # Simple exponential moving average of rewards
        alpha = 0.1  # Learning rate for performance tracking
        self.agent_performance[agent_idx] = (1 - alpha) * self.agent_performance[agent_idx] + alpha * reward
    
    def solve(self, max_episodes: int = 1000, early_stopping: bool = True, 
              timeout: Optional[int] = None, gan_retrain_interval: int = 100) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """
        Solve the SAT problem using adaptive GAN-powered multi-agent Q-learning.
        
        Args:
            max_episodes: Maximum number of episodes
            early_stopping: Whether to stop early when a solution is found
            timeout: Timeout in seconds
            gan_retrain_interval: Episode interval for GAN retraining
            
        Returns:
            Tuple of (solution, stats) where solution is the variable assignment
            and stats contains statistics about the solving process
        """
        start_time = time.time()
        episodes_completed = 0
        solved = False
        timed_out = False
        
        # Initialize with a GAN-generated assignment if possible, otherwise random
        if self.gan is not None:
            current_state = self._get_gan_assignment()
            self.gan_generations_used += 1
            is_gan_exploration = True
        else:
            current_state = [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
            is_gan_exploration = False
        
        for episode in range(max_episodes):
            episodes_completed = episode + 1
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                timed_out = True
                break
            
            # For exploration, use adaptive strategy
            if random.random() < self.epsilon:
                previous_state = current_state.copy()
                current_state = self._adaptive_exploration()
                is_gan_exploration = self.gan is not None and episodes_completed > 1 and self.gan_generations_used > self.exploration_attempts - 1
                
                # Update exploration success metrics
                if episodes_completed > 1:
                    self._update_exploration_success(previous_state, current_state, is_gan_exploration)
            
            # Adapt exploration strategy periodically
            if episode % self.adaptation_interval == 0 and episode > 0:
                self._reinitialize_weak_agents()
                
                # Print debug info if significant episodes have passed
                if episode % 50 == 0:
                    print(f"Episode {episode}: GAN exploration ratio = {self.gan_exploration_ratio:.2f}, "
                          f"GAN success = {self.exploration_success_rate:.2f}, "
                          f"Random success = {self.random_success_rate:.2f}")
            
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
                
                # Track agent performance
                self._update_agent_performance(agent_idx, reward)
                
                # Track best solution
                self._track_best_solution(next_state)
                
                # Record promising solutions for GAN retraining
                satisfaction_ratio = count_satisfied_clauses(self.clauses, next_state) / len(self.clauses)
                self._record_solution(next_state, satisfaction_ratio)
                
                # Update current state
                current_state = next_state
                
                # Check if problem is solved
                if is_satisfied(self.clauses, current_state):
                    solved = True
                    
                    # Always record full solutions
                    self._record_solution(current_state, 1.0)
                    
                    if early_stopping:
                        break
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Periodically retrain the GAN
            if episode % gan_retrain_interval == 0 and episode > 0:
                self._retrain_gan_if_needed()
            
            # Early stopping if solved
            if solved and early_stopping:
                break
        
        # Final GAN retraining if we've found a solution
        if solved and self.gan is not None:
            self._retrain_gan_if_needed(force=True)
        
        # Prepare solution
        solution = self.best_assignment if self.best_assignment else current_state
        
        # Calculate statistics
        end_time = time.time()
        solve_time = end_time - start_time
        
        stats = {
            'solved': solved,
            'episodes': episodes_completed,
            'time': solve_time,
            'best_satisfied': self.best_satisfied,
            'best_satisfaction_ratio': self.best_satisfaction_ratio,
            'final_epsilon': self.epsilon,
            'timed_out': timed_out,
            'gan_generated_solutions': self.gan_generated_solutions,
            'gan_generations_used': self.gan_generations_used,
            'solutions_collected': len(self.solution_history),
            'gan_exploration_ratio': self.gan_exploration_ratio,
            'exploration_success_rate': self.exploration_success_rate,
            'random_success_rate': self.random_success_rate,
            'agent_performance': self.agent_performance
        }
        
        # Save the GAN model if it was used and improved
        if self.gan is not None and solved:
            try:
                self.gan.save("models/sat_gan_adaptive_improved.pth")
                print("Saved improved adaptive GAN model to models/sat_gan_adaptive_improved.pth")
            except Exception as e:
                print(f"Failed to save improved GAN model: {str(e)}")
        
        return solution, stats

if __name__ == "__main__":
    # Example usage
    from sat_problems import generate_sat_problem
    
    # Generate a small SAT problem
    n_vars = 20
    n_clauses = 85
    clauses = generate_sat_problem(n_vars, n_clauses)
    
    # First phase: train the GAN with some initial solutions
    print("Phase 1: Generating initial solutions for GAN training...")
    from sat_problems import random_walksat
    
    solutions = []
    for _ in range(50):
        solution, solved = random_walksat(clauses, n_vars, max_flips=1000)
        solutions.append(solution)
    
    print(f"Generated {len(solutions)} initial solutions for GAN training")
    
    # Create and train the GAN
    from sat_gan import SATGAN
    gan = SATGAN(n_vars, clauses, latent_dim=50, clause_weight=0.5)
    
    print("Training initial GAN model...")
    gan.train(
        solutions=solutions,
        batch_size=16,
        epochs=50,
        eval_interval=10
    )
    
    # Phase 2: Use GAN-powered RL to solve the problem
    print("\nPhase 2: Solving with GAN-powered RL...")
    
    # Create solver
    solver = AdaptiveMultiQLearningSATGAN(
        n_vars, 
        clauses, 
        gan_model=gan,
        gan_exploration_ratio=0.7,
        gan_temperature=1.2
    )
    
    # Solve
    solution, stats = solver.solve(max_episodes=500, early_stopping=True, gan_retrain_interval=50)
    
    # Print results
    print("\nResults:")
    print(f"Solved: {stats['solved']}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Time: {stats['time']:.2f} seconds")
    print(f"Satisfaction: {stats['best_satisfied']}/{len(clauses)} clauses ({stats['best_satisfaction_ratio']:.2%})")
    print(f"GAN generations used: {stats['gan_generations_used']}")
    print(f"Solutions collected for training: {stats['solutions_collected']}")
    print(f"Final GAN exploration ratio: {stats.get('gan_exploration_ratio', 'N/A')}")
    
    if stats['solved']:
        print(f"Solution: {solution}")