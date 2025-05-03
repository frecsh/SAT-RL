#!/usr/bin/env python3
"""
Multi-agent Q-learning implementation for solving SAT problems.
This version integrates a SAT solver oracle for guidance.
"""

import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any, Optional
from src.sat_problems import count_satisfied_clauses, is_satisfied, random_walksat
from src.agents.multi_q_sat import MultiQLearningSAT

class MultiQLearningSATOracle(MultiQLearningSAT):
    """
    Multi-agent Q-learning implementation with oracle guidance.
    The oracle (a traditional SAT solver) provides guidance to help agents learn faster.
    """
    
    def __init__(self, n_vars: int, clauses: List[List[int]], 
                 n_agents: int = 5, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01,
                 oracle_weight: float = 0.35, oracle_interval: int = 10):
        """
        Initialize the oracle-guided multi-agent Q-learning SAT solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses, where each clause is a list of literals
            n_agents: Number of agents (default: 5)
            learning_rate: Learning rate for Q-learning (default: 0.1)
            discount_factor: Discount factor for future rewards (default: 0.95)
            epsilon: Initial exploration rate (default: 0.1)
            epsilon_decay: Rate at which epsilon decays (default: 0.995)
            min_epsilon: Minimum exploration rate (default: 0.01)
            oracle_weight: Weight of oracle guidance (default: 0.35)
            oracle_interval: Interval between oracle consultations (default: 10)
        """
        super().__init__(n_vars, clauses, n_agents, learning_rate, 
                        discount_factor, epsilon, epsilon_decay, min_epsilon)
        self.oracle_weight = oracle_weight
        self.oracle_interval = oracle_interval
        
        # Track oracle suggestions
        self.oracle_suggestions = []
        self.oracle_consultations = 0
        
        # Track clause difficulty to guide oracle suggestions
        self.clause_difficulty = np.ones(len(clauses)) / len(clauses)
        self.clause_satisfaction_count = np.zeros(len(clauses))
        self.clause_check_count = 0
    
    def _update_clause_difficulty(self, state: List[int]) -> None:
        """
        Update the difficulty estimates for each clause based on the current state.
        Clauses that are frequently unsatisfied are considered more difficult.
        
        Args:
            state: Current variable assignment
        """
        # Convert assignment list to a set for O(1) lookups
        true_lits = set(state)
        
        # Update satisfaction counts for each clause
        for i, clause in enumerate(self.clauses):
            # Check if this clause is satisfied
            satisfied = False
            for lit in clause:
                if lit in true_lits:
                    satisfied = True
                    break
            
            # Update counts
            if satisfied:
                self.clause_satisfaction_count[i] += 1
        
        # Increment check count
        self.clause_check_count += 1
        
        # Update difficulty scores
        if self.clause_check_count >= 10:  # Only update periodically
            # Calculate satisfaction ratio for each clause
            satisfaction_ratio = self.clause_satisfaction_count / self.clause_check_count
            
            # Invert to get difficulty (higher = more difficult)
            difficulty = 1.0 - satisfaction_ratio
            
            # Normalize
            total = np.sum(difficulty)
            if total > 0:
                self.clause_difficulty = difficulty / total
            
            # Reset counters for next period
            self.clause_satisfaction_count = np.zeros(len(self.clauses))
            self.clause_check_count = 0
    
    def _consult_oracle(self, current_state: List[int]) -> List[int]:
        """
        Consult the oracle (a traditional SAT solver) for guidance.
        
        Args:
            current_state: Current variable assignment
            
        Returns:
            Suggested variable assignment from oracle
        """
        # Use WalkSAT as our oracle, starting from the current state
        # We limit the number of flips to make it quick
        solution, solved = random_walksat(
            self.clauses, 
            self.n_vars, 
            max_flips=1000,
            random_probability=0.3
        )
        
        # Record the consultation
        self.oracle_consultations += 1
        
        return solution
    
    def _get_oracle_suggestions(self, current_state: List[int]) -> Dict[int, int]:
        """
        Get variable assignment suggestions from the oracle.
        
        Args:
            current_state: Current variable assignment
            
        Returns:
            Dictionary mapping variables to suggested values
        """
        # Consult the oracle
        oracle_assignment = self._consult_oracle(current_state)
        
        # Convert to dictionary for easy lookup
        suggestions = {}
        for lit in oracle_assignment:
            var = abs(lit)
            value = 1 if lit > 0 else -1
            suggestions[var] = value
        
        return suggestions
    
    def _select_action(self, agent_idx: int, state: List[int]) -> int:
        """
        Select an action for an agent using epsilon-greedy policy with oracle guidance.
        
        Args:
            agent_idx: Index of the agent
            state: Current state (variable assignment)
            
        Returns:
            Variable to flip
        """
        available_actions = self._get_available_actions(agent_idx)
        
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.choice(available_actions)
        
        # Check if we have oracle suggestions
        if self.oracle_suggestions:
            # Get Q-values for all available actions
            q_values = {}
            for action in available_actions:
                q_values[action] = self._get_q_value(agent_idx, state, action)
            
            # Get oracle suggestion for this agent's variables
            oracle_action = None
            for var in available_actions:
                # Check if the oracle has a suggestion for this variable
                if var in self.oracle_suggestions:
                    # Get the current value of the variable in the state
                    current_val = None
                    for lit in state:
                        if abs(lit) == var:
                            current_val = 1 if lit > 0 else -1
                            break
                    
                    # Only consider flipping if the oracle suggests a different value
                    if current_val and self.oracle_suggestions[var] != current_val:
                        oracle_action = var
                        break
            
            if oracle_action:
                # Blend Q-values with oracle suggestion
                best_q = max(q_values.values()) if q_values else 0.0
                for action in available_actions:
                    if action == oracle_action:
                        # Boost the Q-value of the oracle's suggestion
                        q_values[action] = (1.0 - self.oracle_weight) * q_values.get(action, 0.0) + self.oracle_weight * best_q * 1.5
            
                # Find the best action(s) after blending
                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if q == max_q]
                
                # Choose randomly among the best actions
                return random.choice(best_actions)
        
        # Fall back to regular Q-learning selection if no oracle guidance
        return super()._select_action(agent_idx, state)
    
    def solve(self, max_episodes: int = 1000, early_stopping: bool = True, 
              timeout: Optional[int] = None) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """
        Solve the SAT problem using multi-agent Q-learning with oracle guidance.
        
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
        
        # Initialize with a random assignment
        current_state = [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
        
        for episode in range(max_episodes):
            episodes_completed = episode + 1
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                timed_out = True
                break
            
            # Consult oracle periodically
            if episode % self.oracle_interval == 0:
                self.oracle_suggestions = self._get_oracle_suggestions(current_state)
            
            # Each episode starts with a random state with probability epsilon,
            # otherwise continues from the current state
            if random.random() < self.epsilon:
                current_state = [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
            
            # Update clause difficulty
            self._update_clause_difficulty(current_state)
            
            # Each agent takes an action in turn
            for agent_idx in range(self.n_agents):
                # Select action (with potential oracle guidance)
                action = self._select_action(agent_idx, current_state)
                
                # Take action
                next_state = self._take_action(current_state, action)
                
                # Calculate reward
                reward = self._get_reward(current_state, next_state)
                
                # Update Q-value
                self._update_q_value(agent_idx, current_state, action, reward, next_state)
                
                # Track best solution
                self._track_best_solution(next_state)
                
                # Update current state
                current_state = next_state
                
                # Check if problem is solved
                if is_satisfied(self.clauses, current_state):
                    solved = True
                    if early_stopping:
                        break
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Early stopping if solved
            if solved and early_stopping:
                break
            
            # Clear oracle suggestions after a few steps to avoid overreliance
            if episode % self.oracle_interval == self.oracle_interval - 1:
                self.oracle_suggestions = {}
        
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
            'oracle_consultations': self.oracle_consultations,
            'oracle_weight': self.oracle_weight
        }
        
        return solution, stats

if __name__ == "__main__":
    # Example usage
    from src.sat_problems import generate_sat_problem
    
    # Generate a small SAT problem
    n_vars = 20
    n_clauses = 85
    clauses = generate_sat_problem(n_vars, n_clauses)
    
    # Create solver with oracle guidance
    solver = MultiQLearningSATOracle(n_vars, clauses, oracle_weight=0.35)
    
    # Solve
    print(f"Solving {n_vars}-variable, {n_clauses}-clause SAT problem with oracle guidance...")
    solution, stats = solver.solve(max_episodes=1000, early_stopping=True)
    
    # Print results
    print(f"Solved: {stats['solved']}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Time: {stats['time']:.2f} seconds")
    print(f"Satisfaction: {stats['best_satisfied']}/{len(clauses)} clauses ({stats['best_satisfaction_ratio']:.2%})")
    print(f"Oracle consultations: {stats['oracle_consultations']}")
    
    if stats['solved']:
        print(f"Solution: {solution}")