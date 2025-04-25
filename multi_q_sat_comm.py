#!/usr/bin/env python3
"""
Multi-agent Q-learning implementation for solving SAT problems.
This version includes communication between agents.
"""

import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any, Optional
from sat_problems import count_satisfied_clauses, is_satisfied
from multi_q_sat import MultiQLearningSAT

class MultiQLearningSATComm(MultiQLearningSAT):
    """
    Multi-agent Q-learning implementation with communication between agents.
    Agents can share experiences with each other based on rewards.
    """
    
    def __init__(self, n_vars: int, clauses: List[List[int]], 
                 n_agents: int = 5, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01,
                 comm_threshold: float = 0.5, comm_weight: float = 0.3):
        """
        Initialize the communicating multi-agent Q-learning SAT solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses, where each clause is a list of literals
            n_agents: Number of agents (default: 5)
            learning_rate: Learning rate for Q-learning (default: 0.1)
            discount_factor: Discount factor for future rewards (default: 0.95)
            epsilon: Initial exploration rate (default: 0.1)
            epsilon_decay: Rate at which epsilon decays (default: 0.995)
            min_epsilon: Minimum exploration rate (default: 0.01)
            comm_threshold: Reward threshold for communication (default: 0.5)
            comm_weight: Weight of communicated Q-values (default: 0.3)
        """
        super().__init__(n_vars, clauses, n_agents, learning_rate, 
                        discount_factor, epsilon, epsilon_decay, min_epsilon)
        self.comm_threshold = comm_threshold
        self.comm_weight = comm_weight
        
        # Track experiences to share
        self.experiences = []
    
    def _update_q_value(self, agent_idx: int, state: List[int], action: int, reward: float, next_state: List[int]) -> None:
        """
        Update the Q-value for a given agent, state, and action.
        
        Args:
            agent_idx: Index of the agent
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after the action
        """
        # Regular Q-value update (same as parent class)
        super()._update_q_value(agent_idx, state, action, reward, next_state)
        
        # If the reward is good enough, share this experience with other agents
        if reward > self.comm_threshold:
            self.experiences.append({
                'agent_idx': agent_idx,
                'state': state.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_state.copy()
            })
    
    def _share_experiences(self) -> None:
        """
        Share experiences between agents.
        """
        if not self.experiences:
            return
        
        # Sort experiences by reward (highest first)
        self.experiences.sort(key=lambda x: x['reward'], reverse=True)
        
        # Share the top experiences with all agents except the originator
        for exp in self.experiences:
            source_agent = exp['agent_idx']
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            
            # Get the Q-value from the source agent
            source_q = self._get_q_value(source_agent, state, action)
            
            # Share with other agents who can perform this action
            for agent_idx in range(self.n_agents):
                if agent_idx != source_agent and action in self._get_available_actions(agent_idx):
                    # Get the current Q-value for the target agent
                    target_q = self._get_q_value(agent_idx, state, action)
                    
                    # Blend the Q-values based on the communication weight
                    new_q = (1 - self.comm_weight) * target_q + self.comm_weight * source_q
                    
                    # Update the target agent's Q-value
                    self._set_q_value(agent_idx, state, action, new_q)
        
        # Clear experiences after sharing
        self.experiences = []
    
    def solve(self, max_episodes: int = 1000, early_stopping: bool = True, 
              timeout: Optional[int] = None, comm_frequency: int = 5) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """
        Solve the SAT problem using multi-agent Q-learning with communication.
        
        Args:
            max_episodes: Maximum number of episodes
            early_stopping: Whether to stop early when a solution is found
            timeout: Timeout in seconds
            comm_frequency: Frequency of communication between agents (in episodes)
            
        Returns:
            Tuple of (solution, stats) where solution is the variable assignment
            and stats contains statistics about the solving process
        """
        start_time = time.time()
        episodes_completed = 0
        solved = False
        timed_out = False
        communications = 0
        
        # Initialize with a random assignment
        current_state = [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
        
        for episode in range(max_episodes):
            episodes_completed = episode + 1
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                timed_out = True
                break
            
            # Each episode starts with a random state with probability epsilon,
            # otherwise continues from the current state
            if random.random() < self.epsilon:
                current_state = [random.choice([var, -var]) for var in range(1, self.n_vars + 1)]
            
            # Each agent takes an action in turn
            for agent_idx in range(self.n_agents):
                # Select action
                action = self._select_action(agent_idx, current_state)
                
                # Take action
                next_state = self._take_action(current_state, action)
                
                # Calculate reward
                reward = self._get_reward(current_state, next_state)
                
                # Update Q-value (and potentially record experience for sharing)
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
            
            # Share experiences periodically
            if episode % comm_frequency == 0 and episode > 0:
                self._share_experiences()
                communications += 1
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Early stopping if solved
            if solved and early_stopping:
                break
        
        # Final communication
        if self.experiences:
            self._share_experiences()
            communications += 1
        
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
            'communications': communications,
            'comm_threshold': self.comm_threshold,
            'comm_weight': self.comm_weight
        }
        
        return solution, stats

if __name__ == "__main__":
    # Example usage
    from sat_problems import generate_sat_problem
    
    # Generate a small SAT problem
    n_vars = 20
    n_clauses = 85
    clauses = generate_sat_problem(n_vars, n_clauses)
    
    # Create solver with communication
    solver = MultiQLearningSATComm(n_vars, clauses, comm_threshold=0.5, comm_weight=0.3)
    
    # Solve
    print(f"Solving {n_vars}-variable, {n_clauses}-clause SAT problem with communication...")
    solution, stats = solver.solve(max_episodes=1000, early_stopping=True)
    
    # Print results
    print(f"Solved: {stats['solved']}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Time: {stats['time']:.2f} seconds")
    print(f"Satisfaction: {stats['best_satisfied']}/{len(clauses)} clauses ({stats['best_satisfaction_ratio']:.2%})")
    print(f"Communications: {stats['communications']}")
    
    if stats['solved']:
        print(f"Solution: {solution}")