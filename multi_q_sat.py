#!/usr/bin/env python3
"""
Multi-agent Q-learning implementation for solving SAT problems.
This is the cooperative version where agents share rewards.
"""

import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any, Optional
from sat_problems import count_satisfied_clauses, is_satisfied

class MultiQLearningSAT:
    """
    Multi-agent Q-learning implementation for solving Boolean Satisfiability (SAT) problems.
    This is the cooperative version where all agents work together.
    """
    
    def __init__(self, n_vars: int, clauses: List[List[int]], 
                 n_agents: int = 5, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        Initialize the multi-agent Q-learning SAT solver.
        
        Args:
            n_vars: Number of variables in the SAT problem
            clauses: List of clauses, where each clause is a list of literals
            n_agents: Number of agents (default: 5)
            learning_rate: Learning rate for Q-learning (default: 0.1)
            discount_factor: Discount factor for future rewards (default: 0.95)
            epsilon: Initial exploration rate (default: 0.1)
            epsilon_decay: Rate at which epsilon decays (default: 0.995)
            min_epsilon: Minimum exploration rate (default: 0.01)
        """
        self.n_vars = n_vars
        self.clauses = clauses
        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-tables for each agent
        # We use a dictionary-based sparse representation since the state space is huge
        self.q_tables = [{} for _ in range(self.n_agents)]
        
        # Assign variables to agents (each agent handles a subset)
        self.agent_vars = self._assign_variables_to_agents()
        
        # Track best solution found so far
        self.best_assignment = None
        self.best_satisfied = 0
        self.best_satisfaction_ratio = 0.0
    
    def _assign_variables_to_agents(self) -> List[List[int]]:
        """
        Assign variables to agents, distributing them evenly.
        
        Returns:
            List of lists, where each inner list contains the variables assigned to an agent
        """
        agent_vars = [[] for _ in range(self.n_agents)]
        
        # Assign variables to agents (1-based indexing for variables)
        for var in range(1, self.n_vars + 1):
            agent_idx = (var - 1) % self.n_agents
            agent_vars[agent_idx].append(var)
        
        return agent_vars
    
    def _state_to_string(self, state: List[int]) -> str:
        """
        Convert a state (variable assignment) to a string for Q-table lookup.
        
        Args:
            state: List of variable assignments (positive = True, negative = False)
            
        Returns:
            String representation of the state
        """
        return ','.join(map(str, state))
    
    def _string_to_state(self, state_str: str) -> List[int]:
        """
        Convert a string representation back to a state.
        
        Args:
            state_str: String representation of the state
            
        Returns:
            List of variable assignments
        """
        return list(map(int, state_str.split(',')))
    
    def _get_q_value(self, agent_idx: int, state: List[int], action: int) -> float:
        """
        Get Q-value for a given agent, state, and action.
        
        Args:
            agent_idx: Index of the agent
            state: Current state (variable assignment)
            action: Action to take (flip a specific variable)
            
        Returns:
            Q-value for the given state-action pair
        """
        state_str = self._state_to_string(state)
        
        if state_str not in self.q_tables[agent_idx]:
            # Initialize Q-values for this state
            self.q_tables[agent_idx][state_str] = {}
        
        if action not in self.q_tables[agent_idx][state_str]:
            # Initialize Q-value for this action
            self.q_tables[agent_idx][state_str][action] = 0.0
        
        return self.q_tables[agent_idx][state_str][action]
    
    def _set_q_value(self, agent_idx: int, state: List[int], action: int, value: float) -> None:
        """
        Set Q-value for a given agent, state, and action.
        
        Args:
            agent_idx: Index of the agent
            state: Current state (variable assignment)
            action: Action to take (flip a specific variable)
            value: New Q-value
        """
        state_str = self._state_to_string(state)
        
        if state_str not in self.q_tables[agent_idx]:
            # Initialize Q-values for this state
            self.q_tables[agent_idx][state_str] = {}
        
        self.q_tables[agent_idx][state_str][action] = value
    
    def _get_available_actions(self, agent_idx: int) -> List[int]:
        """
        Get the list of variables that an agent can flip.
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            List of variables that the agent can flip
        """
        return self.agent_vars[agent_idx]
    
    def _select_action(self, agent_idx: int, state: List[int]) -> int:
        """
        Select an action for an agent using epsilon-greedy policy.
        
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
        else:
            # Exploitation: choose the best action based on Q-values
            max_q_value = float('-inf')
            best_actions = []
            
            for action in available_actions:
                q_value = self._get_q_value(agent_idx, state, action)
                
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_actions = [action]
                elif q_value == max_q_value:
                    best_actions.append(action)
            
            # If all Q-values are the same (e.g., all 0), choose randomly
            if not best_actions or max_q_value == float('-inf'):
                return random.choice(available_actions)
            
            # Choose randomly among the best actions
            return random.choice(best_actions)
    
    def _take_action(self, state: List[int], var_to_flip: int) -> List[int]:
        """
        Take an action by flipping a variable in the state.
        
        Args:
            state: Current state (variable assignment)
            var_to_flip: Variable to flip
            
        Returns:
            New state after the action
        """
        new_state = state.copy()
        
        # Find the variable in the state
        for i, var in enumerate(new_state):
            if abs(var) == var_to_flip:
                # Flip the variable
                new_state[i] = -var
                break
        
        return new_state
    
    def _get_reward(self, state: List[int], next_state: List[int]) -> float:
        """
        Calculate the reward for a transition.
        
        Args:
            state: Current state (variable assignment)
            next_state: Next state after the action
            
        Returns:
            Reward for the transition
        """
        # Count satisfied clauses in both states
        current_satisfied = count_satisfied_clauses(self.clauses, state)
        next_satisfied = count_satisfied_clauses(self.clauses, next_state)
        
        # Reward is based on improvement in satisfaction
        reward = next_satisfied - current_satisfied
        
        # Additional reward for finding a complete solution
        if is_satisfied(self.clauses, next_state):
            reward += 10.0
        
        # Scale the reward
        reward_scale = 1.0 / len(self.clauses)
        return reward * reward_scale
    
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
        # Get the current Q-value
        current_q = self._get_q_value(agent_idx, state, action)
        
        # Get the maximum Q-value for the next state
        max_next_q = float('-inf')
        for next_action in self._get_available_actions(agent_idx):
            next_q = self._get_q_value(agent_idx, next_state, next_action)
            max_next_q = max(max_next_q, next_q)
        
        if max_next_q == float('-inf'):
            max_next_q = 0.0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update the Q-value
        self._set_q_value(agent_idx, state, action, new_q)
    
    def _convert_to_assignment(self, state: List[int]) -> List[int]:
        """
        Convert a state to a variable assignment representation.
        
        Args:
            state: Variable assignment state
            
        Returns:
            Variable assignment representation (positive = True, negative = False)
        """
        return state
    
    def _track_best_solution(self, state: List[int]) -> None:
        """
        Track the best solution found so far.
        
        Args:
            state: Current state (variable assignment)
        """
        # Count satisfied clauses
        satisfied = count_satisfied_clauses(self.clauses, state)
        satisfaction_ratio = satisfied / len(self.clauses)
        
        # Update best solution if better
        if satisfied > self.best_satisfied:
            self.best_assignment = state.copy()
            self.best_satisfied = satisfied
            self.best_satisfaction_ratio = satisfaction_ratio
    
    def solve(self, max_episodes: int = 1000, early_stopping: bool = True, 
              timeout: Optional[int] = None) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """
        Solve the SAT problem using multi-agent Q-learning.
        
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
            'timed_out': timed_out
        }
        
        return solution, stats

if __name__ == "__main__":
    # Example usage
    from sat_problems import generate_sat_problem
    
    # Generate a small SAT problem
    n_vars = 20
    n_clauses = 80
    clauses = generate_sat_problem(n_vars, n_clauses)
    
    # Create solver
    solver = MultiQLearningSAT(n_vars, clauses)
    
    # Solve
    print(f"Solving {n_vars}-variable, {n_clauses}-clause SAT problem...")
    solution, stats = solver.solve(max_episodes=1000, early_stopping=True)
    
    # Print results
    print(f"Solved: {stats['solved']}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Time: {stats['time']:.2f} seconds")
    print(f"Satisfaction: {stats['best_satisfied']}/{len(clauses)} clauses ({stats['best_satisfaction_ratio']:.2%})")
    
    if stats['solved']:
        print(f"Solution: {solution}")
