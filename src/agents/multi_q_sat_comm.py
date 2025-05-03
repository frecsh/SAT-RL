#!/usr/bin/env python3
"""
Multi-agent Q-learning implementation for solving SAT problems.
This version includes communication between agents.
"""

import time
import numpy as np
import random
from collections import defaultdict, deque
from src.sat_problems import count_satisfied_clauses, generate_sat_problem

class MultiQLearningSATCommunicative:
    """Multi-agent Q-Learning for SAT problems with communicative agents"""
    
    def __init__(self, num_vars, clauses, n_agents=3, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.2, epsilon_decay=0.99, epsilon_min=0.01, communication_freq=0.3):
        self.num_vars = num_vars
        self.clauses = clauses
        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.communication_freq = communication_freq
        
        # Each agent focuses on a subset of variables
        self.agent_var_assignments = self._assign_variables_to_agents()
        
        # Q-tables for each agent (state-action value function)
        self.q_tables = [defaultdict(lambda: np.zeros(len(agent_vars) * 2)) 
                        for agent_vars in self.agent_var_assignments]
        
        # Shared knowledge base for communication
        self.knowledge_base = {
            "positive_vars": set(),  # Variables that should be positive
            "negative_vars": set(),  # Variables that should be negative
            "good_actions": {},      # State-action pairs that worked well
            "bad_clauses": []        # Clauses that are hard to satisfy
        }
        
        # Best solution found so far
        self.best_solution = None
        self.best_satisfied = 0
        self.best_satisfaction_ratio = 0.0
    
    def _assign_variables_to_agents(self):
        """Assign variables to agents"""
        agent_vars = [[] for _ in range(self.n_agents)]
        
        # First assign based on clauses (variables that appear together)
        var_groups = self._group_related_variables()
        
        # Distribute groups among agents
        for i, group in enumerate(var_groups):
            agent_idx = i % self.n_agents
            for var in group:
                if var not in agent_vars[agent_idx]:
                    agent_vars[agent_idx].append(var)
        
        # Make sure each agent has some variables (add any missing)
        for var in range(1, self.num_vars + 1):
            assigned = False
            for agent_vars_list in agent_vars:
                if var in agent_vars_list:
                    assigned = True
                    break
            if not assigned:
                # Assign to agent with fewest variables
                agent_idx = np.argmin([len(vars_list) for vars_list in agent_vars])
                agent_vars[agent_idx].append(var)
        
        return agent_vars
    
    def _group_related_variables(self):
        """Group variables that appear together in clauses"""
        # Create graph where variables in the same clause are connected
        graph = defaultdict(set)
        for clause in self.clauses:
            for lit1 in clause:
                for lit2 in clause:
                    if lit1 != lit2:
                        graph[abs(lit1)].add(abs(lit2))
        
        # Find connected components to create groups
        visited = set()
        groups = []
        
        for var in range(1, self.num_vars + 1):
            if var in visited:
                continue
            
            group = []
            queue = [var]
            visited.add(var)
            
            while queue:
                current = queue.pop(0)
                group.append(current)
                
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            groups.append(group)
        
        return groups
    
    def _get_state_key(self, solution):
        """Convert solution to a hashable state representation"""
        # Convert solution to tuple so it can be used as a dictionary key
        return tuple(solution)
    
    def _choose_actions(self, solution):
        """Have agents select actions using epsilon-greedy policy with communication influence"""
        actions = []
        
        for agent_idx, agent_vars in enumerate(self.agent_var_assignments):
            # Skip agents that have no variables assigned
            if len(agent_vars) == 0:
                continue
                
            state_key = self._get_state_key(solution)
            q_values = self.q_tables[agent_idx][state_key].copy()  # Copy to modify
            
            # Ensure q_values is not empty
            if len(q_values) == 0:
                # Create a default action space if none exists
                self.q_tables[agent_idx][state_key] = np.zeros(2)  # Minimum action space
                q_values = self.q_tables[agent_idx][state_key].copy()
            
            # Apply communication knowledge to adjust Q-values
            for var_idx, var in enumerate(agent_vars):
                # Check if knowledge base suggests this variable should be positive
                if var in self.knowledge_base["positive_vars"]:
                    # Increase Q-value for setting var to True
                    if var_idx * 2 + 1 < len(q_values):
                        q_values[var_idx * 2 + 1] += 0.2
                
                # Check if knowledge base suggests this variable should be negative
                if var in self.knowledge_base["negative_vars"]:
                    # Increase Q-value for setting var to False
                    if var_idx * 2 < len(q_values):
                        q_values[var_idx * 2 + 0] += 0.2
            
            # Check if this state-action is known to be good
            if state_key in self.knowledge_base["good_actions"]:
                for action in self.knowledge_base["good_actions"][state_key]:
                    if action < len(q_values):
                        q_values[action] += 0.3
            
            # Epsilon-greedy action selection with communication-adjusted Q-values
            if np.random.rand() < self.epsilon:
                # Random action
                var_idx = np.random.randint(len(agent_vars))
                flip_value = np.random.choice([0, 1])  # 0: set to false, 1: set to true
                action = var_idx * 2 + flip_value
            else:
                # Greedy action using adjusted Q-values
                action = np.argmax(q_values)
            
            actions.append((agent_idx, action))
        
        # Ensure we have at least one action
        if not actions:
            # If no agent had actions, create a default action
            actions = [(0, 0)]
            
        return actions
    
    def _update_q_value(self, agent_idx, state, action, next_state, reward):
        """Update Q-value for an agent"""
        # Make sure the action is within bounds for this agent's Q-table
        q_values = self.q_tables[agent_idx][state]
        if len(q_values) == 0:
            # Create a default action space if none exists
            self.q_tables[agent_idx][state] = np.zeros(2)  # Minimum action space
            q_values = self.q_tables[agent_idx][state]
            
        if action >= len(q_values):
            # Handle out-of-bounds actions by clipping to valid range
            action = len(q_values) - 1
        
        # Get current Q-value
        q_current = q_values[action]
        
        # Get maximum Q-value for next state
        next_q_values = self.q_tables[agent_idx][next_state]
        if len(next_q_values) == 0:
            # Create a default action space for next state if none exists
            self.q_tables[agent_idx][next_state] = np.zeros(2)
            next_q_values = self.q_tables[agent_idx][next_state]
            
        q_next_max = np.max(next_q_values)
        
        # Q-learning update rule
        q_new = q_current + self.learning_rate * (
            reward + self.discount_factor * q_next_max - q_current)
        
        # Update Q-value
        self.q_tables[agent_idx][state][action] = q_new
        
        # Share this knowledge if it was valuable
        if reward > 0.1 and np.random.rand() < self.communication_freq:
            # Convert action to variable and value
            agent_vars = self.agent_var_assignments[agent_idx]
            var_idx = action // 2
            new_value = action % 2
            
            if var_idx < len(agent_vars):
                var = agent_vars[var_idx]
                # Share this knowledge
                if new_value == 1:
                    self.knowledge_base["positive_vars"].add(var)
                else:
                    self.knowledge_base["negative_vars"].add(var)
                
                # Record good action
                if state not in self.knowledge_base["good_actions"]:
                    self.knowledge_base["good_actions"][state] = []
                self.knowledge_base["good_actions"][state].append(action)
    
    def _execute_action(self, solution, agent_idx, action):
        """Execute an action and return the new solution"""
        new_solution = solution.copy()
        agent_vars = self.agent_var_assignments[agent_idx]
        
        var_idx = action // 2
        new_value = action % 2
        
        if var_idx < len(agent_vars):
            var = agent_vars[var_idx]
            # Set variable to True or False based on new_value
            new_solution[var-1] = 1 if new_value == 1 else -1
        
        return new_solution
    
    def _identify_difficult_clauses(self, solution):
        """Identify clauses that are difficult to satisfy"""
        unsatisfied = []
        for i, clause in enumerate(self.clauses):
            if not any(lit * solution[abs(lit)-1] > 0 for lit in clause):
                unsatisfied.append(i)
        
        # Update knowledge base with difficult clauses
        # (focus on consistent ones)
        for clause_idx in unsatisfied:
            if clause_idx in self.knowledge_base["bad_clauses"]:
                # If this clause is repeatedly unsatisfied, increase its priority
                self.knowledge_base["bad_clauses"].append(clause_idx)
            else:
                self.knowledge_base["bad_clauses"].append(clause_idx)
        
        # Keep the list manageable
        if len(self.knowledge_base["bad_clauses"]) > 20:
            # Count occurrences and keep most frequent
            clause_counts = {}
            for idx in self.knowledge_base["bad_clauses"]:
                clause_counts[idx] = clause_counts.get(idx, 0) + 1
            
            # Only keep clauses with multiple occurrences
            self.knowledge_base["bad_clauses"] = [
                idx for idx, count in clause_counts.items() if count > 1
            ]
        
        return unsatisfied
    
    def _communicate(self):
        """Agents communicate to share valuable knowledge"""
        # Agents share knowledge about their best actions
        # (Already implemented in _update_q_value)
        
        # Agents share information about difficult clauses
        # (Already implemented in _identify_difficult_clauses)
        pass
    
    def solve(self, max_episodes=1000, early_stopping=False):
        """
        Attempt to solve the SAT problem using communicative multi-agent Q-learning.
        
        Args:
            max_episodes: Maximum number of episodes to run
            early_stopping: Whether to stop when a satisfying assignment is found
            
        Returns:
            Tuple of (best_solution, statistics dictionary)
        """
        start_time = time.time()
        episode_rewards = []
        best_reward_progress = []
        solution_found = False
        solution_episode = None
        
        for episode in range(max_episodes):
            # Initialize random solution
            solution = [np.random.choice([-1, 1]) for _ in range(self.num_vars)]
            
            # Initial satisfaction
            satisfied = count_satisfied_clauses(self.clauses, solution)
            initial_reward = satisfied / len(self.clauses)
            total_reward = 0
            
            # Identify difficult clauses for this episode
            difficult_clauses = self._identify_difficult_clauses(solution)
            
            # Each agent takes turns
            for step in range(100):  # Limit steps per episode
                # Communication happens with some probability
                if np.random.rand() < self.communication_freq:
                    self._communicate()
                
                # Have agents choose actions (one action per agent)
                agent_actions = self._choose_actions(solution)
                
                for agent_idx, action in agent_actions:
                    # Get current state key
                    state_key = self._get_state_key(solution)
                    
                    # Execute action
                    new_solution = self._execute_action(solution, agent_idx, action)
                    
                    # Calculate reward
                    old_satisfied = count_satisfied_clauses(self.clauses, solution)
                    new_satisfied = count_satisfied_clauses(self.clauses, new_solution)
                    
                    # Check if the action satisfied previously difficult clauses
                    new_difficult_clauses = self._identify_difficult_clauses(new_solution)
                    difficult_clauses_solved = len(set(difficult_clauses) - set(new_difficult_clauses))
                    
                    # Enhanced reward for solving difficult clauses
                    reward = ((new_satisfied - old_satisfied) / len(self.clauses) + 
                             0.2 * difficult_clauses_solved / max(1, len(difficult_clauses)))
                    
                    total_reward += reward
                    
                    # Get next state key
                    next_state_key = self._get_state_key(new_solution)
                    
                    # Update Q-value (includes communication)
                    self._update_q_value(agent_idx, state_key, action, next_state_key, reward)
                    
                    # Update solution
                    solution = new_solution
                    
                    # Check if problem is solved
                    if new_satisfied == len(self.clauses):
                        self.best_solution = solution.copy()
                        self.best_satisfied = new_satisfied
                        self.best_satisfaction_ratio = 1.0
                        solution_found = True
                        solution_episode = episode
                        break
                
                if solution_found and early_stopping:
                    break
            
            # After each episode, update the best solution found so far
            satisfied = count_satisfied_clauses(self.clauses, solution)
            satisfaction_ratio = satisfied / len(self.clauses)
            
            if satisfied > self.best_satisfied:
                self.best_solution = solution.copy()
                self.best_satisfied = satisfied
                self.best_satisfaction_ratio = satisfaction_ratio
            
            # Track episode rewards
            episode_rewards.append(total_reward)
            best_reward_progress.append(self.best_satisfaction_ratio)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Logging
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{max_episodes}, "
                      f"Best satisfied: {self.best_satisfied}/{len(self.clauses)} "
                      f"({self.best_satisfaction_ratio:.2%}), "
                      f"Epsilon: {self.epsilon:.3f}")
                
                # Print knowledge base stats
                print(f"Knowledge base: {len(self.knowledge_base['positive_vars'])} positive vars, "
                      f"{len(self.knowledge_base['negative_vars'])} negative vars, "
                      f"{len(self.knowledge_base['bad_clauses'])} difficult clauses")
            
            # Early stopping if solution found
            if solution_found and early_stopping:
                print(f"Solution found in episode {episode}!")
                break
        
        runtime = time.time() - start_time
        
        # Return metrics
        metrics = {
            "solution_found": solution_found,
            "solution_episode": solution_episode,
            "runtime": runtime,
            "episode_rewards": episode_rewards,
            "best_reward_progress": best_reward_progress,
            "best_satisfaction_ratio": self.best_satisfaction_ratio,
            "knowledge_base_size": {
                "positive_vars": len(self.knowledge_base["positive_vars"]),
                "negative_vars": len(self.knowledge_base["negative_vars"]),
                "good_actions": len(self.knowledge_base["good_actions"]),
                "bad_clauses": len(self.knowledge_base["bad_clauses"])
            }
        }
        
        return self.best_solution, metrics

# Demo code
if __name__ == "__main__":
    # Demo the communicative multi-agent SAT solver
    n_vars = 20
    n_clauses = 85
    clauses = generate_sat_problem(n_vars, n_clauses)
    
    print(f"Testing Communicative Multi-agent Q-Learning on {n_vars}-variable, {n_clauses}-clause problem")
    
    # Create and run solver
    solver = MultiQLearningSATCommunicative(n_vars, clauses, n_agents=5)
    solution, metrics = solver.solve(max_episodes=100, early_stopping=True)
    
    # Report results
    print(f"\nResults:")
    print(f"Solution found: {metrics['solution_found']}")
    if metrics['solution_found']:
        print(f"Found in episode: {metrics['solution_episode']}")
    print(f"Runtime: {metrics['runtime']:.2f} seconds")
    print(f"Best satisfaction ratio: {metrics['best_satisfaction_ratio']:.2%}")
    print(f"Knowledge base size: {metrics['knowledge_base_size']}")