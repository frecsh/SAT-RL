import time
import numpy as np
import random
from collections import defaultdict, deque
from src.sat_problems import count_satisfied_clauses, generate_sat_problem

class MultiQLearningSATCompetitive:
    """Multi-agent Q-Learning for SAT problems with competitive agents"""
    
    def __init__(self, num_vars, clauses, n_agents=3, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.2, epsilon_decay=0.99, epsilon_min=0.01, competition_factor=0.3):
        self.num_vars = num_vars
        self.clauses = clauses
        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.competition_factor = competition_factor
        
        # Each agent focuses on a subset of variables
        self.agent_var_assignments = self._assign_variables_to_agents()
        
        # Q-tables for each agent (state-action value function)
        self.q_tables = [defaultdict(lambda: np.zeros(len(agent_vars) * 2)) 
                        for agent_vars in self.agent_var_assignments]
        
        # Agent scores for competition
        self.agent_scores = np.zeros(n_agents)
        
        # Best solution found so far
        self.best_solution = None
        self.best_satisfied = 0
        self.best_satisfaction_ratio = 0.0
    
    def _assign_variables_to_agents(self):
        """Assign variables to agents with some overlap for competition"""
        agent_vars = [[] for _ in range(self.n_agents)]
        
        # Each variable is primarily assigned to one agent
        for var in range(1, self.num_vars + 1):
            primary_agent = (var - 1) % self.n_agents
            agent_vars[primary_agent].append(var)
            
            # Randomly assign some variables to multiple agents to create competition
            if np.random.rand() < 0.3:  # 30% chance of overlap
                secondary_agent = (primary_agent + 1) % self.n_agents
                agent_vars[secondary_agent].append(var)
        
        return agent_vars
    
    def _get_state_key(self, solution):
        """Convert solution to a hashable state representation"""
        # Convert solution to tuple so it can be used as a dictionary key
        return tuple(solution)
    
    def _choose_actions(self, solution, active_agents):
        """Have active agents select actions using epsilon-greedy policy"""
        actions = []
        
        for agent_idx in active_agents:
            agent_vars = self.agent_var_assignments[agent_idx]
            state_key = self._get_state_key(solution)
            q_values = self.q_tables[agent_idx][state_key]
            
            # Epsilon-greedy action selection
            if np.random.rand() < self.epsilon:
                # Random action
                var_idx = np.random.randint(len(agent_vars))
                flip_value = np.random.choice([0, 1])  # 0: set to false, 1: set to true
                action = var_idx * 2 + flip_value
            else:
                # Greedy action
                action = np.argmax(q_values)
            
            actions.append((agent_idx, action))
        
        return actions
    
    def _update_q_value(self, agent_idx, state, action, next_state, reward):
        """Update Q-value for an agent"""
        # Make sure the action is within bounds for this agent's Q-table
        q_values = self.q_tables[agent_idx][state]
        if action >= len(q_values):
            # Handle out-of-bounds actions by clipping to valid range
            action = len(q_values) - 1
        
        # Get current Q-value
        q_current = q_values[action]
        
        # Get maximum Q-value for next state
        q_next_max = np.max(self.q_tables[agent_idx][next_state])
        
        # Q-learning update rule
        q_new = q_current + self.learning_rate * (
            reward + self.discount_factor * q_next_max - q_current)
        
        # Update Q-value
        self.q_tables[agent_idx][state][action] = q_new
    
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
    
    def _select_active_agents(self):
        """Select which agents get to act based on performance"""
        # All agents are active in first half of agents
        if np.random.rand() < 0.5:
            return list(range(self.n_agents))
        
        # Otherwise, select based on scores
        probs = np.exp(self.agent_scores) / np.sum(np.exp(self.agent_scores))
        selected = np.random.choice(
            self.n_agents, 
            size=max(1, self.n_agents // 2),  # At least one agent
            p=probs,
            replace=False
        )
        return selected.tolist()
    
    def solve(self, max_episodes=1000, early_stopping=False):
        """
        Attempt to solve the SAT problem using competitive multi-agent Q-learning.
        
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
            initial_satisfaction_ratio = satisfied / len(self.clauses)
            episode_reward = 0
            
            # Select active agents for this episode based on performance
            active_agents = self._select_active_agents()
            
            # Agent performance in this episode
            agent_rewards = [0] * self.n_agents
            
            # Each agent takes turns
            for step in range(100):  # Limit steps per episode
                # Have active agents choose actions
                agent_actions = self._choose_actions(solution, active_agents)
                
                for agent_idx, action in agent_actions:
                    # Get current state key
                    state_key = self._get_state_key(solution)
                    
                    # Execute action
                    new_solution = self._execute_action(solution, agent_idx, action)
                    
                    # Calculate reward
                    old_satisfied = count_satisfied_clauses(self.clauses, solution)
                    new_satisfied = count_satisfied_clauses(self.clauses, new_solution)
                    reward = (new_satisfied - old_satisfied) / len(self.clauses)
                    
                    # Agent gets reward for its action
                    agent_rewards[agent_idx] += reward
                    episode_reward += reward
                    
                    # Other agents get negative reward (competition)
                    for other_idx in range(self.n_agents):
                        if other_idx != agent_idx and other_idx in active_agents:
                            # Reduced negative impact to prevent severe performance degradation
                            agent_rewards[other_idx] -= reward * self.competition_factor
                    
                    # Get next state key
                    next_state_key = self._get_state_key(new_solution)
                    
                    # Update Q-value
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
            
            # Update agent scores based on performance
            for agent_idx in range(self.n_agents):
                # Scores have some persistence (discounted update)
                self.agent_scores[agent_idx] = 0.7 * self.agent_scores[agent_idx] + 0.3 * agent_rewards[agent_idx]
            
            # After each episode, update the best solution found so far
            satisfied = count_satisfied_clauses(self.clauses, solution)
            satisfaction_ratio = satisfied / len(self.clauses)
            
            if satisfied > self.best_satisfied:
                self.best_solution = solution.copy()
                self.best_satisfied = satisfied
                self.best_satisfaction_ratio = satisfaction_ratio
            
            # Track episode rewards
            episode_rewards.append(episode_reward)
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
                # Print agent scores
                print(f"Agent scores: {', '.join([f'{score:.2f}' for score in self.agent_scores])}")
            
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
            "agent_scores": self.agent_scores.tolist()
        }
        
        return self.best_solution, metrics

# Demo code
if __name__ == "__main__":
    # Demo the competitive multi-agent SAT solver
    n_vars = 20
    n_clauses = 85
    clauses = generate_sat_problem(n_vars, n_clauses)
    
    print(f"Testing Competitive Multi-agent Q-Learning on {n_vars}-variable, {n_clauses}-clause problem")
    
    # Create and run solver
    solver = MultiQLearningSATCompetitive(n_vars, clauses, n_agents=5)
    solution, metrics = solver.solve(max_episodes=100, early_stopping=True)
    
    # Report results
    print(f"\nResults:")
    print(f"Solution found: {metrics['solution_found']}")
    if metrics['solution_found']:
        print(f"Found in episode: {metrics['solution_episode']}")
    print(f"Runtime: {metrics['runtime']:.2f} seconds")
    print(f"Best satisfaction ratio: {metrics['best_satisfaction_ratio']:.2%}")
    print(f"Final agent scores: {metrics['agent_scores']}")
