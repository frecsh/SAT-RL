import numpy as np
import random
import time
from main import SATEnv

class CommunicatingQLearningAgent:
    def __init__(self, n_vars, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, 
                 min_exploration_rate=0.01, exploration_decay=0.995, communication_threshold=0.5):
        self.n_vars = n_vars
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
        self.shared_experiences = []
        self.communication_benefit = 0  # Track benefits from communication
        self.communication_history = []  # Track what was communicated
        self.communication_stats = {
            'variables_shared': {},  # Track which variables are shared most
            'rewards_shared': [],    # Track rewards of shared experiences
            'clauses_satisfied': {}  # Track which clauses are satisfied by shared experiences
        }
        self.communication_threshold = communication_threshold  # Configurable threshold
    
    def get_q_value(self, state, action):
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        return self.q_table.get((state_tuple, action_tuple), 0.0)
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            # Explore: choose a random action
            return np.random.randint(0, 2, self.n_vars)
        else:
            # Exploit: choose the best action based on Q-values
            best_action = None
            best_value = float('-inf')
            
            # Try several random actions and pick the best one
            # (full enumeration would be 2^n_vars which is too large)
            for _ in range(10):  # Sample 10 random actions
                action = np.random.randint(0, 2, self.n_vars)
                q_value = self.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action.copy()
            
            return best_action
    
    def update(self, state, action, reward, next_state):
        # Convert to tuples for dictionary keys
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Find maximum Q-value for next state
        max_next_q = 0.0
        for _ in range(10):  # Sample 10 random actions for next state
            next_action = np.random.randint(0, 2, self.n_vars)
            next_q = self.get_q_value(next_state, next_action)
            max_next_q = max(max_next_q, next_q)
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[(state_tuple, action_tuple)] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
        
        # Share experience if it's good
        if reward > self.communication_threshold:
            self.shared_experiences.append((state, action, reward))
    
    def share_experience(self, state, action, reward, clause_info=None):
        """Share a valuable experience with other agents"""
        # Only share experiences with good rewards
        if reward > self.communication_threshold:
            experience = (state, action, reward)
            self.shared_experiences.append(experience)
            
            # Track what's being communicated for visualization
            self.communication_history.append({
                'state': state.copy(),
                'action': action.copy(),
                'reward': reward,
                'clauses_satisfied': clause_info['satisfied_clauses'] if clause_info else None
            })
            
            # Update communication statistics
            self.update_communication_stats(state, action, reward, clause_info)
    
    def update_communication_stats(self, state, action, reward, clause_info=None):
        """Update statistics about what's being communicated"""
        # Track which variables are being shared most often
        for i, bit in enumerate(action):
            var_key = f"{i+1}={bit}"
            if var_key not in self.communication_stats['variables_shared']:
                self.communication_stats['variables_shared'][var_key] = 0
            self.communication_stats['variables_shared'][var_key] += 1
        
        # Track rewards of shared experiences
        self.communication_stats['rewards_shared'].append(reward)
        
        # Track which clauses are being satisfied
        if clause_info and 'satisfied_clause_indices' in clause_info:
            for clause_idx in clause_info['satisfied_clause_indices']:
                if clause_idx not in self.communication_stats['clauses_satisfied']:
                    self.communication_stats['clauses_satisfied'][clause_idx] = 0
                self.communication_stats['clauses_satisfied'][clause_idx] += 1
    
    def learn_from_others(self, agents):
        """Learn from other agents' shared experiences"""
        before_q_sum = sum(self.q_table.values()) if self.q_table else 0
        
        for agent in agents:
            if agent is not self:
                for state, action, reward in agent.shared_experiences:
                    # Learn from other agent's experiences (simplified update)
                    state_tuple = tuple(state)
                    action_tuple = tuple(action)
                    current_q = self.get_q_value(state, action)
                    
                    # Use a simpler update rule for shared experiences
                    new_q = current_q + self.learning_rate * (reward - current_q)
                    self.q_table[(state_tuple, action_tuple)] = new_q
        
        # Calculate benefit from communication
        after_q_sum = sum(self.q_table.values()) if self.q_table else 0
        self.communication_benefit += (after_q_sum - before_q_sum)
        
        # Clear after learning
        self.shared_experiences = []
    
    def q_value_variance(self):
        """Calculate variance in Q-values as a measure of convergence"""
        if not self.q_table:
            return 0
        values = list(self.q_table.values())
        return np.var(values) if values else 0

def main(problem=None, communication_threshold=0.5):
    # Define a SAT problem
    if problem is None:
        # Default harder SAT formula from previous implementation
        harder_formula = [
            [1, -3, 5],      # x1 OR NOT x3 OR x5
            [-1, 2, 4],      # NOT x1 OR x2 OR x4
            [2, -4, -5],     # x2 OR NOT x4 OR NOT x5
            [-2, 3, 5],      # NOT x2 OR x3 OR x5
            [1, -2, -3],     # x1 OR NOT x2 OR NOT x3
            [-1, 3, -5]      # NOT x1 OR x3 OR NOT x5
        ]
        problem = {
            "name": "default_harder",
            "clauses": harder_formula,
            "num_vars": 5
        }
    
    n_vars = problem["num_vars"]
    n_agents = 3
    total_episodes = 1000
    
    # Create agents and environment with configurable communication threshold
    agents = [CommunicatingQLearningAgent(n_vars=n_vars, communication_threshold=communication_threshold) for _ in range(n_agents)]
    env = SATEnv(problem=problem)
    
    # Add metrics tracking
    metrics = {
        "episode_rewards": [],      # Track reward per episode
        "best_reward_progress": [], # Track how best reward improves
        "solution_found": False,    # Whether a solution was found
        "solution_episode": None,   # When solution was found
        "q_table_sizes": [],        # Track Q-table growth
        "exploration_rates": [],    # Track exploration decay
        "q_value_variance": [],     # Track Q-value variance
        "communication_benefits": [],# Track benefits from communication
        "early_stopped": False,     # Whether training was stopped early
        "communication_history": [],# Track all communications
        "communication_threshold": communication_threshold  # Store the threshold used
    }
    
    start_time = time.time()
    
    # Training loop
    best_reward = 0
    best_solution = None
    
    # Add plateau detection
    plateau_counter = 0
    last_best_reward = 0
    
    print("Training communicating Q-learning agents on SAT problem...")
    
    for episode in range(total_episodes):
        episode_rewards = []
        
        for agent_idx, agent in enumerate(agents):
            state = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 10  # Limit steps per episode
            
            while not done and step_count < max_steps:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Standard Q-learning update
                agent.update(state, action, reward, next_state)
                
                # Get detailed clause satisfaction information
                assignment = {i + 1: bool(bit) for i, bit in enumerate(action)}
                satisfied_clause_indices = []
                clauses = problem["clauses"]
                
                for clause_idx, clause in enumerate(clauses):
                    # Check if this clause is satisfied
                    if any((assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]) for lit in clause):
                        satisfied_clause_indices.append(clause_idx)
                
                clause_info = {
                    'satisfied_clauses': len(satisfied_clause_indices),
                    'satisfied_clause_indices': satisfied_clause_indices,
                    'total_clauses': len(clauses)
                }
                
                # Share experience with clause information
                if reward > agent.communication_threshold:
                    agent.share_experience(state, action, reward, clause_info)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Keep track of best solution
                if reward > best_reward:
                    best_reward = reward
                    best_solution = action.copy()
                
                # Early stopping if we found a solution
                if done and reward == 1.0:
                    break
            
            episode_rewards.append(total_reward)
        
        # Communication phase - agents share experiences
        if episode % 5 == 0:  # Every 5 episodes to reduce overhead
            for agent in agents:
                agent.learn_from_others(agents)
                
            # Record communication stats for visualization
            if episode % 100 == 0:
                comm_snapshot = {}
                for i, agent in enumerate(agents):
                    comm_snapshot[f"agent_{i}"] = {
                        "vars_shared": agent.communication_stats['variables_shared'].copy(),
                        "rewards": agent.communication_stats['rewards_shared'].copy(),
                        "clauses": agent.communication_stats['clauses_satisfied'].copy()
                    }
                metrics["communication_history"].append({
                    "episode": episode,
                    "data": comm_snapshot
                })
        
        # Update metrics after each episode
        metrics["episode_rewards"].append(max(episode_rewards))
        metrics["best_reward_progress"].append(best_reward)
        
        if episode % 100 == 0:
            # Track Q-table sizes and exploration rates
            avg_q_size = sum(len(agent.q_table) for agent in agents) / len(agents)
            avg_explore = sum(agent.exploration_rate for agent in agents) / len(agents)
            avg_comm_benefit = sum(agent.communication_benefit for agent in agents) / len(agents)
            
            metrics["q_table_sizes"].append(avg_q_size)
            metrics["exploration_rates"].append(avg_explore)
            metrics["communication_benefits"].append(avg_comm_benefit)
        
        if episode % 10 == 0:
            avg_variance = sum(agent.q_value_variance() for agent in agents) / len(agents)
            metrics["q_value_variance"].append(avg_variance)
        
        # Early stopping if performance plateaus
        if best_reward == last_best_reward:
            plateau_counter += 1
        else:
            plateau_counter = 0
            last_best_reward = best_reward
        
        if plateau_counter >= 5 and episode > 100:
            print(f"Early stopping at episode {episode}: No improvement for 5 consecutive checks")
            metrics["early_stopped"] = True
            metrics["solution_found"] = best_reward == 1.0
            metrics["solution_episode"] = episode + 1
            metrics["runtime"] = time.time() - start_time
            
            # Save agent communication history for visualization
            metrics["agent_communications"] = [agent.communication_history for agent in agents]
            return metrics
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{total_episodes}, Best reward so far: {best_reward}")
            avg_comm_benefit = sum(agent.communication_benefit for agent in agents) / len(agents)
            print(f"Average communication benefit: {avg_comm_benefit:.4f}")
            if best_reward == 1.0:
                print("Solution found!")
                assignment = {i + 1: bool(bit) for i, bit in enumerate(best_solution)}
                print(f"Solution: {assignment}")
                
                metrics["solution_found"] = True
                metrics["solution_episode"] = episode + 1
                metrics["runtime"] = time.time() - start_time
                
                # Save agent communication history for visualization
                metrics["agent_communications"] = [agent.communication_history for agent in agents]
                return metrics  # Return early with metrics
    
    # If we get here, no solution was found
    metrics["runtime"] = time.time() - start_time
    # Save agent communication history for visualization
    metrics["agent_communications"] = [agent.communication_history for agent in agents]
    return metrics

if __name__ == "__main__":
    main()