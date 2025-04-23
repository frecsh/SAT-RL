#mutli agent q learnign sat solver

import numpy as np
import random
import time
from main import SATEnv

class CompetitiveQLearningAgent:
    def __init__(self, n_vars, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        self.n_vars = n_vars
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
        self.competitive_bonus = 0  # Track competitive bonuses received
    
    def get_q_value(self, state, action):
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        return self.q_table.get((state_tuple, action_tuple), 0.0)
    
    def select_action(self, state):
        if random.random() < self.exploration_rate:
            return np.random.randint(0, 2, self.n_vars)
        else:
            best_action = None
            best_value = float('-inf')
            
            for _ in range(10):
                action = np.random.randint(0, 2, self.n_vars)
                q_value = self.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action.copy()
            
            return best_action
    
    def update(self, state, action, reward, next_state, competitive_bonus=0):
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        
        # Add competitive bonus to the reward
        augmented_reward = reward + competitive_bonus
        
        current_q = self.get_q_value(state, action)
        
        max_next_q = 0.0
        for _ in range(10):
            next_action = np.random.randint(0, 2, self.n_vars)
            next_q = self.get_q_value(next_state, next_action)
            max_next_q = max(max_next_q, next_q)
        
        new_q = current_q + self.learning_rate * (augmented_reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[(state_tuple, action_tuple)] = new_q
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
    
    def q_value_variance(self):
        """Calculate variance in Q-values as a measure of convergence"""
        if not self.q_table:
            return 0
        values = list(self.q_table.values())
        return np.var(values) if values else 0

def main(problem=None):
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
    competitive_bonus = 0.2  # Bonus for best performing agent per episode
    
    # Create agents and environment
    agents = [CompetitiveQLearningAgent(n_vars=n_vars) for _ in range(n_agents)]
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
        "early_stopped": False      # Whether training was stopped early
    }
    
    start_time = time.time()
    
    # Training loop
    best_overall_reward = 0
    best_overall_solution = None
    
    # Add plateau detection
    plateau_counter = 0
    last_best_reward = 0
    
    print("Training multiple competitive Q-learning agents on SAT problem...")
    
    for episode in range(total_episodes):
        # Track the performance of each agent in this episode
        episode_performances = []
        
        for agent_idx, agent in enumerate(agents):
            state = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 10
            
            # Store actions and states for later updates with competitive bonus
            trajectory = []
            
            while not done and step_count < max_steps:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store current experience for later update
                trajectory.append((state, action, reward, next_state))
                
                state = next_state
                total_reward += reward
                step_count += 1
            
            # Record performance for competition
            episode_performances.append((agent_idx, total_reward, trajectory))
            
            # Keep track of best solution overall
            if total_reward > best_overall_reward:
                best_overall_reward = total_reward
                # Get the last action from this agent
                best_overall_solution = action.copy()
        
        # Sort performances to find the best agent
        episode_performances.sort(key=lambda x: x[1], reverse=True)
        best_agent_idx = episode_performances[0][0]
        
        # Apply updates to all agents (now with competitive bonus for the winner)
        for idx, (agent_idx, agent_reward, trajectory) in enumerate(episode_performances):
            agent = agents[agent_idx]
            
            # Best agent gets a competitive bonus
            bonus = competitive_bonus if idx == 0 else 0
            
            # Update agent's Q-values with appropriate bonus
            for state, action, reward, next_state in trajectory:
                agent.update(state, action, reward, next_state, competitive_bonus=bonus)
            
            # Track competitive bonuses for reporting
            if idx == 0:
                agent.competitive_bonus += competitive_bonus
        
        # Update metrics after each episode
        metrics["episode_rewards"].append(best_overall_reward)
        metrics["best_reward_progress"].append(best_overall_reward)
        
        if episode % 100 == 0:
            # Track Q-table sizes and exploration rates
            avg_q_size = sum(len(agent.q_table) for agent in agents) / len(agents)
            avg_explore = sum(agent.exploration_rate for agent in agents) / len(agents)
            metrics["q_table_sizes"].append(avg_q_size)
            metrics["exploration_rates"].append(avg_explore)
        
        if episode % 10 == 0:
            avg_variance = sum(agent.q_value_variance() for agent in agents) / len(agents)
            metrics["q_value_variance"].append(avg_variance)
        
        # Early stopping if performance plateaus
        if best_overall_reward == last_best_reward:
            plateau_counter += 1
        else:
            plateau_counter = 0
            last_best_reward = best_overall_reward
        
        if plateau_counter >= 5 and episode > 100:
            print(f"Early stopping at episode {episode}: No improvement for 5 consecutive checks")
            metrics["early_stopped"] = True
            metrics["solution_found"] = best_overall_reward == 1.0
            metrics["solution_episode"] = episode + 1
            metrics["runtime"] = time.time() - start_time
            return metrics
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{total_episodes}, Best reward so far: {best_overall_reward}")
            print(f"Best agent in this batch: Agent {best_agent_idx} (total competitive bonus: {agents[best_agent_idx].competitive_bonus})")
            if best_overall_reward == 1.0:
                metrics["solution_found"] = True
                metrics["solution_episode"] = episode + 1
                metrics["runtime"] = time.time() - start_time
                assignment = {i + 1: bool(bit) for i, bit in enumerate(best_overall_solution)}
                print("Solution found!")
                print(f"Solution: {assignment}")
                return metrics  # Return early with metrics
    
    # If we get here, no solution was found
    metrics["runtime"] = time.time() - start_time
    return metrics

# Include the SATEnvironment class from the previous code
class SATEnvironment:
    def __init__(self, clauses, n_vars):
        self.clauses = clauses
        self.n_vars = n_vars
        self.state = np.zeros(n_vars, dtype=np.int32)
    
    def reset(self):
        self.state = np.zeros(self.n_vars, dtype=np.int32)
        return self.state
    
    def step(self, action):
        self.state = action.copy()
        assignment = {i + 1: bool(bit) for i, bit in enumerate(action)}
        satisfied_clauses = self._count_satisfied_clauses(assignment)
        total_clauses = len(self.clauses)
        reward = satisfied_clauses / total_clauses
        
        if satisfied_clauses == total_clauses:
            reward = 1.0
            done = True
        else:
            done = False
            
        return self.state, reward, done
    
    def _count_satisfied_clauses(self, assignment):
        satisfied = 0
        for clause in self.clauses:
            if any((assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]) for lit in clause):
                satisfied += 1
        return satisfied

if __name__ == "__main__":
    main()
