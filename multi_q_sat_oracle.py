import numpy as np
import random
import time
from main import SATEnv
from sat_oracle import SATOracle

class OracleGuidedQLearningAgent:
    """
    Q-learning agent that receives guidance from a SAT solver oracle,
    focusing on problematic clauses identified by the oracle.
    """
    def __init__(self, n_vars, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, 
                 min_exploration_rate=0.01, exploration_decay=0.995, oracle_weight=0.3):
        self.n_vars = n_vars
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
        self.oracle_weight = oracle_weight  # Weight for oracle feedback in reward calculation
        self.trajectory = []  # Store recent state-action-reward
        self.oracle_suggestions = {}  # Store recent oracle suggestions
        self.critiqued_clauses = set()  # Track clauses that have been critiqued
        
    def get_q_value(self, state, action):
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        return self.q_table.get((state_tuple, action_tuple), 0.0)
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            # Random exploration, but with oracle bias if available
            if self.oracle_suggestions and random.random() < self.oracle_weight:
                # Create an action that follows oracle suggestions for some variables
                action = np.random.randint(0, 2, self.n_vars)
                for var, value in self.oracle_suggestions.items():
                    if var <= self.n_vars:
                        action[var-1] = 1 if value else 0
                return action
            else:
                # Pure random action
                return np.random.randint(0, 2, self.n_vars)
        else:
            # Exploit: choose best action based on Q-values
            best_action = None
            best_value = float('-inf')
            
            for _ in range(10):  # Sample 10 random actions
                action = np.random.randint(0, 2, self.n_vars)
                # Bias toward oracle suggestions
                if self.oracle_suggestions and random.random() < self.oracle_weight:
                    for var, value in self.oracle_suggestions.items():
                        if var <= self.n_vars:
                            action[var-1] = 1 if value else 0
                
                q_value = self.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action.copy()
            
            return best_action
    
    def update(self, state, action, reward, next_state, oracle_critique=None):
        # Store in trajectory for later oracle critique
        self.trajectory.append((state, action, reward))
        
        # Apply oracle critique if available
        adjusted_reward = reward
        if oracle_critique:
            # Extract oracle suggestions
            self.oracle_suggestions = oracle_critique.get("suggestions", {})
            
            # Track critiqued clauses
            unsatisfied = oracle_critique.get("unsatisfied_clauses", [])
            self.critiqued_clauses.update(unsatisfied)
            
            # Adjust reward based on oracle critique - penalize for difficult clauses
            difficulty_ranking = oracle_critique.get("difficulty_ranking", [])
            if difficulty_ranking:
                # Penalize based on clause difficulty
                difficulty_penalty = sum(score for _, score in difficulty_ranking[:3]) / 3
                adjusted_reward = reward * (1 - self.oracle_weight) + \
                                 (1 - difficulty_penalty) * self.oracle_weight
        
        # Standard Q-learning update with adjusted reward
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        
        current_q = self.get_q_value(state, action)
        
        # Find maximum Q-value for next state
        max_next_q = 0.0
        for _ in range(10):  # Sample 10 random actions for next state
            next_action = np.random.randint(0, 2, self.n_vars)
            next_q = self.get_q_value(next_state, next_action)
            max_next_q = max(max_next_q, next_q)
        
        # Q-learning update rule with adjusted reward
        new_q = current_q + self.learning_rate * (adjusted_reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[(state_tuple, action_tuple)] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
        
    def get_trajectory(self):
        """Return the recent trajectory for oracle critique"""
        return self.trajectory.copy()
    
    def clear_trajectory(self):
        """Clear the stored trajectory after oracle critique"""
        self.trajectory = []
    
    def q_value_variance(self):
        """Calculate variance in Q-values as a measure of convergence"""
        if not self.q_table:
            return 0
        values = list(self.q_table.values())
        return np.var(values) if values else 0

def main(problem=None, oracle_weight=0.3):
    """Run oracle-guided Q-learning on a SAT problem"""
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
    oracle_critique_frequency = 10  # How often to get oracle feedback
    
    # Create agents, environment, and oracle
    agents = [OracleGuidedQLearningAgent(n_vars=n_vars, oracle_weight=oracle_weight) for _ in range(n_agents)]
    env = SATEnv(problem=problem)
    oracle = SATOracle(problem)
    
    # Add metrics tracking
    metrics = {
        "episode_rewards": [],      # Track reward per episode
        "best_reward_progress": [], # Track how best reward improves
        "solution_found": False,    # Whether a solution was found
        "solution_episode": None,   # When solution was found
        "q_table_sizes": [],        # Track Q-table growth
        "exploration_rates": [],    # Track exploration decay
        "q_value_variance": [],     # Track Q-value variance
        "oracle_critiques": [],     # Track oracle critiques
        "difficult_clauses": [],    # Track difficult clauses over time
        "early_stopped": False      # Whether training was stopped early
    }
    
    start_time = time.time()
    
    # Training loop
    best_reward = 0
    best_solution = None
    
    # Add plateau detection
    plateau_counter = 0
    last_best_reward = 0
    
    print(f"Training oracle-guided Q-learning agents on {problem['name']} (oracle weight: {oracle_weight})...")
    
    for episode in range(total_episodes):
        episode_rewards = []
        episode_critiques = []
        
        for agent_idx, agent in enumerate(agents):
            state = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 10  # Limit steps per episode
            
            # Clear trajectory for this episode
            agent.clear_trajectory()
            
            while not done and step_count < max_steps:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Only request oracle critique periodically to reduce overhead
                oracle_critique = None
                if episode % oracle_critique_frequency == 0 and step_count == max_steps - 1:
                    oracle_critique = oracle.critique([(state, action, reward)])
                    episode_critiques.append(oracle_critique)
                
                # Update agent with oracle feedback
                agent.update(state, action, reward, next_state, oracle_critique)
                
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
        
        # Update metrics after each episode
        metrics["episode_rewards"].append(max(episode_rewards))
        metrics["best_reward_progress"].append(best_reward)
        
        # Periodically update clause difficulty metrics
        if episode % oracle_critique_frequency == 0:
            metrics["difficult_clauses"].append({
                "episode": episode,
                "difficulties": oracle.clause_difficulty.tolist()
            })
            metrics["oracle_critiques"].append({
                "episode": episode,
                "critiques": episode_critiques
            })
            
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
            return metrics
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{total_episodes}, Best reward so far: {best_reward}")
            if episode_critiques:
                difficult_clauses = episode_critiques[0].get("difficulty_ranking", [])
                print(f"Most difficult clauses: {difficult_clauses}")
                
            if best_reward == 1.0:
                print("Solution found!")
                assignment = {i + 1: bool(bit) for i, bit in enumerate(best_solution)}
                print(f"Solution: {assignment}")
                
                metrics["solution_found"] = True
                metrics["solution_episode"] = episode + 1
                metrics["runtime"] = time.time() - start_time
                return metrics  # Return early with metrics
    
    # If we get here, no solution was found
    metrics["runtime"] = time.time() - start_time
    return metrics

def visualize_oracle_guidance(metrics_file=None):
    """Visualize how oracle guidance affected learning"""
    import matplotlib.pyplot as plt
    import pickle
    
    # Load metrics if provided, otherwise use the last run
    metrics = None
    if metrics_file:
        with open(metrics_file, 'rb') as f:
            metrics = pickle.load(f)
    else:
        # Find the most recent metrics file
        import glob
        import os
        files = glob.glob("oracle_metrics_*.pkl")
        if files:
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, 'rb') as f:
                metrics = pickle.load(f)
    
    if not metrics:
        print("No metrics found to visualize")
        return
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot clause difficulty over time
    if "difficult_clauses" in metrics:
        plt.subplot(2, 2, 1)
        clause_data = metrics["difficult_clauses"]
        episodes = [d["episode"] for d in clause_data]
        
        # Get number of clauses from first data point
        n_clauses = len(clause_data[0]["difficulties"])
        
        # Plot each clause's difficulty over time
        for clause_idx in range(n_clauses):
            difficulties = [d["difficulties"][clause_idx] for d in clause_data]
            plt.plot(episodes, difficulties, marker='o', label=f"Clause {clause_idx}")
            
        plt.xlabel("Episode")
        plt.ylabel("Difficulty Score")
        plt.title("Clause Difficulty Over Time")
        plt.legend()
    
    # Plot learning curve
    plt.subplot(2, 2, 2)
    plt.plot(metrics["best_reward_progress"])
    plt.xlabel("Episode")
    plt.ylabel("Best Reward")
    plt.title("Learning Curve with Oracle Guidance")
    
    # Create heatmap of visited clauses
    if "oracle_critiques" in metrics and metrics["oracle_critiques"]:
        plt.subplot(2, 2, 3)
        critiques = metrics["oracle_critiques"]
        
        # Count how often each clause is satisfied/unsatisfied
        clause_satisfaction = {}
        for critique_data in critiques:
            episode = critique_data["episode"]
            for critique in critique_data["critiques"]:
                if "satisfied_clauses" in critique:
                    for clause_idx in critique["satisfied_clauses"]:
                        key = (episode, clause_idx)
                        clause_satisfaction[key] = clause_satisfaction.get(key, 0) + 1
        
        # Create heatmap data
        if clause_satisfaction:
            # Find max episode and clause index
            max_episode = max(episode for episode, _ in clause_satisfaction.keys())
            max_clause = max(clause_idx for _, clause_idx in clause_satisfaction.keys())
            
            # Create bins for heatmap (every 50 episodes)
            episode_bins = range(0, max_episode+50, 50)
            heatmap_data = np.zeros((len(episode_bins)-1, max_clause+1))
            
            for (episode, clause_idx), count in clause_satisfaction.items():
                bin_idx = episode // 50
                if bin_idx < len(episode_bins)-1:
                    heatmap_data[bin_idx, clause_idx] = count
            
            # Plot heatmap
            plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
            plt.colorbar(label='Satisfaction Count')
            plt.xlabel("Clause Index")
            plt.ylabel("Episode Bin")
            plt.title("Clause Satisfaction Over Time")
            plt.yticks(range(len(episode_bins)-1), [f"{episode_bins[i]}-{episode_bins[i+1]}" for i in range(len(episode_bins)-1)])
    
    plt.tight_layout()
    plt.savefig("oracle_guidance.png")
    print("Saved visualization to oracle_guidance.png")
    plt.close()

if __name__ == "__main__":
    from sat_problems import HARD_PROBLEM
    
    # Run with more oracle weights to find the optimal threshold
    oracle_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for weight in oracle_weights:
        print(f"\nRunning with oracle weight: {weight}")
        metrics = main(HARD_PROBLEM, oracle_weight=weight)
        
        # Save metrics
        import pickle
        with open(f"oracle_metrics_{HARD_PROBLEM['name']}_{weight}.pkl", "wb") as f:
            pickle.dump(metrics, f)
    
    # Visualize results
    visualize_oracle_guidance()