import numpy as np
import time
import matplotlib.pyplot as plt
from main import SATEnv
from sat_problems import MEDIUM_PROBLEM, SMALL_PROBLEM

class ParameterizedQLearningAgent:
    def __init__(self, n_vars, learning_rate, discount_factor, exploration_rate, 
                min_exploration_rate, exploration_decay):
        self.n_vars = n_vars
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
    
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
    
    def update(self, state, action, reward, next_state):
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        
        current_q = self.get_q_value(state, action)
        
        max_next_q = 0.0
        for _ in range(10):
            next_action = np.random.randint(0, 2, self.n_vars)
            next_q = self.get_q_value(next_state, next_action)
            max_next_q = max(max_next_q, next_q)
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[(state_tuple, action_tuple)] = new_q
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)


def run_with_params(problem, learning_rate, discount_factor, exploration_decay, max_episodes=500):
    """Run a single configuration with specified parameters"""
    n_vars = problem["num_vars"]
    n_agents = 3
    
    # Create agents with the specified parameters
    agents = [ParameterizedQLearningAgent(
        n_vars=n_vars,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay=exploration_decay
    ) for _ in range(n_agents)]
    
    # Create environment
    env = SATEnv(problem=problem)
    
    # Training loop
    best_reward = 0
    best_solution = None
    solution_found = False
    episodes_to_solution = max_episodes
    
    start_time = time.time()
    
    for episode in range(max_episodes):
        episode_rewards = []
        
        for agent_idx, agent in enumerate(agents):
            state = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 10
            
            while not done and step_count < max_steps:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Keep track of best solution
                if reward > best_reward:
                    best_reward = reward
                    best_solution = action.copy()
                
                # Early stopping if we found a solution
                if reward == 1.0:
                    solution_found = True
                    if episode < episodes_to_solution:
                        episodes_to_solution = episode
                    break
            
            episode_rewards.append(total_reward)
            
            if solution_found:
                break
        
        if solution_found:
            break
        
        # Early stopping - if we've found a perfect solution
        if best_reward == 1.0:
            solution_found = True
            episodes_to_solution = episode + 1
            break
    
    runtime = time.time() - start_time
    
    return {
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "exploration_decay": exploration_decay,
        "solution_found": solution_found,
        "episodes_to_solution": episodes_to_solution if solution_found else None,
        "best_reward": best_reward,
        "runtime": runtime
    }

def run_parameter_sweep():
    """Test different learning parameters"""
    problem = MEDIUM_PROBLEM
    learning_rates = [0.01, 0.1, 0.3]
    discount_factors = [0.9, 0.95, 0.99]
    exploration_decays = [0.99, 0.995, 0.999]
    
    results = []
    
    print(f"Running parameter sweep on {problem['name']} problem")
    print(f"Testing {len(learning_rates) * len(discount_factors) * len(exploration_decays)} combinations")
    
    for lr in learning_rates:
        for df in discount_factors:
            for decay in exploration_decays:
                config = f"lr={lr}, df={df}, decay={decay}"
                print(f"Testing {config}")
                
                result = run_with_params(
                    problem=problem,
                    learning_rate=lr,
                    discount_factor=df,
                    exploration_decay=decay
                )
                
                results.append(result)
                
                # Report this result
                if result["solution_found"]:
                    print(f"  Solution found in {result['episodes_to_solution']} episodes (time: {result['runtime']:.2f}s)")
                else:
                    print(f"  No solution found. Best reward: {result['best_reward']} (time: {result['runtime']:.2f}s)")
    
    # Sort results by performance
    solved_results = [r for r in results if r["solution_found"]]
    unsolved_results = [r for r in results if not r["solution_found"]]
    
    solved_results.sort(key=lambda x: (x["episodes_to_solution"], x["runtime"]))
    unsolved_results.sort(key=lambda x: (-x["best_reward"], x["runtime"]))
    
    sorted_results = solved_results + unsolved_results
    
    # Print top 5 configurations
    print("\nTop configurations:")
    for i, result in enumerate(sorted_results[:5]):
        if result["solution_found"]:
            print(f"{i+1}. lr={result['learning_rate']}, df={result['discount_factor']}, decay={result['exploration_decay']}: "
                  f"Solution in {result['episodes_to_solution']} episodes, {result['runtime']:.2f}s")
        else:
            print(f"{i+1}. lr={result['learning_rate']}, df={result['discount_factor']}, decay={result['exploration_decay']}: "
                  f"No solution, best reward {result['best_reward']}, {result['runtime']:.2f}s")
    
    # Visualize results
    plot_parameter_results(results)
    
    return results

def plot_parameter_results(results):
    """Visualize parameter sweep results"""
    plt.figure(figsize=(15, 10))
    
    # Create a subplot for learning rates
    plt.subplot(2, 2, 1)
    lr_groups = {}
    for r in results:
        lr = r["learning_rate"]
        if lr not in lr_groups:
            lr_groups[lr] = []
        if r["solution_found"]:
            lr_groups[lr].append(r["episodes_to_solution"])
    
    lr_labels = list(lr_groups.keys())
    lr_episodes = [np.mean(v) if v else float('nan') for v in lr_groups.values()]
    lr_std = [np.std(v) if len(v) > 1 else 0 for v in lr_groups.values()]
    
    plt.bar(lr_labels, lr_episodes, yerr=lr_std)
    plt.xlabel('Learning Rate')
    plt.ylabel('Avg Episodes to Solution')
    plt.title('Learning Rate Effect')
    
    # Create a subplot for discount factors
    plt.subplot(2, 2, 2)
    df_groups = {}
    for r in results:
        df = r["discount_factor"]
        if df not in df_groups:
            df_groups[df] = []
        if r["solution_found"]:
            df_groups[df].append(r["episodes_to_solution"])
    
    df_labels = list(df_groups.keys())
    df_episodes = [np.mean(v) if v else float('nan') for v in df_groups.values()]
    df_std = [np.std(v) if len(v) > 1 else 0 for v in df_groups.values()]
    
    plt.bar(df_labels, df_episodes, yerr=df_std)
    plt.xlabel('Discount Factor')
    plt.ylabel('Avg Episodes to Solution')
    plt.title('Discount Factor Effect')
    
    # Create a subplot for exploration decay
    plt.subplot(2, 2, 3)
    decay_groups = {}
    for r in results:
        decay = r["exploration_decay"]
        if decay not in decay_groups:
            decay_groups[decay] = []
        if r["solution_found"]:
            decay_groups[decay].append(r["episodes_to_solution"])
    
    decay_labels = list(decay_groups.keys())
    decay_episodes = [np.mean(v) if v else float('nan') for v in decay_groups.values()]
    decay_std = [np.std(v) if len(v) > 1 else 0 for v in decay_groups.values()]
    
    plt.bar(decay_labels, decay_episodes, yerr=decay_std)
    plt.xlabel('Exploration Decay')
    plt.ylabel('Avg Episodes to Solution')
    plt.title('Exploration Decay Effect')
    
    # Create a scatterplot of runtime vs episodes
    plt.subplot(2, 2, 4)
    episodes = [r["episodes_to_solution"] for r in results if r["solution_found"]]
    runtimes = [r["runtime"] for r in results if r["solution_found"]]
    
    if episodes:  # Only plot if we have solved instances
        plt.scatter(episodes, runtimes)
        plt.xlabel('Episodes to Solution')
        plt.ylabel('Runtime (s)')
        plt.title('Runtime vs Episodes')
    
    plt.tight_layout()
    plt.savefig("parameter_sweep_results.png")
    plt.close()

import random  # Added missing import

if __name__ == "__main__":
    run_parameter_sweep()