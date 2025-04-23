import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os
from sat_problems import MEDIUM_PROBLEM

def load_metrics(filename):
    """Load metrics from a pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def compare_oracle_weights(problem_name=None):
    """
    Compare results across different oracle weights
    
    Args:
        problem_name: Name of the problem to analyze results for
    """
    if problem_name is None:
        problem_name = MEDIUM_PROBLEM["name"]
    
    # Find all relevant metric files
    files = glob.glob(f"oracle_metrics_{problem_name}_*.pkl")
    
    if not files:
        print(f"No metric files found for problem: {problem_name}")
        return
    
    # Extract weights and load metrics
    results = {}
    for file in files:
        # Extract weight from filename
        weight_str = file.split('_')[-1].split('.')[0]
        try:
            weight = float(weight_str)
            metrics = load_metrics(file)
            results[weight] = metrics
        except (ValueError, IndexError) as e:
            print(f"Error processing file {file}: {e}")
    
    if not results:
        print("No valid results found")
        return
        
    # Sort weights for consistent plotting
    weights = sorted(results.keys())
    print(f"Found results for oracle weights: {weights}")
    
    # Create visualizations comparing performance across weights
    visualize_weight_comparison(results, weights, problem_name)
    
    return results

def visualize_weight_comparison(results, weights, problem_name):
    """Create visualizations comparing different oracle weights"""
    # --- 1. Success rate comparison ---
    success_rates = []
    for weight in weights:
        metrics = results[weight]
        success_rates.append(1 if metrics["solution_found"] else 0)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.bar([str(w) for w in weights], success_rates)
    plt.xlabel('Oracle Weight')
    plt.ylabel('Solution Found (1/0)')
    plt.title(f'Solution Success by Oracle Weight ({problem_name})')
    plt.ylim(0, 1.1)
    
    # --- 2. Episodes to solution comparison ---
    episodes = []
    for weight in weights:
        metrics = results[weight]
        if metrics["solution_found"]:
            episodes.append(metrics["solution_episode"])
        else:
            episodes.append(None)  # No solution found
    
    plt.subplot(2, 2, 2)
    # Filter out None values for plotting
    valid_weights = [w for w, e in zip(weights, episodes) if e is not None]
    valid_episodes = [e for e in episodes if e is not None]
    
    if valid_episodes:  # If we have any solutions to plot
        plt.bar([str(w) for w in valid_weights], valid_episodes)
        plt.xlabel('Oracle Weight')
        plt.ylabel('Episodes to Solution')
        plt.title(f'Episodes to Solution by Oracle Weight ({problem_name})')
    else:
        plt.text(0.5, 0.5, "No solutions found", 
                 horizontalalignment='center', verticalalignment='center')
    
    # --- 3. Runtime comparison ---
    runtimes = [results[weight]["runtime"] for weight in weights]
    
    plt.subplot(2, 2, 3)
    plt.bar([str(w) for w in weights], runtimes)
    plt.xlabel('Oracle Weight')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'Runtime by Oracle Weight ({problem_name})')
    
    # --- 4. Best reward achieved ---
    rewards = []
    for weight in weights:
        metrics = results[weight]
        if "best_reward_progress" in metrics and metrics["best_reward_progress"]:
            rewards.append(max(metrics["best_reward_progress"]))
        else:
            rewards.append(0)
    
    plt.subplot(2, 2, 4)
    plt.bar([str(w) for w in weights], rewards)
    plt.xlabel('Oracle Weight')
    plt.ylabel('Best Reward')
    plt.title(f'Best Reward Achieved by Oracle Weight ({problem_name})')
    
    plt.tight_layout()
    plt.savefig(f"oracle_weight_comparison_{problem_name}.png")
    print(f"Saved comparison to oracle_weight_comparison_{problem_name}.png")
    
def visualize_clause_difficulty_comparison(results, weights, problem_name):
    """Compare clause difficulty patterns across oracle weights"""
    plt.figure(figsize=(15, 10))
    
    # Get number of clauses from the first result
    first_metrics = results[weights[0]]
    if "difficult_clauses" not in first_metrics or not first_metrics["difficult_clauses"]:
        print("No clause difficulty data found")
        return
        
    n_clauses = len(first_metrics["difficult_clauses"][0]["difficulties"])
    
    # Create subplot for each clause
    for clause_idx in range(min(n_clauses, 9)):  # Show at most 9 clauses
        plt.subplot(3, 3, clause_idx + 1)
        
        for weight in weights:
            metrics = results[weight]
            if "difficult_clauses" in metrics and metrics["difficult_clauses"]:
                # Extract difficulty trajectory for this clause
                clause_data = metrics["difficult_clauses"]
                episodes = [d["episode"] for d in clause_data]
                difficulties = [d["difficulties"][clause_idx] for d in clause_data]
                
                # Plot difficulty over time for this weight
                plt.plot(episodes, difficulties, marker='o', label=f"Weight {weight}")
        
        plt.title(f"Clause {clause_idx}")
        plt.xlabel("Episode")
        plt.ylabel("Difficulty")
    
    # Add a legend in the last subplot
    plt.subplot(3, 3, 9)
    plt.legend()
    plt.title("Legend")
    
    plt.tight_layout()
    plt.savefig(f"oracle_clause_difficulty_{problem_name}.png")
    print(f"Saved clause difficulty comparison to oracle_clause_difficulty_{problem_name}.png")

def compare_learning_curves(results, weights, problem_name):
    """Compare learning curves across oracle weights"""
    plt.figure(figsize=(12, 8))
    
    for weight in weights:
        metrics = results[weight]
        if "best_reward_progress" in metrics:
            reward_progress = metrics["best_reward_progress"]
            plt.plot(reward_progress, label=f"Weight {weight}")
    
    plt.xlabel("Episode")
    plt.ylabel("Best Reward")
    plt.title(f"Learning Curves by Oracle Weight ({problem_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"oracle_learning_curves_{problem_name}.png")
    print(f"Saved learning curves to oracle_learning_curves_{problem_name}.png")

if __name__ == "__main__":
    results = compare_oracle_weights()
    
    if results:
        weights = sorted(results.keys())
        problem_name = MEDIUM_PROBLEM["name"]
        
        # Create additional visualizations
        visualize_clause_difficulty_comparison(results, weights, problem_name)
        compare_learning_curves(results, weights, problem_name)