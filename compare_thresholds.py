import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from multi_q_sat_comm import main as run_communicating_agents
from sat_problems import MEDIUM_PROBLEM, PHASE_PROBLEM

def compare_communication_thresholds(problem=None, thresholds=None, runs_per_threshold=3):
    """
    Run experiments with different communication thresholds and compare results
    
    Args:
        problem: The SAT problem to solve
        thresholds: List of communication thresholds to test
        runs_per_threshold: Number of runs for each threshold to average results
    """
    if problem is None:
        problem = MEDIUM_PROBLEM
    
    if thresholds is None:
        thresholds = [0.0, 0.25, 0.5, 0.75, 0.9]
    
    problem_name = problem["name"]
    results = {}
    
    print(f"Testing {len(thresholds)} communication thresholds on {problem_name}")
    print(f"Running {runs_per_threshold} trials per threshold")
    
    for threshold in thresholds:
        print(f"\n--- Testing communication threshold: {threshold} ---")
        threshold_results = []
        
        for run in range(runs_per_threshold):
            print(f"  Run {run+1}/{runs_per_threshold}")
            start_time = time.time()
            metrics = run_communicating_agents(problem, communication_threshold=threshold)
            runtime = time.time() - start_time
            metrics["runtime"] = runtime
            threshold_results.append(metrics)
            
            # Report this run's results
            if metrics["solution_found"]:
                print(f"  Solution found in {metrics['solution_episode']} episodes (time: {runtime:.2f}s)")
            else:
                print(f"  No solution found (time: {runtime:.2f}s)")
        
        # Store results for this threshold
        results[threshold] = threshold_results
        
        # Save intermediate results in case of crash
        with open(f"threshold_results_{problem_name}.pkl", "wb") as f:
            pickle.dump(results, f)
    
    # Analyze and visualize results
    visualize_threshold_comparison(results, problem_name)
    
    return results

def visualize_threshold_comparison(results, problem_name):
    """Create visualizations comparing different communication thresholds"""
    thresholds = sorted(results.keys())
    
    # --- 1. Success rate comparison ---
    success_rates = []
    for threshold in thresholds:
        threshold_results = results[threshold]
        success_rate = sum(1 for m in threshold_results if m["solution_found"]) / len(threshold_results)
        success_rates.append(success_rate)
    
    plt.figure(figsize=(10, 6))
    plt.bar([str(t) for t in thresholds], success_rates)
    plt.xlabel('Communication Threshold')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate by Communication Threshold ({problem_name})')
    plt.ylim(0, 1.1)
    for i, rate in enumerate(success_rates):
        plt.text(i, rate + 0.05, f'{rate*100:.0f}%', ha='center')
    plt.tight_layout()
    plt.savefig(f"threshold_success_{problem_name}.png")
    plt.close()
    
    # --- 2. Episodes to solution comparison ---
    plt.figure(figsize=(10, 6))
    all_episodes = []
    labels = []
    
    for threshold in thresholds:
        threshold_results = results[threshold]
        episodes = [m["solution_episode"] for m in threshold_results if m["solution_found"]]
        if episodes:  # If any solutions were found
            all_episodes.append(episodes)
            labels.append(str(threshold))
    
    if all_episodes:  # If we have any solutions to plot
        plt.boxplot(all_episodes, labels=labels)
        plt.xlabel('Communication Threshold')
        plt.ylabel('Episodes to Solution')
        plt.title(f'Episodes to Solution by Threshold ({problem_name})')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"threshold_episodes_{problem_name}.png")
    plt.close()
    
    # --- 3. Communication volume comparison ---
    plt.figure(figsize=(10, 6))
    comm_volumes = []
    comm_stds = []
    
    for threshold in thresholds:
        threshold_results = results[threshold]
        volumes = []
        
        for metrics in threshold_results:
            if "agent_communications" in metrics:
                total_comms = sum(len(agent_comm) for agent_comm in metrics["agent_communications"])
                volumes.append(total_comms)
        
        if volumes:
            avg_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            comm_volumes.append(avg_volume)
            comm_stds.append(std_volume)
        else:
            comm_volumes.append(0)
            comm_stds.append(0)
    
    plt.bar([str(t) for t in thresholds], comm_volumes, yerr=comm_stds)
    plt.xlabel('Communication Threshold')
    plt.ylabel('Communication Volume')
    plt.title(f'Communication Volume by Threshold ({problem_name})')
    plt.tight_layout()
    plt.savefig(f"threshold_volume_{problem_name}.png")
    plt.close()
    
    # --- 4. Runtime comparison ---
    runtimes = []
    runtime_stds = []
    
    for threshold in thresholds:
        threshold_results = results[threshold]
        threshold_runtimes = [m["runtime"] for m in threshold_results]
        runtimes.append(np.mean(threshold_runtimes))
        runtime_stds.append(np.std(threshold_runtimes))
    
    plt.figure(figsize=(10, 6))
    plt.bar([str(t) for t in thresholds], runtimes, yerr=runtime_stds)
    plt.xlabel('Communication Threshold')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'Average Runtime by Threshold ({problem_name})')
    plt.tight_layout()
    plt.savefig(f"threshold_runtime_{problem_name}.png")
    plt.close()

def run_phase_transition_test():
    """Run a specific test on the phase transition problem"""
    # Phase transition problems have clause-to-variable ratio around 4.2-4.3
    # These are known to be especially challenging for SAT solvers
    print("Testing communication thresholds on phase transition problem...")
    thresholds = [0.3, 0.5, 0.7]  # Use fewer thresholds to save time
    compare_communication_thresholds(PHASE_PROBLEM, thresholds, runs_per_threshold=2)

if __name__ == "__main__":
    # Decide which problem to test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "phase":
        run_phase_transition_test()
    else:
        # Use default medium problem with a range of thresholds
        thresholds = [0.0, 0.25, 0.5, 0.75, 0.9]
        compare_communication_thresholds(MEDIUM_PROBLEM, thresholds, runs_per_threshold=3)