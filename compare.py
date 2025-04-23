import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multi_q_sat import main as cooperative_main
from multi_q_sat_comp import main as competitive_main
from multi_q_sat_comm import main as communicative_main
from sat_problems import PROBLEM_COLLECTION

def run_comparison(problems=None, n_runs=3, include_communicative=True):
    if problems is None:
        from sat_problems import SMALL_PROBLEM, MEDIUM_PROBLEM
        problems = [SMALL_PROBLEM, MEDIUM_PROBLEM]
    
    all_results = {}
    
    for problem in problems:
        print(f"\n{'='*50}")
        print(f"= Testing problem: {problem['name']} ({problem['num_vars']} variables, {len(problem['clauses'])} clauses)")
        print(f"{'='*50}")
        
        coop_metrics = []
        comp_metrics = []
        
        print("\nRunning cooperative agents...")
        for i in range(n_runs):
            start_time = time.time()
            metrics = cooperative_main(problem)
            runtime = time.time() - start_time
            metrics["runtime"] = runtime
            coop_metrics.append(metrics)
            
            # Report this run's results
            if metrics["solution_found"]:
                print(f"Run {i+1}: Solution found in {metrics['solution_episode']} episodes (time: {runtime:.2f}s)")
            else:
                print(f"Run {i+1}: No solution found (time: {runtime:.2f}s)")
        
        print("\nRunning competitive agents...")
        for i in range(n_runs):
            start_time = time.time()
            metrics = competitive_main(problem)
            runtime = time.time() - start_time
            metrics["runtime"] = runtime
            comp_metrics.append(metrics)
            
            # Report this run's results
            if metrics["solution_found"]:
                print(f"Run {i+1}: Solution found in {metrics['solution_episode']} episodes (time: {runtime:.2f}s)")
            else:
                print(f"Run {i+1}: No solution found (time: {runtime:.2f}s)")
        
        # Add communicative agents if requested
        if include_communicative:
            comm_metrics = []
            
            print("\nRunning communicative agents...")
            for i in range(n_runs):
                start_time = time.time()
                metrics = communicative_main(problem)
                runtime = time.time() - start_time
                metrics["runtime"] = runtime
                comm_metrics.append(metrics)
                
                # Report this run's results
                if metrics["solution_found"]:
                    print(f"Run {i+1}: Solution found in {metrics['solution_episode']} episodes (time: {runtime:.2f}s)")
                else:
                    print(f"Run {i+1}: No solution found (time: {runtime:.2f}s)")
            
            # Store communicative results
            all_results[problem["name"]] = {
                "cooperative": coop_metrics,
                "competitive": comp_metrics,
                "communicative": comm_metrics,
                "problem": problem
            }
            
            # Analyze communicative vs others
            analyze_three_way(coop_metrics, comp_metrics, comm_metrics, problem)
        else:
            # Store results for this problem
            all_results[problem["name"]] = {
                "cooperative": coop_metrics,
                "competitive": comp_metrics,
                "problem": problem
            }
            
            # Analyze results for this problem
            analyze_results(coop_metrics, comp_metrics, problem)
    
    # Save all metrics for further analysis
    with open("sat_solving_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Create summary plots across all problems
    plot_problem_comparison(all_results)

def analyze_results(coop_metrics, comp_metrics, problem):
    print(f"\n--- Results Summary for {problem['name']} ---")
    
    # Calculate success rate
    coop_success = sum(1 for m in coop_metrics if m["solution_found"]) / len(coop_metrics)
    comp_success = sum(1 for m in comp_metrics if m["solution_found"]) / len(comp_metrics)
    print(f"Success rate: Cooperative: {coop_success*100:.1f}%, Competitive: {comp_success*100:.1f}%")
    
    # Calculate average episodes to solution (only for successful runs)
    coop_episodes = [m["solution_episode"] for m in coop_metrics if m["solution_found"]]
    comp_episodes = [m["solution_episode"] for m in comp_metrics if m["solution_found"]]
    
    if coop_episodes:
        print(f"Cooperative: Avg episodes to solution: {np.mean(coop_episodes):.1f} ± {np.std(coop_episodes):.1f}")
    else:
        print("Cooperative: No successful solutions")
        
    if comp_episodes:
        print(f"Competitive: Avg episodes to solution: {np.mean(comp_episodes):.1f} ± {np.std(comp_episodes):.1f}")
    else:
        print("Competitive: No successful solutions")
    
    # Calculate average runtime
    coop_runtime = [m["runtime"] for m in coop_metrics]
    comp_runtime = [m["runtime"] for m in comp_metrics]
    print(f"Avg runtime: Cooperative: {np.mean(coop_runtime):.2f}s, Competitive: {np.mean(comp_runtime):.2f}s")
    
    # Add statistical significance tests
    try:
        from scipy import stats
        
        # t-test for episode counts
        if coop_episodes and comp_episodes:
            t_stat, p_val = stats.ttest_ind(coop_episodes, comp_episodes)
            print(f"Episode count t-test: p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
        
        # t-test for runtime
        t_stat, p_val = stats.ttest_ind(coop_runtime, comp_runtime)
        print(f"Runtime t-test: p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
    except ImportError:
        print("SciPy not installed - skipping statistical tests")
    
    # Create visualizations for this problem
    plot_problem_results(coop_metrics, comp_metrics, problem)

def analyze_three_way(coop_metrics, comp_metrics, comm_metrics, problem):
    print(f"\n--- Results Summary for {problem['name']} ---")
    
    # Calculate success rate
    coop_success = sum(1 for m in coop_metrics if m["solution_found"]) / len(coop_metrics)
    comp_success = sum(1 for m in comp_metrics if m["solution_found"]) / len(comp_metrics)
    comm_success = sum(1 for m in comm_metrics if m["solution_found"]) / len(comm_metrics)
    print(f"Success rate: Cooperative: {coop_success*100:.1f}%, Competitive: {comp_success*100:.1f}%, Communicative: {comm_success*100:.1f}%")
    
    # Calculate average episodes to solution (only for successful runs)
    coop_episodes = [m["solution_episode"] for m in coop_metrics if m["solution_found"]]
    comp_episodes = [m["solution_episode"] for m in comp_metrics if m["solution_found"]]
    comm_episodes = [m["solution_episode"] for m in comm_metrics if m["solution_found"]]
    
    if coop_episodes:
        print(f"Cooperative: Avg episodes to solution: {np.mean(coop_episodes):.1f} ± {np.std(coop_episodes):.1f}")
    else:
        print("Cooperative: No successful solutions")
        
    if comp_episodes:
        print(f"Competitive: Avg episodes to solution: {np.mean(comp_episodes):.1f} ± {np.std(comp_episodes):.1f}")
    else:
        print("Competitive: No successful solutions")
    
    if comm_episodes:
        print(f"Communicative: Avg episodes to solution: {np.mean(comm_episodes):.1f} ± {np.std(comm_episodes):.1f}")
    else:
        print("Communicative: No successful solutions")
    
    # Calculate average runtime
    coop_runtime = [m["runtime"] for m in coop_metrics]
    comp_runtime = [m["runtime"] for m in comp_metrics]
    comm_runtime = [m["runtime"] for m in comm_metrics]
    print(f"Avg runtime: Cooperative: {np.mean(coop_runtime):.2f}s, Competitive: {np.mean(comp_runtime):.2f}s, Communicative: {np.mean(comm_runtime):.2f}s")
    
    # Add statistical significance tests
    try:
        from scipy import stats
        
        # t-test for episode counts
        if coop_episodes and comp_episodes and comm_episodes:
            t_stat, p_val = stats.ttest_ind(coop_episodes, comp_episodes)
            print(f"Episode count t-test (Coop vs Comp): p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
            
            t_stat, p_val = stats.ttest_ind(coop_episodes, comm_episodes)
            print(f"Episode count t-test (Coop vs Comm): p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
            
            t_stat, p_val = stats.ttest_ind(comp_episodes, comm_episodes)
            print(f"Episode count t-test (Comp vs Comm): p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
        
        # t-test for runtime
        t_stat, p_val = stats.ttest_ind(coop_runtime, comp_runtime)
        print(f"Runtime t-test (Coop vs Comp): p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
        
        t_stat, p_val = stats.ttest_ind(coop_runtime, comm_runtime)
        print(f"Runtime t-test (Coop vs Comm): p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
        
        t_stat, p_val = stats.ttest_ind(comp_runtime, comm_runtime)
        print(f"Runtime t-test (Comp vs Comm): p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
    except ImportError:
        print("SciPy not installed - skipping statistical tests")
    
    # Create visualizations for this problem
    plot_problem_results_three_way(coop_metrics, comp_metrics, comm_metrics, problem)

def plot_problem_results(coop_metrics, comp_metrics, problem):
    """Create visualizations for a single problem's results"""
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Results for {problem['name']} ({problem['num_vars']} variables, {len(problem['clauses'])} clauses)")
    
    # Plot success rate
    plt.subplot(2, 2, 1)
    coop_success = sum(1 for m in coop_metrics if m["solution_found"]) / len(coop_metrics)
    comp_success = sum(1 for m in comp_metrics if m["solution_found"]) / len(comp_metrics)
    plt.bar(['Cooperative', 'Competitive'], [coop_success, comp_success])
    plt.ylabel('Success Rate')
    plt.title('Solution Success Rate')
    
    # Plot average runtime
    plt.subplot(2, 2, 2)
    coop_runtime = [m["runtime"] for m in coop_metrics]
    comp_runtime = [m["runtime"] for m in comp_metrics]
    plt.bar(['Cooperative', 'Competitive'], [np.mean(coop_runtime), np.mean(comp_runtime)])
    plt.ylabel('Seconds')
    plt.title('Average Runtime')
    
    # Plot learning curves if available
    plt.subplot(2, 2, 3)
    for i, metrics in enumerate(coop_metrics):
        if "episode_rewards" in metrics and metrics["episode_rewards"]:
            plt.plot(metrics["episode_rewards"], label=f"Coop Run {i+1}", alpha=0.5)
    for i, metrics in enumerate(comp_metrics):
        if "episode_rewards" in metrics and metrics["episode_rewards"]:
            plt.plot(metrics["episode_rewards"], label=f"Comp Run {i+1}", alpha=0.5, linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Progress')
    plt.legend()
    
    # Add a fourth plot for best reward progress
    plt.subplot(2, 2, 4)
    
    # Plot best reward progress for cooperative agents
    for i, metrics in enumerate(coop_metrics):
        if "best_reward_progress" in metrics and metrics["best_reward_progress"]:
            plt.plot(metrics["best_reward_progress"], 
                     label=f"Coop {i+1}", alpha=0.5)
            
    # Plot best reward progress for competitive agents
    for i, metrics in enumerate(comp_metrics):
        if "best_reward_progress" in metrics and metrics["best_reward_progress"]:
            plt.plot(metrics["best_reward_progress"], 
                     label=f"Comp {i+1}", alpha=0.5, linestyle='--')
            
    plt.xlabel('Episode')
    plt.ylabel('Best Reward So Far')
    plt.title('Solution Quality Progress')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results_{problem['name']}.png")
    plt.close()

def plot_problem_results_three_way(coop_metrics, comp_metrics, comm_metrics, problem):
    """Create visualizations for a single problem's results with three agent types"""
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Results for {problem['name']} ({problem['num_vars']} variables, {len(problem['clauses'])} clauses)")
    
    # Plot success rate
    plt.subplot(2, 2, 1)
    coop_success = sum(1 for m in coop_metrics if m["solution_found"]) / len(coop_metrics)
    comp_success = sum(1 for m in comp_metrics if m["solution_found"]) / len(comp_metrics)
    comm_success = sum(1 for m in comm_metrics if m["solution_found"]) / len(comm_metrics)
    plt.bar(['Cooperative', 'Competitive', 'Communicative'], [coop_success, comp_success, comm_success])
    plt.ylabel('Success Rate')
    plt.title('Solution Success Rate')
    
    # Plot average runtime
    plt.subplot(2, 2, 2)
    coop_runtime = [m["runtime"] for m in coop_metrics]
    comp_runtime = [m["runtime"] for m in comp_metrics]
    comm_runtime = [m["runtime"] for m in comm_metrics]
    plt.bar(['Cooperative', 'Competitive', 'Communicative'], [np.mean(coop_runtime), np.mean(comp_runtime), np.mean(comm_runtime)])
    plt.ylabel('Seconds')
    plt.title('Average Runtime')
    
    # Plot learning curves if available
    plt.subplot(2, 2, 3)
    for i, metrics in enumerate(coop_metrics):
        if "episode_rewards" in metrics and metrics["episode_rewards"]:
            plt.plot(metrics["episode_rewards"], label=f"Coop Run {i+1}", alpha=0.5)
    for i, metrics in enumerate(comp_metrics):
        if "episode_rewards" in metrics and metrics["episode_rewards"]:
            plt.plot(metrics["episode_rewards"], label=f"Comp Run {i+1}", alpha=0.5, linestyle='--')
    for i, metrics in enumerate(comm_metrics):
        if "episode_rewards" in metrics and metrics["episode_rewards"]:
            plt.plot(metrics["episode_rewards"], label=f"Comm Run {i+1}", alpha=0.5, linestyle='-.')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Progress')
    plt.legend()
    
    # Add a fourth plot for best reward progress
    plt.subplot(2, 2, 4)
    
    # Plot best reward progress for cooperative agents
    for i, metrics in enumerate(coop_metrics):
        if "best_reward_progress" in metrics and metrics["best_reward_progress"]:
            plt.plot(metrics["best_reward_progress"], 
                     label=f"Coop {i+1}", alpha=0.5)
            
    # Plot best reward progress for competitive agents
    for i, metrics in enumerate(comp_metrics):
        if "best_reward_progress" in metrics and metrics["best_reward_progress"]:
            plt.plot(metrics["best_reward_progress"], 
                     label=f"Comp {i+1}", alpha=0.5, linestyle='--')
            
    # Plot best reward progress for communicative agents
    for i, metrics in enumerate(comm_metrics):
        if "best_reward_progress" in metrics and metrics["best_reward_progress"]:
            plt.plot(metrics["best_reward_progress"], 
                     label=f"Comm {i+1}", alpha=0.5, linestyle='-.')
            
    plt.xlabel('Episode')
    plt.ylabel('Best Reward So Far')
    plt.title('Solution Quality Progress')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results_{problem['name']}_three_way.png")
    plt.close()

def plot_problem_comparison(all_results):
    """Create summary plots comparing results across all problems"""
    problem_names = list(all_results.keys())
    coop_success_rates = []
    comp_success_rates = []
    comm_success_rates = []
    coop_runtimes = []
    comp_runtimes = []
    comm_runtimes = []
    
    for name in problem_names:
        results = all_results[name]
        
        # Calculate success rates
        coop_success = sum(1 for m in results["cooperative"] if m["solution_found"]) / len(results["cooperative"])
        comp_success = sum(1 for m in results["competitive"] if m["solution_found"]) / len(results["competitive"])
        coop_success_rates.append(coop_success)
        comp_success_rates.append(comp_success)
        
        if "communicative" in results:
            comm_success = sum(1 for m in results["communicative"] if m["solution_found"]) / len(results["communicative"])
            comm_success_rates.append(comm_success)
        else:
            comm_success_rates.append(0)
        
        # Calculate average runtimes
        coop_runtime = np.mean([m["runtime"] for m in results["cooperative"]])
        comp_runtime = np.mean([m["runtime"] for m in results["competitive"]])
        coop_runtimes.append(coop_runtime)
        comp_runtimes.append(comp_runtime)
        
        if "communicative" in results:
            comm_runtime = np.mean([m["runtime"] for m in results["communicative"]])
            comm_runtimes.append(comm_runtime)
        else:
            comm_runtimes.append(0)
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Success rate comparison
    plt.subplot(2, 1, 1)
    x = np.arange(len(problem_names))
    width = 0.25
    plt.bar(x - width, coop_success_rates, width, label='Cooperative')
    plt.bar(x, comp_success_rates, width, label='Competitive')
    plt.bar(x + width, comm_success_rates, width, label='Communicative')
    plt.xlabel('Problem')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Problem')
    plt.xticks(x, problem_names, rotation=45, ha='right')
    plt.legend()
    
    # Runtime comparison
    plt.subplot(2, 1, 2)
    plt.bar(x - width, coop_runtimes, width, label='Cooperative')
    plt.bar(x, comp_runtimes, width, label='Competitive')
    plt.bar(x + width, comm_runtimes, width, label='Communicative')
    plt.xlabel('Problem')
    plt.ylabel('Runtime (seconds)')
    plt.title('Average Runtime by Problem')
    plt.xticks(x, problem_names, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("problem_comparison.png")
    plt.close()

if __name__ == "__main__":
    run_comparison(PROBLEM_COLLECTION[:3], n_runs=2)  # Start with first 3 problems