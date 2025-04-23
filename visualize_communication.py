import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import seaborn as sns
from collections import Counter

# Define image directory
IMAGES_DIR = 'images'
os.makedirs(IMAGES_DIR, exist_ok=True)

def load_metrics(filename):
    """Load metrics from a pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def visualize_variable_communication(metrics, problem_name=""):
    """Visualize which variables are being communicated most frequently"""
    if "agent_communications" not in metrics:
        print("No communication data found in the metrics")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Collect all variable communications across all agents
    var_counts = Counter()
    for agent_comm in metrics["agent_communications"]:
        for comm in agent_comm:
            action = comm['action']
            for i, bit in enumerate(action):
                var_counts[f"{i+1}={bit}"] += 1
    
    # Display the top 15 most communicated variable assignments
    top_vars = var_counts.most_common(15)
    vars_labels = [v[0] for v in top_vars]
    vars_counts = [v[1] for v in top_vars]
    
    plt.bar(vars_labels, vars_counts)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Variable Assignment')
    plt.ylabel('Communication Count')
    plt.title(f'Most Frequently Communicated Variable Assignments {problem_name}')
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, f"comm_variables_{problem_name}.png")
    plt.savefig(output_path)
    print(f"Saved variable communication visualization to {output_path}")
    plt.close()

def visualize_clause_satisfaction(metrics, problem=None):
    """Visualize which clauses are being satisfied by communicated experiences"""
    if "agent_communications" not in metrics:
        print("No communication data found in the metrics")
        return
    
    # Get problem name for plot titles
    problem_name = problem["name"] if problem else ""
        
    plt.figure(figsize=(12, 8))
    
    # Collect all clause satisfactions across all agents
    clause_counts = Counter()
    for agent_comm in metrics["agent_communications"]:
        for comm in agent_comm:
            if 'clauses_satisfied' in comm and comm['clauses_satisfied'] is not None:
                satisfied_clauses = comm.get('satisfied_clauses', [])
                # If we have individual clause indices
                if isinstance(satisfied_clauses, list):
                    for clause_idx in satisfied_clauses:
                        clause_counts[clause_idx] += 1
    
    # If we have clause data, visualize it
    if clause_counts:
        clauses_labels = [f"Clause {i}" for i in sorted(clause_counts.keys())]
        clauses_counts = [clause_counts[i] for i in sorted(clause_counts.keys())]
        
        plt.bar(clauses_labels, clauses_counts)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Clause')
        plt.ylabel('Times Satisfied in Communications')
        plt.title(f'Clause Satisfaction in Communication ({problem_name})')
        plt.tight_layout()
        output_path = os.path.join(IMAGES_DIR, f"comm_clauses_{problem_name}.png")
        plt.savefig(output_path)
        print(f"Saved clause communication visualization to {output_path}")
        plt.close()
    else:
        print("No clause satisfaction data found")

def visualize_communication_heatmap(metrics, problem=None):
    """Create a heatmap showing variable-clause satisfaction correlation"""
    if "agent_communications" not in metrics or "agent_communications" not in metrics:
        print("No communication data found in the metrics")
        return
    
    # Get problem name and details
    problem_name = problem["name"] if problem else ""
    num_vars = problem["num_vars"] if problem else 10
    num_clauses = len(problem["clauses"]) if problem else 10
    
    # Create a heatmap matrix (variables x clauses)
    # For each variable, how often it contributes to satisfying each clause
    heatmap_data = np.zeros((num_vars, num_clauses))
    
    # Process communication history
    for agent_idx, agent_comm in enumerate(metrics["agent_communications"]):
        for comm in agent_comm:
            action = comm['action']
            for i, bit in enumerate(action):
                # For each variable assignment
                if 'clauses_satisfied' in comm and isinstance(comm.get('clauses_satisfied'), list):
                    for clause_idx in comm['clauses_satisfied']:
                        if clause_idx < num_clauses:  # Make sure it's in range
                            heatmap_data[i, clause_idx] += 1
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu", 
                   xticklabels=[f"C{i}" for i in range(num_clauses)],
                   yticklabels=[f"V{i+1}" for i in range(num_vars)])
    plt.xlabel('Clauses')
    plt.ylabel('Variables')
    plt.title(f'Variable-Clause Satisfaction Correlation ({problem_name})')
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, f"comm_heatmap_{problem_name}.png")
    plt.savefig(output_path)
    print(f"Saved communication heatmap to {output_path}")
    plt.close()

def visualize_reward_distribution(metrics, problem=None):
    """Visualize the distribution of rewards in communications"""
    if "agent_communications" not in metrics:
        print("No communication data found in the metrics")
        return
    
    problem_name = problem["name"] if problem else ""
    
    # Collect all rewards from communications
    rewards = []
    for agent_comm in metrics["agent_communications"]:
        for comm in agent_comm:
            rewards.append(comm['reward'])
    
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.title(f'Reward Distribution in Communications ({problem_name})')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(IMAGES_DIR, f"comm_rewards_{problem_name}.png")
    plt.savefig(output_path)
    print(f"Saved reward distribution to {output_path}")
    plt.close()

def visualize_communication_over_time(metrics, problem=None):
    """Visualize how communication patterns change over time"""
    if "communication_history" not in metrics:
        print("No communication history found in the metrics")
        return
    
    problem_name = problem["name"] if problem else ""
    
    # Extract communication frequency over training episodes
    episodes = []
    comm_counts = []
    
    for snapshot in metrics["communication_history"]:
        episode = snapshot["episode"]
        episodes.append(episode)
        
        # Count total communications in this snapshot
        total_comms = 0
        for agent_id, agent_data in snapshot["data"].items():
            if "rewards" in agent_data:
                total_comms += len(agent_data["rewards"])
        
        comm_counts.append(total_comms)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, comm_counts, marker='o')
    plt.xlabel('Training Episode')
    plt.ylabel('Number of Communications')
    plt.title(f'Communication Frequency Over Time ({problem_name})')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(IMAGES_DIR, f"comm_over_time_{problem_name}.png")
    plt.savefig(output_path)
    print(f"Saved communication over time visualization to {output_path}")
    plt.close()

def visualize_all_communication_metrics(metrics, problem=None):
    """Create all communication visualizations"""
    problem_name = problem["name"] if problem else ""
    
    print(f"Generating communication visualizations for {problem_name}...")
    visualize_variable_communication(metrics, problem_name)
    visualize_clause_satisfaction(metrics, problem)
    visualize_communication_heatmap(metrics, problem) 
    visualize_reward_distribution(metrics, problem)
    visualize_communication_over_time(metrics, problem)

def run_communication_visualization():
    """Run a communicating agent and visualize the results"""
    from multi_q_sat_comm import main as run_communicating_agents
    from sat_problems import MEDIUM_PROBLEM
    
    print("Running communicating agents to collect communication data...")
    metrics = run_communicating_agents(MEDIUM_PROBLEM)
    
    # Save the metrics to a file
    with open(f"comm_metrics_{MEDIUM_PROBLEM['name']}.pkl", "wb") as f:
        pickle.dump(metrics, f)
    
    # Visualize the communication data
    visualize_all_communication_metrics(metrics, MEDIUM_PROBLEM)

if __name__ == "__main__":
    # See if we have saved metrics
    problem_name = "medium_standard"
    pickle_path = f"comm_metrics_{problem_name}.pkl"
    if os.path.exists(pickle_path):
        print(f"Loading existing metrics from {pickle_path}")
        with open(pickle_path, "rb") as f:
            metrics = pickle.load(f)
        from sat_problems import MEDIUM_PROBLEM
        visualize_all_communication_metrics(metrics, MEDIUM_PROBLEM)
    else:
        print("No saved metrics found. Running new communication experiment.")
        run_communication_visualization()