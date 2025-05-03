#!/usr/bin/env python3
"""
SAT RL Demo - Demonstration script for SAT Reinforcement Learning agents
"""

import argparse
import time
import os
from src.agents.deep_q_sat_agent import DeepQSATAgent
from src.agents.curriculum_sat_learner import CurriculumSATLearner
from sat_rl_logger import SATRLLogger
import matplotlib.pyplot as plt

def run_demo(agent_type, n_vars, n_clauses, episodes, method=None, visualize=False):
    """Run a demonstration with the specified agent type"""
    print(f"Running demo with {agent_type} agent on {n_vars}-variable, {n_clauses}-clause SAT problem")
    
    # Create logs directory
    logs_dir = f"sat_rl_logs/{agent_type}/{int(time.time())}"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up problem and agent with logging enabled
    if agent_type == "dqn":
        agent = DeepQSATAgent(
            state_size=n_vars, 
            action_size=n_vars * 2,  # For each variable: set true or false
            enable_logging=True,
            logs_dir=logs_dir
        )
        
        # Run training with logging
        print("Training DQN agent...")
        rewards, steps = agent.train(episodes=episodes, verbose=True)
        
        # Export final logs and visualize results if requested
        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(rewards)
            plt.title("Rewards per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            
            plt.subplot(1, 2, 2)
            plt.plot(steps)
            plt.title("Steps per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            
            plt.tight_layout()
            plt.savefig(f"{logs_dir}/training_progress.png")
            print(f"Training visualization saved to {logs_dir}/training_progress.png")
        
    elif agent_type == "curriculum":
        # Default settings for curriculum learning
        initial_ratio = 3.0
        target_ratio = 4.2
        step_size = 0.2
        
        # Initialize with logging enabled
        curriculum = CurriculumSATLearner(
            n_vars=n_vars, 
            initial_ratio=initial_ratio,
            target_ratio=target_ratio,
            step_size=step_size,
            enable_logging=True,
            logs_dir=logs_dir
        )
        
        # Run curriculum learning
        print(f"Running curriculum learning from ratio {initial_ratio} to {target_ratio}...")
        solution, satisfied = curriculum.solve_with_curriculum(max_attempts=5)
        
        print(f"Curriculum learning complete. Satisfied {satisfied} out of {int(target_ratio * n_vars)} clauses.")
    
    elif agent_type == "oracle":
        # Oracle method implementation
        print(f"Running Oracle method for SAT solving ({method} strategy)...")
        
        try:
            from sat_oracle_solver import SATOracleSolver
            oracle = SATOracleSolver(
                n_vars=n_vars,
                n_clauses=n_clauses,
                method=method,
                logs_dir=logs_dir
            )
            
            # Run the oracle solver
            start_time = time.time()
            solution, satisfied, total = oracle.solve()
            solve_time = time.time() - start_time
            
            # Report results
            print(f"Oracle solver complete in {solve_time:.2f} seconds")
            print(f"Satisfied {satisfied} out of {total} clauses ({satisfied/total*100:.1f}%)")
            
            # Export visualization if requested
            if visualize and hasattr(oracle, "get_history"):
                history = oracle.get_history()
                plt.figure(figsize=(10, 5))
                plt.plot(history["satisfaction_ratio"])
                plt.title("Satisfaction Ratio Over Time")
                plt.xlabel("Iteration")
                plt.ylabel("Clauses Satisfied (%)")
                plt.tight_layout()
                plt.savefig(f"{logs_dir}/oracle_progress.png")
                print(f"Oracle visualization saved to {logs_dir}/oracle_progress.png")
                
        except ImportError:
            print("ERROR: Could not import SATOracleSolver. Make sure the module is available.")
            
    else:
        print(f"Unknown agent type: {agent_type}")

def analyze_logs(log_file_path):
    """Analyze logs from a previously run experiment"""
    print(f"Analyzing logs from {log_file_path}")
    
    # Create a logger instance just for analysis
    logger = SATRLLogger(log_to_file=False)
    
    # Load logs from file
    loaded = logger.load_traces_from_csv(log_file_path)
    if not loaded:
        print(f"Failed to load logs from {log_file_path}")
        return
    
    # Print statistics
    stats = logger.get_statistics()
    print("\nLog Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate visualizations
    vis_dir = os.path.dirname(log_file_path) + "/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate reward over time plot
    logger.visualize_rewards(save_path=f"{vis_dir}/rewards.png")
    print(f"Reward visualization saved to {vis_dir}/rewards.png")
    
    # Generate action distribution plot
    logger.visualize_action_distribution(save_path=f"{vis_dir}/actions.png")
    print(f"Action distribution visualization saved to {vis_dir}/actions.png")
    
    # Generate state transitions plot if available
    logger.visualize_state_transitions(save_path=f"{vis_dir}/state_transitions.png")
    print(f"State transition visualization saved to {vis_dir}/state_transitions.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAT RL demonstrations")
    parser.add_argument("--agent", type=str, choices=["dqn", "curriculum", "oracle"], 
                        help="Agent type to use (dqn, curriculum, or oracle)")
    parser.add_argument("--method", type=str, default="dpll", 
                        help="Method to use for oracle agent (dpll, cdcl, walksat, etc.)")
    parser.add_argument("--vars", type=int, default=20, 
                        help="Number of variables")
    parser.add_argument("--clauses", type=int, default=80,
                        help="Number of clauses")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Analyze an existing log file instead of running training")
    
    args = parser.parse_args()
    
    # If method is specified but agent isn't, set agent type based on method
    if args.agent is None and args.method:
        if args.method in ["dpll", "cdcl", "walksat", "oracle", "multiagent"]:
            args.agent = "oracle"
        elif args.method in ["curriculum"]:
            args.agent = "curriculum"
        else:
            args.agent = "dqn"
    
    # Default to dqn if no agent type is specified
    if args.agent is None:
        args.agent = "dqn"
    
    if args.analyze:
        analyze_logs(args.analyze)
    else:
        run_demo(args.agent, args.vars, args.clauses, args.episodes, args.method, args.visualize)