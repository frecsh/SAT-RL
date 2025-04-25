#!/usr/bin/env python3
"""
Benchmark script to compare different SAT solving approaches:
1. Multi-agent Q-learning (original)
2. Multi-agent Q-learning with improved Progressive GAN

This script generates SAT problems of varying complexity and evaluates
the performance of different solving methods.
"""

import time
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import argparse

from sat_problems import generate_sat_problem, is_satisfied, count_satisfied_clauses
from multi_q_sat import MultiQLearningSAT
from multi_q_sat_gan_improved import MultiQLearningSATProgressiveGAN


def run_benchmark(problem_sizes: List[Tuple[int, int]], 
                 num_trials: int = 5,
                 max_episodes: int = 500,
                 timeout: int = 120,
                 output_dir: str = "results"):
    """
    Run benchmarks for different SAT solving methods.
    
    Args:
        problem_sizes: List of (n_vars, n_clauses) tuples to test
        num_trials: Number of trials per problem size
        max_episodes: Maximum episodes per trial
        timeout: Timeout in seconds
        output_dir: Directory to save results
        
    Returns:
        DataFrame with benchmark results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Results will be stored in this list
    results = []
    
    # Loop over all problem sizes
    for n_vars, n_clauses in problem_sizes:
        print(f"\n{'-'*80}")
        print(f"Benchmarking problem size: {n_vars} variables, {n_clauses} clauses")
        print(f"{'-'*80}")
        
        # Run multiple trials for this problem size
        for trial in range(num_trials):
            print(f"\nTrial {trial+1}/{num_trials}")
            
            # Generate a SAT problem
            clauses = generate_sat_problem(n_vars, n_clauses)
            
            # Benchmarks for each solver
            solvers = {
                "Q-Learning": lambda: MultiQLearningSAT(
                    n_vars, clauses, n_agents=5, learning_rate=0.1, 
                    discount_factor=0.95, epsilon=0.2, epsilon_decay=0.995
                ),
                "ProgressiveGAN+Q-Learning": lambda: MultiQLearningSATProgressiveGAN(
                    n_vars, clauses, n_agents=5, learning_rate=0.1,
                    discount_factor=0.95, epsilon=0.1, epsilon_decay=0.995,
                    gan_exploration_ratio=0.7, gan_training_interval=40
                )
            }
            
            # Run each solver
            for solver_name, create_solver in solvers.items():
                print(f"Running {solver_name}...")
                
                # Create solver
                solver = create_solver()
                
                # Solve
                try:
                    start_time = time.time()
                    solution, stats = solver.solve(max_episodes=max_episodes, 
                                                 early_stopping=True, 
                                                 timeout=timeout)
                    solve_time = time.time() - start_time
                    
                    # Extract results
                    solved = stats['solved']
                    episodes = stats['episodes']
                    ratio = stats['best_satisfaction_ratio']
                    
                    # Print results
                    print(f"  {solver_name}: {'Solved' if solved else 'Not solved'} "
                          f"in {episodes} episodes, {solve_time:.2f}s, {ratio:.2%} satisfaction")
                    
                    # Add to results
                    result = {
                        'solver': solver_name,
                        'n_vars': n_vars,
                        'n_clauses': n_clauses,
                        'trial': trial + 1,
                        'solved': int(solved),
                        'episodes': episodes,
                        'time': solve_time,
                        'satisfaction_ratio': ratio,
                        'timed_out': int(stats.get('timed_out', False))
                    }
                    
                    # Add GAN-specific metrics if available
                    if solver_name == "ProgressiveGAN+Q-Learning":
                        result.update({
                            'gan_uses': stats.get('gan_uses', 0),
                            'gan_success_rate': stats.get('gan_success_rate', 0.0),
                            'gan_training_count': stats.get('gan_training_count', 0),
                            'solution_buffer_size': stats.get('solution_buffer_size', 0)
                        })
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error running {solver_name}: {str(e)}")
                    
                    # Add error result
                    results.append({
                        'solver': solver_name,
                        'n_vars': n_vars,
                        'n_clauses': n_clauses,
                        'trial': trial + 1,
                        'solved': 0,
                        'episodes': max_episodes,
                        'time': timeout,
                        'satisfaction_ratio': 0.0,
                        'timed_out': 1,
                        'error': str(e)
                    })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(output_dir, f"sat_benchmark_results.csv")
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Create summary figures
    create_summary_plots(df, output_dir)
    
    return df


def create_summary_plots(df: pd.DataFrame, output_dir: str):
    """
    Create summary plots from benchmark results.
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    # Ensure plots directory exists
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Aggregate results by problem size and solver
    summary = df.groupby(['n_vars', 'n_clauses', 'solver']).agg({
        'solved': 'mean',  # Solve rate
        'episodes': 'mean',  # Average episodes
        'time': 'mean',  # Average time
        'satisfaction_ratio': 'mean',  # Average satisfaction ratio
        'timed_out': 'mean'  # Timeout rate
    }).reset_index()
    
    # Generate problem size labels
    summary['problem_size'] = summary.apply(
        lambda x: f"{int(x['n_vars'])}v-{int(x['n_clauses'])}c", axis=1
    )
    
    # Plot solve rate
    plt.figure(figsize=(12, 6))
    problem_sizes = sorted(summary['problem_size'].unique())
    
    for solver in summary['solver'].unique():
        solver_data = summary[summary['solver'] == solver]
        solve_rates = [
            solver_data[solver_data['problem_size'] == size]['solved'].values[0] 
            if size in solver_data['problem_size'].values else 0
            for size in problem_sizes
        ]
        plt.plot(problem_sizes, solve_rates, marker='o', label=solver)
    
    plt.title('SAT Problem Solve Rate by Solver')
    plt.xlabel('Problem Size (variables-clauses)')
    plt.ylabel('Solve Rate')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'solve_rate.png'))
    
    # Plot average time to solve
    plt.figure(figsize=(12, 6))
    
    for solver in summary['solver'].unique():
        solver_data = summary[summary['solver'] == solver]
        avg_times = [
            solver_data[solver_data['problem_size'] == size]['time'].values[0]
            if size in solver_data['problem_size'].values else 0
            for size in problem_sizes
        ]
        plt.plot(problem_sizes, avg_times, marker='o', label=solver)
    
    plt.title('Average Time to Solve by Solver')
    plt.xlabel('Problem Size (variables-clauses)')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'avg_time.png'))
    
    # Plot average satisfaction ratio
    plt.figure(figsize=(12, 6))
    
    for solver in summary['solver'].unique():
        solver_data = summary[summary['solver'] == solver]
        sat_ratios = [
            solver_data[solver_data['problem_size'] == size]['satisfaction_ratio'].values[0]
            if size in solver_data['problem_size'].values else 0
            for size in problem_sizes
        ]
        plt.plot(problem_sizes, sat_ratios, marker='o', label=solver)
    
    plt.title('Average Satisfaction Ratio by Solver')
    plt.xlabel('Problem Size (variables-clauses)')
    plt.ylabel('Satisfaction Ratio')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'satisfaction_ratio.png'))
    
    # Save summary to CSV
    summary_file = os.path.join(output_dir, "benchmark_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SAT solving approaches")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per problem size")
    parser.add_argument("--episodes", type=int, default=500, help="Max episodes per trial")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    # Define problem sizes to test: (n_vars, n_clauses)
    # Each tuple represents a SAT problem configuration
    problem_sizes = [
        (10, 40),   # Easy
        (20, 85),   # Medium
        (30, 125),  # Hard
        (40, 170),  # Very hard
        (50, 210)   # Extremely hard
    ]
    
    print(f"Starting benchmark with {args.trials} trials per problem size")
    print(f"Max episodes: {args.episodes}, Timeout: {args.timeout}s")
    
    # Run benchmarks
    results = run_benchmark(
        problem_sizes=problem_sizes,
        num_trials=args.trials,
        max_episodes=args.episodes,
        timeout=args.timeout,
        output_dir=args.output
    )
    
    print("\nBenchmark completed!")