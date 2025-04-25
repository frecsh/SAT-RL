#!/usr/bin/env python3
"""
Benchmark comparison pipeline for SAT solvers.
Compares SAT+RL approaches against traditional SAT solvers like MiniSAT and Glucose.
Includes GAN-powered solvers in the comparison.
"""

import os
import time
import argparse
import resource
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import subprocess
import tempfile
import psutil

# Import your SAT+RL implementation
from multi_q_sat import MultiQLearningSAT
from multi_q_sat_comm import MultiQLearningSATComm
from multi_q_sat_oracle import MultiQLearningSATOracle
from multi_q_sat_gan import MultiQLearningSATGAN, AdaptiveMultiQLearningSATGAN
from sat_problems import generate_sat_problem, load_sat_from_file
from sat_utils import clause_to_variable_ratio
from sat_gan import SATGAN

# Configure argument parser
parser = argparse.ArgumentParser(description='Benchmark SAT solvers')
parser.add_argument('--benchmark_dir', type=str, default='benchmarks', 
                    help='Directory containing SAT benchmark files')
parser.add_argument('--output_dir', type=str, default='results/benchmarks',
                    help='Directory to save benchmark results')
parser.add_argument('--minisat_path', type=str, default='minisat',
                    help='Path to MiniSAT executable')
parser.add_argument('--glucose_path', type=str, default='glucose',
                    help='Path to Glucose executable')
parser.add_argument('--timeout', type=int, default=300,
                    help='Timeout in seconds for each solver run')
parser.add_argument('--runs', type=int, default=5,
                    help='Number of runs for each solver-problem pair')
parser.add_argument('--rl_episodes', type=int, default=1000,
                    help='Number of episodes for RL-based solvers')
parser.add_argument('--phase_transition', action='store_true',
                    help='Run phase transition analysis')
parser.add_argument('--gan_model_path', type=str, default=None,
                    help='Path to pre-trained GAN model to use')
parser.add_argument('--pre_train_gan', action='store_true',
                    help='Pre-train GAN models before benchmarking')

class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.solvers = self._initialize_solvers()
        os.makedirs(args.output_dir, exist_ok=True)
        self.results = []

    def _initialize_solvers(self):
        """Initialize all solvers to be benchmarked."""
        return {
            'MiniSAT': self._run_minisat,
            'Glucose': self._run_glucose,
            'SAT+RL (Cooperative)': self._run_rl_coop,
            'SAT+RL (Communicative)': self._run_rl_comm,
            'SAT+RL (Oracle)': self._run_rl_oracle,
            'GAN-Q-Learning': self._run_gan_q_learning,
            'Adaptive GAN-Q-Learning': self._run_adaptive_gan_q_learning
        }
    
    def _run_minisat(self, problem_path):
        """Run MiniSAT on the given problem."""
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as output_file:
            output_path = output_file.name
            
        try:
            result = subprocess.run(
                [self.args.minisat_path, problem_path, output_path],
                capture_output=True,
                timeout=self.args.timeout
            )
            solved = b"SATISFIABLE" in result.stdout
            timed_out = False
        except subprocess.TimeoutExpired:
            solved = False
            timed_out = True
            
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Clean up temp file
        if os.path.exists(output_path):
            os.unlink(output_path)
            
        return {
            'solved': solved,
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'timed_out': timed_out
        }
    
    def _run_glucose(self, problem_path):
        """Run Glucose on the given problem."""
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = subprocess.run(
                [self.args.glucose_path, problem_path],
                capture_output=True,
                timeout=self.args.timeout
            )
            solved = b"s SATISFIABLE" in result.stdout
            timed_out = False
        except subprocess.TimeoutExpired:
            solved = False
            timed_out = True
            
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        return {
            'solved': solved,
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'timed_out': timed_out
        }
    
    def _run_rl_coop(self, problem_path):
        """Run cooperative RL solver on the given problem."""
        clauses, n_vars = load_sat_from_file(problem_path)
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Create and run the solver
        solver = MultiQLearningSAT(n_vars=n_vars, clauses=clauses)
        solution, stats = solver.solve(max_episodes=self.args.rl_episodes, 
                                      early_stopping=True, 
                                      timeout=self.args.timeout)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        return {
            'solved': stats['solved'],
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'timed_out': stats.get('timed_out', False),
            'episodes': stats.get('episodes', self.args.rl_episodes)
        }

    def _run_rl_comm(self, problem_path):
        """Run communicative RL solver on the given problem."""
        clauses, n_vars = load_sat_from_file(problem_path)
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Create and run the solver with communication threshold of 0.5 (optimal per README)
        solver = MultiQLearningSATComm(n_vars=n_vars, clauses=clauses, comm_threshold=0.5)
        solution, stats = solver.solve(max_episodes=self.args.rl_episodes, 
                                      early_stopping=True, 
                                      timeout=self.args.timeout)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        return {
            'solved': stats['solved'],
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'timed_out': stats.get('timed_out', False),
            'episodes': stats.get('episodes', self.args.rl_episodes)
        }
    
    def _run_rl_oracle(self, problem_path):
        """Run oracle-guided RL solver on the given problem."""
        clauses, n_vars = load_sat_from_file(problem_path)
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Create and run the solver with oracle weight of 0.35 (optimal per README)
        solver = MultiQLearningSATOracle(n_vars=n_vars, clauses=clauses, oracle_weight=0.35)
        solution, stats = solver.solve(max_episodes=self.args.rl_episodes, 
                                      early_stopping=True, 
                                      timeout=self.args.timeout)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        return {
            'solved': stats['solved'],
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'timed_out': stats.get('timed_out', False),
            'episodes': stats.get('episodes', self.args.rl_episodes)
        }

    def _run_gan_q_learning(self, problem_path):
        """Run GAN-powered Q-learning solver on the given problem."""
        clauses, n_vars = load_sat_from_file(problem_path)
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Load or pre-train GAN model
        gan_model = None
        if self.args.gan_model_path and os.path.exists(self.args.gan_model_path):
            gan_model = SATGAN.load(self.args.gan_model_path, clauses)
        elif self.args.pre_train_gan:
            gan_model = SATGAN(n_vars=n_vars, clauses=clauses, latent_dim=50, clause_weight=0.5)
            gan_model.train(solutions=[], batch_size=16, epochs=50, eval_interval=10)
        
        # Create and run the solver
        solver = MultiQLearningSATGAN(n_vars=n_vars, clauses=clauses, gan_model=gan_model)
        solution, stats = solver.solve(max_episodes=self.args.rl_episodes, 
                                      early_stopping=True, 
                                      timeout=self.args.timeout)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        return {
            'solved': stats['solved'],
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'timed_out': stats.get('timed_out', False),
            'episodes': stats.get('episodes', self.args.rl_episodes)
        }

    def _run_adaptive_gan_q_learning(self, problem_path):
        """Run adaptive GAN-powered Q-learning solver on the given problem."""
        clauses, n_vars = load_sat_from_file(problem_path)
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Load or pre-train GAN model
        gan_model = None
        if self.args.gan_model_path and os.path.exists(self.args.gan_model_path):
            gan_model = SATGAN.load(self.args.gan_model_path, clauses)
        elif self.args.pre_train_gan:
            gan_model = SATGAN(n_vars=n_vars, clauses=clauses, latent_dim=50, clause_weight=0.5)
            gan_model.train(solutions=[], batch_size=16, epochs=50, eval_interval=10)
        
        # Create and run the solver
        solver = AdaptiveMultiQLearningSATGAN(n_vars=n_vars, clauses=clauses, gan_model=gan_model)
        solution, stats = solver.solve(max_episodes=self.args.rl_episodes, 
                                      early_stopping=True, 
                                      timeout=self.args.timeout)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        return {
            'solved': stats['solved'],
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'timed_out': stats.get('timed_out', False),
            'episodes': stats.get('episodes', self.args.rl_episodes)
        }
        
    def run_benchmark(self):
        """Run the benchmark on all problems in the benchmark directory."""
        problem_files = []
        
        # Collect all .cnf files
        for root, _, files in os.walk(self.args.benchmark_dir):
            for file in files:
                if file.endswith('.cnf'):
                    problem_files.append(os.path.join(root, file))
        
        print(f"Found {len(problem_files)} benchmark problems")
        
        # Run benchmarks
        for problem_path in tqdm(problem_files, desc="Benchmarking"):
            # Get problem characteristics
            clauses, n_vars = load_sat_from_file(problem_path)
            ratio = clause_to_variable_ratio(clauses, n_vars)
            problem_name = os.path.basename(problem_path)
            
            for solver_name, solver_func in self.solvers.items():
                for run in range(self.args.runs):
                    print(f"Running {solver_name} on {problem_name} (Run {run+1}/{self.args.runs})")
                    
                    try:
                        result = solver_func(problem_path)
                        
                        self.results.append({
                            'problem': problem_name,
                            'solver': solver_name,
                            'run': run + 1,
                            'variables': n_vars,
                            'clauses': len(clauses),
                            'ratio': ratio,
                            'solved': result['solved'],
                            'time': result['time'],
                            'memory': result['memory'],
                            'timed_out': result.get('timed_out', False),
                            'episodes': result.get('episodes', None)
                        })
                    except Exception as e:
                        print(f"Error with {solver_name} on {problem_name}: {str(e)}")
                        # Log the error but continue with other benchmarks
                        self.results.append({
                            'problem': problem_name,
                            'solver': solver_name,
                            'run': run + 1,
                            'variables': n_vars,
                            'clauses': len(clauses),
                            'ratio': ratio,
                            'error': str(e)
                        })
        
        # Save results to CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(self.args.output_dir, 'benchmark_results.csv'), index=False)
        
        # Generate visualizations
        self.generate_visualizations(results_df)
        
    def phase_transition_analysis(self):
        """Analyze solver performance across the phase transition."""
        ratios = np.linspace(2.0, 6.0, 20)  # Range covering phase transition (~4.2)
        variables = 50  # Fixed number of variables
        
        results = []
        
        for ratio in tqdm(ratios, desc="Phase Transition Analysis"):
            # Number of clauses = ratio * variables
            n_clauses = int(ratio * variables)
            
            # Generate a random SAT problem with the specified ratio
            clauses = generate_sat_problem(variables, n_clauses)
            
            # Write problem to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as temp_file:
                temp_file.write(f"p cnf {variables} {n_clauses}\n")
                for clause in clauses:
                    temp_file.write(" ".join(map(str, clause)) + " 0\n")
                problem_path = temp_file.name
            
            # Run each solver on this problem
            for solver_name, solver_func in self.solvers.items():
                for run in range(self.args.runs):
                    try:
                        result = solver_func(problem_path)
                        
                        results.append({
                            'ratio': ratio,
                            'solver': solver_name,
                            'run': run + 1,
                            'variables': variables,
                            'clauses': n_clauses,
                            'solved': result['solved'],
                            'time': result['time'],
                            'memory': result['memory'],
                            'timed_out': result.get('timed_out', False)
                        })
                    except Exception as e:
                        print(f"Error with {solver_name} at ratio {ratio}: {str(e)}")
            
            # Clean up temp file
            os.unlink(problem_path)
        
        # Save results to CSV
        phase_df = pd.DataFrame(results)
        phase_df.to_csv(os.path.join(self.args.output_dir, 'phase_transition_results.csv'), index=False)
        
        # Generate phase transition visualization
        self.generate_phase_transition_plot(phase_df)

    def generate_visualizations(self, results_df):
        """Generate visualizations from benchmark results."""
        # Set overall style
        sns.set(style="whitegrid")
        
        # 1. Time to solution by problem size and solver
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x="variables", y="time", hue="solver", data=results_df[results_df['solved']])
        ax.set_title("Time to Solution by Problem Size")
        ax.set_xlabel("Number of Variables")
        ax.set_ylabel("Time (s)")
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, "time_by_size.png"))
        plt.close()
        
        # 2. Memory usage by problem size and solver
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x="variables", y="memory", hue="solver", data=results_df)
        ax.set_title("Memory Usage by Problem Size")
        ax.set_xlabel("Number of Variables")
        ax.set_ylabel("Memory (MB)")
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, "memory_by_size.png"))
        plt.close()
        
        # 3. Success rate by solver
        plt.figure(figsize=(10, 6))
        success_by_solver = results_df.groupby('solver')['solved'].mean()
        ax = success_by_solver.plot(kind='bar')
        ax.set_title("Success Rate by Solver")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, "success_rate.png"))
        plt.close()
        
        # 4. Success rate by problem size
        plt.figure(figsize=(12, 8))
        grouped = results_df.groupby(['variables', 'solver'])['solved'].mean().reset_index()
        ax = sns.lineplot(x="variables", y="solved", hue="solver", data=grouped, markers=True)
        ax.set_title("Success Rate by Problem Size")
        ax.set_xlabel("Number of Variables")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, "success_by_size.png"))
        plt.close()

    def generate_phase_transition_plot(self, phase_df):
        """Generate phase transition visualization."""
        # Success rate by clause-to-variable ratio for each solver
        plt.figure(figsize=(12, 8))
        grouped = phase_df.groupby(['ratio', 'solver'])['solved'].mean().reset_index()
        ax = sns.lineplot(x="ratio", y="solved", hue="solver", data=grouped, markers=True)
        ax.set_title("Phase Transition Analysis: Success Rate by Clause-to-Variable Ratio")
        ax.set_xlabel("Clause-to-Variable Ratio")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        
        # Add vertical line at the theoretical phase transition point (~4.2)
        plt.axvline(x=4.2, color='red', linestyle='--', alpha=0.7, label="Theoretical Phase Transition")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, "phase_transition.png"))
        plt.close()
        
        # Time to solution by ratio
        plt.figure(figsize=(12, 8))
        solved_df = phase_df[phase_df['solved']]
        if not solved_df.empty:
            ax = sns.lineplot(x="ratio", y="time", hue="solver", data=solved_df)
            ax.set_title("Time to Solution by Clause-to-Variable Ratio")
            ax.set_xlabel("Clause-to-Variable Ratio")
            ax.set_ylabel("Time (s)")
            ax.set_yscale('log')
            plt.axvline(x=4.2, color='red', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.args.output_dir, "time_by_ratio.png"))
            plt.close()

def main():
    args = parser.parse_args()
    runner = BenchmarkRunner(args)
    
    # Run standard benchmarks
    runner.run_benchmark()
    
    # Run phase transition analysis if requested
    if args.phase_transition:
        runner.phase_transition_analysis()
    
if __name__ == "__main__":
    main()