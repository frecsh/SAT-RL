#!/usr/bin/env python3
"""
Analyze and visualize benchmark results from SAT solver comparisons.
This script provides more detailed statistical analysis than the
visualizations generated during benchmark runs.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from collections import defaultdict
from typing import Dict, List

parser = argparse.ArgumentParser(description='Analyze SAT solver benchmark results')
parser.add_argument('--results_file', type=str, required=True,
                   help='Path to CSV file with benchmark results')
parser.add_argument('--output_dir', type=str, default='results/analysis',
                   help='Directory to save analysis results')
parser.add_argument('--phase_file', type=str, default=None,
                   help='Path to CSV file with phase transition results')
parser.add_argument('--format', type=str, choices=['png', 'pdf', 'svg'], default='png',
                   help='Output format for visualizations')

class BenchmarkAnalyzer:
    def __init__(self, results_file: str, output_dir: str, phase_file: str = None, format: str = 'png'):
        self.results_file = results_file
        self.output_dir = output_dir
        self.phase_file = phase_file
        self.format = format
        
        # Load benchmark results
        self.results_df = pd.read_csv(results_file)
        
        # Load phase transition results if provided
        self.phase_df = None
        if phase_file and os.path.exists(phase_file):
            self.phase_df = pd.read_csv(phase_file)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        sns.set(style="whitegrid", font_scale=1.2)
        plt.rcParams['figure.figsize'] = [12, 8]

    def run_analysis(self):
        """Run all analysis functions"""
        # Generate basic statistics
        self.generate_basic_stats()
        
        # Performance comparisons
        self.analyze_time_performance()
        self.analyze_memory_usage()
        self.analyze_success_rates()
        
        # Scalability analysis
        self.analyze_scalability()
        
        # Phase transition analysis
        if self.phase_df is not None:
            self.analyze_phase_transition()
        
        # Statistical tests
        self.run_statistical_tests()
        
        # Summary dashboard
        self.create_summary_dashboard()

    def generate_basic_stats(self):
        """Generate basic statistics on the benchmark results"""
        # Filter out errors
        valid_df = self.results_df[~self.results_df['solved'].isna()]
        
        # Group by solver and compute statistics
        stats_by_solver = valid_df.groupby('solver').agg({
            'solved': ['mean', 'count'],
            'time': ['mean', 'median', 'std', 'min', 'max'],
            'memory': ['mean', 'median', 'std', 'min', 'max'],
            'timed_out': ['mean', 'sum']
        })
        
        # Save statistics to CSV and JSON
        stats_by_solver.to_csv(os.path.join(self.output_dir, f'solver_statistics.csv'))
        
        # Create a more readable JSON version
        stats_dict = {}
        for solver, row in stats_by_solver.iterrows():
            stats_dict[solver] = {
                'success_rate': float(row[('solved', 'mean')]),
                'problem_count': int(row[('solved', 'count')]),
                'avg_time_seconds': float(row[('time', 'mean')]),
                'median_time_seconds': float(row[('time', 'median')]),
                'avg_memory_mb': float(row[('memory', 'mean')]),
                'timeout_rate': float(row[('timed_out', 'mean')]),
                'timeouts': int(row[('timed_out', 'sum')])
            }
        
        with open(os.path.join(self.output_dir, f'solver_statistics.json'), 'w') as f:
            json.dump(stats_dict, f, indent=2)
            
        print(f"Basic statistics saved to {self.output_dir}/solver_statistics.json")

    def analyze_time_performance(self):
        """Analyze and visualize time performance"""
        # Focus on solved problems
        solved_df = self.results_df[self.results_df['solved'] == True]
        
        if solved_df.empty:
            print("Warning: No problems were solved successfully by any solver!")
            return
            
        # Time distribution by solver
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x='solver', y='time', data=solved_df)
        ax.set_title("Solve Time Distribution by Solver")
        ax.set_xlabel("Solver")
        ax.set_ylabel("Time (seconds)")
        ax.set_yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'time_distribution.{self.format}'))
        plt.close()
        
        # Time by problem size
        plt.figure(figsize=(12, 8))
        for solver in solved_df['solver'].unique():
            solver_df = solved_df[solved_df['solver'] == solver]
            plt.scatter(solver_df['variables'], solver_df['time'], alpha=0.7, label=solver)
            
            # Add trend line
            if len(solver_df) > 1:
                x = solver_df['variables']
                y = solver_df['time']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), '--', alpha=0.7)
        
        plt.title("Solve Time vs. Problem Size")
        plt.xlabel("Number of Variables")
        plt.ylabel("Time (seconds)")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'time_vs_size.{self.format}'))
        plt.close()

    def analyze_memory_usage(self):
        """Analyze and visualize memory usage"""
        # Memory usage distribution by solver
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x='solver', y='memory', data=self.results_df)
        ax.set_title("Memory Usage Distribution by Solver")
        ax.set_xlabel("Solver")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'memory_distribution.{self.format}'))
        plt.close()
        
        # Memory usage by problem size
        plt.figure(figsize=(12, 8))
        for solver in self.results_df['solver'].unique():
            solver_df = self.results_df[self.results_df['solver'] == solver]
            plt.scatter(solver_df['variables'], solver_df['memory'], alpha=0.7, label=solver)
            
            # Add trend line
            if len(solver_df) > 1:
                x = solver_df['variables']
                y = solver_df['memory']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), '--', alpha=0.7)
        
        plt.title("Memory Usage vs. Problem Size")
        plt.xlabel("Number of Variables")
        plt.ylabel("Memory Usage (MB)")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'memory_vs_size.{self.format}'))
        plt.close()

    def analyze_success_rates(self):
        """Analyze and visualize success rates"""
        # Success rate by solver
        plt.figure(figsize=(10, 6))
        success_by_solver = self.results_df.groupby('solver')['solved'].mean()
        ax = success_by_solver.plot(kind='bar')
        ax.set_title("Success Rate by Solver")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        for i, v in enumerate(success_by_solver):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'success_rate.{self.format}'))
        plt.close()
        
        # Success rate by problem size
        plt.figure(figsize=(12, 8))
        
        # Bin problems by variable count for clearer visualization
        self.results_df['var_bin'] = pd.cut(
            self.results_df['variables'], 
            bins=[0, 50, 100, 200, 500, 1000, float('inf')],
            labels=['<50', '50-100', '100-200', '200-500', '500-1000', '>1000']
        )
        
        success_by_size = self.results_df.groupby(['var_bin', 'solver'])['solved'].mean().reset_index()
        
        # Plot
        ax = sns.barplot(x='var_bin', y='solved', hue='solver', data=success_by_size)
        ax.set_title("Success Rate by Problem Size")
        ax.set_xlabel("Number of Variables")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'success_by_size.{self.format}'))
        plt.close()

    def analyze_scalability(self):
        """Analyze solver scalability with problem size"""
        # Prepare data - filter out errors and timeouts
        valid_df = self.results_df[
            (self.results_df['solved'] == True) & 
            (self.results_df['timed_out'] == False)
        ]
        
        if valid_df.empty:
            print("Warning: No valid data for scalability analysis!")
            return
        
        # Compute average time for each solver and problem size
        scalability_df = valid_df.groupby(['solver', 'variables'])['time'].mean().reset_index()
        
        # Plot scalability curve
        plt.figure(figsize=(12, 8))
        
        for solver in scalability_df['solver'].unique():
            solver_df = scalability_df[scalability_df['solver'] == solver]
            # Sort by variable count for proper line plotting
            solver_df = solver_df.sort_values('variables')
            
            plt.plot(solver_df['variables'], solver_df['time'], 'o-', label=solver)
        
        plt.title("Solver Scalability: Time vs Problem Size")
        plt.xlabel("Number of Variables")
        plt.ylabel("Average Time (seconds)")
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'scalability.{self.format}'))
        plt.close()
        
        # Compute scalability factors (how time increases with problem size)
        scalability_stats = {}
        
        for solver in scalability_df['solver'].unique():
            solver_df = scalability_df[scalability_df['solver'] == solver]
            if len(solver_df) > 2:  # Need at least 3 points for meaningful regression
                # Log-log regression to estimate complexity
                x = np.log(solver_df['variables'])
                y = np.log(solver_df['time'])
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                scalability_stats[solver] = {
                    'complexity_factor': round(slope, 2),
                    'r_squared': round(r_value**2, 3),
                    'p_value': p_value
                }
        
        # Save scalability statistics
        with open(os.path.join(self.output_dir, f'scalability_stats.json'), 'w') as f:
            json.dump(scalability_stats, f, indent=2)
        
        print(f"Scalability analysis saved to {self.output_dir}/scalability_stats.json")

    def analyze_phase_transition(self):
        """Analyze solver performance around the phase transition"""
        if self.phase_df is None:
            print("No phase transition data available.")
            return
        
        # Success rate by clause-to-variable ratio
        plt.figure(figsize=(12, 8))
        grouped = self.phase_df.groupby(['ratio', 'solver'])['solved'].mean().reset_index()
        
        ax = sns.lineplot(x='ratio', y='solved', hue='solver', marker='o', data=grouped)
        ax.set_title("Phase Transition Analysis: Success Rate")
        ax.set_xlabel("Clause-to-Variable Ratio")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.05)
        
        # Add vertical line at theoretical phase transition (~4.2)
        plt.axvline(x=4.2, color='red', linestyle='--', alpha=0.7, label="Theoretical Phase Transition")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'phase_transition_success.{self.format}'))
        plt.close()
        
        # Time to solution by ratio (for solved problems)
        solved_phase = self.phase_df[self.phase_df['solved'] == True]
        
        if not solved_phase.empty:
            plt.figure(figsize=(12, 8))
            avg_times = solved_phase.groupby(['ratio', 'solver'])['time'].mean().reset_index()
            
            ax = sns.lineplot(x='ratio', y='time', hue='solver', marker='o', data=avg_times)
            ax.set_title("Phase Transition Analysis: Time to Solution")
            ax.set_xlabel("Clause-to-Variable Ratio")
            ax.set_ylabel("Average Time (seconds)")
            ax.set_yscale('log')
            
            plt.axvline(x=4.2, color='red', linestyle='--', alpha=0.7)
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'phase_transition_time.{self.format}'))
            plt.close()
            
            # Additional analysis: Time complexity growth rate near phase transition
            # This helps visualize how solver difficulty increases approaching the phase transition
            plt.figure(figsize=(12, 8))
            
            # Define regions around phase transition
            pre_transition = (3.5, 4.0)  # Before transition
            transition = (4.0, 4.5)      # Transition zone
            post_transition = (4.5, 5.0) # After transition
            
            for solver in avg_times['solver'].unique():
                solver_data = avg_times[avg_times['solver'] == solver]
                
                # Compute growth rates in different regions
                growth_rates = {}
                for region_name, (lower, upper) in zip(
                    ['pre-transition', 'transition', 'post-transition'],
                    [pre_transition, transition, post_transition]
                ):
                    region_data = solver_data[
                        (solver_data['ratio'] >= lower) & 
                        (solver_data['ratio'] <= upper)
                    ]
                    
                    if len(region_data) > 1:
                        # Compute slope in log space
                        x = region_data['ratio']
                        y = np.log(region_data['time'])
                        if len(x) > 1:  # Need at least 2 points for regression
                            slope, _, _, _, _ = stats.linregress(x, y)
                            growth_rates[region_name] = slope
                
                # Plot growth rate visualization if we have enough data
                if growth_rates:
                    bar_positions = np.arange(len(growth_rates))
                    plt.bar(
                        bar_positions + 0.1 * list(avg_times['solver'].unique()).index(solver),
                        list(growth_rates.values()),
                        width=0.1,
                        label=solver
                    )
            
            if growth_rates:  # Only if we have data to show
                plt.title("Time Complexity Growth Rate Near Phase Transition")
                plt.xlabel("Region")
                plt.ylabel("Growth Rate (higher = more difficult)")
                plt.xticks(np.arange(3), ['Pre-transition\n(3.5-4.0)', 'Transition\n(4.0-4.5)', 'Post-transition\n(4.5-5.0)'])
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'phase_transition_growth_rate.{self.format}'))
                plt.close()
        
        # Find the precise phase transition point based on results
        # (where success rate drops most dramatically)
        pt_analysis = {}
        
        for solver in grouped['solver'].unique():
            solver_data = grouped[grouped['solver'] == solver].sort_values('ratio')
            
            if len(solver_data) > 1:
                # Calculate rate of change in success rate
                solver_data['success_change'] = solver_data['solved'].diff() / solver_data['ratio'].diff()
                
                # Find the ratio with the steepest drop
                min_idx = solver_data['success_change'].argmin()
                if min_idx > 0:  # Ensure we have a valid index
                    pt_ratio = solver_data.iloc[min_idx]['ratio']
                    pt_drop = solver_data.iloc[min_idx]['success_change']
                    
                    pt_analysis[solver] = {
                        'phase_transition_ratio': round(pt_ratio, 2),
                        'success_rate_drop': round(pt_drop, 3)
                    }
                    
        # Analyze specific behaviors near the phase transition
        if not solved_phase.empty:
            # Extract learning metrics if available in the data
            learning_metrics = [col for col in self.phase_df.columns 
                               if col in ['q_table_size', 'exploration_rate', 'q_value_variance', 
                                         'oracle_consultations', 'gan_generations_used']]
            
            if learning_metrics:
                plt.figure(figsize=(15, 10))
                
                # Create subplots for each learning metric
                for i, metric in enumerate(learning_metrics):
                    if metric in self.phase_df.columns:
                        plt.subplot(2, (len(learning_metrics)+1)//2, i+1)
                        
                        # Group by ratio and solver, then calculate mean
                        metric_data = self.phase_df.groupby(['ratio', 'solver'])[metric].mean().reset_index()
                        
                        # Plot the metric
                        sns.lineplot(x='ratio', y=metric, hue='solver', marker='o', data=metric_data)
                        
                        plt.title(f"{metric.replace('_', ' ').title()} vs. Ratio")
                        plt.xlabel("Clause-to-Variable Ratio")
                        plt.axvline(x=4.2, color='red', linestyle='--', alpha=0.5)
                        
                        # Use log scale for certain metrics that might have large variations
                        if metric in ['q_table_size', 'oracle_consultations', 'gan_generations_used']:
                            plt.yscale('log')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'phase_transition_learning_metrics.{self.format}'))
                plt.close()

    def run_statistical_tests(self):
        """Run statistical tests to compare solver performance"""
        # Filter for solved problems
        solved_df = self.results_df[self.results_df['solved'] == True].copy()
        
        if solved_df.empty or len(solved_df['solver'].unique()) < 2:
            print("Not enough data for statistical comparison.")
            return
        
        # Group problems by category for fair comparison
        # Use both problem name and size as grouping factors
        solved_df['problem_group'] = solved_df.apply(
            lambda x: f"{x['problem']}_{x['variables']}_{x['clauses']}", 
            axis=1
        )
        
        # Collect statistics
        stats_results = {
            'time_comparison': {},
            'memory_comparison': {},
            'pairwise_time': defaultdict(dict),
            'pairwise_memory': defaultdict(dict)
        }
        
        # Get list of solvers
        solvers = solved_df['solver'].unique()
        
        # Overall ANOVA for time
        if len(solvers) > 2:
            # Prepare data for ANOVA
            solver_groups = [solved_df[solved_df['solver'] == s]['time'] for s in solvers]
            try:
                f_val, p_val = stats.f_oneway(*solver_groups)
                stats_results['time_comparison']['anova'] = {
                    'f_statistic': float(f_val),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                }
            except:
                print("Could not perform ANOVA on time data.")
        
        # Overall ANOVA for memory
        if len(solvers) > 2:
            # Prepare data for ANOVA
            solver_groups = [solved_df[solved_df['solver'] == s]['memory'] for s in solvers]
            try:
                f_val, p_val = stats.f_oneway(*solver_groups)
                stats_results['memory_comparison']['anova'] = {
                    'f_statistic': float(f_val),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                }
            except:
                print("Could not perform ANOVA on memory data.")
        
        # Pairwise t-tests
        for i, solver1 in enumerate(solvers):
            for solver2 in solvers[i+1:]:
                # Find problems solved by both solvers
                common_problems = set(
                    solved_df[solved_df['solver'] == solver1]['problem_group']
                ).intersection(
                    set(solved_df[solved_df['solver'] == solver2]['problem_group'])
                )
                
                if len(common_problems) > 1:
                    # Get times for common problems
                    times1 = []
                    times2 = []
                    mems1 = []
                    mems2 = []
                    
                    for prob in common_problems:
                        t1 = float(solved_df[(solved_df['solver'] == solver1) & 
                                          (solved_df['problem_group'] == prob)]['time'].iloc[0])
                        t2 = float(solved_df[(solved_df['solver'] == solver2) & 
                                          (solved_df['problem_group'] == prob)]['time'].iloc[0])
                        
                        m1 = float(solved_df[(solved_df['solver'] == solver1) & 
                                          (solved_df['problem_group'] == prob)]['memory'].iloc[0])
                        m2 = float(solved_df[(solved_df['solver'] == solver2) & 
                                          (solved_df['problem_group'] == prob)]['memory'].iloc[0])
                        
                        times1.append(t1)
                        times2.append(t2)
                        mems1.append(m1)
                        mems2.append(m2)
                    
                    # Paired t-test for time
                    t_stat, p_val = stats.ttest_rel(times1, times2)
                    mean_diff = np.mean(np.array(times1) - np.array(times2))
                    
                    stats_results['pairwise_time'][solver1][solver2] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'mean_diff': float(mean_diff),
                        'faster': solver1 if mean_diff < 0 else solver2,
                        'common_problems': len(common_problems)
                    }
                    
                    # Paired t-test for memory
                    t_stat, p_val = stats.ttest_rel(mems1, mems2)
                    mean_diff = np.mean(np.array(mems1) - np.array(mems2))
                    
                    stats_results['pairwise_memory'][solver1][solver2] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'mean_diff': float(mean_diff),
                        'less_memory': solver1 if mean_diff < 0 else solver2,
                        'common_problems': len(common_problems)
                    }
                    
        # Save statistical test results
        with open(os.path.join(self.output_dir, f'statistical_tests.json'), 'w') as f:
            json.dump(stats_results, f, indent=2)
        
        # Create pairwise comparison summary table
        summary_rows = []
        
        for solver1 in solvers:
            for solver2 in solvers:
                if solver1 != solver2:
                    if solver2 in stats_results['pairwise_time'].get(solver1, {}) or \
                       solver1 in stats_results['pairwise_time'].get(solver2, {}):
                        
                        # Find the comparison (could be in either order)
                        if solver2 in stats_results['pairwise_time'].get(solver1, {}):
                            comp = stats_results['pairwise_time'][solver1][solver2]
                            faster = comp['faster']
                            p_val = comp['p_value']
                            mean_diff = comp['mean_diff']
                            common = comp['common_problems']
                        else:
                            comp = stats_results['pairwise_time'][solver2][solver1]
                            faster = comp['faster']
                            p_val = comp['p_value']
                            mean_diff = -comp['mean_diff']  # Flip the difference
                            common = comp['common_problems']
                        
                        # Create row
                        summary_rows.append({
                            'solver1': solver1,
                            'solver2': solver2,
                            'faster': faster,
                            'time_diff': abs(mean_diff),
                            'p_value': p_val,
                            'significant': p_val < 0.05,
                            'common_problems': common
                        })
        
        # Convert to DataFrame and save
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(os.path.join(self.output_dir, f'pairwise_comparison.csv'), index=False)
        
        print(f"Statistical tests saved to {self.output_dir}/statistical_tests.json")

    def create_summary_dashboard(self):
        """Create a summary dashboard visualization"""
        plt.figure(figsize=(15, 10))
        
        # 1. Success Rate
        plt.subplot(2, 2, 1)
        success_by_solver = self.results_df.groupby('solver')['solved'].mean()
        ax = success_by_solver.plot(kind='bar')
        ax.set_title("Success Rate")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        
        # 2. Average Time (for solved problems)
        plt.subplot(2, 2, 2)
        solved_df = self.results_df[self.results_df['solved'] == True]
        if not solved_df.empty:
            time_by_solver = solved_df.groupby('solver')['time'].mean()
            ax = time_by_solver.plot(kind='bar')
            ax.set_title("Average Time (seconds)")
            ax.set_yscale('log')
            plt.xticks(rotation=45)
        
        # 3. Average Memory
        plt.subplot(2, 2, 3)
        memory_by_solver = self.results_df.groupby('solver')['memory'].mean()
        ax = memory_by_solver.plot(kind='bar')
        ax.set_title("Average Memory (MB)")
        ax.set_yscale('log')
        plt.xticks(rotation=45)
        
        # 4. Phase transition point (if available)
        plt.subplot(2, 2, 4)
        if self.phase_df is not None:
            grouped = self.phase_df.groupby(['ratio', 'solver'])['solved'].mean().reset_index()
            ax = sns.lineplot(x='ratio', y='solved', hue='solver', marker='o', data=grouped)
            ax.set_title("Phase Transition")
            ax.set_xlabel("Clause-to-Variable Ratio")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            plt.axvline(x=4.2, color='red', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, "Phase transition data not available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'summary_dashboard.{self.format}'))
        plt.close()
        
        print(f"Summary dashboard saved to {self.output_dir}/summary_dashboard.{self.format}")

def main():
    args = parser.parse_args()
    analyzer = BenchmarkAnalyzer(
        args.results_file, 
        args.output_dir, 
        args.phase_file, 
        args.format
    )
    analyzer.run_analysis()
    print("Analysis complete!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()