#!/usr/bin/env python3
"""
Script to download and prepare benchmark datasets for SAT solver comparisons.
"""

import os
import argparse
from sat_utils import (
    download_sat_benchmarks, 
    categorize_benchmarks, 
    create_benchmark_subset,
    generate_phase_transition_problems
)

parser = argparse.ArgumentParser(description='Download and prepare SAT benchmark datasets')
parser.add_argument('--output_dir', type=str, default='benchmarks',
                   help='Directory to save benchmark files')
parser.add_argument('--additional_sources', type=str, default=None,
                   help='Path to JSON file with additional benchmark sources')
parser.add_argument('--subset_size', type=int, default=10,
                   help='Number of problems per category in the subset')
parser.add_argument('--phase_transition', action='store_true',
                   help='Generate phase transition problems')
parser.add_argument('--variable_counts', type=str, default='50,100,200',
                   help='Comma-separated list of variable counts for phase transition problems')
parser.add_argument('--all', action='store_true',
                    help='Perform all benchmark preparation steps')

def main():
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load additional sources if provided
    additional_sources = None
    if args.additional_sources:
        import json
        with open(args.additional_sources, 'r') as f:
            additional_sources = json.load(f)
    
    # Download benchmarks
    if args.all or input("Download standard benchmark datasets? (y/n): ").lower() == 'y':
        download_sat_benchmarks(args.output_dir, additional_sources)
    
    # Categorize benchmarks
    if args.all or input("Categorize benchmarks by size and complexity? (y/n): ").lower() == 'y':
        categories = categorize_benchmarks(args.output_dir)
        print("Benchmark categories:")
        for category, problems in categories.items():
            print(f"  {category}: {len(problems)} problems")
    
    # Create benchmark subset
    if args.all or input("Create benchmark subset for quicker testing? (y/n): ").lower() == 'y':
        subset_dir = os.path.join(args.output_dir, 'subset')
        create_benchmark_subset(
            args.output_dir, 
            subset_dir, 
            count_per_category=args.subset_size
        )
        print(f"Created benchmark subset in {subset_dir}")
    
    # Generate phase transition problems
    if args.phase_transition or args.all or input("Generate phase transition problems? (y/n): ").lower() == 'y':
        phase_dir = os.path.join(args.output_dir, 'phase_transition')
        variable_counts = [int(v) for v in args.variable_counts.split(',')]
        generate_phase_transition_problems(
            phase_dir,
            variable_counts=variable_counts
        )
        print(f"Generated phase transition problems in {phase_dir}")
    
    print("\nBenchmark preparation complete!")
    print("To run benchmarks, use: python benchmark_comparison.py --benchmark_dir benchmarks/subset")
    print("For phase transition analysis: python benchmark_comparison.py --phase_transition")

if __name__ == "__main__":
    main()