#!/usr/bin/env python
"""
Command-line interface for benchmarking storage backends.

This script provides a convenient way to run performance benchmarks
on different storage backends for experience replay data.

Usage:
    python benchmark_cli.py run --backends npz hdf5 parquet --sizes 10000 100000 --output ./benchmark_results
    python benchmark_cli.py compare --input ./benchmark_results
"""

import argparse
import os
import sys
from pathlib import Path

from satrlgym.utils.performance_benchmark import StorageBenchmark, run_benchmark

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(args: argparse.Namespace) -> None:
    """
    Run benchmarks on specified storage backends.

    Args:
        args: Command-line arguments
    """
    print(f"Running benchmarks on backends: {args.backends}")

    # Set output directory
    output_dir = args.output or os.path.join(os.getcwd(), "benchmark_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Results will be saved to: {output_dir}")

    # Run benchmarks
    results = run_benchmark(
        backend_types=args.backends, sizes=args.sizes, output_dir=output_dir
    )

    # Print summary
    print("\nBenchmark completed successfully!")
    print(f"Summary report: {results['summary_path']}")

    for size in args.sizes:
        key = f"{size}_transitions"
        if key in results:
            print(f"\nResults for {size} transitions:")
            print(f"  - CSV data: {results[key]['csv_path']}")
            print(f"  - Plot: {results[key]['plot_path']}")


def compare_command(args: argparse.Namespace) -> None:
    """
    Compare previously run benchmarks or generate custom plots.

    Args:
        args: Command-line arguments
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load results
    input_dir = args.input
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found.")
        return

    # Find all CSV files
    csv_files = [
        f
        for f in os.listdir(input_dir)
        if f.startswith("benchmark_results_") and f.endswith(".csv")
    ]
    if not csv_files:
        print(f"Error: No benchmark result CSVs found in {input_dir}")
        return

    print(f"Found {len(csv_files)} result files to compare")

    # Load all results
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(input_dir, csv_file))
        # Extract size from filename
        size = int(csv_file.split("_")[2])
        df["dataset_size"] = size
        dataframes.append(df)

    # Combine all data
    combined_df = pd.concat(dataframes)

    # Create comparison plots
    benchmark = StorageBenchmark()

    # Plot by size
    sizes = combined_df["dataset_size"].unique()
    backends = combined_df["storage_type"].unique()

    for size in sizes:
        size_df = combined_df[combined_df["dataset_size"] == size]
        plot_path = os.path.join(input_dir, f"comparison_plot_{size}_transitions.png")
        benchmark.plot_comparison_results(size_df, output_file=plot_path)
        print(f"Generated comparison plot for {size} transitions: {plot_path}")

    # Plot scaling behavior
    plt.figure(figsize=(15, 10))

    # Write throughput scaling
    plt.subplot(2, 2, 1)
    for backend in backends:
        backend_df = combined_df[
            (combined_df["storage_type"] == backend)
            & (combined_df["benchmark_type"] == "write")
        ]
        if not backend_df.empty:
            plt.plot(
                backend_df["dataset_size"],
                backend_df["throughput_transitions_per_second"],
                marker="o",
                label=backend,
            )
    plt.title("Write Throughput Scaling")
    plt.xlabel("Dataset Size (transitions)")
    plt.ylabel("Throughput (transitions/second)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Read throughput scaling
    plt.subplot(2, 2, 2)
    for backend in backends:
        backend_df = combined_df[
            (combined_df["storage_type"] == backend)
            & (combined_df["benchmark_type"] == "read_sequential")
        ]
        if not backend_df.empty:
            plt.plot(
                backend_df["dataset_size"],
                backend_df["throughput_transitions_per_second"],
                marker="o",
                label=backend,
            )
    plt.title("Sequential Read Throughput Scaling")
    plt.xlabel("Dataset Size (transitions)")
    plt.ylabel("Throughput (transitions/second)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Storage efficiency scaling
    plt.subplot(2, 2, 3)
    for backend in backends:
        backend_df = combined_df[
            (combined_df["storage_type"] == backend)
            & (combined_df["benchmark_type"] == "write")
        ]
        if not backend_df.empty:
            plt.plot(
                backend_df["dataset_size"],
                backend_df["storage_efficiency_bytes_per_transition"],
                marker="o",
                label=backend,
            )
    plt.title("Storage Efficiency Scaling")
    plt.xlabel("Dataset Size (transitions)")
    plt.ylabel("Bytes per Transition")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Memory overhead scaling
    plt.subplot(2, 2, 4)
    for backend in backends:
        backend_df = combined_df[
            (combined_df["storage_type"] == backend)
            & (combined_df["benchmark_type"] == "memory")
        ]
        if not backend_df.empty:
            plt.plot(
                backend_df["dataset_size"],
                backend_df["memory_overhead_mb"],
                marker="o",
                label=backend,
            )
    plt.title("Memory Overhead Scaling")
    plt.xlabel("Dataset Size (transitions)")
    plt.ylabel("Memory Overhead (MB)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    scaling_plot_path = os.path.join(input_dir, "scaling_comparison.png")
    plt.savefig(scaling_plot_path, dpi=300, bbox_inches="tight")
    print(f"Generated scaling comparison plot: {scaling_plot_path}")

    # Save the combined data
    combined_csv_path = os.path.join(input_dir, "combined_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined results to: {combined_csv_path}")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark storage backends for experience replay data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Run benchmarks on specified storage backends"
    )
    run_parser.add_argument(
        "--backends",
        nargs="+",
        default=["npz", "hdf5"],
        help="Storage backends to benchmark",
    )
    run_parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10000, 100000],
        help="Dataset sizes to test in transitions",
    )
    run_parser.add_argument(
        "--output", type=str, default=None, help="Directory to save benchmark results"
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare previously run benchmarks"
    )
    compare_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing benchmark results",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "compare":
        compare_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
