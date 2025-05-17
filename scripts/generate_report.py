"""
Automated Reporting & Visualization for SymbolicGym
- Aggregates logs/results from results/ and logs/
- Generates summary tables (solve rates, avg steps, etc.)
- Produces plots (matplotlib/seaborn)
- Exports reports as Markdown/HTML/PDF
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.symbolicgym.utils import logging as logging_utils

# TODO: Add more imports as needed


def main():
    parser = argparse.ArgumentParser(description="SymbolicGym Report Generator")
    parser.add_argument(
        "--input", type=str, required=True, help="Input results directory"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output report directory"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate logs/results
    results_file = input_dir / "sat_benchmark_results.csv"
    if not results_file.exists():
        print(f"No results file found at {results_file}")
        return
    df = pd.read_csv(results_file)

    # Compute summary stats
    summary_lines = []
    summary_lines.append(f"# SymbolicGym Benchmark Summary\n")
    summary_lines.append(f"**Results file:** `{results_file}`\n")
    summary_lines.append(f"**Number of runs:** {len(df)}\n")

    def safe_markdown(stats):
        try:
            return stats.to_markdown()
        except ImportError:
            return str(stats)

    if "solve_rate" in df.columns:
        solve_rate_stats = df["solve_rate"].describe()
        summary_lines.append(f"**Solve Rate:**\n{safe_markdown(solve_rate_stats)}\n")
    if "steps" in df.columns:
        steps_stats = df["steps"].describe()
        summary_lines.append(f"**Steps:**\n{safe_markdown(steps_stats)}\n")
    # Add more metrics as needed

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    if "solve_rate" in df.columns:
        plt.figure()
        df["solve_rate"].plot(title="Solve Rate")
        plt.xlabel("Run")
        plt.ylabel("Solve Rate")
        plt.tight_layout()
        plt.savefig(plots_dir / "solve_rate.png")
        plt.close()
        summary_lines.append(f"![Solve Rate](plots/solve_rate.png)")
    if "steps" in df.columns:
        plt.figure()
        df["steps"].plot(title="Steps")
        plt.xlabel("Run")
        plt.ylabel("Steps")
        plt.tight_layout()
        plt.savefig(plots_dir / "steps.png")
        plt.close()
        summary_lines.append(f"![Steps](plots/steps.png)")

    # Export summary tables and plots
    summary_md = "\n".join(summary_lines)
    with open(output_dir / "summary.md", "w") as f:
        f.write(summary_md)
    print(f"Summary report written to {output_dir / 'summary.md'}")
    print(f"Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
