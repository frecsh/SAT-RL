"""
Reporting utilities for SymbolicGym: save results, generate summaries, dashboard hooks
"""
import os

import pandas as pd


def save_run_results(results, path):
    """Save results for each run (success/failure, steps, key metrics) to CSV/JSON."""
    df = pd.DataFrame(results)
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".json") or path.endswith(".jsonl"):
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unknown file extension for results: {path}")


def generate_quick_summary(results_path):
    """Print or return a quick summary of results (success rate, avg steps, etc.)."""
    df = pd.read_csv(results_path)
    summary = {
        "success_rate": df["solved"].mean() if "solved" in df else None,
        "avg_steps": df["steps"].mean() if "steps" in df else None,
        # ...add more as needed...
    }
    print("Quick Summary:", summary)
    return summary


# (Optional) Add dashboard integration hooks here
