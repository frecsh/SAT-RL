"""Advanced metrics logging for SymbolicGym benchmarking and evaluation."""

import csv
import json
import logging
import os
from datetime import datetime

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    """Unified logging interface for SymbolicGym (wandb or stdout fallback)."""

    def __init__(self, project="SymbolicGym", use_wandb=False):
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(project=project)
        else:
            logging.basicConfig(level=logging.INFO)

    def log(self, data: dict):
        if self.use_wandb:
            wandb.log(data)
        else:
            logging.info(data)

    def close(self):
        if self.use_wandb:
            wandb.finish()


def log_metrics_csv(path, metrics, fieldnames=None):
    """Append a row of metrics to a CSV file."""
    fieldnames = fieldnames or list(metrics.keys())
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def log_metrics_jsonl(path, metrics):
    """Append a row of metrics to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def log_episode_metrics(path, episode_metrics, format="csv"):
    """Log a list of episode metrics (dicts) to file."""
    if format == "csv":
        for m in episode_metrics:
            log_metrics_csv(path, m)
    elif format == "jsonl":
        for m in episode_metrics:
            log_metrics_jsonl(path, m)


def log_coordination_events(path, events):
    """Log agent coordination/communication/conflict events."""
    with open(path, "a") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def visualize_coordination(events, out_path=None):
    """Visualize agent coordination/communication using matplotlib."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Example: plot communication frequency heatmap
    agents = sorted(
        set(e["sender"] for e in events) | set(e["receiver"] for e in events)
    )
    comm_matrix = np.zeros((len(agents), len(agents)))
    agent_idx = {a: i for i, a in enumerate(agents)}
    for e in events:
        comm_matrix[agent_idx[e["sender"]], agent_idx[e["receiver"]]] += 1
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        comm_matrix, annot=True, xticklabels=agents, yticklabels=agents, cmap="Blues"
    )
    plt.title("Agent Communication Frequency")
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()


def log_experiment_metadata(config, seed, output_dir):
    """Log experiment config, seed, and git commit hash for reproducibility."""
    import json
    import subprocess

    meta = {
        "config": config,
        "seed": seed,
        "timestamp": str(datetime.now()),
        "git_commit": None,
    }
    try:
        meta["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        meta["git_commit"] = "unknown"
    with open(os.path.join(output_dir, "experiment_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def log_advanced_metrics(metrics_dict, output_dir, filename="advanced_metrics.jsonl"):
    """Append advanced metrics (peak clause satisfaction, proof/core size, etc.) to a JSONL file."""
    import json

    with open(os.path.join(output_dir, filename), "a") as f:
        f.write(json.dumps(metrics_dict) + "\n")


def aggregate_results(results_dir):
    """Aggregate results from CSV/JSONL logs in results_dir."""
    import pandas as pd

    # Example: aggregate all CSVs in results_dir
    dfs = []
    for file in os.listdir(results_dir):
        if file.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(results_dir, file)))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def plot_performance_curves(df, output_dir, metric="solve_rate"):
    """Plot performance curves (e.g., solve rate over time/instances)."""
    import matplotlib.pyplot as plt

    plt.figure()
    df.groupby("step")[metric].mean().plot()
    plt.title(f"{metric} over steps")
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.savefig(os.path.join(output_dir, f"{metric}_curve.png"))
    plt.close()
