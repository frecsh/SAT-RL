"""
Automated Experiment Runner for SymbolicGym
- Loads experiment config (YAML/JSON)
- Sets global seed, logs config/code version
- Runs batch/parallel experiments (agents x envs x seeds)
- Stores logs/results in results/ and logs/
- Optionally integrates with SLURM/cloud batch jobs
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.symbolicgym.utils import logging as logging_utils
from src.symbolicgym.utils import seed as seed_utils

# TODO: Add more imports as needed (e.g., agent/env loaders)


def run_experiment(config, run_id, output_dir):
    """Run a single experiment instance."""
    # Set global seed
    seed = config.get("seed", 42) + run_id
    seed_utils.set_global_seed(seed)
    # Log config and code version
    logging_utils.log_experiment_metadata(config, seed, output_dir)
    # TODO: Load SAT instances, agent, environment
    # TODO: Run experiment, save logs/results
    pass


def main():
    parser = argparse.ArgumentParser(
        description="SymbolicGym Automated Experiment Runner"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config YAML/JSON"
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="Number of runs (different seeds)"
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of parallel workers"
    )
    parser.add_argument(
        "--output", type=str, default="results/", help="Output directory"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments (batch/parallel)
    pool = mp.Pool(args.parallel)
    jobs = []
    for run_id in range(args.n_runs):
        jobs.append(pool.apply_async(run_experiment, (config, run_id, output_dir)))
    for job in jobs:
        job.get()
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
