"""
Evaluation script for SymbolicGym: batch experiments, metrics aggregation, and baseline automation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

import numpy as np

from src.symbolicgym.utils import logging as logging_utils
from src.symbolicgym.utils import seed as seed_utils
from symbolicgym.envs.sat_env import SatEnv
from symbolicgym.utils.logging import log_metrics_csv
from symbolicgym.utils.sat_generator import load_cnf
from symbolicgym.utils.seed import set_global_seed


def run_baseline(env, agent_type="random", max_steps=100):
    obs, info = env.reset()
    done = False
    steps = 0
    solved = False
    while not done and steps < max_steps:
        if agent_type == "random":
            action = env.action_space.sample()
        elif agent_type == "greedy":
            # Simple greedy: flip variable with most unsatisfied clauses
            clause_sat = obs["clause_satisfaction"]
            unsat_vars = np.where(clause_sat == 0)[0]
            action = unsat_vars[0] if len(unsat_vars) > 0 else env.action_space.sample()
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")
        obs, reward, terminated, truncated, info, *_ = env.step(action)
        done = terminated or truncated
        solved = info.get("solved", False)
        steps += 1
    return {"solved": solved, "steps": steps, "reward": reward}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances", nargs="+", required=True, help="List of CNF files to evaluate."
    )
    parser.add_argument("--agent", choices=["random", "greedy"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_results.csv")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()
    # Set global seed for reproducibility
    seed_utils.set_global_seed(args.seed)
    # Determine output directory and results file
    out_path = Path(args.out)
    if out_path.suffix == ".csv" or (out_path.exists() and out_path.is_file()):
        results_file = out_path
        output_dir = out_path.parent
    else:
        output_dir = out_path
        results_file = output_dir / "sat_benchmark_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Log config and code version
    logging_utils.log_experiment_metadata(vars(args), args.seed, output_dir)
    results = []
    for cnf_file in args.instances:
        formula = load_cnf(cnf_file)
        env = SatEnv(formula=formula)
        metrics = run_baseline(env, agent_type=args.agent, max_steps=args.max_steps)
        metrics["instance"] = os.path.basename(cnf_file)
        results.append(metrics)
        print(f"{cnf_file}: {metrics}")
    log_metrics_csv(results_file, results[0], fieldnames=results[0].keys())
    for m in results[1:]:
        log_metrics_csv(results_file, m, fieldnames=results[0].keys())
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
