"""
Simple script to run SAT environment benchmarks.
"""
import csv
import random
import sys
import time

import numpy as np

# Add the project root to the path for imports
sys.path.append(".")


# Use a local random SAT generator since symbolicgym.domains.sat.utils does not provide one
def generate_random_ksat(n_vars, clause_ratio=4.2, k=3, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    n_clauses = int(n_vars * clause_ratio)
    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = random.sample(range(1, n_vars + 1), min(k, n_vars))
        clause = [v if random.random() > 0.5 else -v for v in vars_in_clause]
        clauses.append(clause)
    return {"num_vars": n_vars, "clauses": clauses}


# Import the SAT environment from symbolicgym (update to correct class if needed)
from symbolicgym.domains.sat import utils as sat_utils

# Placeholder: from symbolicgym.domains.sat.env import SymbolicSatEnv
SymbolicSatEnv = None  # TODO: Replace with actual class


def benchmark_size(size):
    """Run benchmark for a specific problem size."""
    print(f"Problem size: {size} variables")

    # Generate a random SAT formula
    formula = generate_random_ksat(n_vars=size, clause_ratio=4.2, k=3, seed=42)
    num_clauses = len(formula["clauses"])
    print(f"Generated {num_clauses} clauses")

    # Create environment
    env = SymbolicSatEnv(formula=formula)

    # Measure reset time
    start_time = time.time()
    env.reset()
    reset_time = time.time() - start_time
    print(f"Reset time: {reset_time:.6f} seconds")

    # Measure step time over multiple steps
    total_steps = 20
    total_step_time = 0

    for _ in range(total_steps):
        start_time = time.time()
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        step_time = time.time() - start_time
        total_step_time += step_time

        if terminated or truncated:
            break

    avg_step_time = total_step_time / total_steps
    print(f"Average step time: {avg_step_time:.6f} seconds")
    print(
        f"Estimated episode time (50 steps): {reset_time + 50 * avg_step_time:.6f} seconds"
    )
    print("-" * 50)

    return {
        "size": size,
        "clauses": num_clauses,
        "reset_time": reset_time,
        "avg_step_time": avg_step_time,
    }


if __name__ == "__main__":
    print("Running benchmarks with various problem sizes...")

    sizes = [10, 20, 50, 100, 200]
    results = []

    try:
        for size in sizes:
            result = benchmark_size(size)
            results.append(result)

        # Save results to CSV
        with open("benchmark_results.csv", "w", newline="") as csvfile:
            fieldnames = ["size", "clauses", "reset_time", "avg_step_time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Benchmark results saved to benchmark_results.csv")

    except Exception as e:
        import traceback

        print(f"Error during benchmarking: {e}")
        traceback.print_exc()
