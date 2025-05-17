"""Benchmark script to compare the performance of optimized vs base environment.

This script measures the execution time of various operations in both the base
SymbolicSatEnv and the optimized OptimizedSymbolicSatEnv environments.
"""

import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from symbolicgym.domains.sat.env import OptimizedSymbolicSatEnv, SymbolicSatEnv


def generate_random_sat_formula(num_vars: int, num_clauses: int) -> dict[str, Any]:
    """Generate a random SAT formula.

    Args:
        num_vars: Number of variables in the formula
        num_clauses: Number of clauses in the formula

    Returns:
        A dictionary representing the SAT formula
    """
    variables = np.random.choice([True, False], size=num_vars)
    clauses = [
        {
            "literals": np.random.choice(
                range(num_vars), size=3, replace=False
            ).tolist(),
            "negated": np.random.choice([True, False], size=3).tolist(),
        }
        for _ in range(num_clauses)
    ]
    return {"variables": variables, "clauses": clauses}


def benchmark_reset(env: Any, num_trials: int = 100) -> float:
    """Benchmark the reset method of an environment.

    Args:
        env: Environment to benchmark
        num_trials: Number of reset operations to perform

    Returns:
        Average time per reset in milliseconds
    """
    start_time = time.time()
    for _ in range(num_trials):
        env.reset()
    total_time = time.time() - start_time
    return (total_time / num_trials) * 1000  # Convert to milliseconds


def benchmark_step(env: Any, num_steps: int = 1000) -> float:
    """Benchmark the step method of an environment.

    Args:
        env: Environment to benchmark
        num_steps: Number of step operations to perform

    Returns:
        Average time per step in milliseconds
    """
    env.reset()
    start_time = time.time()
    for _ in range(num_steps):
        action = np.random.randint(0, env.action_space.n)
        env.step(action)
    total_time = time.time() - start_time
    return (total_time / num_steps) * 1000  # Convert to milliseconds


def benchmark_environments(
    num_vars_list: list[int] = [10, 50, 100, 200, 500],
    clause_var_ratio: float = 4.3,
    num_trials: int = 5,
) -> pd.DataFrame:
    """Benchmark the base and optimized environments on various problem sizes.

    Args:
        num_vars_list: List of variable counts to test
        clause_var_ratio: Ratio of clauses to variables
        num_trials: Number of trials per problem size

    Returns:
        DataFrame with benchmark results
    """
    # Create results dictionary
    results = {
        "num_vars": [],
        "num_clauses": [],
        "base_reset_time": [],
        "opt_reset_time": [],
        "base_step_time": [],
        "opt_step_time": [],
        "reset_speedup": [],
        "step_speedup": [],
    }

    for num_vars in num_vars_list:
        # Define problem size
        num_clauses = int(num_vars * clause_var_ratio)

        # Average over multiple trials
        base_reset_times = []
        opt_reset_times = []
        base_step_times = []
        opt_step_times = []

        for _ in range(num_trials):
            # Generate a random formula
            formula = generate_random_sat_formula(
                num_vars=num_vars, num_clauses=num_clauses
            )

            # Create environments
            base_env = SymbolicSatEnv(formula=formula, max_steps=num_vars * 10)
            opt_env = OptimizedSymbolicSatEnv(formula=formula, max_steps=num_vars * 10)

            # Benchmark reset
            base_reset_time = benchmark_reset(base_env, num_trials=10)
            opt_reset_time = benchmark_reset(opt_env, num_trials=10)

            # Benchmark step
            base_step_time = benchmark_step(base_env, num_steps=100)
            opt_step_time = benchmark_step(opt_env, num_steps=100)

            # Store results
            base_reset_times.append(base_reset_time)
            opt_reset_times.append(opt_reset_time)
            base_step_times.append(base_step_time)
            opt_step_times.append(opt_step_time)

        # Calculate averages
        avg_base_reset = np.mean(base_reset_times)
        avg_opt_reset = np.mean(opt_reset_times)
        avg_base_step = np.mean(base_step_times)
        avg_opt_step = np.mean(opt_step_times)

        # Calculate speedups
        reset_speedup = avg_base_reset / avg_opt_reset if avg_opt_reset > 0 else 0
        step_speedup = avg_base_step / avg_opt_step if avg_opt_step > 0 else 0

        # Store results
        results["num_vars"].append(num_vars)
        results["num_clauses"].append(num_clauses)
        results["base_reset_time"].append(avg_base_reset)
        results["opt_reset_time"].append(avg_opt_reset)
        results["base_step_time"].append(avg_base_step)
        results["opt_step_time"].append(avg_opt_step)
        results["reset_speedup"].append(reset_speedup)
        results["step_speedup"].append(step_speedup)

        # Print progress
        print(f"Completed benchmarks for {num_vars} variables, {num_clauses} clauses")
        print(
            f"  Reset: Base={avg_base_reset:.2f}ms, Opt={avg_opt_reset:.2f}ms, Speedup={reset_speedup:.2f}x"
        )
        print(
            f"  Step: Base={avg_base_step:.2f}ms, Opt={avg_opt_step:.2f}ms, Speedup={step_speedup:.2f}x"
        )

    # Create DataFrame
    return pd.DataFrame(results)


def plot_results(results: pd.DataFrame, output_dir: str = None) -> None:
    """Plot benchmark results.

    Args:
        results: DataFrame with benchmark results
        output_dir: Directory to save plots (if None, plots are displayed)
    """
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create figure and axes for reset times
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot reset times
    ax.plot(
        results["num_vars"], results["base_reset_time"], "bo-", label="Base Environment"
    )
    ax.plot(
        results["num_vars"],
        results["opt_reset_time"],
        "go-",
        label="Optimized Environment",
    )

    # Set labels and title
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Reset Time (ms)")
    ax.set_title("Environment Reset Performance")
    ax.legend()
    ax.grid(True)

    # Save or show plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, "reset_performance.png"), dpi=300)
    else:
        plt.show()

    # Create figure and axes for step times
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot step times
    ax.plot(
        results["num_vars"], results["base_step_time"], "bo-", label="Base Environment"
    )
    ax.plot(
        results["num_vars"],
        results["opt_step_time"],
        "go-",
        label="Optimized Environment",
    )

    # Set labels and title
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Step Time (ms)")
    ax.set_title("Environment Step Performance")
    ax.legend()
    ax.grid(True)

    # Save or show plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, "step_performance.png"), dpi=300)
    else:
        plt.show()

    # Create figure and axes for speedups
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot speedups
    ax.plot(results["num_vars"], results["reset_speedup"], "ro-", label="Reset Speedup")
    ax.plot(results["num_vars"], results["step_speedup"], "mo-", label="Step Speedup")

    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)

    # Set labels and title
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Speedup Factor (x)")
    ax.set_title("Performance Speedup of Optimized vs. Base Environment")
    ax.legend()
    ax.grid(True)

    # Save or show plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, "speedup.png"), dpi=300)
    else:
        plt.show()


def verify_correctness(
    num_vars: int = 100, num_clauses: int = 430, num_steps: int = 1000
) -> bool:
    """Verify that the optimized environment produces the same results as the base environment.

    Args:
        num_vars: Number of variables in the SAT formula
        num_clauses: Number of clauses in the SAT formula
        num_steps: Number of steps to take in the environments

    Returns:
        True if the environments produce identical results, False otherwise
    """
    # Generate a random formula
    formula = generate_random_sat_formula(num_vars=num_vars, num_clauses=num_clauses)

    # Create environments
    base_env = SymbolicSatEnv(formula=formula, max_steps=num_steps)
    opt_env = OptimizedSymbolicSatEnv(formula=formula, max_steps=num_steps)

    # Reset environments with the same seed
    base_obs, base_info = base_env.reset(seed=42)
    opt_obs, opt_info = opt_env.reset(seed=42)

    # Check that observations match
    if not np.array_equal(base_obs["variables"], opt_obs["variables"]):
        print("Initial variables arrays don't match")
        return False

    if not np.array_equal(base_obs["clauses"], opt_obs["clauses"]):
        print("Initial clauses arrays don't match")
        return False

    # Take steps with the same actions
    for step in range(num_steps):
        # Generate the same action for both environments
        action = step % num_vars

        # Take step in base environment
        (
            base_obs,
            base_reward,
            base_terminated,
            base_truncated,
            base_info,
        ) = base_env.step(action)

        # Take step in optimized environment
        opt_obs, opt_reward, opt_terminated, opt_truncated, opt_info = opt_env.step(
            action
        )

        # Check that rewards match
        if base_reward != opt_reward:
            print(f"Rewards don't match at step {step}")
            print(f"  Base reward: {base_reward}")
            print(f"  Opt reward: {opt_reward}")
            return False

        # Check that termination flags match
        if base_terminated != opt_terminated or base_truncated != opt_truncated:
            print(f"Termination flags don't match at step {step}")
            return False

        # Check that observations match
        if not np.array_equal(base_obs["variables"], opt_obs["variables"]):
            print(f"Variables arrays don't match at step {step}")
            return False

        if not np.array_equal(base_obs["clauses"], opt_obs["clauses"]):
            print(f"Clauses arrays don't match at step {step}")
            return False

        # Check that satisfaction counts match
        if base_info["num_satisfied_clauses"] != opt_info["num_satisfied_clauses"]:
            print(f"Satisfaction counts don't match at step {step}")
            print(f"  Base: {base_info['num_satisfied_clauses']}")
            print(f"  Opt: {opt_info['num_satisfied_clauses']}")
            return False

        # Check for early termination
        if base_terminated or base_truncated:
            break

    return True


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Benchmark optimized SAT environment")
    parser.add_argument(
        "--verify", action="store_true", help="Run correctness verification"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory for plots"
    )
    parser.add_argument(
        "--small", action="store_true", help="Run a small benchmark (faster)"
    )
    args = parser.parse_args()

    # Run correctness verification if requested
    if args.verify:
        print("Verifying correctness...")
        if verify_correctness():
            print("Correctness verification passed!")
        else:
            print("Correctness verification failed!")
            sys.exit(1)

    # Run benchmarks
    print("Running benchmarks...")
    if args.small:
        # Small benchmark with fewer trials and smaller problems
        var_sizes = [10, 20, 50, 100]
        results = benchmark_environments(num_vars_list=var_sizes, num_trials=3)
    else:
        # Full benchmark
        var_sizes = [10, 50, 100, 200, 500, 1000]
        results = benchmark_environments(num_vars_list=var_sizes, num_trials=5)

    # Save results to CSV
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        results.to_csv(os.path.join(args.output, "benchmark_results.csv"), index=False)

    # Plot results
    plot_results(results, output_dir=args.output)
