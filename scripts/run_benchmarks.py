"""
Script to run benchmarks for the SAT environment with different problem sizes.
"""
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path for imports
sys.path.append(".")

# Replace SatGymEnv with SymbolicSatEnv from symbolicgym.domains.sat (update as needed)
# from symbolicgym.domains.sat.env import SymbolicSatEnv
SymbolicSatEnv = None  # TODO: Replace with actual class


# Define our own generator function since the import path is not working
def generate_random_ksat(n_vars, clause_ratio=4.2, k=3, seed=None):
    """Generate a random k-SAT problem.

    Args:
        n_vars: Number of variables
        clause_ratio: Clause to variable ratio
        k: Number of literals per clause
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing the SAT formula
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Calculate number of clauses based on ratio
    n_clauses = int(n_vars * clause_ratio)

    # Generate random clauses
    clauses = []
    for _ in range(n_clauses):
        # Select k distinct variables for this clause
        vars_in_clause = random.sample(range(1, n_vars + 1), min(k, n_vars))

        # Randomly negate some variables
        clause = []
        for var in vars_in_clause:
            if random.random() < 0.5:
                clause.append(-var)
            else:
                clause.append(var)
        clauses.append(clause)

    return {
        "clauses": clauses,
        "num_vars": n_vars,
        "name": f"random_{n_vars}_{n_clauses}",
    }


def benchmark_problem_size(size, num_steps=50, clause_ratio=4.2, k=3, seed=42):
    """Benchmark environment performance with a specific problem size."""
    print(f"Problem size: {size} variables")

    # Generate a random SAT formula
    formula = generate_random_ksat(
        n_vars=size, clause_ratio=clause_ratio, k=k, seed=seed
    )
    num_clauses = len(formula["clauses"])
    print(f"Generated {num_clauses} clauses")

    # Create environment
    env = SymbolicSatEnv(formula=formula, max_steps=num_steps)

    # Measure reset time
    start_time = time.time()
    obs, _ = env.reset()
    reset_time = time.time() - start_time
    print(f"Reset time: {reset_time:.6f} seconds")

    # Measure step time
    total_step_time = 0
    step_times = []
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < num_steps:
        start_time = time.time()
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        step_time = time.time() - start_time

        total_step_time += step_time
        step_times.append(step_time)
        steps += 1

    avg_step_time = total_step_time / steps if steps > 0 else 0
    print(f"Average step time: {avg_step_time:.6f} seconds")
    print(f"Total steps: {steps}")
    print(
        f"Estimated episode time (50 steps): {reset_time + 50 * avg_step_time:.6f} seconds"
    )
    print("-" * 50)

    return {
        "problem_size": size,
        "num_vars": size,
        "num_clauses": num_clauses,
        "reset_time": reset_time,
        "avg_step_time": avg_step_time,
        "step_times": step_times,
        "total_steps": steps,
    }


def run_benchmarks():
    """Run benchmarks for different problem sizes and plot results."""
    # Problem sizes to test
    problem_sizes = [10, 20, 50, 100, 200, 500]
    results = []

    for size in problem_sizes:
        try:
            result = benchmark_problem_size(size)
            results.append(result)
        except Exception as e:
            print(f"Error with size {size}: {e}")
            break

    # Extract data for plotting
    sizes = [r["problem_size"] for r in results]
    reset_times = [r["reset_time"] for r in results]
    step_times = [r["avg_step_time"] for r in results]

    # Create plots
    plt.figure(figsize=(12, 8))

    # Plot reset times
    plt.subplot(2, 1, 1)
    plt.plot(sizes, reset_times, "o-", label="Reset Time")
    plt.ylabel("Time (seconds)")
    plt.title("Environment Reset Time Scaling")
    plt.grid(True)
    plt.xlabel("Problem Size (number of variables)")

    # Plot step times
    plt.subplot(2, 1, 2)
    plt.plot(sizes, step_times, "o-", label="Avg Step Time")
    plt.xlabel("Problem Size (number of variables)")
    plt.ylabel("Time (seconds)")
    plt.title("Environment Step Time Scaling")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("environment_scaling.png")
    print(f"Saved scaling plots to 'environment_scaling.png'")

    # Save CSV data
    with open("environment_scaling.csv", "w") as f:
        f.write("problem_size,reset_time,avg_step_time\n")
        for r in results:
            f.write(f"{r['problem_size']},{r['reset_time']},{r['avg_step_time']}\n")

    print(f"Saved scaling data to 'environment_scaling.csv'")


if __name__ == "__main__":
    print("Running environment scaling benchmarks...")
    try:
        # Test with just one problem size first
        print("Testing with a single problem size (10)...")
        benchmark_problem_size(10)

        # If successful, run the full benchmark
        print("\nRunning full benchmarks...")
        run_benchmarks()
    except Exception as e:
        import traceback

        print(f"Error running benchmarks: {e}")
        traceback.print_exc()
