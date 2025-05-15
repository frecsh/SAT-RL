"""
Profile the SymbolicGym environment to identify performance bottlenecks.
"""
import cProfile
import os
import pstats
import random
import sys
import time
from contextlib import contextmanager

import numpy as np

# Add the project root to the path for imports
sys.path.append(".")

# Replace SatGymEnv with SymbolicSatEnv from symbolicgym.domains.sat (update as needed)
SymbolicSatEnv = None  # TODO: Replace with actual class


@contextmanager
def timer(name="Operation"):
    """Simple context manager to time operations."""
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"{name} took {elapsed_time:.6f} seconds")


def generate_random_ksat(n_vars, clause_ratio=4.2, k=3, seed=None):
    """Generate a random k-SAT problem."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_clauses = int(n_vars * clause_ratio)

    clauses = []
    for _ in range(n_clauses):
        vars_in_clause = random.sample(range(1, n_vars + 1), min(k, n_vars))

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


def profile_env_initialization(formula):
    """Profile environment initialization."""
    with timer("Environment initialization"):
        cProfile.runctx(
            "SymbolicSatEnv(formula=formula)", globals(), locals(), "env_init.prof"
        )

    init_stats = pstats.Stats("env_init.prof")
    init_stats.sort_stats("cumulative").print_stats(20)


def profile_env_reset(env):
    """Profile environment reset."""
    with timer("Environment reset"):
        cProfile.runctx("env.reset(seed=42)", globals(), locals(), "env_reset.prof")

    reset_stats = pstats.Stats("env_reset.prof")
    reset_stats.sort_stats("cumulative").print_stats(20)


def profile_env_step(env, num_steps=50):
    """Profile environment steps."""
    env.reset(seed=42)

    with timer(f"{num_steps} environment steps"):
        cProfile.runctx(
            "for _ in range(num_steps): env.step(env.action_space.sample())",
            globals(),
            locals(),
            "env_step.prof",
        )

    step_stats = pstats.Stats("env_step.prof")
    step_stats.sort_stats("cumulative").print_stats(20)


def profile_specific_functions(formula):
    """Profile specific functions like compute_satisfied_clauses."""
    # TODO: Replace with actual function from symbolicgym or user-defined utility
    compute_satisfied_clauses = None  # Placeholder for user update

    # First, create a random assignment
    assignment = {}
    for var in range(1, formula["num_vars"] + 1):
        assignment[var] = random.choice([True, False])

    clauses = formula["clauses"]

    with timer("compute_satisfied_clauses"):
        cProfile.runctx(
            "compute_satisfied_clauses(clauses, assignment)",
            globals(),
            locals(),
            "satisfaction.prof",
        )

    satisfaction_stats = pstats.Stats("satisfaction.prof")
    satisfaction_stats.sort_stats("cumulative").print_stats(20)


def run_profiling(sizes=[100, 500, 1000]):
    """Run profiling tests on different problem sizes."""
    print(f"{'=' * 20} SAT Environment Profiling {'=' * 20}")

    for size in sizes:
        print(f"\n\n{'=' * 20} Problem Size: {size} variables {'=' * 20}\n")

        # Generate a formula
        formula = generate_random_ksat(n_vars=size, seed=42)
        print(
            f"Generated formula with {size} variables and {len(formula['clauses'])} clauses"
        )

        # Profile environment creation
        print(f"\n{'-' * 40} Profiling Environment Initialization {'-' * 40}")
        profile_env_initialization(formula)

        # Create environment for other tests
        env = SymbolicSatEnv(formula=formula)

        # Profile environment reset
        print(f"\n{'-' * 40} Profiling Environment Reset {'-' * 40}")
        profile_env_reset(env)

        # Profile environment step
        print(f"\n{'-' * 40} Profiling Environment Steps {'-' * 40}")
        profile_env_step(env)

        # Profile specific functions
        print(f"\n{'-' * 40} Profiling Satisfaction Computation {'-' * 40}")
        profile_specific_functions(formula)

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Run profiling
    try:
        if len(sys.argv) > 1:
            sizes = [int(arg) for arg in sys.argv[1:]]
            run_profiling(sizes)
        else:
            run_profiling()
    except Exception as e:
        import traceback

        print(f"Error during profiling: {e}")
        traceback.print_exc()
