"""
WalkSAT solver implementation using the unified solver interface.
"""

import logging
import random
import time
from typing import Any

from .base import SolverBase, SolverResult, SolverStatus
from .config import get_config
from .registry import register_solver

# Set up logging
logger = logging.getLogger(__name__)


@register_solver("walksat")
class WalkSATSolver(SolverBase):
    """
    WalkSAT algorithm implementation for SAT solving.
    """

    def __init__(self, num_vars: int = None, **kwargs):
        """
        Initialize the WalkSAT solver.

        Args:
            num_vars: Number of variables in the problem
            **kwargs: Additional configuration parameters
        """
        # Get default config
        config = get_config()

        # Set defaults from configuration
        self.num_vars = num_vars or config.get("problem.num_vars", 20)
        self.max_flips = config.get("solver.max_iterations", 100000)
        self.max_tries = config.get("solver.max_tries", 10)
        self.random_probability = config.get("solver.walksat.random_probability", 0.5)
        self.default_timeout = config.get("solver.timeout", 30.0)

        # Override defaults with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize state
        self.clauses = []
        self.solution = None
        self.satisfied = False
        self.solve_time = 0.0
        self.flip_count = 0
        self.try_count = 0
        self.interrupted = False

        # Statistics
        self.stats = {
            "flips": 0,
            "tries": 0,
            "satisfied_clauses": 0,
            "total_clauses": 0,
            "solver_name": "walksat",
        }

    def add_clause(self, clause: list[int]) -> None:
        """
        Add a single clause to the solver.

        Args:
            clause: A list of integers representing literals in the clause.
        """
        self.clauses.append(clause)
        self.stats["total_clauses"] = len(self.clauses)

    def add_clauses(self, clauses: list[list[int]]) -> None:
        """
        Add multiple clauses to the solver.

        Args:
            clauses: A list of clauses, where each clause is a list of integers.
        """
        self.clauses.extend(clauses)
        self.stats["total_clauses"] = len(self.clauses)

    def _count_satisfied_clauses(self, assignment: list[int]) -> int:
        """
        Count how many clauses are satisfied by an assignment.

        Args:
            assignment: List of literals (positive for True, negative for False)

        Returns:
            Number of satisfied clauses
        """
        count = 0
        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if (lit > 0 and assignment[var_idx] > 0) or (
                    lit < 0 and assignment[var_idx] < 0
                ):
                    satisfied = True
                    break
            if satisfied:
                count += 1
        return count

    def _get_unsatisfied_clauses(self, assignment: list[int]) -> list[int]:
        """
        Get indices of unsatisfied clauses.

        Args:
            assignment: List of literals

        Returns:
            List of clause indices
        """
        unsatisfied = []
        for i, clause in enumerate(self.clauses):
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if (lit > 0 and assignment[var_idx] > 0) or (
                    lit < 0 and assignment[var_idx] < 0
                ):
                    satisfied = True
                    break
            if not satisfied:
                unsatisfied.append(i)
        return unsatisfied

    def _flip_variable(self, assignment: list[int], var_idx: int) -> list[int]:
        """
        Flip the value of a variable in an assignment.

        Args:
            assignment: List of literals
            var_idx: Variable index to flip (0-based)

        Returns:
            New assignment with the variable flipped
        """
        new_assignment = assignment.copy()
        new_assignment[var_idx] = -new_assignment[var_idx]
        return new_assignment

    def solve(
        self, assumptions: list[int] | None = None, timeout: float | None = None
    ) -> SolverResult:
        """
        Solve the SAT instance using WalkSAT algorithm.

        Args:
            assumptions: Optional list of literal assumptions (not used in WalkSAT)
            timeout: Optional timeout in seconds

        Returns:
            SolverResult containing the solution status and other information
        """
        if assumptions is not None:
            logger.warning("WalkSAT does not support assumptions - ignoring")

        timeout = timeout or self.default_timeout
        start_time = time.time()

        self.try_count = 0
        self.flip_count = 0
        self.interrupted = False
        best_assignment = None
        best_satisfied = 0

        for _ in range(self.max_tries):
            if self.interrupted:
                break

            self.try_count += 1

            # Start with a random assignment
            assignment = [random.choice([-1, 1]) for _ in range(self.num_vars)]

            # Apply assumptions if provided
            if assumptions:
                for lit in assumptions:
                    var_idx = abs(lit) - 1
                    if var_idx < len(assignment):
                        assignment[var_idx] = 1 if lit > 0 else -1

            for _ in range(self.max_flips):
                if self.interrupted:
                    break

                self.flip_count += 1

                # Check if we've found a solution or timed out
                satisfied = self._count_satisfied_clauses(assignment)
                if satisfied > best_satisfied:
                    best_satisfied = satisfied
                    best_assignment = assignment.copy()

                if satisfied == len(self.clauses):
                    self.solution = assignment
                    self.satisfied = True
                    self.solve_time = time.time() - start_time

                    # Update statistics
                    self.stats["flips"] = self.flip_count
                    self.stats["tries"] = self.try_count
                    self.stats["satisfied_clauses"] = satisfied

                    return SolverResult(
                        status=SolverStatus.SATISFIABLE,
                        solution=[
                            i + 1 if lit > 0 else -(i + 1)
                            for i, lit in enumerate(assignment)
                        ],
                        runtime=self.solve_time,
                        satisfied_clauses=satisfied,
                        total_clauses=len(self.clauses),
                        statistics=self.stats,
                    )

                # Check timeout
                if time.time() - start_time > timeout:
                    break

                # Find unsatisfied clauses
                unsatisfied_clauses = self._get_unsatisfied_clauses(assignment)
                if not unsatisfied_clauses:
                    # This shouldn't happen if satisfied < len(clauses), but just in case
                    continue

                # Choose a random unsatisfied clause
                clause_idx = random.choice(unsatisfied_clauses)
                clause = self.clauses[clause_idx]

                # Either flip a random variable or the best variable in the clause
                if random.random() < self.random_probability:
                    # Choose a random variable from the clause
                    lit = random.choice(clause)
                    var_idx = abs(lit) - 1
                    assignment = self._flip_variable(assignment, var_idx)
                else:
                    # Choose the variable that maximizes satisfied clauses
                    best_var_idx = None
                    best_improvement = -1

                    for lit in clause:
                        var_idx = abs(lit) - 1
                        new_assignment = self._flip_variable(assignment, var_idx)
                        new_satisfied = self._count_satisfied_clauses(new_assignment)
                        improvement = new_satisfied - satisfied

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_var_idx = var_idx

                    if best_var_idx is not None:
                        assignment = self._flip_variable(assignment, best_var_idx)

            # Check timeout between tries
            if time.time() - start_time > timeout:
                break

        # If we didn't find a complete solution, store the best partial solution
        self.solution = best_assignment
        self.satisfied = False
        self.solve_time = time.time() - start_time

        # Update statistics
        self.stats["flips"] = self.flip_count
        self.stats["tries"] = self.try_count
        self.stats["satisfied_clauses"] = best_satisfied

        if self.interrupted:
            status = SolverStatus.ERROR
            error_msg = "Solving was interrupted"
        elif time.time() - start_time >= timeout:
            status = SolverStatus.TIMEOUT
            error_msg = f"Timeout reached ({timeout}s)"
        else:
            status = SolverStatus.UNKNOWN
            error_msg = "Maximum tries reached without finding a solution"

        return SolverResult(
            status=status,
            solution=(
                [
                    i + 1 if lit > 0 else -(i + 1)
                    for i, lit in enumerate(best_assignment)
                ]
                if best_assignment
                else None
            ),
            runtime=self.solve_time,
            satisfied_clauses=best_satisfied,
            total_clauses=len(self.clauses),
            statistics=self.stats,
            error_message=error_msg,
        )

    def get_model(self) -> list[int] | None:
        """
        Get the satisfying assignment if one exists.

        Returns:
            List of literals representing the satisfying assignment,
            or None if problem is unsatisfiable or no solution found yet
        """
        if self.solution is None:
            return None

        # Convert to standard format with 1-based indices
        return [i + 1 if lit > 0 else -(i + 1) for i, lit in enumerate(self.solution)]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get solver statistics.

        Returns:
            Dictionary of statistics
        """
        # Update the runtime in case it was called during solving
        if not self.satisfied and self.stats.get("runtime", 0) == 0:
            self.stats["runtime"] = self.solve_time

        return self.stats

    def interrupt(self) -> None:
        """
        Interrupt the solving process.
        """
        logger.debug("Interrupting WalkSAT solver")
        self.interrupted = True

    def configure(self, config: dict[str, Any]) -> None:
        """
        Configure the solver with the given parameters.

        Args:
            config: Dictionary of configuration parameters
        """
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Set {key}={value} for WalkSAT solver")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
