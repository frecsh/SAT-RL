"""
Base interface for all SAT solvers in the framework.
Defines the standardized solver interface that all solver implementations must follow.
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class SolverStatus(Enum):
    """Enum representing the status of a solver run."""

    UNKNOWN = "unknown"
    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    TIMEOUT = "timeout"
    ERROR = "error"


class SolverResult:
    """
    Standardized result object returned by all solvers.
    """

    def __init__(
        self,
        status: SolverStatus = SolverStatus.UNKNOWN,
        solution: list[int] | None = None,
        runtime: float = 0.0,
        satisfied_clauses: int = 0,
        total_clauses: int = 0,
        statistics: dict[str, Any] | None = None,
        unsatisfiable_core: list[int] | None = None,
        error_message: str | None = None,
    ):
        self.status = status
        self.solution = solution
        self.runtime = runtime
        self.satisfied_clauses = satisfied_clauses
        self.total_clauses = total_clauses
        self.statistics = statistics or {}
        self.unsatisfiable_core = unsatisfiable_core
        self.error_message = error_message

    @property
    def is_sat(self) -> bool:
        """Returns True if the problem is satisfiable."""
        return self.status == SolverStatus.SATISFIABLE

    @property
    def is_unsat(self) -> bool:
        """Returns True if the problem is unsatisfiable."""
        return self.status == SolverStatus.UNSATISFIABLE

    @property
    def satisfaction_ratio(self) -> float:
        """Returns the ratio of satisfied clauses."""
        if self.total_clauses == 0:
            return 0.0
        return self.satisfied_clauses / self.total_clauses

    def __str__(self) -> str:
        """String representation of the result."""
        status_str = str(self.status.value).upper()
        if self.status == SolverStatus.SATISFIABLE:
            return f"SAT Result: {status_str} ({self.satisfied_clauses}/{self.total_clauses} clauses, {self.runtime:.4f}s)"
        elif self.status == SolverStatus.UNSATISFIABLE:
            return f"SAT Result: {status_str} (proved in {self.runtime:.4f}s)"
        elif self.status == SolverStatus.TIMEOUT:
            return f"SAT Result: {status_str} ({self.satisfied_clauses}/{self.total_clauses} clauses, {self.runtime:.4f}s)"
        elif self.status == SolverStatus.ERROR:
            return f"SAT Result: ERROR ({self.error_message})"
        else:
            return f"SAT Result: UNKNOWN"


class SolverBase(ABC):
    """
    Abstract base class for SAT solver implementations.
    All solver implementations must inherit from this class.
    """

    @abstractmethod
    def add_clause(self, clause: list[int]) -> None:
        """
        Add a single clause to the solver.

        Args:
            clause: A list of integers representing literals in the clause.
                   Positive integers represent positive literals, negative integers
                   represent negative literals.
        """

    @abstractmethod
    def add_clauses(self, clauses: list[list[int]]) -> None:
        """
        Add multiple clauses to the solver.

        Args:
            clauses: A list of clauses, where each clause is a list of integers.
        """

    @abstractmethod
    def solve(
        self, assumptions: list[int] | None = None, timeout: float | None = None
    ) -> SolverResult:
        """
        Attempt to solve the SAT instance.

        Args:
            assumptions: Optional list of literal assumptions
            timeout: Optional timeout in seconds

        Returns:
            SolverResult containing the solution status and other information
        """

    @abstractmethod
    def get_model(self) -> list[int] | None:
        """
        Get the satisfying assignment if one exists.

        Returns:
            List of literals representing the satisfying assignment,
            or None if problem is unsatisfiable
        """

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """
        Get solver statistics.

        Returns:
            Dictionary of statistics
        """

    @abstractmethod
    def interrupt(self) -> None:
        """
        Interrupt the solving process.
        """

    @abstractmethod
    def configure(self, config: dict[str, Any]) -> None:
        """
        Configure the solver with the given parameters.

        Args:
            config: Dictionary of configuration parameters
        """

    def solve_with_timeout(
        self, timeout: float, assumptions: list[int] | None = None
    ) -> SolverResult:
        """
        Solve with a timeout. Default implementation uses the solve method.

        Args:
            timeout: Timeout in seconds
            assumptions: Optional list of literal assumptions

        Returns:
            SolverResult
        """
        start_time = time.time()
        result = self.solve(assumptions=assumptions, timeout=timeout)
        end_time = time.time()

        # Update runtime in case the solver didn't track it
        result.runtime = end_time - start_time

        # Check if we timed out
        if end_time - start_time >= timeout:
            result.status = SolverStatus.TIMEOUT

        return result
