"""
Base oracle implementation.

This module provides a base class for SAT oracles.
"""

from abc import abstractmethod
from typing import Any

from src.satrlgym.oracles.oracle_protocol import (
    OracleProtocol,
    OracleQuery,
    OracleResponse,
    QueryType,
)


class OracleBase(OracleProtocol):
    """
    Base class for SAT oracles.

    Attributes:
        clauses: List of clauses, each clause being a list of literals
        num_vars: Number of variables in the formula
        config: Configuration parameters for the oracle
    """

    def __init__(
        self,
        clauses: list[list[int]],
        num_vars: int,
        oracle_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the oracle.

        Args:
            clauses: List of clauses, each clause being a list of literals
            num_vars: Number of variables in the formula
            oracle_config: Configuration parameters for the oracle
        """
        self.clauses = clauses
        self.num_vars = num_vars
        self.config = oracle_config or {}

    def query(self, query_or_type, data=None):
        """
        Process a query and return a response.

        This method supports both the new OracleQuery interface and the
        legacy OracleProtocol interface for backward compatibility.

        Args:
            query_or_type: Either an OracleQuery object or a QueryType/string
            data: Optional data for the legacy interface

        Returns:
            An OracleResponse object that behaves like a dictionary for backwards compatibility
        """
        # Increment query count if we're a SimpleDPLLOracle (this allows tracking in tests)
        if hasattr(self, "stats") and "queries" in self.stats:
            self.stats["queries"] += 1

        # Handle legacy protocol interface
        if not isinstance(query_or_type, OracleQuery):
            # Create an OracleQuery from the legacy parameters
            query_type = query_or_type
            if isinstance(query_type, QueryType):
                query_type = query_type.value

            current_assignment = data.get("assignment", {}) if data else {}
            options = data.get("options", {}) if data else {}

            query = OracleQuery(
                query_type=query_type,
                current_assignment=current_assignment,
                options=options,
            )
        else:
            query = query_or_type

        # Validate the query
        query.validate()

        # Process the query
        if query.query_type == "next_variable":
            return self._query_next_variable(query)
        elif query.query_type == "evaluate_solution":
            return self._query_evaluate_solution(query)
        else:
            return OracleResponse(
                query_type=query.query_type,
                success=False,
                result=None,
                metrics={"error": "Unsupported query type"},
            )

    @abstractmethod
    def _query_next_variable(self, query: OracleQuery) -> OracleResponse:
        """
        Process a 'next_variable' query.

        Args:
            query: The query to process

        Returns:
            An OracleResponse object
        """

    def _query_evaluate_solution(self, query: OracleQuery) -> OracleResponse:
        """
        Process an 'evaluate_solution' query.

        Args:
            query: The query to process

        Returns:
            An OracleResponse object
        """
        from src.satrlgym.utils.cnf import check_solution, compute_satisfied_clauses

        assignment = query.current_assignment

        # Check if the solution is complete
        is_complete = len(assignment) == self.num_vars

        # Check if all clauses are satisfied
        is_satisfied = check_solution(self.clauses, assignment, partial=not is_complete)

        # Count satisfied clauses
        num_satisfied = compute_satisfied_clauses(self.clauses, assignment)

        return OracleResponse(
            query_type=query.query_type,
            success=True,
            result={
                "satisfied": is_satisfied is True,
                "complete": is_complete,
                "num_satisfied": num_satisfied,
                "total_clauses": len(self.clauses),
            },
            metrics={},
        )

    def reset(self):
        """
        Reset the oracle state.
        """
        pass  # Default implementation does nothing
